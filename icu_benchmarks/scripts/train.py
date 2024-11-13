import logging
from time import perf_counter

import click
import gin
import mlflow
import numpy as np
import pandas as pd
import tabmat
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

from icu_benchmarks.gin import GeneralizedLinearRegressor
from icu_benchmarks.load import load
from icu_benchmarks.mlflow_utils import log_df, setup_mlflow

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@gin.configurable
def sources(sources=gin.REQUIRED):  # noqa D
    return sources


@gin.configurable
def outcome(outcome=gin.REQUIRED):  # noqa D
    return outcome


@gin.configurable
def targets(targets=gin.REQUIRED):  # noqa D
    return targets


@gin.configurable
def l1_ratios(l1_ratios=gin.REQUIRED):  # noqa D
    return l1_ratios


def metrics(y, yhat, prefix):  # noqa D
    return {
        f"{prefix}/roc": roc_auc_score(y, yhat) if np.unique(y).size > 1 else 0,
        f"{prefix}/accuracy": (
            accuracy_score(y, yhat >= 0.5) if np.unique(y).size > 1 else 0
        ),
        f"{prefix}/log_loss": log_loss(y, yhat) if np.unique(y).size > 1 else 0,
        f"{prefix}/average_prc": (
            average_precision_score(y, yhat) if np.unique(y).size > 1 else 0
        ),
        f"{prefix}/auprc": (
            auc(*precision_recall_curve(y, yhat)[1::-1]) if np.unique(y).size > 1 else 0
        ),
        f"{prefix}/brier": np.mean((y - yhat) ** 2) if np.unique(y).size > 1 else 0,
    }


@click.command()
@click.option("--config", type=click.Path(exists=True))
def main(config: str):  # noqa D
    gin.parse_config_file(config)

    tags = {
        "outcome": outcome(),
        "sources": sources(),
        "targets": targets(),
        "l1_ratios": l1_ratios(),
        "parent_run": None,
    }

    parent_run = setup_mlflow(tags=tags)

    tic = perf_counter()
    df, y, weights = load(sources=sources(), outcome=outcome(), split="train")
    toc = perf_counter()

    logger.info(f"Loading data ({df.shape}) took {toc - tic:.1f} seconds")

    continuous_variables = [col for col, dtype in df.schema.items() if dtype.is_float()]
    other = [col for col in df.columns if col not in continuous_variables]
    
    imputer = SimpleImputer(strategy="mean", copy=False, keep_empty_features=True)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous",
                imputer,
                continuous_variables,
            ),
            ("other", "passthrough", other),
        ],
        sparse_threshold=0,
        verbose=1,
    ).set_output(transform="polars")

    tic = perf_counter()
    df = tabmat.from_df(preprocessor.fit_transform(df))
    toc = perf_counter()
    logger.info(f"Preprocessing data took {toc - tic:.1f} seconds")

    glms = []
    for l1_ratio in l1_ratios():
        logger.info(f"Fitting the glm with l1_ratio={l1_ratio:.1f}")
        glm = GeneralizedLinearRegressor(l1_ratio=l1_ratio)
        tic = perf_counter()
        glm.fit(df, y, sample_weight=weights)
        toc = perf_counter()
        logger.info(
            f"Fitting the glm with l1_ratio={l1_ratio:.1f} took {toc - tic:.1f} seconds"
        )
        glms.append(glm)

    params = {
        "gradient_tol": glm.gradient_tol,
        "max_iter": glm.max_iter,
        "sources": ",".join(sources()),
    }
    mlflow.log_params(params)

    log_df(
        pd.DataFrame(
            {"stds": glm.col_stds_, "means": glm.col_means_},
            index=glm.feature_names_,
        ),
        "col_stats.csv",
    )

    runs = []
    for l1_ratio_idx, l1_ratio in enumerate(l1_ratios()):
        glm = glms[l1_ratio_idx]
        for alpha_idx, alpha in enumerate(glm._alphas):
            with mlflow.start_run(nested=True) as run:
                mlflow.log_params(
                    {
                        **params,
                        "alpha": alpha,
                        "log_alpha": np.log(alpha),
                        "alpha_idx": alpha_idx,
                        "sparsity": np.mean(glm.coef_path_[alpha_idx] == 0),
                        "l1_ratio": l1_ratio,
                        "l1_ratio_idx": l1_ratio_idx,
                        "l1_alpha": alpha * l1_ratio,
                        "l2_alpha": alpha * (1 - l1_ratio),
                    },
                    synchronous=False,
                )
                mlflow.set_tags({**tags, "parent_run": parent_run.info.run_id})
                mlflow.log_text(glm.coef_table().to_markdown(), "coefficients.md")
                log_df(glm.coef_table(), "coefficients.csv")

                runs.append(
                    {
                        "run_id": run.info.run_id,
                        "alpha": alpha,
                        "alpha_idx": alpha_idx,
                        "l1_ratio": l1_ratio,
                        "l1_ratio_idx": l1_ratio_idx,
                    }
                )

    for target in targets():
        for split in ["train", "val", "test"]:
            logger.info(f"Scoring on {target}/{split}")
            tic = perf_counter()
            df, y, _ = load(sources=[target], outcome=outcome(), split=split)
            toc = perf_counter()
            logger.info(f"Loading data ({df.shape}) took {toc - tic:.1f} seconds")

            if df.shape[0] == 0:
                logger.warning(f"No data for {target}/{split}")
                continue

            tic = perf_counter()
            df = tabmat.from_df(preprocessor.transform(df))
            toc = perf_counter()
            logger.info(f"Preprocessing data took {toc - tic:.1f} seconds")

            for run in runs:
                glm = glms[run["l1_ratio_idx"]]
                glm.intercept_ = glm.intercept_path_[run["alpha_idx"]]
                glm.coef_ = glm.coef_path_[run["alpha_idx"]]
                yhat = glm.predict(df)

                mlflow.log_metrics(
                    metrics(y, yhat, f"{target}/{split}"),
                    run_id=run["run_id"],
                    synchronous=False,
                )

    # This needs to be at the end of the script to log all relevant information
    mlflow.log_text(gin.operative_config_str(), "config.gin")


if __name__ == "__main__":
    main()

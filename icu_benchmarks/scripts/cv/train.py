import logging
from time import perf_counter

import click
import gin
import mlflow
import numpy as np
import polars as pl
import tabmat
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from icu_benchmarks.constants import TASKS
from icu_benchmarks.gin import GeneralizedLinearRegressor
from icu_benchmarks.load import load
from icu_benchmarks.metrics import metrics
from icu_benchmarks.mlflow_utils import log_df, log_pickle, setup_mlflow

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@gin.configurable
def outcome(outcome=gin.REQUIRED):  # noqa D
    return outcome


@gin.configurable
def sources(sources=gin.REQUIRED):  # noqa D
    return sources


@gin.configurable
def targets(targets=gin.REQUIRED):  # noqa D
    return targets


@gin.configurable
def l1_ratios(l1_ratios=gin.REQUIRED):  # noqa D
    return l1_ratios


@click.command()
@click.option("--config", type=click.Path(exists=True))
def main(config: str):  # noqa D
    gin.parse_config_file(config)
    task = TASKS[outcome()]
    tags = {"outcome": outcome(), "sources": sources(), "targets": targets()}
    _ = setup_mlflow(tags=tags)

    tic = perf_counter()
    df, y, weights, dataset = load(sources=sources(), outcome=outcome(), split="train")
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
        glm = GeneralizedLinearRegressor(l1_ratio=l1_ratio, family=task["family"])
        tic = perf_counter()
        glm.fit(df, y, sample_weight=weights)
        toc = perf_counter()
        logger.info(
            f"Fitting the glm with l1_ratio={l1_ratio:.1f} took {toc - tic:.1f} seconds"
        )
        glms.append(glm)

    results = []

    for l1_ratio_idx, l1_ratio in enumerate(l1_ratios()):
        glm = glms[l1_ratio_idx]
        for alpha_idx, alpha in enumerate(glm._alphas):
            glm.coef_ = glm.coef_path_[alpha_idx]
            glm.intercept_ = glm.intercept_path_[alpha_idx]
            coef_table_path = (
                f"coefficients/alpha_{alpha_idx}_l1_ratio_{l1_ratio_idx}.csv"
            )
            log_df(glm.coef_table(), coef_table_path)
            log_pickle(
                Pipeline([("preprocessor", preprocessor), ("glm", glm)]), "model.pickle"
            )

            results.append(
                {
                    "alpha": alpha,
                    "alpha_idx": alpha_idx,
                    "l1_ratio": l1_ratio,
                    "l1_ratio_idx": l1_ratio_idx,
                    "coef_table_path": coef_table_path,
                    "sparsity": np.mean(glm.coef_ == 0),
                }
            )

    for target in targets():
        for split in ["train", "val", "test"]:
            logger.info(f"Scoring on {target}/{split}")
            tic = perf_counter()
            df, y, _, _ = load(sources=[target], outcome=outcome(), split=split)
            toc = perf_counter()
            logger.info(f"Loading data ({df.shape}) took {toc - tic:.1f} seconds")

            if df.shape[0] == 0:
                logger.warning(f"No data for {target}/{split}")
                continue

            tic = perf_counter()
            df = tabmat.from_df(preprocessor.transform(df))
            toc = perf_counter()
            logger.info(f"Preprocessing data took {toc - tic:.1f} seconds")

            for result in results:
                glm = glms[result["l1_ratio_idx"]]
                glm.intercept_ = glm.intercept_path_[result["alpha_idx"]]
                glm.coef_ = glm.coef_path_[result["alpha_idx"]]
                yhat = glm.predict(df)
                result = {
                    **result,
                    **metrics(y, yhat, f"{target}/{split}", TASKS[outcome()]["task"]),
                }

    log_df(pl.DataFrame(results), "results.csv")
    # This needs to be at the end of the script to log all relevant information
    mlflow.log_text(gin.operative_config_str(), "config.gin")


if __name__ == "__main__":
    main()

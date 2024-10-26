import logging
from time import perf_counter

import click
import gin
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from icu_benchmarks.gin import GeneralizedLinearRegressor
from icu_benchmarks.load import load
from icu_benchmarks.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@gin.configurable
def sources(sources=None):  # noqa D
    return sources


@gin.configurable
def outcome(outcome=None):  # noqa D
    return outcome


@gin.configurable
def targets(targets=None):  # noqa D
    return targets


@gin.configurable
def model(model=None):  # noqa D
    return model


def metrics(y, yhat, prefix):  # noqa D
    return {
        f"{prefix}/roc": roc_auc_score(y, yhat),
        f"{prefix}/accuracy": accuracy_score(y, yhat >= 0.5),
        f"{prefix}/log_loss": log_loss(y, yhat),
        f"{prefix}/prc": average_precision_score(y, yhat),
    }


@click.command()
@click.option("--config", type=click.Path(exists=True))
def main(config: str):  # noqa D
    gin.parse_config_file(config)

    tags = {
        "outcome": outcome(),
        "sources": sources(),
        "targets": targets(),
        "parent_run": None,
    }

    parent_run = setup_mlflow(tags=tags)

    tic = perf_counter()
    df, y = load(sources=sources(), outcome=outcome(), split="train")
    toc = perf_counter()
    logger.info(f"Loading data took {toc - tic:.1f} seconds")

    continuous_variables = [col for col in df.columns if df.dtypes[col] == "float64"]
    other = [col for col in df.columns if df.dtypes[col] in ["category", "bool"]]

    imputer = SimpleImputer(strategy="mean", copy=False, keep_empty_features=True)
    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", imputer, continuous_variables),
            ("other", "passthrough", other),
        ]
    ).set_output(transform="pandas")

    glm = GeneralizedLinearRegressor()
    pipeline = Pipeline([("preprocessing", preprocessor), ("glm", glm)])

    tic = perf_counter()
    pipeline.fit(df, y)
    toc = perf_counter()
    logger.info(f"Fitting the pipeline took {toc - tic:.1f} seconds")
    mlflow.log_metric("fit_time", toc - tic)

    params = {
        "l1_ratio": glm.l1_ratio,
        "gradient_tol": glm.gradient_tol,
        "max_iter": glm.max_iter,
        "sources": ",".join(sources()),
    }
    mlflow.log_params(params)

    runs = []
    for alpha_idx, alpha in enumerate(glm._alphas):
        with mlflow.start_run(nested=True) as run:
            mlflow.log_params(
                {
                    **params,
                    "alpha": alpha,
                    "log_alpha": np.log(alpha),
                    "alpha_idx": alpha_idx,
                }
            )
            mlflow.set_tags({**tags, "parent_run": parent_run.info.run_id})

            runs.append(
                {
                    "run_id": run.info.run_id,
                    "alpha": alpha,
                    "idx": alpha_idx,
                }
            )

    for target in targets():
        splits = ["train", "val", "test"] if target in sources() else ["val", "test"]
        for split in splits:
            df, y = load(sources=[target], outcome=outcome(), split=split)

            if len(df) == 0:
                logger.warning(f"No data for {target}/{split}")
                continue

            df = pipeline[:-1].transform(df)

            for run in runs:
                # logger.info(f"Starting run {run['run_id']} for {target}/{split}")
                # with mlflow.start_run(run_id=run["run_id"], nested=True):
                pipeline[-1].intercept_ = pipeline[-1].intercept_path_[run["idx"]]
                pipeline[-1].coef_ = pipeline[-1].coef_path_[run["idx"]]
                yhat = pipeline[-1].predict(df)

                mlflow.log_metrics(
                    metrics(y, yhat, f"{target}/{split}"), run_id=run["run_id"]
                )

    # This needs to be at the end of the script to log all relevant information
    mlflow.log_text(gin.operative_config_str(), "config.gin")


if __name__ == "__main__":
    main()

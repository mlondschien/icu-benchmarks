import logging
from itertools import product
from time import perf_counter

import click
import gin
import mlflow
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from icu_benchmarks.constants import TASKS
from icu_benchmarks.load import load
from icu_benchmarks.metrics import metrics
from icu_benchmarks.mlflow_utils import log_df, log_pickle, setup_mlflow
from icu_benchmarks.models import AnchorRegression, DataSharedLasso  # noqa F401

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
def parameters(parameters=gin.REQUIRED):
    """If parameters is a dictionary, create list of records with all combinations."""
    if isinstance(parameters, dict):
        keys, values = parameters.keys(), parameters.values()
        return [dict(zip(keys, combination)) for combination in product(*values)]
    else:
        return parameters


@gin.configurable
def model(model=gin.REQUIRED):  # noqa D
    return model


@click.command()
@click.option("--config", type=click.Path(exists=True))
def main(config: str):  # noqa D
    gin.parse_config_file(config)
    task = TASKS[outcome()]
    tags = {
        "outcome": outcome(),
        "sources": sources(),
        "targets": targets(),
        "parameter_names": np.unique([k for p in parameters() for k in p.keys()]),
    }

    _ = setup_mlflow(tags=tags)

    tic = perf_counter()
    df, y, weights, dataset = load(sources=sources(), outcome=outcome(), split="train")
    toc = perf_counter()
    logger.info(f"Loading data ({df.shape}) took {toc - tic:.1f} seconds")

    continuous_variables = [col for col, dtype in df.schema.items() if dtype.is_float()]
    bool_variables = [col for col in df.columns if df[col].dtype == pl.Boolean]
    other = [
        col for col in df.columns if col not in continuous_variables + bool_variables
    ]

    scaler = SimpleImputer(strategy="mean", copy=False, keep_empty_features=True)
    imputer = StandardScaler(copy=False)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous",
                Pipeline([("impute", imputer), ("scale", scaler)]),
                continuous_variables,
            ),
            ("bool", "passthrough", bool_variables),
            (
                "other",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                other,
            ),
        ],
        sparse_threshold=0,
        verbose=1,
    ).set_output(transform="polars")

    tic = perf_counter()
    df = preprocessor.fit_transform(df)

    toc = perf_counter()
    logger.info(f"Preprocessing data took {toc - tic:.1f} seconds")

    glms = []
    for parameter in parameters():
        logger.info(f"Fitting the glm with {parameter}")
        glm = model()(family=task["family"])
        glm.set_params(**parameter)
        tic = perf_counter()

        glm.fit(df, y, sample_weight=weights, dataset=dataset)
        toc = perf_counter()
        logger.info(f"Fitting the glm with {parameter} took {toc - tic:.1f} seconds")
        glms.append(glm)

    results = []

    for parameter_idx, parameter in enumerate(parameters()):
        glm = glms[parameter_idx]
        for alpha_idx, alpha in enumerate(glm._alphas):
            glm.coef_ = glm.coef_path_[alpha_idx]
            glm.intercept_ = glm.intercept_path_[alpha_idx]
            suffix = "_".join(f"{key}={value}" for key, value in parameter.items())
            coef_table_path = f"coefficients/alpha={alpha_idx}_{suffix}.csv"
            log_df(glm.coef_table(), coef_table_path)
            log_pickle(
                Pipeline([("preprocessor", preprocessor), ("glm", glm)]),
                f"model_{suffix}.pkl",
            )

            results.append(
                {
                    **{
                        "alpha": alpha,
                        "alpha_idx": alpha_idx,
                        "coef_table_path": coef_table_path,
                        "sparsity": np.mean(glm.coef_ == 0),
                        "parameter_idx": parameter_idx,
                    },
                    **parameter,
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
            df = preprocessor.transform(df)
            toc = perf_counter()
            logger.info(f"Preprocessing data took {toc - tic:.1f} seconds")

            for result_idx, result in enumerate(results):
                glm = glms[result["parameter_idx"]]
                glm.intercept_ = glm.intercept_path_[result["alpha_idx"]]
                glm.coef_ = glm.coef_path_[result["alpha_idx"]]
                yhat = glm.predict(df)
                results[result_idx] = {
                    **result,
                    **metrics(y, yhat, f"{target}/{split}/", TASKS[outcome()]["task"]),
                }

    log_df(pl.DataFrame(results), "results.csv")
    # This needs to be at the end of the script to log all relevant information
    mlflow.log_text(gin.operative_config_str(), "config.gin")


if __name__ == "__main__":
    main()

import logging
import os
from itertools import product
from time import perf_counter

import click
import gin
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from icu_benchmarks.constants import TASKS
from icu_benchmarks.load import load
from icu_benchmarks.metrics import metrics
from icu_benchmarks.mlflow_utils import log_df, setup_mlflow
from icu_benchmarks.models import (  # noqa F401
    AnchorRegression,
    DataSharedLasso,
    EmpiricalBayesCV,
    PipelineCV,
)

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
def target(target=gin.REQUIRED):  # noqa D
    return target


@gin.configurable
def parameters(parameters=gin.REQUIRED):
    """If parameters is a dictionary, create list of records with all combinations."""
    if isinstance(parameters, dict):
        return ParameterGrid(parameters)
    else:
        return parameters


@gin.configurable
def n_samples(n_samples=gin.REQUIRED):  # noqa D
    return n_samples


@gin.configurable
def seeds(seeds=gin.REQUIRED):  # noqa D
    return seeds


@gin.configurable
def model(model=gin.REQUIRED):  # noqa D
    return model


@gin.configurable
def predict_kwargs(predict_kwargs=gin.REQUIRED):
    """If kwargs is a dictionary, create list of records with all combinations."""
    if isinstance(predict_kwargs, dict):
        return ParameterGrid(predict_kwargs)
    else:
        return predict_kwargs


@click.command()
@click.option("--config", type=click.Path(exists=True))
def main(config: str):  # noqa D
    gin.parse_config_file(config)

    task = TASKS[outcome()]["task"]
    tags = {
        "outcome": outcome(),
        "target": target(),
        "parameter_names": np.unique([k for p in parameters() for k in p.keys()]),
    }

    _ = setup_mlflow(tags=tags)

    tic = perf_counter()
    df, y, _, hashes = load(outcome=outcome(), split="train", other_columns=["hash"])
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

    hashes = hashes.sort()
    data = {}
    for seed in seeds():
        sampled_hashes = hashes.sample(max(n_samples()), seed=seed, shuffle=True)
        for n in n_samples():
            mask = hashes.is_in(sampled_hashes[:n])
            data[n, seed] = (
                df.filter(mask),
                y[mask],
                hashes.filter(mask),
            )

    df_test, y_test, _ = load(split="test", outcome=outcome())

    jobs = []
    for parameter_idx, parameter in enumerate(parameters()):
        glm = model()(**parameter)
        pipeline = PipelineCV(steps=[("preprocessor", preprocessor), ("model", glm)])
        details = {
            "model_idx": parameter_idx,
            **parameter,
        }
        jobs.append(
            delayed(_fit)(
                pipeline,
                data,
                df_test,
                y_test,
                n_samples(),
                seeds(),
                task,
                predict_kwargs(),
                details,
            )
        )

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    with Parallel(n_jobs=-1, prefer="processes") as parallel:
        parallel_results = parallel(jobs)

    del df, y

    results: list[dict] = sum(parallel_results, [])
    log_df(pl.DataFrame(results), "n_samples_results.csv")


def _fit(
    refit_model,
    data_train,
    df_test,
    y_test,
    n_samples,
    seeds,
    task,
    predict_kwargs,
    details,
):
    results = {}
    for n_sample in n_samples:
        results[n_sample] = {}
        for seed in seeds:
            results[n_sample][seed] = {}
            df, y, groups = data_train[n_sample, seed]
            yhat = refit_model.fit_predict_cv(
                df, y, groups=groups, predict_kwargs=predict_kwargs
            )
            results[n_sample][seed]["scores_cv"] = [
                metrics(y, yhat[:, idx], "", task) for idx in range(len(predict_kwargs))
            ]

            yhat_test = refit_model.predict_with_kwargs(
                df_test, predict_kwargs=predict_kwargs
            )
            results[n_sample][seed]["scores_test"] = [
                metrics(y_test, yhat_test[:, idx], "", task)
                for idx in range(len(predict_kwargs))
            ]

    out = [
        {
            **details,
            **{
                f"cv_{n}_{seed}/{k}": v
                for n, seed in product(n_samples, seeds)
                for k, v in results[n][seed]["scores_cv"][idx].items()
            },
            **{
                f"test_{n}_{seed}/{k}": v
                for n, seed in product(n_samples, seeds)
                for k, v in results[n][seed]["scores_test"][idx].items()
            },
            **predict_kwarg,
        }
        for idx, predict_kwarg in enumerate(predict_kwargs)
    ]
    return out


if __name__ == "__main__":
    main()

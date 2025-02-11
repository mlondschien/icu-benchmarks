import logging
import os
from time import perf_counter

import click
import gin
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid

from icu_benchmarks.constants import TASKS
from icu_benchmarks.load import load
from icu_benchmarks.metrics import metrics
from icu_benchmarks.mlflow_utils import log_df, setup_mlflow
from icu_benchmarks.models import PipelineCV
from icu_benchmarks.preprocessing import get_preprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@gin.configurable
def get_outcome(outcome=gin.REQUIRED):  # noqa D
    return outcome


@gin.configurable
def get_target(target=gin.REQUIRED):  # noqa D
    return target


@gin.configurable
def get_parameters(parameters=gin.REQUIRED):
    """If parameters is a dictionary, create list of records with all combinations."""
    if isinstance(parameters, dict):
        return ParameterGrid(parameters)
    else:
        return parameters


@gin.configurable
def get_n_samples(n_samples=gin.REQUIRED):  # noqa D
    return n_samples


@gin.configurable
def get_seeds(seeds=gin.REQUIRED):  # noqa D
    return seeds


@gin.configurable
def get_model(model=gin.REQUIRED):  # noqa D
    return model


@gin.configurable
def get_predict_kwargs(predict_kwargs=gin.REQUIRED):
    """If kwargs is a dictionary, create list of records with all combinations."""
    if isinstance(predict_kwargs, dict):
        return ParameterGrid(predict_kwargs)
    else:
        return predict_kwargs


@click.command()
@click.option("--config", type=click.Path(exists=True))
def main(config: str):  # noqa D
    gin.parse_config_file(config)

    task = TASKS[get_outcome()]["task"]
    tags = {
        "outcome": get_outcome(),
        "parameter_names": np.unique([k for p in get_parameters() for k in p.keys()]),
        "target": get_target(),
    }

    _ = setup_mlflow(tags=tags)

    tic = perf_counter()
    df, y, _, hashes = load(
        outcome=get_outcome(), split="train_val", other_columns=["stay_id_hash"]
    )
    toc = perf_counter()
    logger.info(f"Loading data ({df.shape}) took {toc - tic:.1f} seconds")

    preprocessor = get_preprocessing(get_model(), df)

    unique_hashes = hashes.unique().sort()
    data = {}
    n_samples = np.unique(np.clip(get_n_samples(), 0, len(unique_hashes)))
    for seed in get_seeds():
        sampled_hashes = unique_hashes.sample(max(n_samples), seed=seed, shuffle=True)
        for n in n_samples:
            mask = hashes.is_in(sampled_hashes[:n])
            data[n, seed] = (
                df.filter(mask),
                y[mask],
                hashes.filter(mask),
            )

    df_test, y_test, _ = load(split="test", outcome=get_outcome())

    jobs = []
    for parameter_idx, parameter in enumerate(get_parameters()):
        glm = get_model()(**parameter)
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
                task,
                get_predict_kwargs(),
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
    model,
    data_train,
    df_test,
    y_test,
    task,
    kwargs,
    details,
):
    results = []
    for (n_samples, seed), (df, y, groups) in data_train.items():
        yhat = model.fit_predict_cv(df, y, groups=groups, predict_kwargs=kwargs)
        cv_scores = [metrics(y, yhat[:, idx], "", task) for idx in range(len(kwargs))]

        yhat_test = model.predict_with_kwargs(df_test, predict_kwargs=kwargs)
        test_scores = [
            metrics(y_test, yhat_test[:, idx], "", task) for idx in range(len(kwargs))
        ]
        results += [
            {
                "n_target": n_samples,
                "seed": seed,
                "metric": metric,
                "cv_value": cv_scores[idx][metric],
                "test_value": test_scores[idx][metric],
                **kwarg,
                **details,
            }
            for metric in cv_scores[0].keys()
            for idx, kwarg in enumerate(kwargs)
        ]
    return results


if __name__ == "__main__":
    main()

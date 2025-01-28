import logging
import os
import pickle
import tempfile
from itertools import product
from pathlib import Path

import click
import gin
import polars as pl
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid

from icu_benchmarks.constants import TASKS
from icu_benchmarks.load import load
from icu_benchmarks.metrics import metrics
from icu_benchmarks.mlflow_utils import get_run, log_df
from icu_benchmarks.models import (  # noqa F401
    EmpiricalBayesCV,
    PriorPassthroughCV,
    RefitInterceptModelCV,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@gin.configurable
def refit_parameters(refit_parameters=gin.REQUIRED):
    """If parameters is a dictionary, create list of records with all combinations."""
    if isinstance(refit_parameters, dict):
        return ParameterGrid(refit_parameters)
    else:
        return refit_parameters


@gin.configurable
def predict_kwargs(predict_kwargs=gin.REQUIRED):
    """If kwargs is a dictionary, create list of records with all combinations."""
    if isinstance(predict_kwargs, dict):
        return ParameterGrid(predict_kwargs)
    else:
        return predict_kwargs


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
def name(name="refit"):  # noqa D
    return name


@click.command()
@click.option("--config", type=click.Path(exists=True))
def main(config: str):  # noqa D
    gin.parse_config_file(config)

    client, run = get_run()
    run_id = run.info.run_id

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(client.download_artifacts(run_id, "models", tmpdir))
        with open(model_dir / "preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        priors = []
        for model_idx in range(len(list(model_dir.glob("model_*.pkl")))):
            with open(model_dir / f"model_{model_idx}.pkl", "rb") as f:
                priors.append(pickle.load(f))

    outcome = run.data.tags["outcome"]

    df, y, _, hashes = load(split="train", outcome=outcome, other_columns=["hash"])
    df = preprocessor.transform(df)

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

    df_test, y_test, _ = load(split="test", outcome=outcome)
    df_test = preprocessor.transform(df_test)

    jobs = []
    for model_idx, prior in enumerate(priors):
        for refit_parameter in refit_parameters():
            refit_model = model()(prior=prior, **refit_parameter)
            details = {
                "model_idx": model_idx,
                **refit_parameter,
            }
            jobs.append(
                delayed(_refit)(
                    refit_model,
                    data,
                    df_test,
                    y_test,
                    n_samples(),
                    seeds(),
                    TASKS[outcome]["task"],
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
    log_df(pl.DataFrame(results), f"{name()}_results.csv", client=client, run_id=run_id)


def _refit(
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

    return [
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


if __name__ == "__main__":
    main()

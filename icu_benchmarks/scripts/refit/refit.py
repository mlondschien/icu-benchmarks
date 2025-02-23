import logging
import os
import pickle
import tempfile
from pathlib import Path

import click
import gin
import numpy as np
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
    RefitLGBMModelCV,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@gin.configurable
def get_refit_parameters(refit_parameters=gin.REQUIRED):
    """If parameters is a dictionary, create list of records with all combinations."""
    if isinstance(refit_parameters, dict):
        return ParameterGrid(refit_parameters)
    else:
        return refit_parameters


@gin.configurable
def get_predict_kwargs(predict_kwargs=gin.REQUIRED):
    """If kwargs is a dictionary, create list of records with all combinations."""
    if isinstance(predict_kwargs, dict):
        return ParameterGrid(predict_kwargs)
    else:
        return predict_kwargs


@gin.configurable
def get_n_target(n_target=gin.REQUIRED):  # noqa D
    return n_target


@gin.configurable
def get_seeds(seeds=gin.REQUIRED):  # noqa D
    return seeds


@gin.configurable
def get_model(model=gin.REQUIRED):  # noqa D
    return model


@gin.configurable
def get_name(name="refit"):  # noqa D
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

    df, y, _, hashes = load(
        split="train_val", outcome=outcome, other_columns=["stay_id_hash"]
    )
    df = preprocessor.transform(df)

    unique_hashes = hashes.unique().sort()
    data = {}
    n_target = np.unique(np.clip(get_n_target(), 0, len(unique_hashes)))
    for seed in get_seeds():
        sampled_hashes = unique_hashes.sample(max(n_target), seed=seed, shuffle=True)
        # For a single seed, the data increases monotonicly. We pass the data, including
        # y and "hashes" for group CV for the largest value of n_target. For smaller
        # values, the data can be reconstructed via a mask. This uses memory much more
        # efficiently than passing the training data for each value of n_target.
        mask = hashes.is_in(sampled_hashes)
        data[seed] = {
            "df": df.filter(mask),
            "y": y[mask],
            "hashes": hashes.filter(mask),
            "masks": {},
        }
        for n in n_target:
            data[seed]["masks"][n] = data[seed]["hashes"].is_in(sampled_hashes[:n])

    df_test, y_test, _ = load(split="test", outcome=outcome)
    df_test = preprocessor.transform(df_test)

    jobs = []
    for model_idx, prior in enumerate(priors):
        if get_name() in ["refit_linear", "refit_lgbm"]:
            value = getattr(prior, "gamma", None) or getattr(prior, "ratio", 1.0)
            if np.abs(np.log2(value) - np.round(np.log2(value))) > 0.1:
                continue

        for refit_parameter in get_refit_parameters():
            if get_name() == "refit_linear":
                value = refit_parameter["prior_alpha"]
                if np.abs(np.log10(value) - np.round(np.log10(value))) > 0.1:
                    continue

            for seed in get_seeds():
                model = get_model()(prior=prior, **refit_parameter)
                details = {
                    "model_idx": model_idx,
                    "seed": seed,
                    **refit_parameter,
                }
                jobs.append(
                    delayed(_fit)(
                        model,
                        data[seed],
                        df_test,
                        y_test,
                        TASKS[outcome]["task"],
                        get_predict_kwargs(),
                        details,
                    )
                )

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    logger.info(f"Number of jobs: {len(jobs)}")

    with Parallel(n_jobs=-1, prefer="processes") as parallel:
        parallel_results = parallel(jobs)

    del df, y

    results: list[dict] = sum(parallel_results, [])
    log_df(
        pl.DataFrame(results), f"{get_name()}_results.csv", client=client, run_id=run_id
    )


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
    for n_target, mask in data_train["masks"].items():
        df = data_train["df"].filter(mask)
        y = data_train["y"][mask]
        groups = data_train["hashes"].filter(mask)

        yhat = model.fit_predict_cv(df, y, groups=groups, predict_kwargs=kwargs)
        cv_scores = [metrics(y, yhat[:, idx], "", task) for idx in range(len(kwargs))]

        yhat_test = model.predict_with_kwargs(df_test, predict_kwargs=kwargs)
        test_scores = [
            metrics(y_test, yhat_test[:, idx], "", task) for idx in range(len(kwargs))
        ]
        results += [
            {
                "n_target": n_target,
                "metric": metric,
                "cv_value": cv_scores[idx][metric],
                "test_value": test_scores[idx][metric],
                **details,
                # kwarg needs to come after details. E.g., for linear, details has
                # "alpha": [1, 0.1, ...], and kwarg has "alpha": 1.
                **kwarg,
            }
            for metric in cv_scores[0].keys()
            for idx, kwarg in enumerate(kwargs)
        ]
    return results


if __name__ == "__main__":
    main()

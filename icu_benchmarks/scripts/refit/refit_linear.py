import logging
import os
import pickle
import tempfile
from itertools import product
from pathlib import Path
from time import perf_counter

import click
import gin
import polars as pl
import tabmat
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid

from icu_benchmarks.constants import TASKS
from icu_benchmarks.load import load
from icu_benchmarks.metrics import metrics
from icu_benchmarks.mlflow_utils import get_run, log_df
from icu_benchmarks.models import EmpiricalBayesRidgeCV

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
def refit_alphas(refit_alphas=gin.REQUIRED):  # noqa D
    return refit_alphas


@gin.configurable
def n_samples(n_samples=gin.REQUIRED):  # noqa D
    return n_samples


@gin.configurable
def seeds(seeds=gin.REQUIRED):  # noqa D
    return seeds


@click.command()
@click.option("--config", type=click.Path(exists=True))
@click.option("--n_cpus", type=int, default=None)
def main(config: str, n_cpus: int):  # noqa D
    gin.parse_config_file(config)

    client, run = get_run()
    run_id = run.info.run_id

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(client.download_artifacts(run_id, "models", tmpdir))
        with open(model_dir / "preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        models = []
        for model_idx in range(len(list(model_dir.glob("model_*.pkl")))):
            with open(model_dir / f"model_{model_idx}.pkl", "rb") as f:
                models.append(pickle.load(f))

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
                tabmat.from_df(df.filter(mask)),
                y[mask],
                hashes.filter(mask),
            )

    df_test, y_test, _ = load(split="test", outcome=outcome)
    df_test = preprocessor.transform(df_test)
    df_test_size = df_test.estimated_size("gb")
    df_test = tabmat.from_df(df_test, sparse_threshold=1)

    jobs = []
    for model_idx, model in enumerate(models):
        for alpha_idx in range(len(model.alpha)):
            model.coef_ = model.coef_path_[alpha_idx]
            model.intercept_ = model.intercept_path_[alpha_idx]
            for refit_parameter in refit_parameters():
                refit_model = EmpiricalBayesRidgeCV(
                    prior=model,
                    alpha=refit_alphas(),
                    alpha_search=True,
                    copy_X=False,
                    **refit_parameter,
                )
                details = {
                    "model_idx": model_idx,
                    "alpha_idx": alpha_idx,
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
                        details,
                    )
                )

    n_jobs = n_cpus // min(1, (6 + df_test_size) // 4)
    print(n_jobs)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    tic = perf_counter()

    with Parallel(n_jobs=n_jobs, prefer="processes") as parallel:
        refit_results = parallel(jobs)
    toc = perf_counter()
    print(toc - tic)

    del df, y

    results = []
    for refit_result in refit_results:
        # for n_sample in n_samples():
        #     for seed in seeds():
        # model = refit_result[n_sample][seed].pop("refit_model")
        # refit_result[n_sample][seed]["alphas"] = model._alphas

        # yhats = model.predict(df_test, alpha_index=range(len(model._alphas)))
        # yhats = [yhats[:, idx] for idx in range(yhats.shape[1])]
        # refit_result[n_sample][seed]["scores_test"] = [
        #     metrics(
        #         y_test,
        #         yhat,
        #         "",
        #         TASKS[outcome]["task"],
        #     )
        #     for yhat in yhats
        # ]

        results += [
            {
                "refit_alpha_idx": alpha_idx,
                "refit_alpha": alpha,
                **refit_result["details"],
                **{
                    f"cv_{n_sample}_{seed}/{k}": v
                    for n_sample, seed in product(n_samples(), seeds())
                    for k, v in refit_result[n_sample][seed]["scores_cv"][
                        alpha_idx
                    ].items()
                },
                **{
                    f"test_{n_sample}_{seed}/{k}": v
                    for n_sample, seed in product(n_samples(), seeds())
                    for k, v in refit_result[n_sample][seed]["scores_test"][
                        alpha_idx
                    ].items()
                },
            }
            for alpha_idx, alpha in enumerate(refit_alphas())
        ]

    log_df(pl.DataFrame(results), "refit_results.csv", client=client, run_id=run_id)


def _refit(refit_model, data_train, df_test, y_test, n_samples, seeds, task, details):
    results = {"details": details}
    for n_sample in n_samples:
        results[n_sample] = {}
        for seed in seeds:
            results[n_sample][seed] = {}
            df_train, y_train, groups = data_train[n_sample, seed]
            alpha_index = list(range(len(refit_model.alpha)))
            yhat = refit_model.refit_predict_cv(
                df_train, y_train, groups=groups, alpha_index=alpha_index
            )
            results[n_sample][seed]["scores_cv"] = [
                metrics(y_train, yhat[:, idx], "", task) for idx in alpha_index
            ]

            yhat_test = refit_model.predict(df_test, alpha_index=alpha_index)
            results[n_sample][seed]["scores_test"] = [
                metrics(y_test, yhat_test[:, idx], "", task) for idx in alpha_index
            ]

            results[n_sample][seed]["refit_alpha"] = refit_model._alphas
            # results[n_sample][seed]["scores_test"] = metrics(y_test, refit_model.predict(df_test), "", task)
    return results
    # return {
    #     "refit_model": refit_model,
    #     **details,
    #     **{
    #         f"cv_{n_sample}_{seed}/{k}": v
    #         for n_sample, seed in product(n_samples(), seeds())
    #         for k, v in results[n_sample][seed]["scores_cv"].items()
    #     }
    # }


if __name__ == "__main__":
    main()

import logging
import pickle
import tempfile
from itertools import product
from pathlib import Path

import click
import gin
import polars as pl
from sklearn.model_selection import ParameterGrid

from icu_benchmarks.constants import TASKS
from icu_benchmarks.load import load
from icu_benchmarks.metrics import metrics
from icu_benchmarks.mlflow_utils import get_run, log_df
from icu_benchmarks.models import RefitLGBMModelCV

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
def num_iterations(num_iterations=gin.REQUIRED):  # noqa D
    return num_iterations


@gin.configurable
def n_samples(n_samples=gin.REQUIRED):  # noqa D
    return n_samples


@gin.configurable
def seeds(seeds=gin.REQUIRED):  # noqa D
    return seeds


@gin.configurable
def outcome(outcome=gin.REQUIRED):  # noqa D
    return outcome


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

        models = []
        for model_idx in range(len(list(model_dir.glob("model_*.pkl")))):
            with open(model_dir / f"model_{model_idx}.pkl", "rb") as f:
                models.append(pickle.load(f))

    df, y, _, hashes = load(split="train", outcome=outcome(), other_columns=["hash"])
    df = preprocessor.transform(df)

    df_test, y_test, _ = load(split="test", outcome=outcome())
    df_test = preprocessor.transform(df_test)

    train_hashes = [
        hashes.sample(max(n_samples()), seed=seed, shuffle=True, with_replacement=False)
        for seed in seeds()
    ]

    refit_results: list[dict] = []
    for model_idx, model in enumerate(models):
        logger.info(f"Refitting model {model_idx}/{len(models)}")
        for refit_parameter in refit_parameters():
            logger.info(f"Refitting with {refit_parameter}")
            model_results: dict[str, dict[str, dict[str, list]]] = {}
            for n_sample in n_samples():
                model_results[n_sample] = {}
                for seed in seeds():
                    model_results[n_sample][seed] = {}
                    mask = hashes.is_in(train_hashes[seed][:n_sample])

                    df_train = df.filter(mask)
                    y_train = y[mask]
                    groups = hashes.filter(mask)

                    refit_model = RefitLGBMModelCV(prior=model, **refit_parameter)
                    yhats_cv = refit_model.refit_predict_cv(
                        df_train, y_train, groups=groups, num_iteration=num_iterations()
                    )
                    model_results[n_sample][seed]["scores_cv"] = [
                        metrics(y_train, yhat, "", TASKS[outcome()]["task"])
                        for yhat in yhats_cv
                    ]
                    model_results[n_sample][seed]["scores_test"] = [
                        metrics(
                            y_test,
                            model.predict(df_test, num_iteration=num_iteration),
                            "",
                            TASKS[outcome()]["task"],
                        )
                        for num_iteration in num_iterations()
                    ]

            refit_results += [
                {
                    **{
                        "model_idx": model_idx,
                        "num_iteration": num_iteration,
                    },
                    **refit_parameter,
                    **{
                        f"cv_{n_sample}_{seed}/{k}": v
                        for n_sample, seed in product(n_samples(), seeds())
                        for k, v in model_results[n_sample][seed]["scores_cv"][
                            num_iteration_idx
                        ].items()
                    },
                }
                for num_iteration_idx, num_iteration in enumerate(num_iterations())
            ]
    log_df(
        pl.DataFrame(refit_results), "refit_results.csv", client=client, run_id=run_id
    )


if __name__ == "__main__":
    main()

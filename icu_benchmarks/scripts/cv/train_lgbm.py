import logging
from time import perf_counter

import click
import gin
import mlflow
import numpy as np
import polars as pl

from icu_benchmarks.constants import TASKS
from icu_benchmarks.load import load
from icu_benchmarks.metrics import metrics
from icu_benchmarks.mlflow_utils import log_df, log_lgbm_model, setup_mlflow
from icu_benchmarks.models import LGBMAnchorModel  # noqa F401

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
def parameters(parameters=gin.REQUIRED):  # noqa D
    return parameters


@click.command()
@click.option("--config", type=click.Path(exists=True))
def main(config: str):  # noqa D
    gin.parse_config_file(config)
    # task = TASKS[outcome()]
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

    models = []
    for parameter in parameters():
        logger.info(f"Fitting the lgbm model with {parameter}")
        tic = perf_counter()
        objective = parameter.pop("objective")
        model = LGBMAnchorModel(params=parameter, objective=objective)
        model.fit(df, y, sample_weight=weights, dataset=dataset)
        toc = perf_counter()
        logger.info(f"Fitting the glm with {parameter} took {toc - tic:.1f} seconds")
        models.append(model)

    results = []

    for parameter_idx, parameter in enumerate(parameters()):
        model = models[parameter_idx]
        suffix = "_".join(f"{key}={value}" for key, value in parameter.items())
        log_lgbm_model(model.model, f"models/model_{suffix}.pkl")

        results.append(
            {
                **{
                    "model_path": f"models/model_{suffix}.pkl",
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

            for result_idx, result in enumerate(results):
                model = models[result["parameter_idx"]]
                yhat = model.predict(df)
                results[result_idx] = {
                    **result,
                    **metrics(y, yhat, f"{target}/{split}", TASKS[outcome()]["task"]),
                }

    log_df(pl.DataFrame(results), "results.csv")
    # This needs to be at the end of the script to log all relevant information
    mlflow.log_text(gin.operative_config_str(), "config.gin")


if __name__ == "__main__":
    main()

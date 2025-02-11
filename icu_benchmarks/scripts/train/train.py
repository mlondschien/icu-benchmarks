import logging
from itertools import product
from time import perf_counter

import click
import gin
import mlflow
import numpy as np
import polars as pl
from sklearn.model_selection import ParameterGrid

from icu_benchmarks.constants import TASKS
from icu_benchmarks.load import load
from icu_benchmarks.metrics import metrics
from icu_benchmarks.mlflow_utils import log_df, log_dict, log_pickle, setup_mlflow
from icu_benchmarks.models import (  # noqa F401
    AnchorRegression,
    DataSharedLasso,
    GeneralizedLinearModel,
    LGBMAnchorModel,
)
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
def get_sources(sources=gin.REQUIRED):  # noqa D
    return sources


@gin.configurable
def get_targets(targets=gin.REQUIRED):  # noqa D
    return targets


@gin.configurable
def get_parameters(parameters=gin.REQUIRED):
    """If parameters is a dictionary, create list of records with all combinations."""
    if isinstance(parameters, dict):
        keys, values = parameters.keys(), parameters.values()
        return [dict(zip(keys, combination)) for combination in product(*values)]
    else:
        return parameters


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

    outcome, sources, targets = get_outcome(), get_sources(), get_targets()

    tags = {
        "outcome": outcome,
        "sources": sources,
        "targets": targets,
        "parameter_names": np.unique([k for p in get_parameters() for k in p.keys()]),
        "summary_run": False,
    }

    _ = setup_mlflow(tags=tags)

    log_dict(get_parameters(), "parameters.json")
    log_dict(get_predict_kwargs(), "predict_kwargs.json")

    tic = perf_counter()
    df, y, weights, dataset = load(
        sources=sources,
        outcome=outcome,
        split="train_val",
        other_columns=["dataset"],
    )
    toc = perf_counter()
    logger.info(f"Loading data ({df.shape}) took {toc - tic:.1f} seconds")

    preprocessor = get_preprocessing(get_model(), df)

    tic = perf_counter()
    df = preprocessor.fit_transform(df)
    toc = perf_counter()
    logger.info(f"Preprocessing data took {toc - tic:.1f} seconds")
    log_pickle(preprocessor, "models/preprocessor.pkl")

    models = []
    model_dict = {}
    results = []
    for parameter_idx, parameter in enumerate(get_parameters()):
        logger.info(f"Fitting the glm with {parameter}")
        model = get_model()(**parameter)
        tic = perf_counter()

        model.fit(df, y, sample_weight=weights, dataset=dataset)
        toc = perf_counter()
        logger.info(f"Fitting the model with {parameter} took {toc - tic:.1f} seconds")
        models.append(model)

        model_dict[parameter_idx] = parameter

        log_pickle(model, f"models/model_{parameter_idx}.pkl")

        for predict_kwarg in get_predict_kwargs():
            results.append(
                {
                    "parameter_idx": parameter_idx,
                    "predict_kwarg": predict_kwarg,
                    **parameter,
                    **predict_kwarg,
                }
            )

    log_dict(model_dict, "models.json")

    for target in targets:
        for split in ["train_val", "test"]:
            logger.info(f"Scoring on {target}/{split}")
            tic = perf_counter()
            df, y, _ = load(sources=[target], outcome=outcome, split=split)
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
                model = models[result["parameter_idx"]]
                yhat = model.predict(df, **result["predict_kwarg"])
                results[result_idx] = {
                    **result,
                    **metrics(y, yhat, f"{target}/{split}/", TASKS[outcome]["task"]),
                }

    for result_idx in range(len(results)):
        del results[result_idx]["predict_kwarg"]
        if "objective" in results[result_idx]:
            del results[result_idx]["objective"]

    log_df(pl.DataFrame(results), "results.csv")
    # This needs to be at the end of the script to log all relevant information
    mlflow.log_text(gin.operative_config_str(), "config.gin")


if __name__ == "__main__":
    main()

import logging
from itertools import product
from time import perf_counter

import click
import gin
import mlflow
import numpy as np
import polars as pl
from anchorboosting import AnchorBooster  # noqa F401
from sklearn.model_selection import ParameterGrid
import json
from mlflow.tracking import MlflowClient
import tempfile
import pickle
from pathlib import Path
from icu_benchmarks.constants import ANCHORS, TASKS
from icu_benchmarks.gin import load
from icu_benchmarks.metrics import metrics
from icu_benchmarks.mlflow_utils import log_df, log_dict, log_pickle, setup_mlflow
from icu_benchmarks.models import (  # noqa F401
    AnchorRegression,
    DataSharedLasso,
    FFill,
    GeneralizedLinearModel,
)
from icu_benchmarks.preprocessing import get_preprocessing
from icu_benchmarks.utils import get_model_matrix

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
@click.option("--anchor_formula", type=str, default=None)
@click.option("--continue_run", type=str, default=None)
def main(config: str = "", anchor_formula: str = "", continue_run=None):  # noqa D
    gin.parse_config_file(config)

    outcome, sources, targets = get_outcome(), get_sources(), get_targets()

    tags = {
        "outcome": outcome,
        "sources": sources,
        "targets": targets,
        "parameter_names": np.unique([k for p in get_parameters() for k in p.keys()]),
        "summary_run": False,
        "anchor_formula": anchor_formula,
    }

    _ = setup_mlflow(tags=tags)

    log_dict(get_parameters(), "parameters.json")
    log_dict(get_predict_kwargs(), "predict_kwargs.json")

    tic = perf_counter()
    df, y, df_anchor = load(
        sources=sources,
        outcome=outcome,
        split="train_val",
        other_columns=[x for x in ANCHORS if x in anchor_formula + " dataset"],
    )
    toc = perf_counter()
    logger.info(f"Loading data ({df.shape}) took {toc - tic:.1f} seconds")

    preprocessor = get_preprocessing(get_model(), df)

    df_anchor = df_anchor.fill_null(2010).fill_null("other").to_pandas()
    Z = get_model_matrix(df_anchor, anchor_formula)

    tic = perf_counter()
    df = preprocessor.fit_transform(df)
    toc = perf_counter()
    logger.info(f"Preprocessing data took {toc - tic:.1f} seconds")
    log_pickle(preprocessor, "models/preprocessor.pkl")

    if any(x in str(get_model()) for x in ["GeneralizedLinear", "AnchorRegression", "DataSharedLasso"]):
        from anchorboosting.models import Proj
        X = df.to_numpy()
        proj = Proj(Z)
        X_proj = np.zeros_like(X)
        for j in range(X.shape[1]):
            X_proj[:, j] = proj(X[:, j] - X[:, j].mean())
        y_proj = proj(y - y.mean(axis=0))

    models = []
    if continue_run is not None:
        client = MlflowClient()

        run = client.get_run(continue_run)
        if not run.data.tags["sources"] == str(sources):
            raise ValueError

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(client.download_artifacts(continue_run, "models", tmpdir))
            for model_idx in range(len(list(model_dir.glob("model_*.pkl")))):
                with open(model_dir / f"model_{model_idx}.pkl", "rb") as f:
                    models.append(pickle.load(f))
    
    model_dict = {}
    results = []
    for parameter_idx, parameter in enumerate(get_parameters()):
        logger.info(f"Fitting model with {parameter}")
        if len(models) > parameter_idx:
            logger.info(f"Using existing model {parameter_idx}")
            model = models[parameter_idx]
        else:
            model = get_model()(**parameter)

            categorical_features = [c for c in df.columns if "categorical" in c]
            tic = perf_counter()

            if "AnchorBooster" in str(model):
                model.fit(             df, y, Z=Z, categorical_feature=categorical_features)
            else:
                model.fit(            df, y, Z=Z, X_proj=X_proj, y_proj=y_proj)

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
            df, y, df_anchor = load(
                sources=[target],
                outcome=outcome,
                split=split,
                other_columns=[x for x in ANCHORS if x in anchor_formula + "dataset"],
            )
            df_anchor = df_anchor.fill_null(2010).fill_null("other").to_pandas()
            Z = get_model_matrix(df_anchor, anchor_formula)

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
                    **metrics(
                        y,
                        yhat,
                        f"{target}/{split}/",
                        TASKS[outcome]["task"],
                        Z=Z,
                    ),
                }

    for result_idx in range(len(results)):
        del results[result_idx]["predict_kwarg"]
        if "objective" in results[result_idx]:
            del results[result_idx]["objective"]

    log_df(pl.DataFrame(results), "results.csv")
    # This needs to be at the end of the script to log all relevant information
    mlflow.log_text(gin.operative_config_str(), "config.txt")


if __name__ == "__main__":
    main()

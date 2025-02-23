import logging
from itertools import product
from time import perf_counter
from icu_benchmarks.metrics import metrics
import click
import gin
import mlflow
import numpy as np
import polars as pl
from sklearn.model_selection import ParameterGrid, GroupKFold
from icu_benchmarks.constants import TASKS, METRIC_NAMES
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

@gin.configurable
def get_n_splits(n_splits=gin.REQUIRED):
    return n_splits

@gin.configurable
def get_split_by(split_by=gin.REQUIRED):
    return split_by

@click.command()
@click.option("--config", type=click.Path(exists=True))
def main(config: str):  # noqa D
    gin.parse_config_file(config)

    outcome, sources, targets = get_outcome(), get_sources(), get_targets()
    split_by, n_splits = get_split_by(), get_n_splits()

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
    df, y, weights, groups = load(
        sources=sources,
        outcome=outcome,
        split="train_val",
        other_columns=[split_by],
    )
    toc = perf_counter()
    logger.info(f"Loading data ({df.shape}) took {toc - tic:.1f} seconds")

    cv_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(GroupKFold(n_splits=n_splits).split(df, y, groups)):
        logger.info(f"Fold: {fold_idx}.")
        df_train, df_val = df[train_idx, :], df[val_idx, :]
        y_train, y_val = y[train_idx], y[val_idx]
        groups_train = groups[train_idx]

        preprocessor = get_preprocessing(get_model(), df)
        df_train = preprocessor.fit_transform(df_train)
        df_val = preprocessor.transform(df_val)

        for parameter_idx, parameter in enumerate(get_parameters()):

            logger.info(f"Fitting the model with {parameter}")
            model = get_model()(**parameter)
            
            tic = perf_counter()
            model.fit(df_train, y_train, sample_weight=weights, dataset=groups_train)
            toc = perf_counter()

            for predict_kwarg_idx, predict_kwarg in enumerate(get_predict_kwargs()):
                y_val_hat = model.predict(df_val, **predict_kwarg)
                cv_results.append(
                    {
                        "fold_idx": fold_idx,
                        "parameter_idx": parameter_idx,
                        "predict_kwarg_idx": predict_kwarg_idx,
                        **metrics(y_val, y_val_hat, "", TASKS[outcome]["task"]),
                    }
                )

    cv_results = pl.DataFrame(cv_results)
    log_df(cv_results, "cv_results.csv")

    metric_names = [x for x in cv_results.columns if x in METRIC_NAMES]
    cv_results = cv_results.pivot(on="fold_idx", values=metric_names)
    
    results = []

    preprocessor = get_preprocessing(get_model(), df)
    df = preprocessor.fit_transform(df)

    models = []
    results = []
    for parameter_idx, parameter in enumerate(get_parameters()):
        logger.info(f"Fitting the model with {parameter}")
        model = get_model()(**parameter)

        tic = perf_counter()
        model.fit(df, y, sample_weight=weights)
        toc = perf_counter()
        models.append(model)
        for predict_kwarg_idx, predict_kwarg in enumerate(get_predict_kwargs()):
            yhat = model.predict(df, **predict_kwarg)
            results.append(
                {
                    "parameter_idx": parameter_idx,
                    "predict_kwarg_idx": predict_kwarg_idx,
                    **parameter,
                    **predict_kwarg,
                    **metrics(y, yhat, "train/", TASKS[outcome]["task"]),
                }
            )
    
    for target in targets:
        tic = perf_counter()
        df, y, _ = load(sources=[target], outcome=outcome, split="test")
        toc = perf_counter()
        logger.info(f"Loading data ({df.shape}) took {toc - tic:.1f} seconds")

        if df.shape[0] == 0:
            logger.warning(f"No data for {target}")
            continue

        tic = perf_counter()
        df = preprocessor.transform(df)
        toc = perf_counter()

        logger.info(f"Preprocessing data took {toc - tic:.1f} seconds")

        idx = 0
        for model, parameter in zip(models, get_parameters()):
            for predict_kwarg in get_predict_kwargs():
                yhat = model.predict(df, **predict_kwarg)
                results[idx] = {
                    **results[idx],
                    **metrics(y, yhat, f"{target}/test/", TASKS[outcome]["task"]),
                }
                idx += 1

    results = pl.DataFrame(results)
    results = results.join(cv_results, on=["parameter_idx", "predict_kwarg_idx"])

    log_df(pl.DataFrame(results), "results.csv")
    # This needs to be at the end of the script to log all relevant information
    mlflow.log_text(gin.operative_config_str(), "config.txt")


if __name__ == "__main__":
    main()

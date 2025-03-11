import json
import os
import pickle
import tempfile

import gin
import mlflow
import numpy as np
import pandas as pd
import polars as pl


class NumpyEncoder(json.JSONEncoder):
    """
    Allow storing dict of numpy arrays as json.

    https://github.com/thepushkarp/til/blob/main/python/store-dictionary-with-numpy-arrays-as-json.md
    """

    def default(self, obj):  # noqa D
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)


def log_dict(dict, name):
    """Log a dictionary as json to MLflow."""
    target_dir, name = os.path.split(name)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/{name}"
        with open(path, "w") as f:
            json.dump(dict, f, cls=NumpyEncoder)
        mlflow.log_artifact(path, target_dir)


def log_fig(fig, name, client=None, run_id=None, **kwargs):
    """Log a matplotlib figure to MLflow."""
    target_dir, name = os.path.split(name)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/{name}"
        fig.savefig(path, **kwargs)

        if client is not None:
            client.log_artifact(run_id, path, target_dir)
        else:
            mlflow.log_artifact(path, target_dir)


def log_df(df, name, client=None, run_id=None):
    """Log a pandas dataframe to MLflow."""
    target_dir, name = os.path.split(name)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/{name}"

        if name.endswith(".csv") and isinstance(df, (pd.DataFrame, pd.Series)):
            df.to_csv(path)
        elif name.endswith(".csv") and isinstance(df, (pl.DataFrame, pl.Series)):
            df.write_csv(path)
        elif name.endswith(".parquet") and isinstance(df, (pd.DataFrame, pd.Series)):
            df.to_parquet(path)
        elif name.endswith(".parquet") and isinstance(df, (pl.DataFrame, pl.Series)):
            df.write_parquet(path)
        else:
            raise ValueError(f"Unknown file extension: {name} or type: {type(df)}")

        if client is not None:
            client.log_artifact(run_id, path, target_dir)
        else:
            mlflow.log_artifact(path, target_dir)


def log_lgbm_model(model, name):
    """Log a lightgbm model to MLflow."""
    target_dir, name = os.path.split(name)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/{name}"
        model.booster_.save_model(path)
        mlflow.log_artifact(path, target_dir)


def log_pickle(object, name):
    """Log a pickle object to MLflow."""
    target_dir, name = os.path.split(name)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/{name}"
        with open(path, "wb") as f:
            pickle.dump(object, f)
        mlflow.log_artifact(path, target_dir)


def log_markdown(df, name, client=None, run_id=None):
    """Log a polars dataframe as markdown to MLflow."""
    with pl.Config() as cfg:
        cfg.set_tbl_rows(-1)
        cfg.set_tbl_cols(-1)
        cfg.set_tbl_formatting("MARKDOWN")
        text = repr(df)

    target_dir, name = os.path.split(name)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/{name}"
        with open(path, "w") as f:
            f.write(text)

        if client is not None:
            client.log_artifact(run_id, path, target_dir)
        else:
            mlflow.log_artifact(path, target_dir)


@gin.configurable
def setup_mlflow(
    tracking_uri: str,
    experiment_name: str,
    tags: dict[str, str] | None = None,
):
    """
    Set up an MLflow run.

    Returns
    -------
    run_id : str
        The ID of the MLflow run.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.set_experiment(experiment_name).experiment_id
    run_id = mlflow.start_run(experiment_id=experiment_id)

    for key, value in (tags or {}).items():
        mlflow.set_tag(key, value)

    return run_id


@gin.configurable
def get_run(tracking_uri, run_id):
    """Set up client with tracking_uri, then get the run with run_id."""
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id=run_id)
    return client, run


def get_target_run(client, experiment_name, create_if_not_exists=True):
    """
    Get run with sources tag equal '' in experiment.

    If such a run does not exist, create it.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No experiment {experiment_name}.")

    target_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.sources = ''",
    )
    if len(target_run) == 1:
        return experiment, target_run[0]
    elif len(target_run) == 0 and create_if_not_exists:
        target_run = client.create_run(
            experiment_id=experiment.experiment_id,
            tags={"sources": "", "summary_run": True},
        )
        return experiment, target_run
    else:
        raise ValueError(
            f"Expected exactly one target run for {experiment_name}. Got {len(target_run)}."
        )

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
        return json.JSONEncoder.default(self, obj)


def log_dict(dict, name):
    """Log a dictionary as json to MLflow."""
    target_dir, name = os.path.split(name)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/{name}"
        with open(path, "w") as f:
            json.dump(dict, f, cls=NumpyEncoder)
        mlflow.log_artifact(path, target_dir)


def log_fig(fig, name):
    """Log a matplotlib figure to MLflow."""
    target_dir, name = os.path.split(name)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/{name}"
        fig.savefig(path)
        mlflow.log_artifact(path, target_dir)


def log_df(df, name):
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

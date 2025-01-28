import logging
from time import perf_counter
from icu_benchmarks.models import DataSharedLasso
import click
import gin
import mlflow
import numpy as np
import polars as pl
import tabmat
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from icu_benchmarks.constants import TASKS, DATASETS
from icu_benchmarks.gin import GeneralizedLinearRegressor
from icu_benchmarks.load import load
from icu_benchmarks.metrics import metrics
from icu_benchmarks.mlflow_utils import log_df, log_pickle, setup_mlflow
from icu_benchmarks.benchmarks import severinghaus_spo2_to_po2, severinghaus_spo2_to_po2_from_table, severinghaus_po2_to_spo2
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
from mlflow.tracking import MlflowClient


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

@click.command()
@click.option("--target_experiment", type=str)
@click.option("--tracking_uri", type=str, default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns.db")
@click.option("--data_dir", type=str, default="/cluster/work/math/lmalte/data")
def main(target_experiment, tracking_uri, data_dir):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)

    target_experiment = client.get_experiment_by_name(target_experiment)
    target_run = client.search_runs(
        experiment_ids=[target_experiment.experiment_id], filter_string="tags.sources = ''"
    )
    if len(target_run) > 0:
        target_run = target_run[0]
    else:
        target_run = client.create_run(experiment_id=target_experiment.experiment_id, tags={"sources": ""})

    results = []
    for target in DATASETS:
        df, y, _ = load(sources=[target], outcome="log_po2", split="test", variables=["spo2", "sao2"], data_dir=data_dir)

        sao2 = df.select(pl.when(pl.col("sao2_all_missing_h8")).then(pl.col("spo2_mean_h8")).otherwise(pl.col("sao2_ffilled")).fill_null(pl.col("sao2_ffilled").mean())).to_numpy()
        yhat = np.log(severinghaus_po2_to_spo2(sao2 / 100))
        results.append(
            {
                "target": target,
                **metrics(y, yhat, "", "regression"),
            }
        )
        print(results[-1])


if __name__ == "__main__":
    main()

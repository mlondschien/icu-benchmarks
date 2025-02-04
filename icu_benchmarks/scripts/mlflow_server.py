import logging

import click
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.benchmarks import severinghaus_spo2_to_po2
from icu_benchmarks.constants import DATASETS
from icu_benchmarks.load import load
from icu_benchmarks.metrics import metrics
from icu_benchmarks.mlflow_utils import log_df
from icu_benchmarks.slurm_utils import setup_mlflow_server

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns.db",
)
@click.option("--hours", type=int, default=24)
@click.option(
    "--artifact_location",
    type=str,
    default="file:///cluster/work/math/lmalte/mlflow/artifacts",
)
def main(tracking_uri, hours, artifact_location):
    _, _ = setup_mlflow_server(
        tracking_uri=tracking_uri,
        experiment_name=None,
        artifact_location=artifact_location,
        hours=hours,
        verbose=True,
        experiment_note=None,
    )


if __name__ == "__main__":
    main()

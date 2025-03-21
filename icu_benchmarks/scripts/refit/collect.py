import json
import logging
import tempfile

import click
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import GREATER_IS_BETTER
from icu_benchmarks.mlflow_utils import get_target_run, log_df

SOURCES = ["mimic-carevue", "miiv", "eicu", "aumc", "sic", "hirid"]

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option("--experiment_name", type=str)
@click.option("--result_name", type=str)
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns3.db",
)
def main(experiment_name: str, result_name: str, tracking_uri: str):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment, target_run = get_target_run(client, experiment_name)

    logger.info(f"logging to {target_run.info.run_id}")

    _, run = get_target_run(client, experiment_name, create_if_not_exists=False)
    run_id = run.info.run_id

    results = []
    with tempfile.TemporaryDirectory() as f:
        for file in client.list_artifacts(run_id, path=f"refit/{result_name}"):
            client.download_artifacts(run_id, f"{file.path}/results.csv", f)
            results.append(pl.read_csv(f"{f}/{file.path}/results.csv"))

    results = pl.concat(results, how="diagonal")
    mult = pl.when(pl.col("metric").is_in(GREATER_IS_BETTER)).then(1).otherwise(-1)

    group_by = ["target", "metric", "n_target", "seed"]
    summary = (
        results.group_by(group_by)
        .agg(pl.all().top_k_by(k=1, by=pl.col("cv_value") * mult))
        .explode([x for x in results.columns if x not in group_by])
    )
    log_df(summary, f"{result_name}_results.csv", client, run_id=target_run.info.run_id)


if __name__ == "__main__":
    main()

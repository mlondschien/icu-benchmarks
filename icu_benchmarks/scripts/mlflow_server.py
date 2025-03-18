import logging

import click

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
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns3.db",
)
@click.option("--hours", type=int, default=24)
@click.option(
    "--artifact_location",
    type=str,
    default="file:///cluster/work/math/lmalte/mlflow/artifacts",
)
def main(tracking_uri, hours, artifact_location):  # noqa D
    _, _ = setup_mlflow_server(
        tracking_uri=tracking_uri,
        experiment_name="plots",
        artifact_location=artifact_location,
        hours=hours,
        verbose=True,
        experiment_note=None,
    )


if __name__ == "__main__":
    main()

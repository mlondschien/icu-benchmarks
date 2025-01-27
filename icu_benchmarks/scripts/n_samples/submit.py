import subprocess
from itertools import product
from pathlib import Path

import click

from icu_benchmarks.constants import DATASETS, OBSERVATIONS_PER_GB, TASKS
from icu_benchmarks.slurm_utils import setup_mlflow_server

SOURCES = [
    "miiv",
    "mimic-carevue",
    "hirid",
    "eicu",
    "aumc",
    "sic",
]


@click.command()
@click.option("--config", type=click.Path(exists=True))
@click.option("--hours", type=int, default=24)
@click.option("--experiment_name", type=str, default=None)
@click.option("--experiment_note", type=str, default=None)
@click.option("--outcome", type=str, default=None)
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns.db",
)
@click.option(
    "--artifact_location",
    type=str,
    default="file:///cluster/work/math/lmalte/mlflow/artifacts",
)
@click.option("--script", type=str, default="train_linear.py")
def main(
    config: str,
    hours: int,
    experiment_name: str,
    experiment_note: str,
    outcome: str,
    tracking_uri: str,
    artifact_location: str,
    script: str,
):  # noqa D
    ip, port = setup_mlflow_server(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_location=artifact_location,
        hours=hours,
        verbose=True,
        experiment_note=experiment_note,
    )

    outcomes = [outcome]

    refit_config = Path(__file__).parents[3] / "configs" / "refit" / "refit.gin"
    config_text = Path(config).read_text()

    for source, outcome in product(SOURCES, outcomes):
        n_samples = TASKS[outcome]["n_samples"][source]

        log_dir = Path("logs") / experiment_name / outcome / source
        log_dir.mkdir(parents=True, exist_ok=True)
        config_file = log_dir / "config.gin"

        with config_file.open("w") as f:
            f.write(
                f"""{refit_config.read_text()}

{config_text}

outcome.outcome = '{outcome}'
target.target = {DATASETS}

icu_benchmarks.mlflow_utils.setup_mlflow.experiment_name = "{experiment_name}"
icu_benchmarks.mlflow_utils.setup_mlflow.tracking_uri = "http://{ip}:{port}"


icu_benchmarks.load.load.sources = ["{source}"]
icu_benchmarks.load.load.variables = {TASKS[outcome].get('variables')}
icu_benchmarks.load.load.horizons = {TASKS[outcome].get('horizons')}
"""
            )

        required_memory = n_samples / OBSERVATIONS_PER_GB
        n_cpus = min(64, max(4, required_memory))

        command_file = log_dir / "command.sh"
        with command_file.open("w") as f:
            f.write(
                f"""#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task={int(n_cpus)}
#SBATCH --time={hours}:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name="{outcome}_{source}"
#SBATCH --output="{log_dir}/slurm.out"

python icu_benchmarks/scripts/n_samples/{script} --config {config_file.resolve()}"""
            )

        subprocess.run(["sbatch", str(command_file.resolve())])


if __name__ == "__main__":
    main()

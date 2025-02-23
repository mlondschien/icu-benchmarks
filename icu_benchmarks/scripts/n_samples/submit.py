import subprocess
from itertools import product
from pathlib import Path

import click

from icu_benchmarks.constants import TASKS
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
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns2.db",
)
@click.option(
    "--artifact_location",
    type=str,
    default="file:///cluster/work/math/lmalte/mlflow/artifacts",
)
def main(
    config: str,
    hours: int,
    experiment_name: str,
    experiment_note: str,
    outcome: str,
    tracking_uri: str,
    artifact_location: str,
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
        log_dir = Path("logs2") / experiment_name / source
        log_dir.mkdir(parents=True, exist_ok=True)
        config_file = log_dir / "config.gin"

        with config_file.open("w") as f:
            f.write(
                f"""{refit_config.read_text()}

{config_text}

get_outcome.outcome = '{outcome}'
get_target.target = '{source}'

icu_benchmarks.mlflow_utils.setup_mlflow.experiment_name = "{experiment_name}"
icu_benchmarks.mlflow_utils.setup_mlflow.tracking_uri = "http://{ip}:{port}"

FAMILY = "{TASKS[outcome]["family"]}"
TASK = "{TASKS[outcome]["task"]}"

icu_benchmarks.load.load.sources = ["{source}"]
icu_benchmarks.load.load.variables = {TASKS[outcome].get('variables')}
icu_benchmarks.load.load.horizons = {TASKS[outcome].get('horizons')}
"""
            )

        n_cpus = 16
        command_file = log_dir / "command.sh"
        with command_file.open("w") as f:
            f.write(
                f"""#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task={int(n_cpus)}
#SBATCH --time={hours}:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name="{experiment_name}_n_{source}"
#SBATCH --output="{log_dir}/slurm.out"

python icu_benchmarks/scripts/n_samples/n_samples.py --config {config_file.resolve()}"""
            )

        subprocess.run(["sbatch", str(command_file.resolve())])


if __name__ == "__main__":
    main()

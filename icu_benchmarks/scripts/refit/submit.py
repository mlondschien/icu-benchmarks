import json
import subprocess
from pathlib import Path

import click
import numpy as np
from mlflow.tracking import MlflowClient

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
    tracking_uri: str,
    artifact_location: str,
):  # noqa D
    ip, port = setup_mlflow_server(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_location=artifact_location,
        hours=hours,
        verbose=True,
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    runs = client.search_runs(experiment_ids=[experiment_id], max_results=10_000)

    refit_config = Path(__file__).parents[3] / "configs" / "refit" / "refit.gin"
    config_text = Path(config).read_text()
    stem = Path(config).stem

    for run in runs:
        if run.data.tags["sources"] == "":
            continue

        alpha_max = TASKS[run.data.tags["outcome"]]["alpha_max"]
        alpha = np.geomspace(alpha_max, alpha_max * 1e-6, 13)

        sources = sorted(json.loads(run.data.tags["sources"].replace("'", '"')))
        if len(sources) != len(SOURCES) - 1:
            continue

        target = [d for d in SOURCES if d not in sources][0]

        outcome = run.data.tags["outcome"]
        log_dir = Path("logs2") / experiment_name / "_".join(sources)
        log_dir = log_dir / f"refit_{stem}"
        log_dir.mkdir(parents=True, exist_ok=True)
        refit_config_file = log_dir / "config.gin"

        with refit_config_file.open("w") as f:
            f.write(
                f"""ALPHA = {alpha.tolist()}
FAMILY = "{TASKS[outcome]["family"]}"
TASK = "{TASKS[outcome]["task"]}"

{refit_config.read_text()}

{config_text}

get_name.name = "{stem}"
icu_benchmarks.mlflow_utils.setup_mlflow.experiment_name = "{experiment_name}"
icu_benchmarks.mlflow_utils.setup_mlflow.tracking_uri = "http://{ip}:{port}"

icu_benchmarks.load.load.variables = {TASKS[outcome].get("variables")}
icu_benchmarks.load.load.horizons = {TASKS[outcome].get("horizons")}
icu_benchmarks.load.load.sources = ["{target}"]

get_run.run_id = "{run.info.run_id}"
get_run.tracking_uri = "http://{ip}:{port}"
"""
            )

        command_file = log_dir / "command.sh"
        with command_file.open("w") as f:
            f.write(
                f"""#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time={hours}:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name="{experiment_name}/{stem}/{target}"
#SBATCH --output="{log_dir}/slurm.out"

python icu_benchmarks/scripts/refit/refit.py --config {refit_config_file.resolve()}"""
            )

        subprocess.run(["sbatch", str(command_file.resolve())])


if __name__ == "__main__":
    main()

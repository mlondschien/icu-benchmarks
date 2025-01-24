import json
import subprocess
from pathlib import Path
import numpy as np
import click
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
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns.db",
)
@click.option(
    "--artifact_location",
    type=str,
    default="file:///cluster/work/math/lmalte/mlflow/artifacts",
)
@click.option("--script", type=str, default="refit_lgbm.py")
def main(
    config: str,
    hours: int,
    experiment_name: str,
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
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    runs = client.search_runs(experiment_ids=[experiment_id], max_results=10_000)

    config_text = Path(config).read_text()

    for run in runs:
        alpha_max = TASKS[run.data.tags["outcome"]]["alpha_max"]
        alpha = np.geomspace(alpha_max, alpha_max * 1e-8, 20)

        sources = sorted(json.loads(run.data.tags["sources"].replace("'", '"')))
        if len(sources) != len(SOURCES) - 1:
            continue

        target = [d for d in SOURCES if d not in sources][0]

        outcome = run.data.tags["outcome"]
        log_dir = Path("logs") / experiment_name / outcome / "_".join(sources)
        (log_dir / "refit").mkdir(parents=True, exist_ok=True)
        refit_config_file = log_dir / "refit" / "config.gin"
        # train_config_text = (log_dir / "config.gin").read_text()

        with refit_config_file.open("w") as f:
            f.write(
                f"""{config_text}

ALPHA = {alpha.tolist()}

icu_benchmarks.mlflow_utils.setup_mlflow.experiment_name = "{experiment_name}"
icu_benchmarks.mlflow_utils.setup_mlflow.tracking_uri = "http://{ip}:{port}"

icu_benchmarks.load.load.variables = {TASKS[outcome].get('variables')}
icu_benchmarks.load.load.sources = ["{target}"]

get_run.run_id = "{run.info.run_id}"
get_run.tracking_uri = "http://{ip}:{port}"
"""
            )

        command_file = log_dir / "refit" / "command.sh"
        with command_file.open("w") as f:
            f.write(
                f"""#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time={hours}:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name="{outcome}_{'_'.join(sorted(sources))}"
#SBATCH --output="{log_dir / "refit"}/slurm.out"

python icu_benchmarks/scripts/refit/refit_linear.py --config {refit_config_file.resolve()}"""
            )

        subprocess.run(["sbatch", str(command_file.resolve())])


if __name__ == "__main__":
    main()

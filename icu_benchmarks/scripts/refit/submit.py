import json
import subprocess
from pathlib import Path

import click
import numpy as np
from mlflow.tracking import MlflowClient
from icu_benchmarks.mlflow_utils import get_target_run
from icu_benchmarks.constants import TASKS
from icu_benchmarks.slurm_utils import setup_mlflow_server

SOURCES6 = ["miiv", "mimic-carevue", "hirid", "eicu", "aumc", "sic"]

ALL_SOURCES = [
    "miiv",
    "mimic-carevue",
    "hirid",
    "eicu",
    "aumc",
    "sic",
    "nwicu",
    "zigong",
    "picdb",
]


@click.command()
@click.option("--config", type=click.Path(exists=True))
@click.option("--hours", type=int, default=24)
@click.option("--experiment_name", type=str, default=None)
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns3.db",
)
@click.option(
    "--artifact_location",
    type=str,
    default="file:///cluster/work/math/lmalte/mlflow/artifacts",
)
@click.option("--mlflow_server", type=str, default=None)
def main(
    config: str,
    hours: int,
    experiment_name: str,
    tracking_uri: str,
    artifact_location: str,
    mlflow_server: str = None,
):  # noqa D
    if mlflow_server is None:
        ip, port = setup_mlflow_server(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            artifact_location=artifact_location,
            hours=hours,
            verbose=True,
        )
    else:
        ip, port = mlflow_server.split(":")
    
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    _ , _= get_target_run(client, experiment_name)

    refit_config = Path(__file__).parents[3] / "configs" / "refit" / "refit.gin"
    config_text = Path(config).read_text()
    stem = Path(config).stem

    for target in ALL_SOURCES:
        sources = [s for s in SOURCES6 if s != target]
        filter_string = " AND ".join([f"tags.sources LIKE '%{s}%'" for s in sources])
        runs = client.search_runs(
            experiment_ids=[experiment_id], filter_string=filter_string
        )
        if len(runs) == 0:
            raise ValueError(f"No runs found for {target}.")
        
        for seed in range(0, 20):
            for run in runs:
                run_sources = sorted(json.loads(run.data.tags["sources"].replace("'", '"')))

                if len(run_sources) != len(sources):
                    continue

                alpha_max = TASKS[run.data.tags["outcome"]]["alpha_max"]
                alpha = np.geomspace(alpha_max, alpha_max * 1e-6, 13)

                outcome = run.data.tags["outcome"]
                log_dir = Path("logs3") / experiment_name / f"refit_{stem}" / target / str(seed)
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
get_target.target = "{target}"

icu_benchmarks.mlflow_utils.setup_mlflow.experiment_name = "{experiment_name}"
icu_benchmarks.mlflow_utils.setup_mlflow.tracking_uri = "http://{ip}:{port}"

icu_benchmarks.mlflow_utils.get_target_run.experiment_name = "{experiment_name}"
icu_benchmarks.mlflow_utils.get_target_run.create_if_not_exists = True

load.variables = {TASKS[outcome].get("variables")}
load.horizons = {TASKS[outcome].get("horizons")}
load.sources = ["{target}"]

get_seeds.seeds = [{seed}]

get_run.run_id = "{run.info.run_id}"
get_run.tracking_uri = "http://{ip}:{port}"
"""
                )

            command_file = log_dir / "command.sh"
            with command_file.open("w") as f:
                f.write(
                    f"""#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time={hours}:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name="{experiment_name}/{stem}/{target}/{seed}"
#SBATCH --output="{log_dir}/slurm.out"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python icu_benchmarks/scripts/refit/refit.py --config {refit_config_file.resolve()}"""
                )
            subprocess.run(["sbatch", str(command_file.resolve())])


if __name__ == "__main__":
    main()

import subprocess
from pathlib import Path

import click
import numpy as np

from icu_benchmarks.constants import DATASETS, TASKS
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
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns3.db",
)
@click.option(
    "--artifact_location",
    type=str,
    default="file:///cluster/work/math/lmalte/mlflow/artifacts",
)
@click.option(
    "--anchor_formula",
    type=str,
    default="",
)
@click.option(
    "--mlflow_server",
    type=str,
    default=None,
)
@click.option("--cpus", type=int, default=32)
@click.option("--memory", type=int, default=2)
@click.option("--suffix", type=str, default="")
def main(
    config: str,
    hours: int,
    experiment_name: str,
    experiment_note: str,
    outcome: str,
    tracking_uri: str,
    artifact_location: str,
    anchor_formula: str,
    suffix: str = "",
    mlflow_server: str | None = None,
    cpus: int = 32,
    memory: int = 2,
):  # noqa D
    if mlflow_server is None:
        ip, port = setup_mlflow_server(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            artifact_location=artifact_location,
            hours=hours,
            verbose=True,
            experiment_note=experiment_note,
        )
    else:
        ip, port = mlflow_server.split(":")

    # "<" avoids duplicates in the n-2. The "=" includes n-1 lists.
    list_of_sources = [
        [source for source in SOURCES if source != dataset1 and source != dataset2]
        for dataset1 in SOURCES
        for dataset2 in SOURCES
        if dataset1 <= dataset2
    ] + [SOURCES]
    train_config = Path(__file__).parents[3] / "configs" / "train" / "train.gin"
    config_text = Path(config).read_text()

    for sources in list_of_sources:
        alpha_max = TASKS[outcome]["alpha_max"]
        alpha = np.geomspace(alpha_max, alpha_max * 1e-6, 13)

        log_dir = Path("logs3") / experiment_name / ("_".join(sorted(sources)) + suffix)
        log_dir.mkdir(parents=True, exist_ok=True)
        config_file = log_dir / "config.gin"

        with config_file.open("w") as f:
            f.write(
                f"""{train_config.read_text()}

{config_text}

get_sources.sources = {sources}
get_outcome.outcome = '{outcome}'
get_targets.targets = {DATASETS}

icu_benchmarks.mlflow_utils.setup_mlflow.experiment_name = "{experiment_name}"
icu_benchmarks.mlflow_utils.setup_mlflow.tracking_uri = "http://{ip}:{port}"

TASK = "{TASKS[outcome]["task"]}"
FAMILY = "{TASKS[outcome]["family"]}"
ALPHA = {alpha.tolist()}

load.variables = {TASKS[outcome].get("variables")}
load.horizons = {TASKS[outcome].get("horizons")}
"""
            )

        command_file = log_dir / "command.sh"
        with command_file.open("w") as f:
            f.write(
                f"""#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={hours}:00:00
#SBATCH --mem-per-cpu={memory}G
#SBATCH --job-name="{experiment_name}_{"_".join(sorted(sources))}"
#SBATCH --output="{log_dir}/slurm.out"

export OMP_NUM_THREADS={cpus // 2}
export MKL_NUM_THREADS={cpus // 2}
export NUMEXPR_NUM_THREADS={cpus // 2}
export OPENBLAS_NUM_THREADS={cpus // 2}

python icu_benchmarks/scripts/train/train.py --config {config_file.resolve()} --anchor_formula "{anchor_formula}" """
            )

        subprocess.run(["sbatch", str(command_file.resolve())])


if __name__ == "__main__":
    main()

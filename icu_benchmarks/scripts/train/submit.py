import subprocess
from itertools import product
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
@click.option("--style", type=str, default="cv")
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
    style: str,
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

    if style == "cv":
        # "<" avoids duplicates in the n-2. The "=" includes n-1 lists.
        list_of_sources = [
            [source for source in SOURCES if source != dataset1 and source != dataset2]
            for dataset1 in SOURCES
            for dataset2 in SOURCES
            if dataset1 <= dataset2
        ] + [SOURCES]
    elif style == "1v1":
        # list_of_sources = [[source] for source in SOURCES]
        list_of_sources = [
            [source for source in SOURCES if source != dataset] for dataset in SOURCES
        ]
    else:
        raise ValueError(f"Unknown style {style}")

    outcomes = [outcome]

    train_config = Path(__file__).parents[3] / "configs" / "train" / "train.gin"
    config_text = Path(config).read_text()

    if style == "1v1":
        config_path = Path(__file__).parents[3] / "configs" / "train" / "1v1.gin"
        config_text = f"{config_path.read_text()}\n{config_text}"

    for sources, outcome in product(list_of_sources, outcomes):
        alpha_max = TASKS[outcome]["alpha_max"]
        alpha = np.geomspace(alpha_max, alpha_max * 1e-6, 13)

        log_dir = Path("logs2") / experiment_name / "_".join(sorted(sources))
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

icu_benchmarks.load.load.variables = {TASKS[outcome].get("variables")}
icu_benchmarks.load.load.horizons = {TASKS[outcome].get("horizons")}
"""
            )

        # required_memory = n_samples / OBSERVATIONS_PER_GB
        # n_cpus = min(64, max(4, required_memory))

        script = "train.py" if style != "1v1" else "train_1v1.py"
        command_file = log_dir / "command.sh"
        with command_file.open("w") as f:
            f.write(
                f"""#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time={hours}:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name="{experiment_name}_{"_".join(sorted(sources))}"
#SBATCH --output="{log_dir}/slurm.out"

python icu_benchmarks/scripts/train/{script} --config {config_file.resolve()}"""
            )

        subprocess.run(["sbatch", str(command_file.resolve())])


if __name__ == "__main__":
    main()

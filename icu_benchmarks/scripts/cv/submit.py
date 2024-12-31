import subprocess
from pathlib import Path

import click

from icu_benchmarks.constants import OBSERVATIONS_PER_GB, TASKS
from icu_benchmarks.slurm_utils import setup_mlflow_server

DATASETS = [
    "miiv",
    "mimic-carevue",
    "hirid",
    "eicu",
    "aumc",
    "sic",
]
OUTCOMES = ["mortality_at_24h"]


@click.command()
@click.option("--config", type=click.Path(exists=True))
@click.option("--logs", type=click.Path())
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
def main(
    config: str,
    logs: str,
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

    list_of_sources = [
        [source for source in DATASETS if source != dataset1 and source != dataset2]
        for dataset1 in DATASETS
        for dataset2 in DATASETS
        if dataset1 <= dataset2
    ]

    for sources, outcome in zip(list_of_sources, OUTCOMES):
        n_samples = sum(TASKS[outcome]["n_samples"][source] for source in sources)

        log_dir = Path(logs) / outcome / "_".join(sorted(sources))
        log_dir.mkdir(parents=True, exist_ok=True)
        config_file = log_dir / "config.gin"

        with config_file.open("w") as f:
            f.write(
                f"""
include '{config}'
sources.sources = ['{", ".join(sources)}']
outcome.outcome = '{outcome}'
targets.targets = {DATASETS}

icu_benchmarks.mlflow_utils.setup_mlflow.experiment_name = "{experiment_name}"
icu_benchmarks.mlflow_utils.setup_mlflow.tracking_uri = "http://{ip}:{port}"

GeneralizedLinearRegressor.alpha_max = {TASKS[outcome]['alpha_max']}
GeneralizedLinearRegressor.family = '{TASKS[outcome]['family']}'
"""
            )

            required_memory = n_samples / OBSERVATIONS_PER_GB
            n_cpus = min(64, max(4, required_memory))

            command_file = log_dir / "command.sh"
            with command_file.open("w") as f:
                f.write(
                    f"""#!/bin/sh
python icu_benchmarks/scripts/cv/train.py --config {config_file.resolve()}"""
                )

            process = [
                "sbatch",
                "--ntasks=1",
                f"--cpus-per-task={int(n_cpus)}",
                "--mem-per-cpu=8G",
                f"--time={hours}:00:00",
                f"--output={log_dir}/slurm.out",
                f"--job-name={outcome}_{'_'.join(sorted(sources))}",
            ] + [str(command_file.resolve())]
            print(" ".join(process))
            subprocess.run(process)


if __name__ == "__main__":
    main()

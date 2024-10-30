import shlex
import subprocess
import tempfile
from pathlib import Path

import click

DATASETS = ["aumc", "eicu", "hirid", "miiv", "picdb", "sic", "zigong"]

# approximate sizes in GB of the datasets
TASKS = {
    "remaining_los": 75,
    "mortality_at_24h": 10,
    "los_at_24h": 10,
    "decompensation_at_24h": 75,
    "respiratory_failure_at_24h": 20,
    "circulatory_failure_at_8h": 30,
    "kidney_failure_at_48h": 40,
}


@click.command()
@click.option("--config", type=click.Path(exists=True))
@click.option("--hours", type=int, default=24)
@click.option("--args", type=str, default="")
@click.option("--dry", is_flag=True)
def main(config: str, hours: int, args: str, dry: bool):  # noqa D
    for dataset in DATASETS:
        for task, size in TASKS.items():
            tmpdir = Path(tempfile.mkdtemp()) / "config.gin"
            tmpdir.touch()

            with tmpdir.open("a") as f:
                f.write(f"include '{config}'\n\n")
                f.write(f"sources.sources = ['{dataset}']\n")
                f.write(f"outcome.outcome = '{task}'\n")

            n_tasks = max(4, size // 4)

            process = (
                [
                    "sbatch",
                    f"--ntasks={n_tasks}",
                    "--mem-per-cpu=8G",
                    f"--time={hours}:00:00",
                ]
                + shlex.split(args)
                + [f"--wrap='python icu_benchmarks/scripts/train.py --config {tmpdir}'"]
            )

            print(" ".join(process))
            if not dry:
                subprocess.run(process)


if __name__ == "__main__":
    main()

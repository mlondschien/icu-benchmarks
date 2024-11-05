import os
import shlex
import socket
import subprocess
import tempfile
from pathlib import Path
from time import sleep

import click

DATASETS = ["aumc", "eicu", "hirid", "miiv", "picdb", "sic", "zigong"]
OUTCOMES = [
    # "remaining_los",
    "mortality_at_24h",
    # "los_at_24h",
    "decompensation_at_24h",
    "respiratory_failure_at_24h",
    "circulatory_failure_at_8h",
    "kidney_failure_at_48h",
]

SIZES = {
    "mimic": {
        "remaining_los": [5179853, 33.3],
        "mortality_at_24h": [48560, 0.3],
        "los_at_24h": [48560, 0.3],
        "decompensation_at_24h": [4020520, 25.8],
        "respiratory_failure_at_24h": [997507, 6.4],
        "circulatory_failure_at_8h": [309693, 2.0],
        "kidney_failure_at_48h": [2948173, 18.9],
    },
    "ehrshot": {
        "remaining_los": [0, 0.0],
        "mortality_at_24h": [32740, 0.2],
        "los_at_24h": [0, 0.0],
        "decompensation_at_24h": [2902373, 18.6],
        "respiratory_failure_at_24h": [70040, 0.4],
        "circulatory_failure_at_8h": [11453, 0.1],
        "kidney_failure_at_48h": [60993, 0.4],
    },
    "miived": {
        "remaining_los": [1550213, 10.0],
        "mortality_at_24h": [11733, 0.1],
        "los_at_24h": [11733, 0.1],
        "decompensation_at_24h": [1550213, 10.0],
        "respiratory_failure_at_24h": [0, 0.0],
        "circulatory_failure_at_8h": [0, 0.0],
        "kidney_failure_at_48h": [0, 0.0],
    },
    "miiv": {
        "remaining_los": [5236473, 33.6],
        "mortality_at_24h": [56887, 0.4],
        "los_at_24h": [56887, 0.4],
        "decompensation_at_24h": [3801287, 24.4],
        "respiratory_failure_at_24h": [971420, 6.2],
        "circulatory_failure_at_8h": [489247, 3.1],
        "kidney_failure_at_48h": [3950100, 25.4],
    },
    "eicu": {
        "remaining_los": [11623907, 74.7],
        "mortality_at_24h": [132760, 0.9],
        "los_at_24h": [132760, 0.9],
        "decompensation_at_24h": [11623907, 74.7],
        "respiratory_failure_at_24h": [2105320, 13.5],
        "circulatory_failure_at_8h": [331540, 2.1],
        "kidney_failure_at_48h": [5857953, 37.6],
    },
    "hirid": {
        "remaining_los": [1716120, 11.0],
        "mortality_at_24h": [16900, 0.1],
        "los_at_24h": [16900, 0.1],
        "decompensation_at_24h": [1075833, 6.9],
        "respiratory_failure_at_24h": [599713, 3.9],
        "circulatory_failure_at_8h": [646907, 4.2],
        "kidney_failure_at_48h": [980813, 6.3],
    },
    "aumc": {
        "remaining_los": [1789927, 11.5],
        "mortality_at_24h": [13133, 0.1],
        "los_at_24h": [13133, 0.1],
        "decompensation_at_24h": [1789927, 11.5],
        "respiratory_failure_at_24h": [854267, 5.5],
        "circulatory_failure_at_8h": [275340, 1.8],
        "kidney_failure_at_48h": [1475073, 9.5],
    },
    "sic": {
        "remaining_los": [1716120, 11.0],
        "mortality_at_24h": [19100, 0.1],
        "los_at_24h": [19100, 0.1],
        "decompensation_at_24h": [1254313, 8.1],
        "respiratory_failure_at_24h": [421500, 2.7],
        "circulatory_failure_at_8h": [1072500, 6.9],
        "kidney_failure_at_48h": [1277720, 8.2],
    },
    "zigong": {
        "remaining_los": [367333, 2.4],
        "mortality_at_24h": [2380, 0.0],
        "los_at_24h": [2380, 0.0],
        "decompensation_at_24h": [362853, 2.3],
        "respiratory_failure_at_24h": [0, 0.0],
        "circulatory_failure_at_8h": [2633, 0.0],
        "kidney_failure_at_48h": [0, 0.0],
    },
    "picdb": {
        "remaining_los": [1614980, 10.4],
        "mortality_at_24h": [9200, 0.1],
        "los_at_24h": [9200, 0.1],
        "decompensation_at_24h": [1614980, 10.4],
        "respiratory_failure_at_24h": [216947, 1.4],
        "circulatory_failure_at_8h": [9727, 0.1],
        "kidney_failure_at_48h": [40920, 0.3],
    },
}


def free_port():
    """Return free port."""
    # https://stackoverflow.com/a/1365284/10586763
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]


@click.command()
@click.option("--config", type=click.Path(exists=True))
@click.option("--logs", type=click.Path(exists=True))
@click.option("--hours", type=int, default=24)
@click.option("--args", type=str, default="")
@click.option("--experiment_name", type=str, default=None)
def main(config: str, logs: str, hours: int, args: str, experiment_name: str):  # noqa D
    tmpdir = Path(tempfile.mkdtemp(dir=os.environ["SCRATCH"]))
    print(f"Using temporary directory {tmpdir}")
    ip_file = tmpdir / "ip"
    port = free_port()
    cmd_file = tmpdir / "cmd.sh"
    tracking_uri = "sqlite:////cluster/work/math/lmalte/mlflow/mlruns.db"

    # - set tracking uri and set the experiment. This ensures the experiment exists when
    #   the first run wants to log to it. If not, many runs started at the same time
    #   will try to create the same experiment.
    # - write the ip address to a file so that we can use it to set up port forwarding
    # - start the mlflow server
    # flake8: noqa E702
    with cmd_file.open("w") as f:
        f.write(
            f"""#!/bin/sh
python -c "import mlflow; mlflow.set_tracking_uri('{tracking_uri}'); mlflow.set_experiment('{experiment_name}')"
echo $(hostname -i) > {ip_file.resolve()}
mlflow server --port {port} --host 0.0.0.0 --backend-store-uri {tracking_uri} --default-artifact-root=file:/cluster/work/math/lmalte/mlflow/artifacts"""
        )

    cmd = [
        "sbatch",
        "--ntasks=1",
        "--cpus-per-task=4",
        f"--time={hours}:00:00",
        "--mem-per-cpu=4G",
        "--job-name=mlflow_server",
        str(cmd_file.resolve()),
    ]
    subprocess.Popen(cmd)

    # Wait until the job has started and the ip address is written to the file
    while not ip_file.is_file():
        sleep(0.1)

    with ip_file.open() as f:
        ip = f.read().strip()

    # If run on a local machine, you can view the mlflow server by running the following
    # -N ensures no shell is started and & runs the command in the background
    # local:ip:remote
    # ps | grep "ssh euler -L" | sed 's/\|/ /'| awk '{print $1}' | xargs kill
    print(f"ssh euler -L {port}:{ip}:{port} -N &")
    print(f"http://localhost:{port}/")

    for dataset in DATASETS:
        for outcome in OUTCOMES:
            if dataset == "zigong" and outcome in [
                "respiratory_failure_at_24h",
                "circulatory_failure_at_8h",
                "kidney_failure_at_48h",
            ]:
                continue

            dir = Path(logs) / outcome / dataset
            dir.mkdir(parents=True, exist_ok=True)
            config_file = dir / "config.gin"

            with config_file.open("w") as f:
                f.write(
                    f"""
include '{config}'
sources.sources = ['{dataset}']
outcome.outcome = '{outcome}'
icu_benchmarks.mlflow_utils.setup_mlflow.experiment_name = "{experiment_name}"
icu_benchmarks.mlflow_utils.setup_mlflow.tracking_uri = "http://{ip}:{port}"
"""
                )

            required_memory = max(SIZES[d][outcome][1] for d in DATASETS) * 5
            n_cpus = max(4, required_memory // 8)

            command_file = dir / "command.sh"
            with command_file.open("w") as f:
                f.write(
                    f"""#!/bin/sh
python icu_benchmarks/scripts/train.py --config {config_file.resolve()}"""
                )

            process = (
                [
                    "sbatch",
                    "--ntasks=1",
                    f"--cpus-per-task={int(n_cpus)}",
                    "--mem-per-cpu=8G",
                    f"--time={hours}:00:00",
                    f"--output={dir}/slurm.out",
                    f"--job-name={dataset}_{outcome}",
                ]
                + shlex.split(args)
                + [str(command_file.resolve())]
            )

            subprocess.run(process)


if __name__ == "__main__":
    main()

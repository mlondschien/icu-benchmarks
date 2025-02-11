import os
import socket
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Optional

from icu_benchmarks.constants import DATASETS, OBSERVATIONS_PER_GB, OUTCOMES, TASKS


def free_port():
    """Return free port."""
    # https://stackoverflow.com/a/1365284/10586763
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]


def setup_mlflow_server(
    tracking_uri: str,
    experiment_name: str,
    artifact_location: str,
    hours: int = 24,
    tmpdir: Optional[Path] = None,
    verbose: bool = False,
    experiment_note=None,
):
    r"""
    Set up an mlflow server on euler and return the ip address and port.

    This function starts a slurm job that runs an mlflow server. Once granted resources,
    the machine running the slurm job does the following:
     - it runs a python script that sets the tracking uri and creates an
     experiment with `experiment_name` if it does not exist.
     - it selects a free port.
     - it writes its ip address and the selected port to a file `ip`.
     - it starts the mlflow server with the selected port.

    The function waits until the ip address is written to the file by the machine
    running the mlflow server and reads it. It writes the command to forward the port
    to the local machine and the current time to a file `.mlflow_server` in the current
    directory. Finally, it returns the ip address and port.

    Note, the command to ssh into euler is of the form
    `ssh euler -L <local_port>:<ip>:<port> -N &`. if <local_port> is not free, choose
    another one. The mlflow GUI can be accessed at
    `http://localhost:<another_local_port>/`.
    One can kill all processes running port forwarding with
    `ps | grep "ssh euler -L" | sed 's/\|/ /'| awk '{print $1}' | xargs kill`.

    Parameters
    ----------
    tracking_uri : str
        The tracking uri to use for the mlflow server. E.g.,
        "sqlite:////cluster/work/math/lmalte/mlflow/mlruns.db".
    experiment_name : str
        The name of the experiment to use.
    artifact_location : str
        The location to store the artifacts. E.g.,
        "file:///cluster/work/math/lmalte/mlflow/artifacts".
    hours : int, optional, default=24
        The number of hours to run the server for.
    tmpdir : Path, optional
        The temporary directory to use. If None, uses `os.environ["SCRATCH"]` (euler).
    verbose : bool, optional, default=False
        Whether to print the command to forward the port and the url to access the
        mlflow server.

    Returns
    -------
    ip : str
        The ip address of the machine running the mlflow server.
    port : int
        The port the mlflow server is running on.
    """
    tmpdir = Path(tempfile.mkdtemp(dir=tmpdir or os.environ["SCRATCH"]))
    if verbose:
        print(f"Using temporary directory {tmpdir}")

    ip_file = tmpdir / "ip"
    port = free_port()
    python_script = tmpdir / "setup_mlflow.py"

    # - set tracking uri and set the experiment. This ensures the experiment exists when
    #   the first run wants to log to it. If not, many runs started at the same time
    #   will try to create the same experiment.
    if experiment_name is not None:
        python_script_string = f"""import mlflow

mlflow.set_tracking_uri('{tracking_uri}')
experiment = mlflow.get_experiment_by_name('{experiment_name}')
if experiment is None:
    experiment_id = mlflow.create_experiment('{experiment_name}', artifact_location='{artifact_location}')
else:
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_id=experiment_id)"""

        # https://stackoverflow.com/a/60732403/10586763
        if experiment_note is not None:
            python_script_string += f"""
mlflow.set_experiment_tag('mlflow.note.content', '{experiment_note}')"""

        with python_script.open("w") as f:
            f.write(python_script_string)

    # - write the ip address to a file so that we can use it to set up port forwarding
    # - start the mlflow server
    # flake8: noqa E702
    # where to store artifacts: https://stackoverflow.com/a/75073333/10586763
    # if experiment exists / else https://github.com/mlflow/mlflow/issues/2464
    cmd_file = tmpdir / "cmd.sh"
    with cmd_file.open("w") as f:
        f.write(
            f"""#!/bin/sh
python {python_script.resolve()}
echo $(hostname -i) > {ip_file.resolve()}
mlflow server --port {port} --host 0.0.0.0 --backend-store-uri {tracking_uri} --default-artifact-root={artifact_location}" --artifacts-destinations {artifact_location}" --serve-artifacts"""
        )

    cmd = [
        "sbatch",
        "--ntasks=1",
        "--cpus-per-task=2",
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

    server_file = Path(".") / ".mlflow_server"
    server_file.touch()
    # -N ensures no shell is started and & runs the command in the background
    with open(server_file, "w") as f:
        f.write(
            f"""
ssh euler -L {port+1}:{ip}:{port} "sleep 1; exit" && ssh euler -L {port}:{ip}:{port} -N &
http://localhost:{port}/
{datetime.now()}
"""
        )

    if verbose:
        print(f"ssh euler -L {port + 1}:{ip}:{port} -N &")
        print(f"http://localhost:{port + 1}/")

    return ip, port

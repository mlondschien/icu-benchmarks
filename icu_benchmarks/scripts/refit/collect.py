import json
import logging
import re
import tempfile

import click
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient
from icu_benchmarks.plotting import PARAMETER_NAMES
from icu_benchmarks.mlflow_utils import log_df
GREATER_IS_BETTER = ["accuracy", "roc", "auprc", "r2"]
SOURCES = ["mimic-carevue", "miiv", "eicu", "aumc", "sic", "hirid"]

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option("--experiment_name", type=str)
@click.option("--result_name", type=str)
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns.db",
)
def main(experiment_name: str, result_name:str, tracking_uri: str):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    target_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.sources = ''",
    )
    if len(target_run) > 0:
        target_run = target_run[0]
    else:
        target_run = client.create_run(
            experiment_id=experiment.experiment_id, tags={"sources": "", "summary_run": True}
        )

    logger.info(f"logging to {target_run.info.run_id}")

    summarized = []

    experiment_id = experiment.experiment_id

    runs = client.search_runs(experiment_ids=[experiment_id])

    all_results = []
    for run in runs:
        sources = run.data.tags.get("sources", "")

        if sources != "":
            sources = json.loads(run.data.tags["sources"].replace("'", '"'))
            if len(sources) != 5:
                continue

        if "target" in run.data.tags:
            target = run.data.tags["target"]
        else:
            target = [t for t in SOURCES if t not in sources][0]

        run_id = run.info.run_id
        result_file = f"{result_name}_results.csv"
        with tempfile.TemporaryDirectory() as f:
            if result_file not in [x.path for x in client.list_artifacts(run_id)]:
                logger.warning(f"Run {run_id} has no {result_file}")
                continue

            client.download_artifacts(run_id, result_file, f)
            results = pl.read_csv(f"{f}/{result_file}")

        results = results.with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.lit(sources).alias("sources"),
            pl.lit(run.data.tags["outcome"]).alias("outcome"),
            pl.lit(result_name).alias("result_name"),
            pl.lit(target).alias("target"),
        )

        all_results.append(results)

        parameter_names = [p for p in PARAMETER_NAMES if p in results.columns]

        metrics = map(re.compile(r"\/(.+)$").search, results.columns)
        metrics = np.unique([m.groups()[0] for m in metrics if m is not None])

        for result in all_results:
            target = result["target"].unique()[0]
            for metric in metrics:
                for n_target in [5, 10, 20, 40, 80, 160, 320, 640, 1280]:
                    for seed in [0, 1, 2, 3, 4]:
                        if metric in GREATER_IS_BETTER:
                            df = result[
                                result[f"cv_{n_target}_{seed}/{metric}"].arg_max()
                            ]
                        else:
                            df = result[
                                result[f"cv_{n_target}_{seed}/{metric}"].arg_min()
                            ]

                        summarized.append(
                            {
                                "target": target,
                                "metric": metric,
                                "cv_value": df[f"cv_{n_target}_{seed}/{metric}"].item(),
                                "test_value": df[
                                    f"test_{n_target}_{seed}/{metric}"
                                ].item(),
                                **{p: df[p].item() for p in parameter_names},
                                "seed": seed,
                                "n_target": n_target,
                                "result_name": result_name,
                            }
                        )
            
    result = pl.DataFrame(summarized)
    log_df(result, f"{result_name}_results.csv", client, run_id=target_run.info.run_id)


if __name__ == "__main__":
    main()

import json
import logging
import tempfile

import click
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import GREATER_IS_BETTER
from icu_benchmarks.mlflow_utils import get_target_run, log_df

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
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns3.db",
)
def main(experiment_name: str, result_name: str, tracking_uri: str):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment, target_run = get_target_run(client, experiment_name)

    logger.info(f"logging to {target_run.info.run_id}")

    experiment_id = experiment.experiment_id

    runs = client.search_runs(experiment_ids=[experiment_id])

    all_results = []
    for run in runs:
        if "target" in run.data.tags:
            target = run.data.tags.get("target")
            if target == "":
                continue
        else:
            sources = run.data.tags.get("sources", "")
            if sources == "":
                continue

            sources = json.loads(run.data.tags["sources"].replace("'", '"'))
            if len(sources) != 5:
                continue
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
            pl.lit(result_name).alias("result_name"),
            pl.lit(target).alias("target"),
        )
        all_results.append(results)

    results = pl.concat(all_results, how="diagonal")
    metrics = results["metric"].unique().to_list()

    mult = pl.when(pl.col("metric").is_in(GREATER_IS_BETTER)).then(1).otherwise(-1)
    results = results.with_columns((pl.col("cv_value") * mult).alias("cv_value"))
    results = results.pivot(values=["cv_value", "test_value"], on=["metric"], separator="/")
    group_by = ["target", "n_target", "seed"]
    
    summaries = []
    for metric in metrics:
        summaries.append(
            results.group_by(group_by)
            .agg(pl.all().top_k_by(k=1, by=pl.col(f"cv_value/{metric}")))
            .explode([x for x in results.columns if x not in group_by]).with_columns(pl.lit(metric).alias("cv_metric"))
        )
    
    summary = pl.concat(summaries, how="diagonal")
    log_df(summary, f"{result_name}_results.csv", client, run_id=target_run.info.run_id)


if __name__ == "__main__":
    main()

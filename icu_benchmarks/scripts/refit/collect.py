import logging
import tempfile

import click
import polars as pl
from mlflow.tracking import MlflowClient
import json
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
    _, target_run = get_target_run(client, experiment_name)

    logger.info(f"logging to {target_run.info.run_id}")

    experiment, run = get_target_run(client, experiment_name, create_if_not_exists=False)
    run_id = run.info.run_id

    all_results = []
    with tempfile.TemporaryDirectory() as f:
        for file in client.list_artifacts(run_id, path=f"refit/{result_name}"):
            client.download_artifacts(run_id, f"{file.path}/results.csv", f)
            all_results.append(pl.read_csv(f"{f}/{file.path}/results.csv"))

    results = pl.concat(all_results, how="diagonal")
    metrics = results["metric"].unique().to_list()
    if "random_state" in results.columns:
        results = results.filter(pl.col("random_state").eq(0))

    run = [x for x in client.search_runs(experiment_ids=[experiment.experiment_id]) if x.data.tags.get("sources", "") != ""][0]

    with tempfile.TemporaryDirectory() as f:
        client.download_artifacts(run.info.run_id, f"models.json", f)
        with open(f"{f}/models.json", "r") as f:
            models = json.load(f)
    results = results.join(
        pl.DataFrame([{"model_idx": int(i), **params} for i, params in models.items()]),
        on="model_idx",
    )
    if "gamma" in results.columns:
        results = results.filter(pl.col("gamma").eq(8))
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

    if "gamma" in results.columns:
        log_df(pl.concat(summaries, how="diagonal"), f"{result_name}8_results.csv", client, run_id=target_run.info.run_id)
    else:
        log_df(pl.concat(summaries, how="diagonal"), f"{result_name}_results.csv", client, run_id=target_run.info.run_id)


if __name__ == "__main__":
    main()

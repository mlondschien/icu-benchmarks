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
    _, target_run = get_target_run(client, experiment_name)

    logger.info(f"logging to {target_run.info.run_id}")

    experiment, run = get_target_run(
        client, experiment_name, create_if_not_exists=False
    )
    run_id = run.info.run_id

    all_results = []
    with tempfile.TemporaryDirectory() as f:
        for folder in client.list_artifacts(run_id, path=f"refit/{result_name}"):
            for file in client.list_artifacts(run_id, path=folder.path):
                client.download_artifacts(run_id, file.path, f)
                all_results.append(pl.read_csv(f"{f}/{file.path}"))

    results = pl.concat(all_results, how="diagonal")

    # Filter gamma = 1, 2, 4, ...
    if "gamma" in results.columns:
        col = pl.col("gamma").log() / pl.lit(2).log()
        results = results.filter((col - col.round(0)).abs().le(0.01))
    if "max_depth" in results.columns:
        results = results.filter(pl.col("gamma").le(16.0))
        results = results.filter(pl.col("max_depth").eq(3))
    if "num_iteration" in results.columns:
        results = results.filter(pl.col("num_iteration").eq(1000))

    results = results.filter(pl.col("seed") <= 19)
    group_by = [
        x
        for x in [
            "model_idx",
            "target",
            "gamma",
            "decay_rate",
            "metric",
            "n_target",
            "alpha",
            "l1_ratio",
            "prior_alpha",
            "alpha_index",
        ]
        if x in results.columns
    ]
    nunique = results.group_by(group_by).len()
    if not nunique.select(pl.col("len").eq(20).all()).item():
        raise ValueError

    metrics = results["metric"].unique().to_list()

    metric = "auprc" if "auprc" in metrics else "mse"
    cv_metric = "log_loss" if metric == "auprc" else "mse"

    if "random_state" in results.columns:
        results = results.filter(pl.col("random_state").eq(0))

    mult = pl.when(pl.col("metric").is_in(GREATER_IS_BETTER)).then(1).otherwise(-1)
    results = results.with_columns((pl.col("cv_value") * mult).alias("cv_value"))
    results = results.pivot(
        values=["cv_value", "test_value"], on=["metric"], separator="/"
    )
    group_by = ["target", "n_target", "seed"]

    summary = (
        results.group_by(group_by)
        .agg(pl.all().top_k_by(k=1, by=pl.col(f"cv_value/{cv_metric}")))
        .explode([x for x in results.columns if x not in group_by])
        .with_columns(pl.lit(cv_metric).alias("cv_metric"))
    )

    log_df(
        summary,
        f"{result_name}_results.csv",
        client,
        run_id=target_run.info.run_id,
    )


if __name__ == "__main__":
    main()

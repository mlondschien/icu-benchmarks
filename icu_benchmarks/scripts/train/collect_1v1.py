import json
import logging
import re
import tempfile

import click
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import DATASETS, GREATER_IS_BETTER
from icu_benchmarks.mlflow_utils import log_df

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option("--experiment_name", type=str)
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns2.db",
)
def main(experiment_name: str, tracking_uri: str):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    if "mlflow.note.content" in experiment.tags:
        print(experiment.tags["mlflow.note.content"])

    experiment_id = experiment.experiment_id

    runs = client.search_runs(experiment_ids=[experiment_id], max_results=10_000)

    all_results = []
    for run in runs:
        if run.data.tags["sources"] == "":
            continue
        sources = json.loads(run.data.tags["sources"].replace("'", '"'))

        # if len(sources) != 1:
        #     continue

        run_id = run.info.run_id
        with tempfile.TemporaryDirectory() as f:
            if "results.csv" not in [x.path for x in client.list_artifacts(run_id)]:
                logger.warning(f"Run {run_id} ({sources[0]}) has no results.csv")
                continue

            client.download_artifacts(run_id, "results.csv", f)
            results = pl.read_csv(f"{f}/results.csv")

        results = results.with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.lit(sources[0]).alias("source"),
            pl.lit(sources).alias("sources"),
            pl.lit(run.data.tags["outcome"]).alias("outcome"),
        )

        all_results.append(results)

    results = pl.concat(all_results)

    metrics = map(re.compile(r"^[a-z]+\/test\/(.+)$").match, results.columns)
    metrics = np.unique([m.groups()[0] for m in metrics if m is not None])

    cv_results = []
    for metric in metrics:
        cv_columns = [x for x in results.columns if re.match(rf"^cv/{metric}_\d+$", x)]
        if len(cv_columns) == 0:
            logger.info(f"No cv columns for {metric}.")
            continue

        results = results.with_columns(
            pl.mean_horizontal(c for c in cv_columns).alias("cv_value")
        )
        grouped = (
            results.group_by("sources")
            .agg(
                pl.all().top_k_by(
                    k=1, by="cv_value", reverse=metric not in GREATER_IS_BETTER
                )
            )
            .explode([x for x in results.columns if x != "sources"])
        )
        grouped = grouped.with_columns(pl.lit(metric).alias("metric"))
        grouped = grouped.with_columns(
            pl.coalesce(
                pl.when(~pl.col("sources").list.contains(t)).then(pl.lit(t))
                for t in DATASETS
            ).alias("target"),
            pl.coalesce(
                pl.when(~pl.col("sources").list.contains(t)).then(
                    pl.col(f"{t}/test/{metric}")
                )
                for t in DATASETS
            ).alias("test_value"),
            *[pl.col(f"{t}/test/{metric}").alias(f"{t}/test_value") for t in DATASETS],
        )
        cv_results.append(grouped)
    cv_results = pl.concat(cv_results).drop("sources")

    target_run = client.search_runs(
        experiment_ids=[experiment_id], filter_string="tags.sources = ''"
    )
    if len(target_run) > 0:
        target_run = target_run[0]
    else:
        target_run = client.create_run(
            experiment_id=experiment_id, tags={"sources": ""}
        )

    print(f"logging to {target_run.info.run_id}")
    log_df(
        pl.DataFrame(cv_results),
        "1v1_results.csv",
        client,
        target_run.info.run_id,
    )


if __name__ == "__main__":
    main()

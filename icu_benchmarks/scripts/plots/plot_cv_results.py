import logging
import re
import tempfile

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.mlflow_utils import log_fig
from icu_benchmarks.plotting import PARAMETER_NAMES, plot_by_x

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
def main(experiment_name, tracking_uri):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)


    if "mlflow.note.content" in experiment.tags:
        print(experiment.tags["mlflow.note.content"])

    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id], filter_string="tags.sources != ''"
    )

    all_results = []
    for run in runs:
        run_id = run.info.run_id
        with tempfile.TemporaryDirectory() as f:
            if "results.csv" not in [x.path for x in client.list_artifacts(run_id)]:
                logger.warning(f"Run {run_id} has no results.csv")
                continue

            client.download_artifacts(run_id, "results.csv", f)
            results = pl.read_csv(f"{f}/results.csv")

        results = results.with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.lit(run.data.tags["sources"]).alias("sources"),
            pl.lit(run.data.tags["outcome"]).alias("outcome"),
        )
        # results = results.drop(pl.col(col) for col in results.columns if "/r2" in col)
        # results = results.drop(pl.col(c) for c in results.columns if "/val/" in c)
        # results = results.rename(
        #     {
        #         c: c.replace("/train/", "/train_val/")
        #         for c in results.columns
        #         if "/train/" in c
        #     }
        # )
        all_results.append(results)

    results = pl.concat(all_results, how="diagonal")

    run = client.search_runs(
        experiment_ids=[experiment_id], filter_string="tags.sources = ''"
    )
    if len(run) > 0:
        run = run[0]
    else:
        run = client.create_run(experiment_id=experiment_id, tags={"sources": ""})

    print(f"logging to {run.info.run_id}")
    metrics = map(re.compile(r"^[a-z]+\/test\/(.+)$").match, results.columns)
    metrics = np.unique([m.groups()[0] for m in metrics if m is not None])
    for metric in metrics:
        for x in [p for p in PARAMETER_NAMES if p in results.columns]:
            fig = plot_by_x(results, x, metric)
            log_fig(
                fig,
                f"plot_by_x/{metric}_{x}.png",
                client,
                run.info.run_id,
            )
            log_fig(
                fig,
                f"plot_by_x/{metric}_{x}.pdf",
                client,
                run.info.run_id,
            )
            plt.close(fig)


if __name__ == "__main__":
    main()

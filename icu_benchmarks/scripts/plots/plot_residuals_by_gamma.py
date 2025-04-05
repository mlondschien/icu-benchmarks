import json
import logging
import tempfile

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from icu_benchmarks.constants import GREATER_IS_BETTER, METRICS, PARAMETERS
from icu_benchmarks.mlflow_utils import get_target_run, log_fig
from icu_benchmarks.plotting import cv_results
import gin
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@gin.configurable()
def get_config(config):  # noqa D
    return config

@click.command()
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns3.db",
)
@click.option("--config", type=click.Path(exists=True))
def main(tracking_uri, config):  # noqa D
    gin.parse_config_file(config)
    CONFIG = get_config()
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment, run = get_target_run(client, CONFIG["experiment_name"])

    _, target_run = get_target_run(client, CONFIG["target_experiment"])

    all_results = []
    for run in client.search_runs(experiment_ids=[experiment.experiment_id]):
        run_id = run.info.run_id
        with tempfile.TemporaryDirectory() as f:
            if "results.csv" not in [x.path for x in client.list_artifacts(run_id)]:
                logger.warning(f"Run {run_id} has no results.csv")
                continue

            client.download_artifacts(run_id, "results.csv", f)
            results = pl.read_csv(f"{f}/results.csv")

        sources = json.loads(run.data.tags["sources"].replace("'", '"'))
        if len(sources) < 4:
            continue
        results = results.with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.lit(sources).alias("sources"),
            pl.lit(run.data.tags["outcome"]).alias("outcome"),
        )
        all_results.append(results)

    results = pl.concat(all_results, how="diagonal")

    ncols = int(len(CONFIG["panels"]) / 2)
    fig, axes = plt.subplots(
        2, ncols, figsize=(3 * ncols, 5), constrained_layout=True, gridspec_kw={"hspace": 0.02}#, sharex=True
    )

    for idx, (panel, ax) in enumerate(zip(CONFIG["panels"], axes.flat)):
        target = panel["target"]

        if target == "empty":
            ax.set_visible(False)
            continue

        cv = cv_results(
            results.filter(~pl.col("sources").list.contains(target)),
            CONFIG["metric"],
        ).group_by("gamma").agg(pl.all().top_k_by(k=1, by="__cv_value", reverse=CONFIG["metric"] not in GREATER_IS_BETTER)).select(pl.all().explode()).sort("gamma")

        ax.plot(
            cv["gamma"],
            cv[f"{target}/test/mean_residual"],
            color="black",
            label="mean" if idx == 0 else None,
        )

        ax.plot(
            cv["gamma"],
            np.zeros_like(cv["gamma"]),
            color="grey",
            ls="dotted",
            alpha=0.5,
        )

        color = "tab:blue"

        ax.plot(
            cv["gamma"],
            cv[f"{target}/test/quantile_0.5"],
            color=color,
            label="median" if idx == 0 else None,
        )

        ax.fill_between(
            cv["gamma"],
            cv[f"{target}/test/quantile_0.1"],
            cv[f"{target}/test/quantile_0.25"],
            color=color,
            alpha=0.1,
            label="10% - 90%" if idx == 0 else None,
        )

        ax.fill_between(
            cv["gamma"],
            cv[f"{target}/test/quantile_0.75"],
            cv[f"{target}/test/quantile_0.9"],
            color=color,
            alpha=0.1,
            label=None,
        )

        ax.fill_between(
            cv["gamma"],
            cv[f"{target}/test/quantile_0.25"],
            cv[f"{target}/test/quantile_0.75"],
            color=color,
            alpha=0.2,
            label="25% - 75%" if idx == 0 else None,
        )

        best = cv.select(
            pl.col("gamma").top_k_by(k=1, by="__cv_value", reverse=True)
        ).item()
        ax.axvline(best, color="black", ls="dashed", alpha=0.2)

        ax.set_xlabel("gamma")
        ax.set_ylabel("residuals")

        ax.set_title(panel["title"])
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.label_outer()
        ax.yaxis.set_tick_params(labelleft=True)  # manually add y ticks again
        # ax.xaxis.set_tick_params(labelbottom=True)

    fig.legend(loc="outside lower center", ncols=4)
    log_fig(
        fig,
        f"{CONFIG['filename']}.png",
        client,
        target_run.info.run_id,
    )
    log_fig(
        fig,
        f"{CONFIG['filename']}.pdf",
        client,
        target_run.info.run_id,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()

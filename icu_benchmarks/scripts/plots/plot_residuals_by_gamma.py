import json
import logging
import tempfile

import click
import gin
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import NullFormatter, StrMethodFormatter
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import GREATER_IS_BETTER
from icu_benchmarks.mlflow_utils import get_target_run, log_fig
from icu_benchmarks.plotting import cv_results

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
    metric = CONFIG["metric"]
    cv_metric = CONFIG.get("cv_metric", metric)

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

    if "filter" in CONFIG.keys():
        results = results.filter(**CONFIG["filter"])

    ncols = 3
    fig = plt.figure(figsize=(3.5 * ncols, 7))
    gs = gridspec.GridSpec(
        ncols + 1, 3, height_ratios=[1, 1, -0.1, 1], wspace=0.18, hspace=0.3
    )
    axes = [fig.add_subplot(gs[i, j]) for i in [0, 1, 3] for j in range(ncols)]

    for idx, (panel, ax) in enumerate(zip(CONFIG["panels"], axes)):
        target = panel["target"]

        if target == "empty":
            ax.set_visible(False)
            continue
        cv = results.filter(~pl.col("sources").list.contains(target))
        cv = (
            cv_results(
                cv,
                [cv_metric],
            )
            .group_by("gamma")
            .agg(
                pl.all().top_k_by(
                    k=1,
                    by=f"__cv_{cv_metric}",
                    reverse=cv_metric not in GREATER_IS_BETTER,
                )
            )
            .select(pl.all().explode())
            .sort("gamma")
        )

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
        # twin_ax = ax.twinx()
        # twin_ax.plot(
        #     cv["gamma"],
        #     cv[f"{target}/test/mse"],
        #     color="red",
        #     alpha=1,
        # )
        # twin_ax.plot(
        #     cv["gamma"],
        #     cv[f"{target}/test/mse"] - cv[f"{target}/test/mean_residual"].pow(2),
        #     color="red",
        #     alpha=1,
        # )
        # twin_ax.plot(
        #     cv["gamma"],
        #     cv[f"{target}/test/mean_residual"].pow(2),
        #     color="red",
        #     alpha=1,
        # )

        best = cv.select(
            pl.col("gamma").top_k_by(k=1, by=f"__cv_{cv_metric}", reverse=True)
        ).item()
        ax.axvline(best, color="black", ls="dashed", alpha=0.2)

        ax.set_xlabel("gamma", labelpad=2)
        ax.set_ylabel("residuals")

        ax.set_title(panel["title"], y=0.985)
        ax.set_xscale("log")

        def my_formatter(x, _):
            return f"{x:.0f}".lstrip("0")

        # ax.xaxis.set_tick_params(labelbottom=True)

        if panel["target"] == "empty":
            continue

        # if panel.get("yticks") is not None:
        #     ax.set_yticks(panel["yticks"])
        #     ax.set_yticklabels(panel.get("yticklabels", panel["yticks"]))
        # else:
        #     ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        ax.tick_params(axis="y", which="both", pad=1)
        ax.tick_params(axis="x", which="both", pad=2)

        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_tick_params(labelleft=True)  # manually add y ticks again

        ax.label_outer()
        ax.yaxis.set_tick_params(labelleft=True)  # manually add x & y ticks again
        ax.xaxis.set_tick_params(labelbottom=True)

    fig.align_ylabels(axes)

    line = plt.Line2D(
        [0.07, 0.925],
        [0.37, 0.37],
        transform=fig.transFigure,
        color="black",
        linewidth=2,
        alpha=0.5,
    )
    _ = plt.text(
        0.915,
        0.56,
        "core datasets",
        transform=fig.transFigure,
        fontsize=12,
        rotation=90,
        alpha=0.6,
    )
    _ = plt.text(
        0.915,
        0.165,
        "truly OOD",
        transform=fig.transFigure,
        fontsize=12,
        rotation=90,
        alpha=0.6,
    )

    fig.add_artist(line)
    # fig.legend(handles=legend_elements, loc="outside lower center", ncols=4)
    if CONFIG.get("title") is not None:
        fig.suptitle(CONFIG["title"], size="x-large", y=0.94)
    fig.legend(loc="outside lower center", ncols=4)

    log_fig(
        fig,
        f"{CONFIG['filename']}.pdf",
        client,
        run_id=target_run.info.run_id,
        bbox_inches="tight",
    )
    log_fig(
        fig,
        f"{CONFIG['filename']}.png",
        client,
        run_id=target_run.info.run_id,
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":
    main()

import logging
import tempfile

import click
import gin
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter, StrMethodFormatter
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import GREATER_IS_BETTER
from icu_benchmarks.mlflow_utils import log_fig
from icu_benchmarks.utils import fit_monotonic_spline

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

N_TARGET_VALUES = [
    25,
    35,
    50,
    70,
    100,
    140,
    200,
    280,
    400,
    560,
    800,
    1120,
    1600,
    2250,
    3200,
    4480,
    6400,
    8960,
    12800,
    17920,
    25600,
    35840,
    51200,
    71680,
    102400,
]


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
    client = MlflowClient(tracking_uri=tracking_uri)
    gin.parse_config_file(config)
    CONFIG = get_config()

    experiment = client.get_experiment_by_name(CONFIG["target_experiment"])

    if experiment is None:
        raise ValueError("target experiment not found")

    experiment_id = experiment.experiment_id

    target_runs = client.search_runs(
        experiment_ids=[experiment_id], filter_string="tags.sources = ''"
    )

    if len(target_runs) != 1:
        raise ValueError(f"Expected exactly one run. Got {target_runs:}")
    target_run = target_runs[0]

    ncols = int(len(CONFIG["panels"]) / 3)
    fig = plt.figure(figsize=(3.2 * ncols, 5))
    gs = gridspec.GridSpec(
        ncols + 2, 3, height_ratios=[1, 1, -0.15, 1, 0], wspace=0.19, hspace=0.4
    )
    axes = [fig.add_subplot(gs[i, j]) for i in [0, 1, 3] for j in range(ncols)]

    metric = CONFIG["metric"]
    cv_metric = CONFIG["cv_metric"]

    legend_handles = []
    labels = []

    for line in CONFIG["lines"]:
        experiment = client.get_experiment_by_name(line["experiment_name"])
        if experiment is None:
            raise ValueError(f"Did not find experiment {line['experiment_name']}.")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id], filter_string="tags.sources = ''"
        )
        if len(runs) != 1:
            raise ValueError(
                f"Expected exactly one run for {line['experiment_name']}. Got {runs:}"
            )

        run = runs[0]

        with tempfile.TemporaryDirectory() as f:
            client.download_artifacts(
                run.info.run_id, f"{line['result_name']}_results.csv", f
            )
            df = pl.read_csv(f"{f}/{line['result_name']}_results.csv")

        df = df.filter(pl.col("cv_metric") == cv_metric)

        if "filter" in line.keys():
            df = df.filter(**line["filter"])

        for panel, ax in zip(CONFIG["panels"], axes):
            if panel["source"] == "empty":
                continue

            xmin, xmax = panel["xlim"]
            column = f"{panel['source']}/test/{metric}"

            data = df.filter(pl.col("target") == panel["source"])
            if len(data) == 0:
                logger.warning(f"No data for {line}: {panel['source']}.")
                continue
            if len(data) == 1:
                ax.hlines(
                    data[column].first(),
                    xmin=xmin,
                    xmax=xmax,
                    # label=line["legend"] if idx == 0 else None,
                    color=line["color"],
                    alpha=line.get("alpha", 1),
                    ls=line["ls"],
                )
                continue

            x_new = np.asarray([x for x in N_TARGET_VALUES if xmin <= x <= xmax])
            y_new = np.empty((len(x_new), len(data["seed"].unique())))
            data = data.filter(pl.col("n_target").is_in(x_new)).sort("n_target")
            seeds = data["seed"].unique().to_numpy()
            for idx, seed in enumerate(seeds):
                y_new[:, idx] = fit_monotonic_spline(
                    data.filter(pl.col("seed").eq(seed))["n_target"].log(),
                    data.filter(pl.col("seed").eq(seed))[f"test_value/{metric}"],
                    np.log(x_new[:]),
                    increasing=metric in GREATER_IS_BETTER,
                )
            quantiles = np.quantile(y_new, [0.1, 0.5, 0.9], axis=1)
            idx = x_new.searchsorted(data["n_target"].max())

            ax.plot(
                x_new[: idx + 1],
                quantiles[1, : idx + 1],
                color=line["color"],
                ls=line["ls"],
                alpha=line.get("alpha", 1),
            )
            ax.plot(
                x_new[idx:],
                quantiles[1, idx:],
                color=line["color"],
                ls="dotted",
                alpha=line.get("alpha", 1),
            )
            ax.fill_between(
                x_new,
                quantiles[0, :],
                quantiles[2, :],
                color=line["color"],
                alpha=line.get("alpha", 1) * 0.1,
                hatch=line.get("hatch", None),
            )

        ls = "dashed" if isinstance(line["ls"], tuple) else line["ls"]
        handle = Line2D([], [], color=line["color"], ls=ls, alpha=line.get("alpha", 1))
        if len(data) > 1:
            legend_handles.append(
                (handle, Patch(color=line["color"], alpha=0.1 * line.get("alpha", 1)))
            )
        else:
            legend_handles.append(handle)
        labels.append(line["legend"])

    for panel, ax in zip(CONFIG["panels"], axes):
        ax.set_xscale("log")
        delta = np.pow(panel["xlim"][1] / panel["xlim"][0], 0.02)
        ax.set_xlim(panel["xlim"][0] / delta, panel["xlim"][1] * delta)
        if "ylim" in panel:
            ax.set_ylim(*panel["ylim"])
        if panel.get("yticks") is not None:
            ax.set_yticks(panel["yticks"])
            ax.set_yticklabels(panel.get("yticklabels", panel["yticks"]), fontsize=10)
        else:
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        if ax.get_subplotspec().colspan == range(0, 1):
            ax.set_ylabel(CONFIG["ylabel"], fontsize=10)
        if ax.get_subplotspec().rowspan == range(3, 4):
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks(panel["xticks"])
            ax.set_xticklabels(panel.get("xticklabels", panel["xticks"]), fontsize=10)
            ax.set_xlabel("number of patients from target", labelpad=3.5, fontsize=10)
        else:
            ax.xaxis.set_major_formatter(NullFormatter())
        ax.tick_params(axis="y", which="both", pad=0)
        ax.tick_params(axis="x", which="both", pad=2)

        ax.grid(visible=True, axis="x", alpha=0.2)
        ax.set_title(panel["title"], y=0.965, fontsize=10)

    fig.align_xlabels(axes)
    fig.align_ylabels(axes)
    line = plt.Line2D(
        [0.07, 0.925],
        [0.402, 0.402],
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
        fontsize=11,
        rotation=90,
        alpha=1.0,
        color="grey",
    )
    _ = plt.text(
        0.915,
        0.195,
        "truly OOD",
        transform=fig.transFigure,
        fontsize=11,
        rotation=90,
        alpha=1.0,
        color="grey",
    )
    fig.add_artist(line)

    fig.legend(
        legend_handles,
        labels,
        ncols=3,
        loc="center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        fontsize=10,
        handletextpad=0.8,
        columnspacing=1,
        labelspacing=0.5,
    )
    if CONFIG.get("title") is not None:
        fig.suptitle(CONFIG["title"], size=12, y=0.955)
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


if __name__ == "__main__":
    main()

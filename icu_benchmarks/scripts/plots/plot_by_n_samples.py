import logging
import tempfile

import click
import gin
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient
from icu_benchmarks.utils import fit_monotonic_spline
from icu_benchmarks.mlflow_utils import log_fig
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, StrMethodFormatter
import matplotlib.gridspec as gridspec
from icu_benchmarks.constants import GREATER_IS_BETTER

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

N_TARGET_VALUES = [25, 35, 50, 70, 100, 140, 200, 280, 400, 560, 800, 1120, 1600, 2250, 3200, 4480, 6400, 8960, 12800, 17920, 25600, 35840, 51200, 71680, 102400]
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
    fig = plt.figure(figsize=(3.5 * ncols, 7))
    gs = gridspec.GridSpec(ncols + 2, 3, height_ratios=[1, 1, -0.15, 1, -0.1], wspace=0.13, hspace=0.4)
    axes = [
        fig.add_subplot(
            gs[i, j]
        ) for i in [0, 1, 3] for j in range(ncols)
    ]

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
            xmin, xmax = panel["xlim"]
            column = f"{panel['source']}/test/{metric}"
            if panel["source"] == "empty":
                # ax.set_visible(False)
                continue

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
            for seed in data["seed"].unique():
                y_new[:, seed] = fit_monotonic_spline(
                    data.filter(pl.col("seed").eq(seed))["n_target"].log(),
                    data.filter(pl.col("seed").eq(seed))[f"test_value/{metric}"],
                    np.log(x_new),
                    increasing=metric in GREATER_IS_BETTER,
                )
                # ax.plot(x_new, y_new[:, seed], color=line["color"], alpha=0.1)
                # ax.scatter(
                #     data.filter(pl.col("seed").eq(seed))["n_target"],
                #     data.filter(pl.col("seed").eq(seed))[f"test_value/{metric}"],
                #     alpha=0.1, color=line["color"],
                # )

            quantiles = np.quantile(y_new, [0.1, 0.5, 0.9], axis=1)
            # data = (
            #     data.group_by("n_target")
            #     .agg(
            #         [
            #             pl.col(f"test_value/{metric}").median().alias("__score"),
            #             pl.col(f"test_value/{metric}").quantile(0.2).alias("__min"),
            #             pl.col(f"test_value/{metric}").quantile(0.8).alias("__max"),
            #         ]
            #     )
            #     .sort("n_target")
            # )
            idx = x_new.searchsorted(data["n_target"].max())
            ax.plot(
                x_new[:idx + 1],
                # data["n_target"],
                # data["__score"],
                quantiles[1, :idx + 1],
                # label=line["legend"] if idx == 0 else None,
                color=line["color"],
                ls=line["ls"],
                alpha=line.get("alpha", 1),
            )
            ax.plot(
                x_new[idx:],
                # data["n_target"],
                # data["__score"],
                quantiles[1, idx:],
                # label=line["legend"] if idx == 0 else None,
                color=line["color"],
                ls="dotted",
                alpha=line.get("alpha", 1),
            )
            ax.fill_between(
                x_new,
                # data["n_target"],
                quantiles[0, :],
                quantiles[2, :],
                # data["__min"],
                # data["__max"],
                color=line["color"],
                alpha=line.get("alpha", 1) * 0.1,
                hatch=line.get("hatch", None),
            )
            # ax.fill_between(
            #     x_new[idx:],
            #     # data["n_target"],
            #     quantiles[0, idx:],
            #     quantiles[2, idx:],
            #     # data["__min"],
            #     # data["__max"],
            #     color=line["color"],
            #     alpha=0.1, #line.get("alpha", 1) * 0.1,
            #     # hatch=".",
            # )
        
        ls = "dashed" if isinstance(line["ls"], tuple) else line["ls"]
        handle = Line2D([], [], color=line["color"], ls=ls, alpha=line.get("alpha", 1))
        if len(data) > 1:
            legend_handles.append((handle, Patch(color=line["color"], alpha=0.1*line.get("alpha", 1))))
        else:
            legend_handles.append(handle)
        labels.append(line["legend"])

    for panel, ax in zip(CONFIG["panels"], axes):
        ax.set_xscale("log")
        delta = np.pow(panel["xlim"][1] / panel["xlim"][0], 0.02)
        ax.set_xlim(panel["xlim"][0] / delta, panel["xlim"][1] * delta)
        ax.set_ylim(*panel["ylim"])
        if panel.get("yticks") is not None:
            ax.set_yticks(panel["yticks"])
            ax.set_yticklabels(panel.get("yticklabels", panel["yticks"]))
        else:
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        if ax.get_subplotspec().colspan == range(0, 1):
            ax.set_ylabel(CONFIG["ylabel"])
        if ax.get_subplotspec().rowspan == range(3, 4):
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks(panel["xticks"])
            ax.set_xticklabels(panel.get("xticklabels", panel["xticks"]))
            ax.set_xlabel("number of patient stays from target", labelpad=3.5)
        else:
            ax.xaxis.set_major_formatter(NullFormatter())
        ax.tick_params(axis="y", which="both", pad=1)
        ax.tick_params(axis="x", which="both", pad=2)

        ax.grid(visible=True, axis="x", alpha=0.2)
        # ax.yaxis.set_tick_params(labelleft=True)  # manually add x & y ticks again
        # ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_title(panel["title"], y=0.985)
    
    fig.align_xlabels(axes)
    fig.align_ylabels(axes)
    line = plt.Line2D([0.07, 0.925], [0.388, 0.388], transform=fig.transFigure, color="black", linewidth=2, alpha=0.5)
    _ = plt.text(0.915, 0.565, "core datasets", transform=fig.transFigure, fontsize=12, rotation=90, alpha=0.65)
    _ = plt.text(0.915, 0.185, "truly OOD", transform=fig.transFigure, fontsize=12, rotation=90, alpha=0.65)
    fig.add_artist(line)

    fig.legend(legend_handles, labels, loc="outside lower center", ncols=2)
    if CONFIG.get("title") is not None:
        fig.suptitle(CONFIG["title"], size="x-large", y=0.94)
    log_fig(fig, f"{CONFIG['filename']}.png", client, run_id=target_run.info.run_id, bbox_inches='tight')
    log_fig(fig, f"{CONFIG['filename']}.pdf", client, run_id=target_run.info.run_id, bbox_inches='tight')


if __name__ == "__main__":
    main()

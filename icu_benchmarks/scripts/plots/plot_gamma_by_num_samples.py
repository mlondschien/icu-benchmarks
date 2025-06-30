import logging
import tempfile

import click
import gin
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import DATASETS, GREATER_IS_BETTER
from icu_benchmarks.mlflow_utils import get_results, get_target_run, log_fig
from icu_benchmarks.plotting import SOURCE_COLORS, VERY_SHORT_DATASET_NAMES, cv_results
from icu_benchmarks.utils import fit_monotonic_spline

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@gin.configurable()
def get_config(config):  # noqa D
    return config


N_TARGET_VALUES = np.geomspace(10, 1e6, 200)


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
    _, target_run = get_target_run(client, CONFIG["target_experiment_name"])

    fig = plt.figure(figsize=(7, 2.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    for idx, panel in enumerate(CONFIG["panels"]):
        results = get_results(
            client, panel["experiment_name"], f"{panel['result_name']}.csv"
        )

        if "num_iteration" in results.columns:
            results = results.filter(pl.col("num_iteration").eq(1000))
        if "max_depth" in results.columns:
            results = results.filter(pl.col("max_depth").eq(3))
        if "alpha" in results.columns:
            col = pl.col("alpha").log10() - pl.col("alpha").min().log10()
            results = results.filter((col - 3).abs().le(0.01))
            results = results.filter(pl.col("l1_ratio").eq(0.5))

        _, n_samples_run = get_target_run(client, panel["n_samples_experiment_name"])

        with tempfile.TemporaryDirectory() as f:
            file = f"{panel['n_samples_result_name']}.csv"
            client.download_artifacts(n_samples_run.info.run_id, file, f)
            n_samples = pl.read_csv(f"{f}/{file}")

        if panel.get("break_y_axis", False):
            local_gs = gridspec.GridSpecFromSubplotSpec(
                subplot_spec=gs[idx],
                ncols=1,
                nrows=2,
                hspace=0.1,
                height_ratios=[np.log(i[1]) - np.log(i[0]) for i in panel["ylim"]],
            )
            big_ax = plt.Subplot(fig, gs[idx])
            [sp.set_visible(False) for sp in big_ax.spines.values()]
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax.patch.set_facecolor("none")

            fig.add_subplot(big_ax)

            local_axes = []
            for igs, ylim, yticks, yticklabels in zip(
                local_gs, panel["ylim"], panel["yticks"], panel["yticklabels"]
            ):
                ax = plt.Subplot(fig, igs)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(*panel["xlim"])
                ax.set_ylim(ylim)
                ax.yaxis.set_major_locator(FixedLocator(yticks))
                ax.yaxis.set_major_formatter(FixedFormatter(yticklabels))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.yaxis.offsetText.set_visible(False)
                ax.yaxis.get_offset_text().set_visible(False)
                ax.set_xticks(panel["xticks"])
                ax.set_xticklabels(panel["xticks"])
                ax.xaxis.tick_bottom()
                ax.yaxis.tick_left()
                fig.add_subplot(ax)
                ax.xaxis.set_minor_locator(NullLocator())
                local_axes.append(ax)

            local_axes[1].set_xlabel("$\\gamma$")
            local_axes[0].spines["bottom"].set_visible(False)
            plt.setp(local_axes[0].xaxis.get_minorticklabels(), visible=False)
            plt.setp(local_axes[0].xaxis.get_minorticklines(), visible=False)
            plt.setp(local_axes[0].xaxis.get_majorticklabels(), visible=False)
            plt.setp(local_axes[0].xaxis.get_majorticklines(), visible=False)

            local_axes[1].spines["top"].set_visible(False)

            kwargs = dict(
                marker=[(-1, -0.5), (1, 0.5)],
                markersize=10,
                linestyle="none",
                color="k",
                mec="k",
                mew=1,
                clip_on=False,
            )
            local_axes[0].plot(
                [0, 1], [0, 0], transform=local_axes[0].transAxes, **kwargs
            )
            local_axes[1].plot(
                [0, 1], [1, 1], transform=local_axes[1].transAxes, **kwargs
            )
            if idx == 0:
                labelpad = 28
        else:
            big_ax = fig.add_subplot(gs[idx])
            local_axes = [big_ax]
            big_ax.set_yscale("log")
            big_ax.set_xscale("log")
            big_ax.set_xlim(*panel["xlim"])
            big_ax.set_xticks(panel["xticks"])
            big_ax.set_xticklabels(panel["xticks"])
            big_ax.set_ylim(*panel["ylim"])
            big_ax.set_yticks(panel["yticks"])
            big_ax.set_yticklabels(panel["yticklabels"])
            big_ax.set_xlabel("$\\gamma$")
            big_ax.xaxis.set_minor_locator(NullLocator())
            if idx == 0:
                labelpad = None
        if idx == 0:
            big_ax.set_ylabel("equivalent number of patients", labelpad=labelpad)
        big_ax.set_title(panel["title"], fontsize=10)

        for ldx, ax in enumerate(local_axes):
            for target in DATASETS:
                cv = results.filter(~pl.col("sources").list.contains(target))
                cv = cv_results(cv, [panel["cv_metric"]]).sort(by="gamma")

                n_target = n_samples.filter(
                    pl.col("target").eq(target)
                    & pl.col("cv_metric").eq(panel["cv_metric"])
                )
                n_target = n_target.sort(pl.col("n_target"))

                x = np.empty((len(N_TARGET_VALUES), n_target["seed"].n_unique()))
                for seed in n_target.select("seed").unique().to_numpy().flatten():
                    x[:, seed] = fit_monotonic_spline(
                        n_target.filter(pl.col("seed") == seed)["n_target"].log(),
                        n_target.filter(pl.col("seed") == seed)[
                            f"test_value/{panel['metric']}"
                        ],
                        np.log(N_TARGET_VALUES),
                        increasing=panel["metric"] in GREATER_IS_BETTER,
                    )

                mult = 1 if panel["metric"] in GREATER_IS_BETTER else -1
                x = np.median(x, axis=1)
                y = np.interp(
                    mult * cv[f"{target}/test/{panel['metric']}"].to_numpy(),
                    mult * x,
                    np.log(N_TARGET_VALUES),
                    left=np.nan,
                    right=np.nan,
                )
                ax.plot(
                    cv["gamma"].to_numpy(),
                    np.exp(y),
                    color=SOURCE_COLORS[target],
                    label=VERY_SHORT_DATASET_NAMES[target] if idx == ldx == 0 else None,
                    zorder=10 - idx,
                )
                cv = cv.with_columns(y=np.exp(y))
                best = cv.top_k(
                    1,
                    by=f"__cv_{panel['cv_metric']}",
                    reverse=panel["cv_metric"] not in GREATER_IS_BETTER,
                )
                ax.scatter(
                    best["gamma"].item(),
                    best["y"].item(),
                    color=SOURCE_COLORS[target],
                    marker="*",
                    s=100,
                    zorder=20 - idx,
                )
    fig.suptitle(CONFIG["title"], fontsize=11, y=1.05)
    fig.legend(
        loc="center",
        ncol=1,
        bbox_to_anchor=(1.01, 0.5),
        fontsize=10,
        frameon=False,
    )
    log_fig(
        fig,
        f"{CONFIG['filename']}.pdf",
        client,
        target_run.info.run_id,
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()

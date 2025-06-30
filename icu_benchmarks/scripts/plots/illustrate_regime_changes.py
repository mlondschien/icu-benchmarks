import json
import logging
import re
import tempfile

import click
import gin
from matplotlib.transforms import Bbox
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import NullFormatter, StrMethodFormatter, NullLocator
from mlflow.tracking import MlflowClient
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from matplotlib.patches import Patch
from icu_benchmarks.constants import GREATER_IS_BETTER, PARAMETERS
from icu_benchmarks.mlflow_utils import get_target_run, log_fig
from icu_benchmarks.plotting import cv_results
from icu_benchmarks.utils import fit_monotonic_spline, find_intersection

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

@click.command()
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns3.db",
)
def main(tracking_uri):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)

    _, target_run = get_target_run(client, "plots")
    
    fig, axes = plt.subplots(ncols=2, figsize=(6, 1.4))
    _, cv_run = get_target_run(client, "crea_algbm_7_d")
    with tempfile.TemporaryDirectory() as f:
        client.download_artifacts(cv_run.info.run_id, "cv_results.csv", f)
        cv = pl.read_csv(f"{f}/cv_results.csv")
        cv = cv.filter(pl.col("cv_metric") == "mse")

    _, refit_run = get_target_run(client, "crea_algbm_7_d")
    with tempfile.TemporaryDirectory() as f:
        client.download_artifacts(refit_run.info.run_id, "refit_lgbm_results.csv", f)
        refit = pl.read_csv(f"{f}/refit_lgbm_results.csv")
        refit = refit.filter(pl.col("cv_metric") == "mse")
    
    _, n_samples_run = get_target_run(client, "crea_algbm_n_samples_7_1")
    with tempfile.TemporaryDirectory() as f:
        client.download_artifacts(n_samples_run.info.run_id, "n_samples_results.csv", f)
        n_samples = pl.read_csv(f"{f}/n_samples_results.csv")
        n_samples = n_samples.filter(pl.col("cv_metric") == "mse")

    panels = [
        {
            "target":"eicu",
            "title":"eICU",
            "xlim":(25, 102400),
            "ylim":(0.0745, 0.0815),
            "xticks":[25, 100, 1000, 10000, 100000],
            "xticklabels":["25", "100", "1k", "10k", "100k"],
            "yticks":[0.075, 0.076, 0.077, 0.078, 0.079, 0.08, 0.081],
            "yticklabels":["", ".076", "", ".078", "", ".08", ""],
        },
        {
            "target": "picdb",
            "title": "PICdb",
            "xlim": (25, 10000),
            "ylim": (0.111, 0.195),
            "xticks": [25, 100, 1000, 10000],
            "xticklabels": ["25", "100", "1k", "10k"],
            "yticks": [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19],
            "yticklabels": [".12", "", ".14", "", ".16", "", ".18", ""],
        },
    ]

    for idx, panel in enumerate(panels):
        ax = axes[idx]
        # ax.set_title(panel['title'], fontsize=10, pad=15)
        ax.set_xscale("log")
        ax.set_xlim(*panel["xlim"])
        ax.set_ylim(*panel["ylim"])
        ax.set_xlabel("num. patients from target domain", fontsize=10)
        if idx == 0:
            ax.set_ylabel("MSE", fontsize=10)
        ax.set_xticks(panel["xticks"])
        ax.set_xticklabels(panel["xticklabels"], fontsize=10)
        ax.set_yticks(panel["yticks"])
        ax.set_yticklabels(panel["yticklabels"], fontsize=10)

        cv_val = cv.filter(pl.col("target").eq(panel['target']))[f"{panel['target']}/test/mse"].item()
        color, ls = "#004488", "dashed"
        ax.hlines(
            cv_val,
            xmin=panel["xlim"][0],
            xmax=panel["xlim"][1],
            color=color,
            ls=ls,
        )
        handle_1 = Line2D([], [], color=color, ls=ls)
        x = np.asarray(N_TARGET_VALUES)

        y_refit = np.empty((len(x), 20))
        color, ls = "#004488", "solid"
        for seed in range(20):
            df = refit.filter(pl.col("seed").eq(seed) & pl.col("target").eq(panel['target']))
            y_refit[:, seed] = fit_monotonic_spline(
                df["n_target"].log(),
                df["test_value/mse"],
                np.log(x),
                increasing=False,
            )
        refit_q = np.quantile(y_refit, [0.1, 0.5, 0.9], axis=1)
        refit_std = np.std(y_refit, axis=1)

        idx = x.searchsorted(df["n_target"].max())
        ax.plot(x[: idx + 1], refit_q[1, : idx + 1], color=color, ls=ls)
        handle_2 = (Line2D([], [], color=color, ls=ls), Patch(color=color, alpha=0.1))
        ax.plot(x[idx:], refit_q[1, idx:], color=color, ls="dotted")
        ax.fill_between(x, refit_q[0, :], refit_q[2, :], color=color, alpha=0.1)

        y_n_samples = np.empty((len(x), 20))
        color, ls = "#DDAA33", "dashdot"
        for seed in range(20):
            df = n_samples.filter(pl.col("seed").eq(seed) & pl.col("target").eq(panel["target"]))
            y_n_samples[:, seed] = fit_monotonic_spline(
                df["n_target"].log(),
                df["test_value/mse"],
                np.log(x),
                increasing=False,
            )

        n_samples_q = np.quantile(y_n_samples, [0.1, 0.5, 0.9], axis=1)
        idx = x.searchsorted(df["n_target"].max())

        ax.plot(x[: idx + 1], n_samples_q[1, : idx + 1], color=color, ls=ls)
        handle_3 = (Line2D([], [], color=color, ls=ls), Patch(color=color, alpha=0.1))
        ax.plot(x[idx:], n_samples_q[1, idx:], color=color, ls="dotted")
        ax.fill_between(x, n_samples_q[0, :], n_samples_q[2, :], color=color, alpha=0.1)

        cv_vs_n_samples = np.exp(find_intersection(n_samples_q[1, :] - cv_val, np.log(x), increasing=False))
        cv_vs_refit = np.exp(
            find_intersection(
                refit_q[1, :] + 0.1 * refit_std - cv_val,
                np.log(x),
                increasing=False,
            )
        )
        if cv_vs_refit == 0:
            cv_vs_refit = 25
        n_samples_vs_refit = np.exp(
            find_intersection(
                n_samples_q[1, :] - refit_q[1, :],
                np.log(x),
                increasing=False,
            )
        )

        ax.vlines(
            [cv_vs_refit, n_samples_vs_refit],
            ymin=panel["ylim"][0],
            ymax=panel["ylim"][1],
            color="grey",
            ls="dashed",
            alpha=0.5,
        )
        ax.vlines(
            [cv_vs_n_samples],
            ymin=cv_val,
            ymax=panel["ylim"][1],
            color="grey",
            ls="dotted",
            alpha=0.5,
        )
        y = -0.08 * panel["ylim"][0] + 1.08 * panel["ylim"][1]
        ax.scatter([cv_vs_n_samples], [y], color="black", marker="X", s=40,clip_on=False)
        ax.scatter([cv_vs_refit], [y], color="black", marker="o", s=40,clip_on=False)
        ax.scatter([n_samples_vs_refit], [y], color="black", marker="s", s=40,clip_on=False)
        # dummy = ax.scatter([n_samples_vs_refit], [y + 0.1 * panel["ylim"][1]], color="black", marker="s", alpha=0, s=40,clip_on=False)
        
        y_text = 0.8 * panel["ylim"][1] + 0.2 * panel["ylim"][0]
        if panel["target"] != "picdb":
            ax.text(
                np.sqrt(panel["xlim"][0] * cv_vs_refit),
                y_text,
                "(a)",
                fontsize=10,
                ha="center",
                va="center",
                weight="bold"
            )

        # x_text = 4200 if panel["target"] == "eicu" else 120
        x_text = np.sqrt(cv_vs_refit * n_samples_vs_refit)
        ax.text(
            x_text,
            y_text,
            "(b)",
            fontsize=10,
            ha="center",
            va="center",
            weight="bold"
        )
        ax.text(
            np.sqrt(panel["xlim"][1] * n_samples_vs_refit),
            y_text,
            "(c)",
            fontsize=10,
            ha="center",
            va="center",
            weight="bold"
        )
    # fig.suptitle("log(creatinine) in 24h", fontsize=11, y=1.2)
    legend = fig.legend(
        [handle_1, handle_2, handle_3],
        [
            "anchor boosting fit on source",
            "anchor boosting refit on target",
            "standard boosting fit on target",
        ],
        ncols=1, loc="center",  bbox_to_anchor=(1.15, 0.5) , frameon=False, fontsize=10, handletextpad=0.8, columnspacing=1, labelspacing=0.5)
    plt.draw()
    # fig.subplots_adjust(top=5)
    # fig.constrained_layout(rect=(0, 0, 1.5, 1.1))
    log_fig(
        fig, "illustration.pdf", client, run_id=target_run.info.run_id,  bbox_extra_artists=[legend], bbox_inches=Bbox([[0.1, -0.3], [8.25, 1.4]]),# (-1, -1, 2, 2),# , bbox_extra_artists=[legend]
    )

if __name__ == "__main__":
    main()
import logging
import tempfile

import click
import gin
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import FancyArrowPatch, Patch
from mlflow.tracking import MlflowClient
from icu_benchmarks.constants import GREATER_IS_BETTER, VERY_SHORT_DATASET_NAMES
from icu_benchmarks.mlflow_utils import get_target_run, log_fig
from icu_benchmarks.utils import fit_monotonic_spline, find_intersection
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

N_TARGET_VALUES = np.array(
    [
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
    client = MlflowClient(tracking_uri=tracking_uri)
    gin.parse_config_file(config)

    CONFIG = get_config()

    _, target_run = get_target_run(client, CONFIG["target_experiment"])

    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True, sharex=True)

    datasets = CONFIG["datasets"]
    datasets.reverse()

    legend_handles = []
    labels = []
    for experient_idx, experiment in enumerate(CONFIG["experiments"]):
        legend_handles.append(Patch(color=experiment["color"]))
        labels.append(experiment["name"])

        group_shift = -0.3 + 0.6 * experient_idx / (len(CONFIG["experiments"]) - 1)

        _, run = get_target_run(
            client, experiment["cv_experiment_name"], create_if_not_exists=False
        )
        with tempfile.TemporaryDirectory() as f:
            client.download_artifacts(run.info.run_id, "cv_results.csv", f)
            cv_data = pl.read_csv(f"{f}/cv_results.csv")

        _, run = get_target_run(
            client, experiment["n_samples_experiment_name"], create_if_not_exists=False
        )
        with tempfile.TemporaryDirectory() as f:
            client.download_artifacts(run.info.run_id, "n_samples_results.csv", f)
            n_samples_data = pl.read_csv(f"{f}/n_samples_results.csv")

        _, run = get_target_run(
            client, experiment["refit_experiment_name"], create_if_not_exists=False
        )
        with tempfile.TemporaryDirectory() as f:
            file_name = f"{experiment['refit_result_name']}_results.csv"
            client.download_artifacts(run.info.run_id, file_name, f)
            refit_data = pl.read_csv(f"{f}/{file_name}")

        cv_metric = experiment["cv_metric"]
        metric = experiment["metric"]
        column = f"test_value/{metric}"

        n_samples_data = n_samples_data.filter(pl.col("cv_metric").eq(cv_metric))
        refit_data = refit_data.filter(pl.col("cv_metric").eq(cv_metric))
        cv_data = cv_data.filter(pl.col("cv_metric").eq(cv_metric)).with_columns(
            pl.coalesce(
                pl.when(pl.col("target").eq(t)).then(pl.col(f"{t}/test/{metric}"))
                for t in CONFIG["datasets"]
            ).alias(column)
        )

        for target_idx, target in enumerate(datasets):
            df = n_samples_data.filter(pl.col("target") == target)
            seeds = n_samples_data["seed"].unique().to_list()
            n_samples_new = np.empty((len(N_TARGET_VALUES), len(seeds)))
            for seed_idx, seed in enumerate(seeds):
                n_samples_new[:, seed_idx] = fit_monotonic_spline(
                    df.filter(pl.col("seed") == seed)["n_target"].log().to_numpy(),
                    df.filter(pl.col("seed") == seed)[column].to_numpy(),
                    np.log(N_TARGET_VALUES),
                    increasing=metric in GREATER_IS_BETTER,
                )

            df = refit_data.filter(pl.col("target") == target)
            seeds = refit_data["seed"].unique().to_list()
            refit_new = np.empty((len(N_TARGET_VALUES), len(seeds)))
            for seed_idx, seed in enumerate(seeds):
                refit_new[:, seed_idx] = fit_monotonic_spline(
                    df.filter(pl.col("seed") == seed)["n_target"].log().to_numpy(),
                    df.filter(pl.col("seed") == seed)[column].to_numpy(),
                    np.log(N_TARGET_VALUES),
                    increasing=metric in GREATER_IS_BETTER,
                )

            n_samples = np.median(n_samples_new, axis=1)

            refit = np.median(refit_new, axis=1)
            refit_std = np.std(refit_new, axis=1)
            mult = 0.02 if metric in GREATER_IS_BETTER else -0.1
            
            cv = cv_data.filter(pl.col("target").eq(target))[column].item()
            cv_vs_n_samples = np.exp(
                find_intersection(
                n_samples - cv,
                np.log(N_TARGET_VALUES),
                increasing=metric in GREATER_IS_BETTER,
                )
            )
            cv_vs_refit = np.exp(
                find_intersection( refit - mult * refit_std - cv, np.log(N_TARGET_VALUES), increasing=metric in GREATER_IS_BETTER)
            )
            n_samples_vs_refit = np.exp(
                find_intersection(
                    n_samples - refit,
                    np.log(N_TARGET_VALUES),
                    increasing=metric in GREATER_IS_BETTER,
                )
            )

            n_max = df["n_target"].max()

            if np.isfinite(cv_vs_refit) and cv_vs_refit > 0:
                ax.scatter(
                    cv_vs_refit,
                    target_idx + group_shift,
                    marker="o",
                    color=experiment["color"],
                    alpha=1,
                    s=25,
                    zorder=4,
                )
            elif cv_vs_refit == 0:
                cv_vs_refit = np.min(N_TARGET_VALUES)

            if cv_vs_n_samples > 0 and cv_vs_n_samples <= n_max * 4:
                ax.scatter(
                    cv_vs_n_samples,
                    target_idx + group_shift,
                    marker="X",
                    color=experiment["color"],
                    alpha=1,
                    s=25,
                    zorder=4,
                )

            if n_samples_vs_refit > 0 and n_samples_vs_refit <= n_max * 4:
                ax.scatter(
                    n_samples_vs_refit,
                    target_idx + group_shift,
                    marker="s",
                    color=experiment["color"],
                    alpha=1,
                    s=25,
                    zorder=4,
                )
            elif n_samples_vs_refit > 0:
                n_samples_vs_refit = n_max * 4

            if not np.isfinite(cv_vs_refit):
                continue
            if not np.isfinite(n_samples_vs_refit):
                continue

            if n_samples_vs_refit > n_max:
                patch = FancyArrowPatch(
                    posA=(cv_vs_refit, target_idx + group_shift),
                    posB=(n_max, target_idx + group_shift),
                    arrowstyle="-",
                    ls="solid",
                    color=experiment["color"],
                    lw=1.5,
                    alpha=0.5,
                )
                ax.add_patch(patch)

                patch = FancyArrowPatch(
                    posA=(n_max, target_idx + group_shift),
                    posB=(n_samples_vs_refit, target_idx + group_shift),
                    arrowstyle="-",
                    ls=(0, (4, 4)),
                    color=experiment["color"],
                    lw=1.5,
                    alpha=0.5,
                )
                ax.add_patch(patch)
            else:
                patch = FancyArrowPatch(
                    posA=(cv_vs_refit, target_idx + group_shift),
                    posB=(n_samples_vs_refit, target_idx + group_shift),
                    arrowstyle="-",
                    color=experiment["color"],
                    lw=1,
                    alpha=0.5,
                )
                ax.add_patch(patch)

    legend_handles = [legend_handles[1], legend_handles[0], legend_handles[3], legend_handles[2]]
    labels = [labels[1], labels[0], labels[3], labels[2]]
    fig.legend(
        legend_handles,
        labels,
        loc="outside lower center",
        ncols=int(len(legend_handles) / 2),
        fontsize=10,
        frameon=False,
    )
    ax.set_xscale("log")
    ax.set_xlim(CONFIG["xlim"])
    ax.set_xticks(CONFIG["xticks"])
    ax.set_xticklabels(CONFIG["xticklabels"], fontsize=10)
    ax.set_xlabel("number of patients from target", labelpad=3.5, fontsize=10)

    ax.set_yticks(np.arange(len(CONFIG["datasets"])))
    ax.set_yticklabels(
        [VERY_SHORT_DATASET_NAMES[x] for x in datasets],
        fontsize=11,
    )
    log_fig(
        fig, f"{CONFIG['filename']}.pdf", client, run_id=target_run.info.run_id, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()





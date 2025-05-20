import logging
import tempfile

import click
import gin
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.mlflow_utils import log_fig, get_target_run
from icu_benchmarks.constants import GREATER_IS_BETTER, SHORT_DATASET_NAMES
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import matplotlib.gridspec as gridspec
from icu_benchmarks.utils import fit_monotonic_spline, find_intersection
from matplotlib.patches import FancyArrowPatch

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

N_TARGET_VALUES = np.array([25, 35, 50, 70, 100, 140, 200, 280, 400, 560, 800, 1120, 1600, 2250, 3200, 4480, 6400, 8960, 12800, 17920, 25600, 35840, 51200, 71680, 102400])

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

    fig, ax = plt.subplots(
        figsize=(3.5 * 2, 5), constrained_layout=True, sharex=True
    )

    rng = np.random.default_rng(0)
    # ncols = int(len(CONFIG["panels"]) / 3)
    # fig = plt.figure(figsize=(3.5 * ncols, 7))# , constrained_layout=True,sharex=True)
    # gs = gridspec.GridSpec(ncols + 2, 3, height_ratios=[1, 1, -0.27, 1, -0.2], wspace=0.13, hspace=0.68)
    # axes = [
    #     fig.add_subplot(
    #         gs[i, j] #, sharex=gs[0,j] if i > 0 else None, sharey=gs[0,j] if j > 0 else None
    #     ) for i in [0, 1, 3] for j in range(ncols)
    # ]

    # metric = CONFIG["metric"]
    # cv_metric = CONFIG["cv_metric"]

    # legend_handles = []
    # labels = []
    
    datasets = CONFIG["datasets"]
    datasets.reverse()

    legend_handles = []
    labels = []
    for experient_idx, experiment in enumerate(CONFIG["experiments"]):
        legend_handles.append((Patch(color=experiment["color"])))
        labels.append(experiment["name"])

        group_shift = -0.3 + 0.6 * experient_idx / (len(CONFIG["experiments"]) - 1)
        # scatter = 0.4 / (len(CONFIG["experiments"]) - 1) / 3

        _, run = get_target_run(client, experiment["cv_experiment_name"], create_if_not_exists=False)
        with tempfile.TemporaryDirectory() as f:
            client.download_artifacts(run.info.run_id, f"cv_results.csv", f)
            cv_data = pl.read_csv(f"{f}/cv_results.csv")

        _, run = get_target_run(client, experiment["n_samples_experiment_name"], create_if_not_exists=False)
        with tempfile.TemporaryDirectory() as f:
            client.download_artifacts(run.info.run_id, f"n_samples_results.csv", f)
            n_samples_data = pl.read_csv(f"{f}/n_samples_results.csv")

        _, run = get_target_run(client, experiment["refit_experiment_name"], create_if_not_exists=False)
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

        data = cv_data.join(n_samples_data, on=["target"], how="full", validate="1:m", suffix="_n_samples")
        data = data.join(refit_data, on=["target", "n_target", "seed"], how="full", validate="1:1", suffix="_refit")

        val = (pl.col("n_target") / 100).log() / np.log(np.sqrt(2))
        data = data.filter((val - val.round()).abs() < 0.01)

        for target_idx, target in enumerate(datasets):
            df = data.filter(pl.col("target") == target)

            n_samples_new = np.empty((len(N_TARGET_VALUES), len(df["seed"].unique())))
            refit_new = np.empty((len(N_TARGET_VALUES), len(df["seed"].unique())))

            for seed_idx, seed in enumerate(df["seed"].unique()):
                n_samples_new[:, seed_idx] = fit_monotonic_spline(
                    df.filter(pl.col("seed") == seed)["n_target"].log().to_numpy(),
                    df.filter(pl.col("seed") == seed)[f"{column}_n_samples"].to_numpy(),
                    np.log(N_TARGET_VALUES),
                    increasing=metric in GREATER_IS_BETTER,
                )
                refit_new[:, seed_idx] = fit_monotonic_spline(
                    df.filter(pl.col("seed") == seed)["n_target"].log().to_numpy(),
                    df.filter(pl.col("seed") == seed)[f"{column}_refit"].to_numpy(),
                    np.log(N_TARGET_VALUES),
                    increasing=metric in GREATER_IS_BETTER,
                )

                # cv_vs_n_samples = np.exp(find_intersection(
                #     np.log(N_TARGET_VALUES),
                #     n_samples_new[:, seed_idx] - df[column].first(),
                #     increasing=metric in GREATER_IS_BETTER,
                # ))
                # cv_vs_refit = np.exp(find_intersection(
                #     np.log(N_TARGET_VALUES),
                #     refit_new[:, seed_idx] - df[column].first(),
                #     increasing=metric in GREATER_IS_BETTER,
                # ))
                # n_samples_vs_refit = np.exp(find_intersection(
                #     np.log(N_TARGET_VALUES),
                #     n_samples_new[:, seed_idx] - refit_new[:, seed_idx],
                #     increasing=metric in GREATER_IS_BETTER,
                # ))

                # seed_shift = 0.3 / (len(CONFIG["experiments"]) - 1) * (seed_idx - 10) / (len(df["seed"].unique()) - 1)
                # for value, nan_value in [
                #     (cv_vs_n_samples, np.max(N_TARGET_VALUES)),
                #     (cv_vs_refit, np.min(N_TARGET_VALUES)),
                #     (n_samples_vs_refit, np.max(N_TARGET_VALUES)),
                # ]:
                #     if np.isnan(value):
                #         ax.scatter(
                #             nan_value,
                #             target_idx + group_shift + seed_shift,
                #             marker="o",
                #             color=experiment["color"],
                #             alpha=0.2,
                #             s=15,
                #             zorder=3,
                #         )
                #     else:
                #         ax.scatter(
                #             value,
                #             target_idx + group_shift,
                #             marker="o",
                #             color=experiment["color"],
                #             alpha=0.2,
                #             s=15,
                #             zorder=3,
                #         )
                # arrowstyle = "-|>, head_width=5, head_length=5"
                # if np.isnan(n_samples_vs_refit):
                #     n_samples_vs_refit = np.max(N_TARGET_VALUES)

                # patch = FancyArrowPatch(
                #     posA=(n_samples_vs_refit * 0.99, target_idx + group_shift + seed_shift),
                #     posB=(n_samples_vs_refit, target_idx + group_shift + seed_shift),
                #     arrowstyle=arrowstyle,
                #     ls="solid",
                #     color="black",
                #     lw=1,
                #     alpha=0.5,
                # )
                # ax.add_patch(patch)

                # if np.isnan(n_samples_vs_refit):
                #     n_samples_vs_refit = np.max(N_TARGET_VALUES)
                # if np.isnan(cv_vs_refit):
                #     cv_vs_refit = np.min(N_TARGET_VALUES)
                # if n_samples_vs_refit > df["n_target"].max():
                #     patch = FancyArrowPatch(
                #         posA=(cv_vs_refit, target_idx + group_shift + seed_shift),
                #         posB=(df["n_target"].max(), target_idx + group_shift + seed_shift),
                #         arrowstyle="-",
                #         ls="solid",
                #         color="black",
                #         lw=1,
                #         alpha=0.5,
                #     )
                #     ax.add_patch(patch)

                #     patch = FancyArrowPatch(
                #         posA=(df["n_target"].max(), target_idx + group_shift + seed_shift),
                #         posB=(n_samples_vs_refit, target_idx + group_shift + seed_shift),
                #         arrowstyle="-",
                #         ls="dashed",
                #         color="black",
                #         lw=1,
                #         alpha=0.5,
                #     )
                #     ax.add_patch(patch)
                # else:
                #     patch = FancyArrowPatch(
                #         posA=(cv_vs_refit, target_idx + group_shift + seed_shift),
                #         posB=(n_samples_vs_refit, target_idx + group_shift + seed_shift),
                #         arrowstyle="-",
                #         # ls="solid",
                #         color="black",
                #         lw=1,
                #         alpha=0.5,
                #     )
                #     ax.add_patch(patch)

            n_samples_new = np.median(n_samples_new, axis=1)
            refit_new = np.median(refit_new, axis=1)

            cv_vs_n_samples = np.exp(find_intersection(
                np.log(N_TARGET_VALUES),
                n_samples_new - df[column].first(),
                increasing=metric in GREATER_IS_BETTER,
            ))
            cv_vs_refit = np.exp(find_intersection(
                np.log(N_TARGET_VALUES),
                refit_new - df[column].first(),
                increasing=metric in GREATER_IS_BETTER,
            ))
            n_samples_vs_refit = np.exp(find_intersection(
                np.log(N_TARGET_VALUES),
                n_samples_new - refit_new,
                increasing=metric in GREATER_IS_BETTER,
            ))

            n_max = df["n_target"].max()
            if np.isnan(cv_vs_refit):
                cv_vs_refit = np.min(N_TARGET_VALUES)
            else:
                ax.scatter(
                    cv_vs_refit,
                    target_idx + group_shift,
                    marker="o",
                    color=experiment["color"],
                    alpha=1,
                    s=50,
                    zorder=4,
                )

            if ~np.isnan(cv_vs_n_samples) and cv_vs_n_samples < n_max:
                ax.scatter(
                    cv_vs_n_samples,
                    target_idx + group_shift,
                    marker="X",
                    color=experiment["color"],
                    alpha=1,
                    s=50,
                    zorder=4,
                )
            
            if n_samples_vs_refit <= n_max * 6:
                ax.scatter(
                    n_samples_vs_refit,
                    target_idx + group_shift,
                    marker="s",
                    color=experiment["color"],
                    alpha=1,
                    s=50,
                    zorder=4,
                )
            else:
                n_samples_vs_refit = n_max * 5.6
            
            # cv_vs_n_samples
            # ax.scatter(
            #     value,
            #     target_idx + group_shift,
            #     marker=marker,
            #     color=experiment["color"],
            #     alpha=alpha,
            #     s=50,
            #     zorder=4,
            # )
            # for value, nan_value, marker in [
            #     (cv_vs_n_samples, np.max(N_TARGET_VALUES), "X"),
            #     (cv_vs_refit, np.min(N_TARGET_VALUES), "o"),
            #     (n_samples_vs_refit, np.max(N_TARGET_VALUES), "s"),
            # ]:
            #     if np.isnan(value):
            #         alpha = 0.5
            #         value = nan_value
            #     elif value > df["n_target"].max():
            #         alpha = 0.5
            #     else:
            #         alpha = 1
            
            #     ax.scatter(
            #         value,
            #         target_idx + group_shift,
            #         marker=marker,
            #         color=experiment["color"],
            #         alpha=alpha,
            #         s=50,
            #         zorder=4,
            #     )
            #     # else:
            #     #     ax.scatter(
            #     #         value,
            #     #         target_idx + group_shift,
            #     #         marker=marker,
            #     #         color=experiment["color"],
            #     #         alpha=1,
            #     #         s=50,
            #     #         zorder=4,
            #     #     )
            # arrowstyle = "-|>, head_width=5, head_length=5"
            # if np.isnan(n_samples_vs_refit):
            #     n_samples_vs_refit = np.max(N_TARGET_VALUES)

            # patch = FancyArrowPatch(
            #     posA=(n_samples_vs_refit * 0.99, target_idx + group_shift),
            #     posB=(n_samples_vs_refit, target_idx + group_shift),
            #     arrowstyle="-",
            #     ls="solid",
            #     color="black",
            #     lw=1,
            #     alpha=0.5,
            # )
            # ax.add_patch(patch)

            # if np.isnan(n_samples_vs_refit):
            #     n_samples_vs_refit = np.max(N_TARGET_VALUES)
            # if np.isnan(cv_vs_refit):
            #     cv_vs_refit = np.min(N_TARGET_VALUES)
            if n_samples_vs_refit > n_max:
                patch = FancyArrowPatch(
                    posA=(cv_vs_refit, target_idx + group_shift),
                    posB=(n_max, target_idx + group_shift),
                    arrowstyle="-",
                    ls="solid",
                    color="black",
                    lw=1,
                    alpha=0.5,
                )
                ax.add_patch(patch)

                patch = FancyArrowPatch(
                    posA=(n_max, target_idx + group_shift),
                    posB=(n_samples_vs_refit, target_idx + group_shift),
                    arrowstyle="-",
                    ls="dashed",
                    color="black",
                    lw=1,
                    alpha=0.5,
                )
                ax.add_patch(patch)
            else:
                patch = FancyArrowPatch(
                    posA=(cv_vs_refit, target_idx + group_shift),
                    posB=(n_samples_vs_refit, target_idx + group_shift),
                    arrowstyle="-",
                    # ls="solid",
                    color="black",
                    lw=1,
                    alpha=0.5,
                )
                ax.add_patch(patch)

            # ax.scatter(
            #     target_idx + group_shift,
            #     cv_vs_refit,
            #     # marker=experiment["marker"],
            #     color=experiment["color"],
            #     alpha=1,
            #     s=50,
            #     zorder=4,
            # )

            # ax.scatter(
            #     target_idx + group_shift,
            #     cv_vs_refit,
            #     # marker=experiment["marker"],
            #     color=experiment["color"],
            #     alpha=1,
            #     s=50,
            #     zorder=4,
            # )




            # for seed in df["seed"].unique():
            #     filtered = df.filter(pl.col("seed") == seed)
            #     n_target_equiv = np.exp(
            #         find_intersection(
            #             filtered["n_target"].log().to_numpy(),
            #             (filtered[f"{column}_right"] - filtered[column]).to_numpy(),
            #             increasing=metric in GREATER_IS_BETTER,
            #         )
            #     )
            #     ax.scatter(
            #         target_idx + group_shift + rng.normal(0, scatter),
            #         n_target_equiv,
            #         marker=experiment["marker"],
            #         color=experiment_group["color"],
            #         alpha=0.1,
            # )

            # n_target_equiv = np.exp(
            #     find_intersection(
            #         median["n_target"].log().to_numpy(),
            #         (median[f"{column}_right"] - median[column]).to_numpy(),
            #         increasing=metric in GREATER_IS_BETTER,
            #     )
            # )
    fig.legend(legend_handles, labels, loc="outside lower center", ncols=int(len(legend_handles)/2))
    ax.set_xscale("log")

    ax.set_xticks([25, 100, 1000, 10_000, 100_000])
    ax.set_xticklabels(["25", "100", "1k", "10k", "100k"])
    ax.set_xlabel("number of patient stays from target", labelpad=3.5)

    ax.set_yticks(np.arange(len(CONFIG["datasets"])))
    ax.set_yticklabels(
        [SHORT_DATASET_NAMES[x] for x in datasets],
        fontsize=12,
    )
    log_fig(fig, "regimes.pdf", client, run_id=target_run.info.run_id, bbox_inches="tight")
    log_fig(fig, "regimes.png", client, run_id=target_run.info.run_id, bbox_inches="tight")

if __name__ == "__main__":
    main()

import json
import logging
import re
import tempfile

import click
import gin
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import NullFormatter, StrMethodFormatter, NullLocator
from mlflow.tracking import MlflowClient
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from icu_benchmarks.constants import GREATER_IS_BETTER, PARAMETERS
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
    client = MlflowClient(tracking_uri=tracking_uri)
    gin.parse_config_file(config)
    CONFIG = get_config()

    metric = CONFIG["metric"]
    param = CONFIG["x"]

    _, target_run = get_target_run(client, CONFIG["target_experiment"])

    ncols = 3
    fig = plt.figure(figsize=(3.2* ncols, 4))
    gs = gridspec.GridSpec(
        ncols + 1, 3, height_ratios=[1, 1, -0.15, 1], wspace=0.2, hspace=0.39
    )
    axes = [fig.add_subplot(gs[i, j]) for i in [0, 1, 3] for j in range(ncols)]
    legend_elements = []

    for line in CONFIG["lines"]:
        cv_metric = line.get("cv_metric", CONFIG.get("cv_metric", metric))
        result_file = f"{line.get('result_name', 'results')}.csv"
        experiment = client.get_experiment_by_name(line["experiment_name"])
        if experiment is None:
            raise ValueError(f"Did not find experiment {line['experiment_name']}.")

        all_results = []
        for run in client.search_runs(experiment_ids=[experiment.experiment_id]):
            if run.data.tags.get("sources", "") == "":
                continue
            sources = json.loads(run.data.tags["sources"].replace("'", '"'))
            if len(sources) == 0:  # not in [4, 5, 6]:
                continue

            run_id = run.info.run_id
            with tempfile.TemporaryDirectory() as f:
                if result_file not in [x.path for x in client.list_artifacts(run_id)]:
                    logger.warning(f"Run {run_id} has no {result_file}")
                    continue
                logger.info(f"Downloading {result_file} for run {run_id}")
                client.download_artifacts(run_id, result_file, f)
                results = pl.read_csv(f"{f}/{result_file}")

            results = results.with_columns(
                pl.lit(sources).alias("sources"),
                pl.lit(run.data.tags["outcome"]).alias("outcome"),
            )

            if "min_gain_to_split" in results.columns:
                results = results.with_columns(
                    pl.col("min_gain_to_split").cast(pl.Float64)
                )

            if param in results.columns:
                results = results.filter(
                    pl.col(param).ge(CONFIG["xlim"][0])
                    & pl.col(param).le(CONFIG["xlim"][1])
                )

            all_results.append(results)

        if len(all_results) == 0:
            continue
        results = pl.concat(all_results, how="diagonal")

        # results = results.filter(pl.col("min_gain_to_split").is_in([0.1]))
        # results = results.with_columns(pl.col("num_leaves").fill_null(31))
        # results = results.filter(pl.col("num_leaves").eq(31))
        # results = results.filter(pl.col("sources").list.len() >= 5)
        col = pl.col("gamma").log() / pl.lit(np.sqrt(2)).log()
        results = results.filter( (col - col.round(0)).abs() < 0.01)
        # results = results.filter(pl.col("num_iteration").is_in([200, 500, 1000]))

        if "filter" in line.keys():
            results = results.filter(**line["filter"])

        if "alpha" in results.columns:
            col = pl.col("alpha").log10() - pl.col("alpha").min().log10()
            results = results.with_columns(
                pl.coalesce(
                    pl.when(np.abs(col - x) < 0.01).then(x) for x in [2, 3, 4]
                ).alias("alpha_index")
            )
        
            if line.get("plot_all", True):
                results = results.filter(pl.col("alpha_index").is_not_null())
            else:
                results = results.filter(pl.col("alpha_index").eq(3) & pl.col("l1_ratio").eq(0.5))

        if "num_iteration" in results.columns:
            results = results.filter(pl.col("num_iteration").is_in([500, 1000, 2000]))
        # if "n_samples" in line.keys():
        #     n_samples_experiment = client.get_experiment_by_name(line["n_samples"])
        #     runs = client.search_runs(
        #         experiment_ids=[n_samples_experiment.experiment_id],
        #         filter_string="tags.sources = ''",
        #     )
        #     with tempfile.TemporaryDirectory() as f:
        #         client.download_artifacts(
        #             runs[0].info.run_id, "n_samples_results.csv", f
        #         )
        #         n_samples_results = pl.read_csv(f"{f}/n_samples_results.csv")
        #     n_samples_results = (
        #         n_samples_results.filter(pl.col("metric").eq(metric))
        #         .group_by(["target", "n_target"])
        #         .agg(pl.col("test_value").median())
        #     )
        # else:
        #     n_samples_results = None

        for panel, ax in zip(CONFIG["panels"], axes):
            target = panel["source"]

            if target == "empty":
                ax.set_visible(False)
                continue

            cv_columns = [
                x for x in results.columns if re.match(rf"^cv/{cv_metric}_\d+$", x)
            ]
            if len(cv_columns) > 0:
                cv = results.filter(~pl.col("sources").list.contains(target))
                cv = cv.with_columns(
                    pl.mean_horizontal(c for c in cv_columns).alias(f"__cv_{metric}")
                )
            else:
                cv = results.filter(~pl.col("sources").list.contains(target))
                # cv = cv.filter(pl.col("sources").list.len() == pl.col("sources").list.len().max())
                if "gamma" in results.columns:
                    sources = results["sources"].explode().unique().to_list()
                    # cv = cv.with_columns(
                    #     (
                    #         pl.col(f"{s}/train_val/{cv_metric}")
                    #         + (pl.col("gamma") - 1)
                    #         * pl.col(f"{s}/train_val/proj_residuals_sq")
                    #     ).alias(f"{s}/train_val/anchor")
                    #     for s in sources
                    # )

                    cv = cv_results(
                        cv,
                        [cv_metric],
                    )
                else:
                    cv = cv_results(
                        cv
                        [cv_metric],
                    )

            if param not in cv.columns or len(cv[param].unique()) == 1:
                best = cv.top_k(
                    1,
                    by=f"__cv_{cv_metric}",
                    reverse=cv_metric not in GREATER_IS_BETTER
                )[0]
                ax.hlines(
                    best[f"{target}/test/{metric}"].item(),
                    *CONFIG["xlim"],
                    color=line["color"],
                    ls=line["ls"],
                    alpha=line["alpha"],
                    zorder=line.get("zorder", 1),
                )
                print(
                    f"{line['experiment_name']}, {target}: {best[f'{target}/test/{metric}'].item()}"
                )
                continue

            grouped = (
                cv.group_by(param)
                .agg(pl.all().top_k_by(k=1, by=f"__cv_{cv_metric}", reverse=True))
                .select(pl.all().explode())
                .sort(param)
            )
            if len(grouped) == 0:
                continue

            if not line.get("plot_all", True):
                ax.plot(
                    grouped[param],
                    grouped[f"{target}/test/{metric}"],
                    color=line["color"],
                    alpha=line["alpha"],
                    ls=line["ls"],
                    zorder=line.get("zorder", 1),
                )

            best = grouped.top_k(
                1,
                by=f"__cv_{cv_metric}",
                reverse=cv_metric not in GREATER_IS_BETTER
            )[0]

            if line.get("plot_quantiles", False):
                ax.plot(
                    grouped[param],
                    grouped[f"{target}/test/{metric}_quantile_0.75"],
                    color="black",
                    alpha=0.5,
                    ls="dashed",
                )
                ax.plot(
                    grouped[param],
                    grouped[f"{target}/test/{metric}_quantile_0.9"],
                    color="black",
                    alpha=0.5,
                    ls="dashed",
                )
                # ax.plot(
                #     cv["gamma"],
                #     cv[f"{target}/test/{metric}_quantile_0.5"],
                #     color="blue",
                #     # label="median" if idx == 0 else None,
                # )
                # ax.fill_between(
                #     cv["gamma"],
                #     cv[f"{target}/test/{metric}_quantile_0.1"],
                #     cv[f"{target}/test/{metric}_quantile_0.25"],
                #     color="blue",
                #     alpha=0.1,
                #     # label="10% - 90%" if idx == 0 else None,
                # )

                # ax.fill_between(
                #     cv["gamma"],
                #     cv[f"{target}/test/{metric}_quantile_0.75"],
                #     cv[f"{target}/test/{metric}_quantile_0.9"],
                #     color="blue",
                #     alpha=0.1,
                #     label=None,
                # )

                # ax.fill_between(
                #     cv["gamma"],
                #     cv[f"{target}/test/{metric}_quantile_0.25"],
                #     cv[f"{target}/test/{metric}_quantile_0.75"],
                #     color="blue",
                #     alpha=0.2,
                #     # label="25% - 75%" if idx == 0 else None,
                # )
        
            if line.get("plot_star", True):
                ax.scatter(
                    best[param].item(),
                    best[f"{target}/test/{metric}"].item(),
                    color=line["color"],
                    marker="*",
                    s=100,
                    alpha=line["alpha"],
                    zorder=line.get("zorder", 1),
                )
            print(
                f"{line['experiment_name']}, {target}: {best[f'{target}/test/{metric}'].item()}"
            )

            # if "plus" in  line["experiment_name"]:
            #     breakpoint()
            if line.get("plot_all", True):
                params = [p for p in PARAMETERS if p in cv.columns]
                # cmap = matplotlib.colormaps["rainbow"]
                cmap = LinearSegmentedColormap.from_list(
                    "tol",
                    # [
                    #     '#1965B0',
                    #     '#437DBF',
                    #     '#5289C7',
                    #     '#6195CF',
                    #     '#7BAFDE',
                    #     '#4EB265',
                    #     '#90C987',
                    #     '#CAE0AB',
                    #     '#F7F056',
                    #     '#F7CB45',
                    #     '#F6C141',
                    #     '#F4A736',
                    #     '#F1932D',
                    #     '#EE8026',
                    #     '#E8601C',
                    #     '#E65518',
                    #     '#DC050C'
                    # ]
                    [
                        '#5568B8',
                        '#4E79C5',
                        '#4D8AC6',
                        '#4E96BC',
                        '#549EB3',
                        '#59A5A9',
                        '#60AB9E',
                        '#69B190',
                        '#77B77D',
                        '#8CBC68',
                        '#A6BE54',
                        '#BEBC48',
                        '#D1B541',
                        '#DDAA3C',
                        '#E49C39',
                        '#E78C35',
                        '#E67932',
                        '#E4632D',
                        '#DF4828',
                        '#DA2222'
                    ]
                )
                
                for _, group in cv.group_by([p for p in params if p != param]):
                    # color = {
                    #      2: "#4EB265",
                    #      3: "#004488",
                    #      # 2: "#7BDED8",
                    #      # 4: "#F7F056",
                    #      4: "#F4A736"
                    # }[group["alpha_index"].first()]
                    # ls={0.01: (1, (1, 1)), 0.5: "solid", 1.0: (0, (2, 2))}[group["l1_ratio"].first()]


                    # #4EB265
                    color = {
                         4: "#4EB265",
                         3: "#004488",
                         # 2: "#7BDED8",
                         # 4: "#F7F056",
                         2: "#F4A736"
                    }[group["max_depth"].first()]
                    ls={500: (1, (1, 1)), 1000: "solid", 2000: (0, (2, 2))}[group["num_iteration"].first()]


                    # np.max([np.max([val, 0]), 1])
                    # val = (
                    #     -(results["min_gain_to_split"] + 0.001).log().min()
                    #     + (group["min_gain_to_split"] + 0.001).log().first()
                    # ) / (
                    #     (results["min_gain_to_split"] + 0.001).log().max() + 0.001
                    #     - (results["min_gain_to_split"] + 0.001).log().min()
                    # )
                    # val = (group["num_leaves"].first() - 8) / (31 - 8)
                    # val = group["num_iteration"].first() - results["num_iteration"].min()
                    # val /= results["num_iteration"].max() - results["num_iteration"].min()
                    # val = 1 - val
                    # val = group["max_depth"].first() / results["max_depth"].max()

                    group = group.sort(param)
                    ax.plot(
                        group[param],
                        group[f"{target}/test/{metric}"],
                        color=color,
                        ls=ls,
                        # color=line["color"],
                        # ls={2: (0, (2, 4)), 3: (2, (4, 2))}[group["max_depth"].first()],
                        # ls=line.get("ls"),
                        # color = {2: "blue", 4: "green", 8: "orange", 16: "red", 31: "purple"}[group["num_leaves"].first()],
                        # color=cmap(val),
                        # ls={100: (0, (2, 4)), 200: (0, (3, 3)), 300: (0, (4, 2)), 400: (0, (5, 1)), 500: "solid"}[group["num_iteration"].first()],
                        # ls={200: (0, (2, 4)), 400: (0, (3, 3)), 600: (0, (4, 2)), 800: (0, (5, 1)), 1000: "solid"}[group["num_iteration"].first()],
                        # ls={0.025: (1, (1, 1)), 0.05: (0, (4, 1)), 0.1: "solid"}[group["learning_rate"].first()],
                        # ls={10.0: (1, (1, 1)), 1.0: (1, (2, 2)), 0.1: (0, (4, 1)), 0.0: "solid"}[group["lambda_l2"].first()],
                        # ls={0.01: (1, (1, 1)), 0.5: (0, (4, 1)), 1.0: "solid"}[group["l1_ratio"].first()],
                        # ls={1: (0, (2, 4)), 2: (0, (5, 2)), 3: "solid"}[group["max_depth"].first()],
                        # ls={2: (0, (3, 3)), 4: (0, (4, 1)), 8: "solid"}[group["max_depth"].first()],
                        # ls={2: (1, (1, 1)), 3: (0, (4, 1)), 4: "solid"}[group["max_depth"].first()],
                        # alpha=0.6 * line["alpha"],
                        alpha=0.6,
                    )

                
            # cv_gamma_1 = cv.filter(pl.col("gamma").eq(1)).top_k(k=1, by="__cv_value", reverse=cv_metric not in GREATER_IS_BETTER)
            # group = cv.filter(pl.all_horizontal(pl.col(x).eq(cv_gamma_1[x].item()) for x in params if x != "gamma")).sort(param)
            # ax.plot(
            #     group[param],
            #     group[f"{target}/test/{metric}"],
            #     color=line["color"],
            #     ls="dotted",
            #     alpha=1 * line["alpha"],
            # )

        if param not in cv.columns or len(cv[param].unique()) == 1:
            legend_elements.append(
                plt.Line2D(
                    [], [], color=line["color"], label=line["label"], ls=line["ls"]
                )
            )
        elif not line.get("plot_all", False):
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=line["color"],
                    label=line["label"],
                    ls=line["ls"],
                    # marker="*",
                )
            )
            if line.get("plot_star", True):
                legend_elements.append(
                    plt.Line2D(
                        [],
                        [],
                        color=line["color"],
                        label=r"$\gamma$ as selected by LOEO-CV",
                        ls="None",
                        marker=r"$\star$",
                        markersize=8,
                    )
                )

    for ax, panel in zip(axes, CONFIG["panels"]):
        if panel["source"] == "empty":
            continue
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        if param in ["gamma", "alpha", "ratio", "learning_rate"]:
            ax.set_xscale("log")

        ax.set_ylabel(CONFIG["ylabel"], fontsize=10)
        if "ylim" in panel:
            ax.set_ylim(*panel["ylim"])

        if panel.get("yticks") is not None:
            ax.set_yticks(panel["yticks"])
            ax.set_yticklabels(panel.get("yticklabels", panel["yticks"]))
        else:
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        if param == "gamma":
            ax.set_xlabel(
                r"$\gamma$",
                labelpad=1,
                fontsize=10,
            )
        else:
            ax.set_xlabel(param, labelpad=1, fontsize=10)
        delta = np.pow(CONFIG["xlim"][1] / CONFIG["xlim"][0], 0.02)
        ax.set_xlim(CONFIG["xlim"][0] / delta, CONFIG["xlim"][1] * delta)

        if "xticks" in CONFIG:
            ax.set_xticks(CONFIG["xticks"])
            ax.set_xticklabels(CONFIG.get("xticklabels", CONFIG["xticks"]))

        ax.set_title(panel["title"], y=0.95, fontsize=10)

        ax.tick_params(axis="y", which="both", pad=1)
        ax.tick_params(axis="x", which="both", pad=2)
        # ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        # ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.label_outer()
        ax.yaxis.set_tick_params(labelleft=True)  # manually add x & y ticks again
        # ax.xaxis.set_tick_params(labelbottom=True)

    fig.align_ylabels(axes)
    line = plt.Line2D(
        [0.07, 0.92],
        [0.372, 0.372],
        transform=fig.transFigure,
        color="grey",
        linewidth=2,
        alpha=0.8,
    )
    _ = plt.text(
        0.915,
        0.54,
        "core datasets",
        transform=fig.transFigure,
        fontsize=11,
        rotation=90,
        alpha=1.0,
        color="grey",
    )
    _ = plt.text(
        0.915,
        0.14,
        "truly OOD",
        transform=fig.transFigure,
        fontsize=11,
        rotation=90,
        alpha=1.0,
        color="grey",
    )

    fig.add_artist(line)
    
    # fig.legend(handles=legend_elements, loc="center", ncols=4, bbox_to_anchor=(0.5, -0.02) ,  frameon=False)
    fig.legend(handles=legend_elements, loc="center", ncols=3, bbox_to_anchor=(0.5, -0.06    ) ,  frameon=False, fontsize=10, handletextpad=0.8, columnspacing=1, labelspacing=0.5)

    if CONFIG.get("title") is not None:
        fig.suptitle(CONFIG["title"], size=12, y=0.97)

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

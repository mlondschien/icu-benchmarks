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
from matplotlib.ticker import NullFormatter, StrMethodFormatter
from mlflow.tracking import MlflowClient

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
    fig = plt.figure(figsize=(3.5 * ncols, 7))
    gs = gridspec.GridSpec(
        ncols + 1, 3, height_ratios=[1, 1, -0.1, 1], wspace=0.18, hspace=0.3
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

                client.download_artifacts(run_id, result_file, f)
                results = pl.read_csv(f"{f}/{result_file}")

            results = results.with_columns(
                pl.lit(sources).alias("sources"),
                pl.lit(run.data.tags["outcome"]).alias("outcome"),
            )
            if "lambda_l2" in results.columns:
                results = results.with_columns(
                    pl.col("lambda_l2").cast(pl.Float64),
                )
            if param in results.columns:
                results = results.filter(
                    pl.col(param).ge(CONFIG["xlim"][0])
                    & pl.col(param).le(CONFIG["xlim"][1])
                )

            all_results.append(results)

        results = pl.concat(all_results, how="diagonal")

        if "filter" in line.keys():
            results = results.filter(**line["filter"])

        if "n_samples" in line.keys():
            n_samples_experiment = client.get_experiment_by_name(line["n_samples"])
            runs = client.search_runs(
                experiment_ids=[n_samples_experiment.experiment_id],
                filter_string="tags.sources = ''",
            )
            with tempfile.TemporaryDirectory() as f:
                client.download_artifacts(
                    runs[0].info.run_id, "n_samples_results.csv", f
                )
                n_samples_results = pl.read_csv(f"{f}/n_samples_results.csv")
            n_samples_results = (
                n_samples_results.filter(pl.col("metric").eq(metric))
                .group_by(["target", "n_target"])
                .agg(pl.col("test_value").median())
            )
        else:
            n_samples_results = None

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
                if "gamma" in results.columns:
                    sources = results["sources"].explode().unique().to_list()
                    results = results.with_columns(
                        (
                            pl.col(f"{s}/train_val/{cv_metric}")
                            + (pl.col("gamma") - 1)
                            * pl.col(f"{s}/train_val/proj_residuals_sq")
                        ).alias(f"{s}/train_val/anchor")
                        for s in sources
                    )

                    cv = cv_results(
                        results.filter(~pl.col("sources").list.contains(target)),
                        ["anchor", cv_metric],
                        n_samples_result=n_samples_results,
                    )

                else:
                    cv = cv_results(
                        results.filter(~pl.col("sources").list.contains(target)),
                        [cv_metric],
                        n_samples_result=n_samples_results,
                    )

            if param not in cv.columns or len(cv[param].unique()) == 1:
                best = cv.top_k(
                    1,
                    by=f"__cv_{cv_metric}",
                    reverse=cv_metric not in GREATER_IS_BETTER
                    and n_samples_results is None,
                )[0]
                ax.hlines(
                    best[f"{target}/test/{metric}"].item(),
                    *CONFIG["xlim"],
                    color=line["color"],
                    ls=line["ls"],
                    alpha=line["alpha"],
                )
                print(
                    f"{line['experiment_name']}, {target}: {best[f'{target}/test/{metric}'].item()}"
                )
                continue

            grouped = (
                cv.group_by(param)
                .agg(pl.all().top_k_by(k=1, by="__cv_anchor", reverse=True))
                .select(pl.all().explode())
                .sort(param)
            )
            ax.plot(
                grouped[param],
                grouped[f"{target}/test/{metric}"],
                color=line["color"],
                alpha=line["alpha"],
                ls=line["ls"],
            )

            best = grouped.top_k(
                1,
                by=f"__cv_{cv_metric}",
                reverse=cv_metric not in GREATER_IS_BETTER
                and n_samples_results is None,
            )[0]
            ax.scatter(
                best[param].item(),
                best[f"{target}/test/{metric}"].item(),
                color=line["color"],
                marker="*",
                s=100,
                alpha=line["alpha"],
            )
            print(
                f"{line['experiment_name']}, {target}: {best[f'{target}/test/{metric}'].item()}"
            )

            if line.get("plot_all", True):
                params = [p for p in PARAMETERS if p in cv.columns]
                cmap = matplotlib.colormaps["rainbow"]
                for _, group in cv.group_by([p for p in params if p != param]):
                    # val = (np.log(group["alpha"].first()) - np.log(cv["alpha"].min())) / np.log(cv["alpha"].max() / cv["alpha"].min())
                    # np.max([np.max([val, 0]), 1])
                    val = (
                        -(results["lambda_l2"] + 0.01).log().min()
                        + (group["lambda_l2"] + 0.01).log().first()
                    ) / (
                        (results["lambda_l2"] + 0.01).log().max()
                        - (results["lambda_l2"] + 0.01).log().min()
                    )
                    group = group.sort(param)
                    ax.plot(
                        group[param],
                        group[f"{target}/test/{metric}"],
                        # color=line["color"],
                        color=cmap(val),
                        # ls={0.025: (1, (1, 1)), 0.05: (0, (4, 1)), 0.1: "solid"}[group["learning_rate"].first()],
                        # ls={10.0: (1, (1, 1)), 1.0: (1, (2, 2)), 0.1: (0, (4, 1)), 0.0: "solid"}[group["lambda_l2"].first()],
                        # ls={0.01: (1, (1, 1)), 0.5: (0, (4, 1)), 1.0: "solid"}[group["l1_ratio"].first()],
                        alpha=0.3 * line["alpha"],
                    )

            # cv_gamma_1 = cv.filter(pl.col("gamma").eq(1)).top_k(k=1, by="__cv_value", reverse=cv_metric not in GREATER_IS_BETTER and n_samples_results is None)
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
        else:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=line["color"],
                    label=line["label"],
                    ls=line["ls"],
                    marker="*",
                )
            )

    for ax, panel in zip(axes, CONFIG["panels"]):
        if panel["source"] == "empty":
            continue
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        if param in ["gamma", "alpha", "ratio", "learning_rate"]:
            ax.set_xscale("log")

        ax.set_ylabel(CONFIG["ylabel"])
        if "ylim" in panel:
            ax.set_ylim(*panel["ylim"])

        if panel.get("yticks") is not None:
            ax.set_yticks(panel["yticks"])
            ax.set_yticklabels(panel.get("yticklabels", panel["yticks"]))
        else:
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlabel(param, labelpad=2)
        delta = np.pow(CONFIG["xlim"][1] / CONFIG["xlim"][0], 0.02)
        ax.set_xlim(CONFIG["xlim"][0] / delta, CONFIG["xlim"][1] * delta)
        ax.set_title(panel["title"], y=0.985)

        ax.tick_params(axis="y", which="both", pad=1)
        ax.tick_params(axis="x", which="both", pad=2)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.label_outer()
        ax.yaxis.set_tick_params(labelleft=True)  # manually add x & y ticks again
        # ax.xaxis.set_tick_params(labelbottom=True)
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
    fig.legend(handles=legend_elements, loc="outside lower center", ncols=4)
    if CONFIG.get("title") is not None:
        fig.suptitle(CONFIG["title"], size="x-large", y=0.94)
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

import json
import logging
import tempfile

import click
import gin
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from icu_benchmarks.constants import GREATER_IS_BETTER, PARAMETERS
from icu_benchmarks.plotting import cv_results
from icu_benchmarks.mlflow_utils import get_target_run, log_fig
import matplotlib.gridspec as gridspec

SOURCES = ["miiv", "mimic-carevue", "aumc", "sic", "eicu", "hirid"]

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
    cv_metric = CONFIG.get("cv_metric", metric)
    param = CONFIG["x"]

    _, target_run = get_target_run(client, CONFIG["target_experiment"])

    ncols = int(len(CONFIG["panels"]) / 3)
    fig = plt.figure(figsize=(3.5 * ncols, 7))
    gs = gridspec.GridSpec(ncols + 1, 3, height_ratios=[1, 1, -0.1, 1], wspace=0.18, hspace=0.3)
    axes = [
        fig.add_subplot(
            gs[i, j]
        ) for i in [0, 1, 3] for j in range(ncols)
    ]
    legend_elements = []

    for line in CONFIG["lines"]:
        experiment = client.get_experiment_by_name(line["experiment_name"])
        if experiment is None:
            raise ValueError(f"Did not find experiment {line['experiment_name']}.")

        all_results = []
        for run in client.search_runs(experiment_ids=[experiment.experiment_id]):
            if run.data.tags["sources"] == "":
                continue
            sources = json.loads(run.data.tags["sources"].replace("'", '"'))
            if len(sources) not in [4, 5, 6]:
                continue

            run_id = run.info.run_id
            with tempfile.TemporaryDirectory() as f:
                if "results.csv" not in [x.path for x in client.list_artifacts(run_id)]:
                    logger.warning(f"Run {run_id} has no results.csv")
                    continue

                client.download_artifacts(run_id, "results.csv", f)
                results = pl.read_csv(f"{f}/results.csv")

            results = results.with_columns(
                pl.lit(sources).alias("sources"),
                pl.lit(run.data.tags["outcome"]).alias("outcome"),
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
            runs = client.search_runs(experiment_ids=[n_samples_experiment.experiment_id], filter_string="tags.sources = ''")
            with tempfile.TemporaryDirectory() as f:
                client.download_artifacts(runs[0].info.run_id, "n_samples_results.csv", f)
                n_samples_results = pl.read_csv(f"{f}/n_samples_results.csv")
            n_samples_results = n_samples_results.filter(pl.col("metric").eq(metric)).group_by(["target", "n_target"]).agg(pl.col("test_value").median())
        else:
            n_samples_results = None

        for panel, ax in zip(CONFIG["panels"], axes):
            target = panel["source"]

            if target == "empty":
                ax.set_visible(False)
                continue

            cv = cv_results(
                results.filter(~pl.col("sources").list.contains(target)),
                cv_metric,
                n_samples_result=n_samples_results,
            )
            
            if param not in cv.columns or len(cv[param].unique()) == 1:
                best = cv.top_k(
                    1, by="__cv_value", reverse=cv_metric not in GREATER_IS_BETTER and n_samples_results is None
                )[0]
                ax.hlines(
                    best[f"{target}/test/{metric}"].item(),
                    *CONFIG["xlim"],
                    color=line["color"],
                    ls=line["ls"],
                    alpha=line["alpha"],
                )
                continue

            grouped = (
                cv.group_by(param)
                .agg(
                    pl.all().top_k_by(
                        k=1, by="__cv_value", reverse=cv_metric not in GREATER_IS_BETTER and n_samples_results is None
                    )
                )
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
                1, by="__cv_value", reverse=cv_metric not in GREATER_IS_BETTER and n_samples_results is None
            )[0]
            ax.scatter(
                best[param].item(),
                best[f"{target}/test/{metric}"].item(),
                color=line["color"],
                marker="*",
                s=100,
                alpha=line["alpha"],
            )

            params = [p for p in PARAMETERS if p in cv.columns]
            for _, group in cv.group_by([p for p in params if p != param]):
                group = group.sort(param)
                ax.plot(
                    group[param],
                    group[f"{target}/test/{metric}"],
                    color=line["color"],
                    ls="solid",
                    alpha=0.1 * line["alpha"],
                )

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
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.label_outer()
        ax.yaxis.set_tick_params(labelleft=True)  # manually add x & y ticks again
        # ax.xaxis.set_tick_params(labelbottom=True)
    fig.align_ylabels(axes)
    line = plt.Line2D([0.07, 0.925], [0.37, 0.37], transform=fig.transFigure, color="black", linewidth=2, alpha=0.5)
    _ = plt.text(0.915, 0.56, "core datasets", transform=fig.transFigure, fontsize=12, rotation=90, alpha=0.6)
    _ = plt.text(0.915, 0.165, "truly OOD", transform=fig.transFigure, fontsize=12, rotation=90, alpha=0.6)
    
    fig.add_artist(line)
    fig.legend(handles=legend_elements, loc="outside lower center", ncols=4)
    if CONFIG.get("title") is not None:
        fig.suptitle(CONFIG["title"], size="x-large", y=0.94)
    log_fig(fig, f"{CONFIG['filename']}.pdf", client, run_id=target_run.info.run_id, bbox_inches='tight')
    log_fig(fig, f"{CONFIG['filename']}.png", client, run_id=target_run.info.run_id, bbox_inches='tight')


if __name__ == "__main__":
    main()

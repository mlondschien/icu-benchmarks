import json
import logging
import tempfile

import click
import gin
import matplotlib.pyplot as plt
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import GREATER_IS_BETTER
from icu_benchmarks.mlflow_utils import get_target_run, log_fig
from icu_benchmarks.plotting import PARAMETER_NAMES

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
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns2.db",
)
@click.option("--config", type=click.Path(exists=True))
def main(tracking_uri, config):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)
    gin.parse_config_file(config)
    CONFIG = get_config()

    metric = CONFIG["metric"]
    param = CONFIG["x"]

    _, target_run = get_target_run(client, CONFIG["target_experiment"])

    fig, axes = plt.subplots(
        2, 3, figsize=(12, 8), constrained_layout=True, gridspec_kw={"hspace": 0.02}
    )

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
        results_n2 = results.filter(pl.col("sources").list.len() == 4)
        results_n1 = results.filter(pl.col("sources").list.len() == 5)
        params = [z for z in PARAMETER_NAMES if z in results.columns]
        for panel, ax in zip(CONFIG["panels"], axes.flat):
            target = panel["source"]
            cv_results = results_n2.filter(~pl.col("sources").list.contains(target))
            cv_results = cv_results.with_columns(
                pl.coalesce(
                    pl.when(~pl.col("sources").list.contains(s)).then(
                        pl.col(f"{s}/train_val/{metric}")
                    )
                    for s in sources
                ).alias("cv_value")
            )
            cv_results = cv_results.group_by(params).agg(pl.mean("cv_value"))
            cv_results = results_n1.filter(
                ~pl.col("sources").list.contains(target)
            ).join(cv_results, on=params, how="full", validate="1:1")

            if param not in params:
                best = cv_results.top_k(
                    1, by="cv_value", reverse=metric not in GREATER_IS_BETTER
                )[0]
                ax.hlines(
                    best[f"{target}/test/{metric}"].item(),
                    *ax.get_xlim(),
                    color=line["color"],
                    ls=line["ls"],
                    alpha=line["alpha"],
                )
                continue

            grouped = (
                cv_results.group_by(param)
                .agg(
                    pl.all().top_k_by(
                        k=1, by="cv_value", reverse=metric not in GREATER_IS_BETTER
                    )
                )
                .select(pl.all().explode())
                .sort(param)
            )

            ax.plot(
                grouped[param],
                grouped[f"{target}/test/{metric}"],
                # label=line["label"],
                color=line["color"],
                alpha=line["alpha"],
                ls=line["ls"],
            )

            best = grouped.top_k(
                1, by="cv_value", reverse=metric not in GREATER_IS_BETTER
            )[0]
            ax.scatter(
                best[param].item(),
                best[f"{target}/test/{metric}"].item(),
                color=line["color"],
                marker="*",
                s=100,
                alpha=line["alpha"],
            )

            for _, group in cv_results.group_by([p for p in params if p != param]):
                group = group.sort(param)
                ax.plot(
                    group[param],
                    group[f"{target}/test/{metric}"],
                    color=line["color"],
                    ls="solid",
                    alpha=0.2 * line["alpha"],
                )
        if param in params:
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
        else:
            legend_elements.append(
                plt.Line2D(
                    [], [], color=line["color"], label=line["label"], ls=line["ls"]
                )
            )

    for ax, panel in zip(axes.flat, CONFIG["panels"]):
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        if param in ["gamma", "alpha", "ratio", "learning_rate"]:
            ax.set_xscale("log")

        ax.set_ylabel(CONFIG["ylabel"])
        ax.set_ylim(*panel["ylim"])
        ax.set_xlabel(param)
        ax.set_xlim(*CONFIG["xlim"])
        ax.set_title(panel["title"])

        ax.label_outer()
        ax.yaxis.set_tick_params(labelleft=True)  # manually add x & y ticks again
        ax.xaxis.set_tick_params(labelbottom=True)

    fig.legend(handles=legend_elements, loc="outside lower center", ncols=4)
    fig.suptitle(CONFIG["title"], size="x-large")
    log_fig(fig, f"{CONFIG['filename']}.png", client, run_id=target_run.info.run_id)
    log_fig(fig, f"{CONFIG['filename']}.pdf", client, run_id=target_run.info.run_id)


if __name__ == "__main__":
    main()

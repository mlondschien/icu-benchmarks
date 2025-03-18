import json
import logging
import tempfile

import click
import gin
import matplotlib.pyplot as plt
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import PARAMETERS
from icu_benchmarks.mlflow_utils import get_target_run, log_fig

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@gin.configurable
def get_config(config):  # noqa D
    return config


@click.command()
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns2.db",
)
# @click.option("--experiment_name", type=str)
@click.option("--config", type=str)
def main(tracking_uri, config):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)

    gin.parse_config_file(config)
    config = get_config()
    experiment, target_run = get_target_run(client, config["experiment_name"])

    all_results = []
    for run in client.search_runs(experiment_ids=[experiment.experiment_id]):
        run_id = run.info.run_id
        with tempfile.TemporaryDirectory() as f:
            if "results.csv" not in [x.path for x in client.list_artifacts(run_id)]:
                logger.warning(f"Run {run_id} has no results.csv")
                continue

            client.download_artifacts(run_id, "results.csv", f)
            results = pl.read_csv(f"{f}/results.csv")

        sources = json.loads(run.data.tags["sources"].replace("'", '"'))
        if len(sources) != 1:
            continue
        results = results.with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.lit(sources).alias("sources"),
            pl.lit(run.data.tags["outcome"]).alias("outcome"),
        )
        all_results.append(results)

    results = pl.concat(all_results, how="diagonal")
    x = config["parameter"]
    metric = config["metric"]
    print(f"logging to {run.info.run_id}")

    sources = results["sources"].explode().unique().to_list()

    # mult = -1 if metric in GREATER_IS_BETTER else 1
    params = [p for p in PARAMETERS if p in results.columns and p != x]
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 8), constrained_layout=True, gridspec_kw={"hspace": 0.02}
    )

    for ax, panel in zip(axes.flat, config["panels"]):
        target = panel["source"]

        filter_ = pl.col("sources").list.contains(target)
        agg = pl.mean_horizontal(pl.col(f"{metric}_{i}") for i in range(5))
        cv_result = results.filter(filter_).with_columns(agg.alias("cv_value"))
        grouped = (
            cv_result.group_by(x)
            .agg(pl.all().top_k_by(k=1, by="cv_value", reverse=True))
            .select(pl.all().explode())
            .sort(x)
        )

        ax.plot(
            grouped[x],
            grouped[f"{target}/test/{metric}"],
            color="tab:red",
            zorder=3,
        )

        best = grouped.top_k(1, by="cv_value", reverse=True)[0]
        ax.scatter(
            best[x].item(),
            best[f"{target}/test/{metric}"].item(),
            color="tab:red",
            zorder=3,
            marker="*",
        )

        for _, group in cv_result.group_by([p for p in params if p != x]):
            group = group.sort(x)
            # var = group[variable].first() / cur_results_n1[variable].max()
            # color = np.clip((var - var_min) / max(0.01, (var_max - var_min)), 0, 1)
            # if 0 <= color <= 1:
            ax.plot(
                group[x],
                group[f"{target}/test/{metric}"],
                color="grey",
                alpha=0.1,
            )

        ax.set_ylim(*panel["ylim"])

        ax.set_title(panel["title"])
        ax.set_xscale("log")
        ax.set_xlabel(x)
        ax.label_outer()
        ax.yaxis.set_tick_params(labelleft=True)  # manually add x & y ticks again
        ax.xaxis.set_tick_params(labelbottom=True)

    # fig.legend(loc="outside lower center", ncols=4)
    log_fig(
        fig,
        f"in_distribution/in_distribution_{config['experiment_name']}_{x}_{metric}.png",
        client,
        target_run.info.run_id,
    )
    log_fig(
        fig,
        f"in_distribution/in_distribution_{config['experiment_name']}_{x}_{metric}.pdf",
        client,
        target_run.info.run_id,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()

import logging
import tempfile

import click
import matplotlib.pyplot as plt
import polars as pl
from mlflow.tracking import MlflowClient
import gin
from icu_benchmarks.mlflow_utils import log_fig
from icu_benchmarks.plotting import COLORS, DATASET_NAMES, METRIC_NAMES

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@gin.configurable()
def get_config(config):
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

    experiment = client.get_experiment_by_name(CONFIG["target_experiment"])

    if experiment is None:
        raise ValueError(f"target experiment not found")

    experiment_id = experiment.experiment_id

    target_runs = client.search_runs(
        experiment_ids=[experiment_id], filter_string="tags.sources = ''"
    )

    if len(target_runs) != 1:
        raise ValueError(f"Expected exactly one run. Got {target_runs:}")
    target_run = target_runs[0]

    fig, axes = plt.subplots(
        2, 3, figsize=(12, 8), constrained_layout=True, gridspec_kw={"hspace": 0.02}
    )

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

        df = df.filter(pl.col("metric") == CONFIG["metric"])

        for idx, (panel, ax) in enumerate(zip(CONFIG["panels"], axes.flat)):
            data = df.filter((pl.col("target") == panel["source"]))
            if len(data) == 1:
                ax.hlines(
                    data["test_value"].first(),
                    xmin=panel["xlim"][0],
                    xmax=panel["xlim"][1],
                    label=line["legend"] if idx == 0 else None,
                    color=line["color"],
                    ls=line["ls"],
                )
                continue

            data = (
                data.group_by("n_target")
                .agg(
                    [
                        pl.col("test_value").median().alias("score"),
                        pl.col("test_value").quantile(0.2).alias("min"),
                        pl.col("test_value").quantile(0.8).alias("max"),
                    ]
                )
                .sort("n_target")
            )
            ax.plot(
                data["n_target"],
                data["score"],
                label=line["legend"] if idx == 0 else None,
                color=line["color"],
                ls=line["ls"],
            )

            ax.fill_between(
                data["n_target"],
                data["min"],
                data["max"],
                color=line["color"],
                alpha=0.1,
                hatch=line.get("hatch", None),
            )

            ax.set_xscale("log")
            ax.set_xlabel("number of patient stays from target")
            ax.set_ylabel(CONFIG["ylabel"])
            ax.label_outer()
            ax.yaxis.set_tick_params(labelleft=True)  # manually add x & y ticks again
            ax.xaxis.set_tick_params(labelbottom=True)

            ax.set_title(panel["title"])
            ax.set_xlim(*panel["xlim"])
            ax.set_ylim(*panel["ylim"])

    fig.legend(loc="outside lower center", ncols=4)
    fig.suptitle(CONFIG["title"], size="x-large")
    log_fig(fig, f"{CONFIG['filename']}.png", client, run_id=target_run.info.run_id)
    log_fig(fig, f"{CONFIG['filename']}.pdf", client, run_id=target_run.info.run_id)


if __name__ == "__main__":
    main()

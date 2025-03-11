import logging
import re
import tempfile
import gin
import click
import matplotlib.pyplot as plt
import numpy as np
import json
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.mlflow_utils import log_fig, get_target_run
from icu_benchmarks.plotting import PARAMETER_NAMES, plot_by_x
from icu_benchmarks.constants import GREATER_IS_BETTER
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib import colormaps

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

    experiment, target_run = get_target_run(client, CONFIG["experiment_name"])

    if "mlflow.note.content" in experiment.tags:
        print(experiment.tags["mlflow.note.content"])

    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id], filter_string="tags.sources != ''"
    )

    all_results = []
    for run in runs:
        run_id = run.info.run_id
        sources = json.loads(run.data.tags["sources"].replace("'", '"'))
        if len(sources) not in [4, 5, 6]:
            continue

        with tempfile.TemporaryDirectory() as f:
            if "results.csv" not in [x.path for x in client.list_artifacts(run_id)]:
                logger.warning(f"Run {run_id} has no results.csv")
                continue

            client.download_artifacts(run_id, "results.csv", f)
            results = pl.read_csv(f"{f}/results.csv")

        results = results.with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.lit(sources).alias("sources"),
            pl.lit(run.data.tags["outcome"]).alias("outcome"),
        )
        all_results.append(results)

    results = pl.concat(all_results, how="diagonal")
    # results = results.filter(pl.col("num_iteration").eq(1000))

    fig, axes = plt.subplots(
        2, 3, figsize=(22, 15), constrained_layout=True, gridspec_kw={"hspace": 0.02}
    )

    # results = results.filter(pl.col("learning_rate").eq(0.1))
    metric = CONFIG["metric"]

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
        cv_results = results_n1.filter(~pl.col("sources").list.contains(target)).join(
            cv_results, on=params, how="full", validate="1:1"
        )

        grouped = (
            cv_results.group_by([CONFIG["x"], CONFIG["y"]])
            .agg(
                pl.all().top_k_by(
                    k=1, by="cv_value", reverse=metric not in GREATER_IS_BETTER
                )
            )
            .select(pl.all().explode())
        )
        best = cv_results.top_k(
            1, by="cv_value", reverse=metric not in GREATER_IS_BETTER
        )[0]
        best_gamma_1 = cv_results.filter(pl.col("gamma").eq(1)).top_k(
            1, by="cv_value", reverse=metric not in GREATER_IS_BETTER
        )[0]

        values = f"{target}/test/{metric}"
        # values = "num_iteration"
        pivot = grouped.pivot(on=[CONFIG["y"]], index=[CONFIG["x"]], values=[values])
        pivot = pivot.select(
            [CONFIG["x"]] + sorted([x for x in pivot.columns if x != CONFIG["x"]])
        ).sort(CONFIG["x"])
        mat = pivot.drop(CONFIG["x"]).to_numpy().T

        for (i, j), z in np.ndenumerate(mat):
            if z > 1000:
                continue
            color = None
            if z == best[values].item():
                color = "red"
            if z == best_gamma_1[values].item():
                color = "green"
            ax.text(j, i, f"{z:.5f}", ha="center", va="center", color=color)

        cmap = colormaps.get_cmap("plasma")
        vmin, vmax = None, None
        cmap = colormaps.get_cmap(
            "plasma" if metric in GREATER_IS_BETTER else "plasma_r"
        )
        if metric in GREATER_IS_BETTER:
            vmax = np.max(mat)
            vmin = vmax - 0.05
        else:
            vmin = np.min(mat)
            vmax = vmin + 0.005

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        ax.imshow(np.clip(mat, vmin, vmax), cmap=cmap, norm=norm)
        ax.set_title(target)
        ax.set_xticks(np.arange(len(pivot[CONFIG["x"]])))
        ax.set_yticks(np.arange(len(pivot.drop(CONFIG["x"]).columns)))
        ax.set_xticklabels(pivot[CONFIG["x"]])
        ax.set_yticklabels(pivot.drop(CONFIG["x"]).columns)
        ax.set_xlabel(CONFIG["x"])
        ax.set_ylabel(CONFIG["y"])
        ax.label_outer()
        ax.yaxis.set_tick_params(labelleft=True)  # manually add x & y ticks again
        ax.xaxis.set_tick_params(labelbottom=True)

    log_fig(fig, f"{CONFIG['filename']}.png", client, target_run.info.run_id)
    log_fig(fig, f"{CONFIG['filename']}.pdf", client, target_run.info.run_id)


if __name__ == "__main__":
    main()

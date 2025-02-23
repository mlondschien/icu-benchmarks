import logging
import tempfile

import click
import matplotlib.pyplot as plt
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.mlflow_utils import log_fig
from icu_benchmarks.plotting import COLORS, DATASET_NAMES, METRIC_NAMES

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

SOURCES = sorted(["mimic-carevue", "miiv", "eicu", "aumc", "sic", "hirid"])

colors = {
    "benchmark": COLORS["grey"],
    "cv": COLORS["black"],
    "passthrough_linear": COLORS["blue"],
    "passthrough_lgbm": COLORS["blue"],
    "refit_linear_light": COLORS["red"],
    "refit_lgbm": COLORS["red"],
    "refit_linear": COLORS["red"],
    "refit_intercept_linear": COLORS["purple"],
    "refit_intercept_lgbm": COLORS["purple"],
    "n_samples": COLORS["green"],
}

LEGEND = {
    "cv": "Model fit on source",
    "n_samples": "Model fit on target",
    "passthrough_linear": "Target data used for model selection",
    "passthrough_lgbm": "Target data used for model selection",
    "refit_linear": "Empirical Bayes: source model refit on target data",
    "refit_lgbm": "Tree leaf values updated with target data",
    "benchmark": "benchmark",
    "refit_intercept_linear": "Intercept refit on target data",
}


@click.command()
@click.option("--target_experiment", type=str)
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns2.db",
)
@click.argument("result_names", type=str, nargs=-1)
def main(result_names, target_experiment, tracking_uri):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(target_experiment)

    if "mortality" in target_experiment:
        TITLE = "mortality after 24 hours"
    elif "pf_ratio" in target_experiment:
        TITLE = "log(pf ratio) after 12 hours"
    else:
        TITLE = ""

    if "_dsl" in target_experiment:
        TITLE += " (DSL)"
    elif "_glm" in target_experiment:
        TITLE += " (GLM)"
    elif "_lgbm" in target_experiment:
        TITLE += " (LGBM)"
    elif "_anchor" in target_experiment:
        TITLE += " (Anchor)"

    if experiment is None:
        raise ValueError(f"Experiment {target_experiment} not found")

    experiment_id = experiment.experiment_id

    target_runs = client.search_runs(
        experiment_ids=[experiment_id], filter_string="tags.sources = ''"
    )

    if len(target_runs) != 1:
        raise ValueError(f"Expected exactly one run. Got {target_runs:}")
    target_run = target_runs[0]

    results = []
    outcome = None
    for result_name in result_names:
        experiment_name, name = result_name.split("/")
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Did not find experiment {experiment_name}.")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id], filter_string="tags.sources = ''"
        )
        if len(runs) != 1:
            raise ValueError(f"Expected exactly one run for {experiment_name}. Got {runs:}")
        run = runs[0]

        with tempfile.TemporaryDirectory() as f:
            client.download_artifacts(run.info.run_id, f"{name}_results.csv", f)
            df = pl.read_csv(f"{f}/{name}_results.csv")

        outcome = outcome or run.data.tags.get("outcome")

        # df = df.rename(
        #     {
        #         # "n_samples": "n_target",
        #         "scores_cv": "cv_value",
        #         "scores_test": "test_value",
        #         "target_value": "test_value",
        #     },
        #     strict=False,
        # )
        df = df.with_columns(
            pl.lit(name).alias("name"),
            # pl.lit(run.data.tags["outcome"]).alias("outcome"),
        )
        results.append(df)

    results = pl.concat(results, how="diagonal_relaxed")
    results = results.with_columns(
        pl.col("n_target").fill_null(results["n_target"].min())
    )

    metrics = results["metric"].unique()
    for metric in metrics:
        fig, axes = plt.subplots(
            2, 3, figsize=(12, 8), constrained_layout=True, gridspec_kw={"hspace": 0.02}
        )
        for idx, (dataset, ax) in enumerate(zip(SOURCES, axes.flat)):
            data = results.filter(
                (pl.col("target") == dataset) & (pl.col("metric") == metric)
            )
            for result_name in sorted(data["name"].unique()):
                kwargs = {}
                if result_name in colors:
                    kwargs["color"] = colors[result_name]

                data_ = data.filter(pl.col("name") == result_name)
                if len(data_) == 1:
                    ax.hlines(
                        data_["test_value"].first(),
                        xmin=data["n_target"].min(),
                        xmax=data["n_target"].max(),
                        label=LEGEND[result_name] if idx == 0 else None,
                        # color=colors[result_name],
                        **kwargs,
                    )
                    continue

                data_ = (
                    data_.group_by("n_target")
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
                    data_["n_target"],
                    data_["score"],
                    label=LEGEND[result_name] if idx == 0 else None,
                    # color=colors[result_name],
                    **kwargs,
                )
                ax.fill_between(
                    data_["n_target"],
                    data_["min"],
                    data_["max"],
                    # color=colors[result_name],
                    alpha=0.1,
                    **kwargs,
                )

            ax.set_xscale("log")
            ax.set_xlabel("number of patient stays from target")
            ax.set_ylabel(f"test {METRIC_NAMES[metric]}")
            ax.label_outer()
            ax.yaxis.set_tick_params(labelleft=True)  # manually add x & y ticks again
            ax.xaxis.set_tick_params(labelbottom=True)
            data = data.group_by(["n_target", "name"]).agg(
                pl.col("test_value").median()
            )
            ymin, ymax = data["test_value"].min(), data["test_value"].max()
            ymin, ymax = ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.05
            ax.set_ylim(ymin, ymax)
            # ax.legend()
            ax.set_title(DATASET_NAMES[dataset])

        # fig.tight_layout()
        fig.legend(loc="outside lower center", ncols=4)
        fig.suptitle(TITLE, size="x-large")
        log_fig(fig, f"n_samples/{metric}.png", client, run_id=target_run.info.run_id)
        log_fig(fig, f"n_samples/{metric}.pdf", client, run_id=target_run.info.run_id)

    #  summary.filter(pl.col("metric") == "r2").group_by(["target", "result_name", "n_target"]).agg(pl.col("test_value").mean()).sort(["target", "result_name", "n_target"])


if __name__ == "__main__":
    main()

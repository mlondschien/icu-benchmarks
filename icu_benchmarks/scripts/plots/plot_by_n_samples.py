import logging
import re
import tempfile

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient
import json
from icu_benchmarks.mlflow_utils import log_fig
from icu_benchmarks.plotting import PARAMETER_NAMES, plot_by_x

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

PARAMETER_NAMES = [
    "alpha_idx",
    "alpha",
    "refit_alpha_idx",
    "refit_alpha",
    "l1_ratio",
    "gamma",
    "num_boost_round",
    "num_iteration",
    "learning_rate",
    "num_leaves",
    "ratio",
    "decay_rate",
]

SOURCES = [
    "aumc",
    "eicu",
    "hirid",
    "mimic-carevue",
    "miiv",
    "sic",
]

GREATER_IS_BETTER = ["accuracy", "roc", "auprc", "r2"]

@click.command()
@click.option("--target_experiment", type=str)
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns.db",
)
@click.argument("result_names", type=str, nargs=-1)
def main(result_names, target_experiment, tracking_uri):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)

    target_experiment = client.get_experiment_by_name(target_experiment)
    target_run = client.search_runs(
        experiment_ids=[target_experiment.experiment_id], filter_string="tags.sources = ''"
    )
    if len(target_run) > 0:
        target_run = target_run[0]
    else:
        target_run = client.create_run(experiment_id=experiment_id, tags={"sources": ""})

    print(f"logging to {target_run.info.run_id}")

    results = []
    summarized = []
    for result_name in result_names:
        experiment_name, result_name = result_name.split("/")

        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            raise ValueError(f"Experiment {experiment_name} not found")

        experiment_id = experiment.experiment_id

        runs = client.search_runs(          experiment_ids=[experiment_id]       )

        all_results = []
        for run in runs:
            sources = run.data.tags.get("sources", "")

            if sources != "":
                sources = json.loads(run.data.tags["sources"].replace("'", '"'))
                if len(sources) != 5:
                    continue

            if "target" in run.data.tags:
                target = run.data.tags["target"]
            else:
                target = [t for t in SOURCES if t not in sources][0]

            run_id = run.info.run_id
            result_file = f"{result_name}_results.csv"
            with tempfile.TemporaryDirectory() as f:
                if result_file not in [x.path for x in client.list_artifacts(run_id)]:
                    logger.warning(f"Run {run_id} has no results.csv")
                    continue

                client.download_artifacts(run_id, result_file, f)
                results = pl.read_csv(f"{f}/{result_file}")


            results = results.with_columns(
                pl.lit(run_id).alias("run_id"),
                pl.lit(sources).alias("sources"),
                pl.lit(run.data.tags["outcome"]).alias("outcome"),
                pl.lit(result_name).alias("result_name"),
                pl.lit(target).alias("target"),
            )

            all_results.append(results)

        parameter_names = [p for p in PARAMETER_NAMES if p in results.columns]

        metrics = map(re.compile(r"\/(.+)$").search, results.columns)
        metrics = np.unique([m.groups()[0] for m in metrics if m is not None])

        for result in all_results:
            target = result["target"].unique()[0]
            for metric in metrics:
                for n_target in [5, 10, 20, 40, 80, 160, 320, 640, 1280]:
                    for seed in [0, 1, 2, 3, 4]:
                        if metric in GREATER_IS_BETTER:
                            df = result[result[f"cv_{n_target}_{seed}/{metric}"].arg_max()]
                        else:
                            df = result[result[f"cv_{n_target}_{seed}/{metric}"].arg_min()]

                        summarized.append(
                            {
                                "target": target,
                                "metric": metric,
                                "cv_value": df[f"cv_{n_target}_{seed}/{metric}"].item(),
                                "test_value": df[f"test_{n_target}_{seed}/{metric}"].item(),
                                **{p: df[p].item() for p in parameter_names},
                                "seed": seed,
                                "n_target": n_target,
                                "result_name": result_name,
                            }
                        )
    summary = pl.DataFrame(summarized)
    pl.Config.set_tbl_rows(100)

    for metric in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for dataset, ax in zip(SOURCES, axes.flat):
            data = summary.filter((pl.col("target") == dataset) & (pl.col("metric") == metric))
            for result_name in data["result_name"].unique():
                data_ = data.filter(pl.col("result_name") == result_name)
                data_ = data_.group_by("n_target").agg([pl.col("test_value").mean().alias("score"), pl.col("test_value").std().alias("std")]).sort("n_target")
                ax.plot(data_["n_target"], data_["score"], label=result_name)
                ax.fill_between(data_["n_target"], data_["score"] - data_["std"], data_["score"] + data_["std"], color="grey", alpha=0.2)
            ax.set_xscale("log")
            ax.legend()
            ax.set_title(dataset)

        log_fig(fig, f"n_samples_{metric}.png", client, run_id=target_run.info.run_id)

    #  summary.filter(pl.col("metric") == "r2").group_by(["target", "result_name", "n_target"]).agg(pl.col("test_value").mean()).sort(["target", "result_name", "n_target"])
if __name__ == "__main__":
    main()

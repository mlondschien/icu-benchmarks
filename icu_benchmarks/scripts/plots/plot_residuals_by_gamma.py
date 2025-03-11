import json
import logging
import tempfile

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import GREATER_IS_BETTER
from icu_benchmarks.mlflow_utils import get_target_run, log_fig
from icu_benchmarks.plotting import DATASET_NAMES, METRICS, PARAMETER_NAMES

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns2.db",
)
@click.option("--experiment_name", type=click.Path(exists=True))
def main(tracking_uri, experiment_name):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment, target_run = get_target_run(client, "experiment_name")

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
        if len(sources) < 4:
            continue
        results = results.with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.lit(sources).alias("sources"),
            pl.lit(run.data.tags["outcome"]).alias("outcome"),
        )
        all_results.append(results)

    results = pl.concat(all_results, how="diagonal")

    print(f"logging to {run.info.run_id}")

    params = [p for p in PARAMETER_NAMES if p in results.columns]

    sources = results["sources"].explode().unique().to_list()
    results_n2 = results.filter(pl.col("sources").list.len() == len(sources) - 2)
    results_n1 = results.filter(pl.col("sources").list.len() == len(sources) - 1)

    metrics = [m for m in METRICS if f"eicu/test/{m}" in results.columns]
    for metric in metrics:
        for x in params:
            mult = -1 if metric in GREATER_IS_BETTER else 1

            cv_results = []

            fig, axes = plt.subplots(
                2,
                3,
                figsize=(12, 8),
                constrained_layout=True,
                gridspec_kw={"hspace": 0.02},
            )

            for idx, (ax, target) in enumerate(zip(axes.flat, sources)):
                filter_ = ~pl.col("sources").list.contains(target)
                cv_results = results_n2.filter(filter_)
                cv_sources = [source for source in sources if source != target]
                columns = (
                    params
                    + [f"{target}/test/{metric}", f"{target}/test/mean_residual"]
                    + [
                        f"{target}/test/quantile_{q}"
                        for q in [0.1, 0.25, 0.5, 0.75, 0.9]
                    ]
                )
                result = results_n1.filter(filter_).select(columns)
                for source in cv_sources:
                    filter_ = ~pl.col("sources").list.contains(source)
                    result = result.join(
                        cv_results.filter(filter_).select(
                            params
                            + [
                                (pl.col(f"{source}/train_val/{metric}") * mult).alias(
                                    f"{source}/cv_value"
                                )
                            ]
                        ),
                        on=params,
                        how="left",
                        validate="1:1",
                    )

                agg = pl.mean_horizontal(
                    [f"{source}/cv_value" for source in cv_sources]
                )
                result = result.with_columns(agg.alias("cv_value"))
                grouped = (
                    result.group_by("x")
                    .agg(pl.all().top_k_by(k=1, by="cv_value", reverse=True))
                    .select(pl.all().explode())
                    .sort("x")
                )

                ax.set_xlabel("x")
                color = "tab:blue"
                ax.set_ylabel("residuals", color=color)
                ax.plot(
                    grouped["x"],
                    grouped[f"{target}/test/mean_residual"],
                    color="black",
                    label="mean" if idx == 0 else None,
                )

                ax.plot(
                    grouped["x"],
                    np.zeros_like(grouped["x"]),
                    color="grey",
                    ls="dotted",
                    alpha=0.5,
                )
                ax.plot(
                    grouped["x"],
                    grouped[f"{target}/test/quantile_0.5"],
                    color=color,
                    label="median" if idx == 0 else None,
                )

                ax.fill_between(
                    grouped["x"],
                    grouped[f"{target}/test/quantile_0.1"],
                    grouped[f"{target}/test/quantile_0.25"],
                    color=color,
                    alpha=0.1,
                    label="10% - 90%" if idx == 0 else None,
                )

                ax.fill_between(
                    grouped["x"],
                    grouped[f"{target}/test/quantile_0.75"],
                    grouped[f"{target}/test/quantile_0.9"],
                    color=color,
                    alpha=0.1,
                    label=None,
                )

                ax.fill_between(
                    grouped["x"],
                    grouped[f"{target}/test/quantile_0.25"],
                    grouped[f"{target}/test/quantile_0.75"],
                    color=color,
                    alpha=0.2,
                    label="25% - 75%" if idx == 0 else None,
                )

                best = grouped.select(
                    pl.col("x").top_k_by(k=1, by="cv_value", reverse=True)
                ).item()
                ax.axvline(best, color="black", ls="dashed", alpha=0.2)

                ax.set_title(DATASET_NAMES["target"])
                ax.set_xscale("log")
                ax.label_outer()
                ax.yaxis.set_tick_params(
                    labelleft=True
                )  # manually add x & y ticks again
                ax.xaxis.set_tick_params(labelbottom=True)

            fig.legend(loc="outside lower center", ncols=4)
            log_fig(
                fig,
                f"residuals/residuals_{experiment_name}_{x}_{metric}.png",
                client,
                target_run.info.run_id,
            )
            log_fig(
                fig,
                f"residuals/residuals_{experiment_name}_{x}_{metric}.pdf",
                client,
                target_run.info.run_id,
            )


if __name__ == "__main__":
    main()

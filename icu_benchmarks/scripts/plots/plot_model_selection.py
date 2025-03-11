import logging
import tempfile
from matplotlib import colormaps
import click
import matplotlib.pyplot as plt
import polars as pl
from mlflow.tracking import MlflowClient
from icu_benchmarks.mlflow_utils import log_fig, get_target_run
from icu_benchmarks.plotting import DATASET_NAMES, PARAMETER_NAMES, METRICS
import numpy as np
from icu_benchmarks.constants import GREATER_IS_BETTER
import json
import matplotlib.cm as cm
from matplotlib import colors

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
@click.option("--experiment_name", type=str)
def main(tracking_uri, experiment_name):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment, target_run = get_target_run(client, experiment_name)

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

            cmap = colormaps.get_cmap("plasma")
            if x in ["gamma", "alpha", "ratio"]:
                norm = colors.LogNorm(vmin=results[x].min(), vmax=results[x].max())
            else:
                norm = colors.Normalize(vmin=results[x].min(), vmax=results[x].max())
            im = cm.ScalarMappable(norm=norm)
            for idx, (ax, target) in enumerate(zip(axes.flat, sources)):
                filter_ = ~pl.col("sources").list.contains(target)
                cv_results = results_n2.filter(filter_)
                cv_sources = [source for source in sources if source != target]
                columns = params + [f"{target}/test/{metric}"]
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

                ax.set_xlabel(f"CV {metric}")
                ax.set_ylabel(f"test {metric}")

                filter_ = pl.col("cv_value").le(pl.col("cv_value").median()) & pl.col(
                    f"{target}/test/{metric}"
                ).le(pl.col(f"{target}/test/{metric}").median())
                im = ax.scatter(
                    result.filter(filter_)["cv_value"] * mult,
                    result.filter(filter_)[f"{target}/test/{metric}"],
                    alpha=0.5,
                    s=6,
                    c=result.filter(filter_)[x],
                    cmap=cmap,
                    norm=norm,
                )

                ax.set_title(DATASET_NAMES[target])
                ax.label_outer()
                ax.yaxis.set_tick_params(
                    labelleft=True
                )  # manually add x & y ticks again
                ax.xaxis.set_tick_params(labelbottom=True)

            fig.colorbar(im, ax=axes.ravel().tolist())
            log_fig(
                fig,
                f"model_selection/model_selection_{experiment_name}_{x}_{metric}.png",
                client,
                target_run.info.run_id,
            )
            log_fig(
                fig,
                f"model_selection/model_selection_{experiment_name}_{x}_{metric}.pdf",
                client,
                target_run.info.run_id,
            )


if __name__ == "__main__":
    main()

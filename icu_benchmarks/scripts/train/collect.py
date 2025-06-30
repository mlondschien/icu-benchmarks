import logging
import re

import click
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import DATASETS, GREATER_IS_BETTER
from icu_benchmarks.mlflow_utils import get_results, log_df
from icu_benchmarks.plotting import PARAMETERS, cv_results

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

ALL_SOURCES = [
    "miiv",
    "mimic-carevue",
    "hirid",
    "eicu",
    "aumc",
    "sic",
    "nwicu",
    "zigong",
    "picdb",
]


@click.command()
@click.option("--experiment_name", type=str)
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns3.db",
)
@click.option("--result_name", default="results")
@click.option("--output_name", default="cv_results")
def main(experiment_name: str, tracking_uri: str, result_name, output_name):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    if "mlflow.note.content" in experiment.tags:
        print(experiment.tags["mlflow.note.content"])

    experiment_id = experiment.experiment_id

    result_file = f"{result_name}.csv"
    output_file = f"{output_name}.csv"

    results = get_results(client, experiment_name, result_file)

    # Filter to max_depth=3, gamma=1, sqrt(2), 2, sqrt(8), ..., num_iteration=1000
    if "gamma" in results.columns:
        col = pl.col("gamma").log() / pl.lit(2).sqrt().log()
        results = results.filter(np.abs(col - col.round(0)).le(0.01))
    if "max_depth" in results.columns:
        results = results.filter(pl.col("max_depth").eq(3))
        results = results.filter(pl.col("gamma").le(16.0))
    if "num_iteration" in results.columns:
        results = results.filter(pl.col("num_iteration").eq(1000))

    if "alpha" in results.columns:
        col = pl.col("alpha").log10() - pl.col("alpha").min().log10()
        results = results.with_columns(
            pl.coalesce(
                pl.when(np.abs(col - x) < 0.01).then(x) for x in [2, 3, 4]
            ).alias("alpha_index")
        )
        results = results.filter(pl.col("alpha_index").eq(3))

    parameter_names = [x for x in PARAMETERS if x in results.columns]

    metrics = map(re.compile(r"^[a-z]+\/test\/(.+)$").match, results.columns)
    metrics = np.unique([m.groups()[0] for m in metrics if m is not None])

    all_targets = map(re.compile(r"^(.+)\/test\/[a-z]+$").match, results.columns)
    all_targets = np.unique([m.groups()[0] for m in all_targets if m is not None])

    out = []
    for target in all_targets:
        ood_results = results.filter(~pl.col("sources").list.contains(target))
        for metric in metrics:
            cv = cv_results(ood_results, [metric])
            if metric in GREATER_IS_BETTER:
                best = cv[cv[f"__cv_{metric}"].arg_max()]
            else:
                best = cv[cv[f"__cv_{metric}"].arg_min()]

            out.append(
                {
                    **{
                        "target": target,
                        "cv_metric": metric,
                        "cv_value": best[f"__cv_{metric}"].item(),
                        "test_value": best[f"{target}/test/{metric}"].item(),
                    },
                    **{p: best[p].item() for p in parameter_names},
                    **{
                        f"{source}/{split}/{m}": best[f"{source}/{split}/{m}"].item()
                        for source in sorted(DATASETS)
                        for m in metrics
                        for split in ["train_val", "test"]
                        if f"{source}/{split}/{m}" in best.columns
                    },
                }
            )

    target_run = client.search_runs(
        experiment_ids=[experiment_id], filter_string="tags.sources = ''"
    )
    if len(target_run) > 0:
        target_run = target_run[0]
    else:
        target_run = client.create_run(
            experiment_id=experiment_id, tags={"sources": ""}
        )

    print(f"logging to {target_run.info.run_id}")
    log_df(
        pl.DataFrame(out),
        output_file,
        client,
        target_run.info.run_id,
    )


if __name__ == "__main__":
    main()

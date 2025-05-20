import json
import logging
import re
import tempfile

import click
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import DATASETS, GREATER_IS_BETTER
from icu_benchmarks.mlflow_utils import log_df
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
    "miived",
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

    runs = client.search_runs(experiment_ids=[experiment_id], max_results=10_000)

    result_file = f"{result_name}.csv"
    output_file = f"{output_name}.csv"

    all_results = []
    for run in runs:
        run_id = run.info.run_id

        if run.data.tags["sources"] == "":
            continue
        else:
            sources = json.loads(run.data.tags["sources"].replace("'", '"'))
            if len(sources) not in [4, 5, 6]:
                continue

        with tempfile.TemporaryDirectory() as f:
            if result_file not in [x.path for x in client.list_artifacts(run_id)]:
                logger.warning(f"Run {run_id} has no {result_file}")
                continue

            client.download_artifacts(run_id, result_file, f)
            results = pl.read_csv(f"{f}/{result_file}")

        results = results.with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.lit(sources).alias("sources"),
            pl.lit(run.data.tags["outcome"]).alias("outcome"),
        )
        all_results.append(results)

    results = pl.concat(all_results)

    results = results.filter(pl.col("gamma").eq(1.0)) ## remove

    parameter_names = [x for x in PARAMETERS if x in results.columns]

    if "random_state" in results.columns:
        results = results.filter(pl.col("random_state").eq(0))

    metrics = map(re.compile(r"^[a-z]+\/test\/(.+)$").match, results.columns)
    metrics = np.unique([m.groups()[0] for m in metrics if m is not None])

    all_targets = map(re.compile(r"^(.+)\/test\/[a-z]+$").match, results.columns)
    all_targets = np.unique([m.groups()[0] for m in all_targets if m is not None])
    
    out = []

    for target in sorted(all_targets):
        ood_results = results.filter(~pl.col("sources").list.contains(target))
        for metric in metrics:
            cv = cv_results(ood_results, [metric])
        
            if metric in GREATER_IS_BETTER:
                best = cv[cv[f"__cv_{metric}"].arg_max()]
            else:
                best = cv[cv[f"__cv_{metric}"].arg_min()]
            model = cv.filter(
                pl.all_horizontal(pl.col(p).eq(best[p]) for p in parameter_names)
            )
            out.append(
                {
                    **{
                        "target": target,
                        "cv_metric": metric,
                        "cv_value": best[f"__cv_{metric}"].item(),
                        "test_value": model[f"{target}/test/{metric}"].item(),
                    },
                    **{p: best[p].item() for p in parameter_names},
                    **{
                        f"{source}/{split}/{m}": model[
                            f"{source}/{split}/{m}"
                        ].item()
                        for source in sorted(DATASETS)
                        for m in metrics
                        for split in ["train_val", "test"]
                        if f"{source}/{split}/{m}" in model.columns
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

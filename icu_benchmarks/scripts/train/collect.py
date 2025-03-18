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
from icu_benchmarks.plotting import PARAMETERS

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option("--experiment_name", type=str)
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns2.db",
)
def main(experiment_name: str, tracking_uri: str):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    if "mlflow.note.content" in experiment.tags:
        print(experiment.tags["mlflow.note.content"])

    experiment_id = experiment.experiment_id

    runs = client.search_runs(experiment_ids=[experiment_id], max_results=10_000)

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

    parameter_names = [x for x in PARAMETERS if x in results.columns]

    for var, val in [
        (None, None),
        # ("learning_rate", 0.02),
        # ("learning_rate", 0.1),
        # ("colsample_bytree", 0.5),
        # ("colsample_bytree", 1.0),
        # ("bagging_fraction", 0.5),
        # ("bagging_fraction", 1.0),
        # ("min_data_in_leaf", 20),
        # ("min_data_in_leaf", 100),
    ]:
        results = pl.concat(all_results)
        # results = results.filter(pl.col("learning_rate").eq(0.1))
        if var is not None:
            results = results.filter(pl.col(var).eq(val))

        sources = results["sources"].explode().unique().to_list()

        metrics = map(re.compile(r"^[a-z]+\/test\/(.+)$").match, results.columns)
        metrics = np.unique([m.groups()[0] for m in metrics if m is not None])

        # results = results.filter(pl.col("gamma").eq(1))

        results_n2 = results.filter(pl.col("sources").list.len() == len(sources) - 2)
        results_n1 = results.filter(pl.col("sources").list.len() == len(sources) - 1)

        cv_results = []

        for target in sorted(sources):
            cv = results_n2.filter(~pl.col("sources").list.contains(target))
            cv_sources = [source for source in sources if source != target]
            for metric in metrics:
                expr = pl.coalesce(
                    pl.when(~pl.col("sources").list.contains(t)).then(
                        pl.col(f"{t}/train_val/{metric}")
                    )
                    for t in cv_sources
                )
                col = f"target/train_val/{metric}"

                cv = cv.with_columns(expr.alias(col))
                cv_grouped = cv.group_by(parameter_names).agg(pl.mean(col))
                if metric in GREATER_IS_BETTER:
                    best = cv_grouped[cv_grouped[col].arg_max()]
                else:
                    best = cv_grouped[cv_grouped[col].arg_min()]

                model = results_n1.filter(
                    ~pl.col("sources").list.contains(target)
                    & pl.all_horizontal(pl.col(p).eq(best[p]) for p in parameter_names)
                )
                cv_results.append(
                    {
                        **{
                            "target": target,
                            "metric": metric,
                            "cv_value": best[col].item(),
                            "test_value": model[f"{target}/test/{metric}"].item(),
                        },
                        **{p: best[p].item() for p in parameter_names},
                        **{
                            f"{source}/train_val/": model[
                                f"{source}/train_val/{metric}"
                            ].item()
                            for source in sorted(DATASETS)
                            if f"{source}/train_val/{metric}" in model.columns
                        },
                    }
                )
        cv_results = pl.DataFrame(cv_results)
        # string = f"{var}:{val}".ljust(22)
        # for target in sorted(sources):
        #     string += f"{cv_results.filter(pl.col('target').eq(target) & pl.col('metric').eq('rmse'))['test_value'].item():7.4f} "
        # print(string)
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
        pl.DataFrame(cv_results),
        "cv_results.csv",
        client,
        target_run.info.run_id,
    )


if __name__ == "__main__":
    main()

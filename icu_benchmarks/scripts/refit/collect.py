import json
import logging
import re
import tempfile

import click
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import DATASETS

GREATER_IS_BETTER = ["accuracy", "roc", "auprc", "r2"]
SOURCES = ["mimic-carevue", "miiv", "eicu", "aumc", "sic", "hirid"]
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
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns.db",
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
        sources = json.loads(run.data.tags["sources"].replace("'", '"'))
        if len(sources) != 5:
            continue

        with tempfile.TemporaryDirectory() as f:
            if "refit_results.csv" not in [
                x.path for x in client.list_artifacts(run_id)
            ]:
                logger.warning(f"Run {run_id} has no results.csv")
                continue

            client.download_artifacts(run_id, "refit_results.csv", f)
            results = pl.read_csv(f"{f}/refit_results.csv")

        results = results.with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.lit(run.data.tags["sources"]).alias("sources"),
            pl.lit(run.data.tags["outcome"]).alias("outcome"),
        )
        if len(results.columns) == 306:
            all_results.append(results)

    parameter_names = [
        x
        for x in [
            "alpha_idx",
            "l1_ratio",
            "gamma",
            "num_boost_round",
            "num_iteration",
            "learning_rate",
            "num_leaves",
            "ratio",
            "decay_rate",
        ]
        if x in results.columns
    ]

    results = pl.concat(all_results)
    results = results.with_columns(
        pl.col("sources").str.replace_all("'", '"').str.json_decode()
    )
    sources = results["sources"].explode().unique().to_list()

    metrics = map(re.compile(r"\/(.+)$").search, results.columns)
    metrics = np.unique([m.groups()[0] for m in metrics if m is not None])

    # results = results.filter(pl.col("gamma").eq(1))

    out = []
    for result in all_results:
        target = [
            t for t in SOURCES if t not in result["sources"].explode().unique()[0]
        ][0]
        for metric in metrics:
            for n_target in [10, 30, 100, 300, 1000]:
                for seed in [0, 1, 2, 3, 4]:
                    if metric in GREATER_IS_BETTER:
                        df = result[result[f"cv_{n_target}_{seed}/{metric}"].arg_max()]
                    else:
                        df = result[result[f"cv_{n_target}_{seed}/{metric}"].arg_min()]

                    out.append(
                        {
                            "target": target,
                            "metric": metric,
                            "cv_value": df[f"cv_{n_target}_{seed}/{metric}"].item(),
                            "test_value": df[f"test_{n_target}_{seed}/{metric}"].item(),
                            **{p: df[p].item() for p in parameter_names},
                            "seed": seed,
                            "n_target": n_target,
                        }
                    )

    out = pl.DataFrame(out)

    results_n2 = results.filter(pl.col("sources").list.len() == len(sources) - 2)
    results_n1 = results.filter(pl.col("sources").list.len() == len(sources) - 1)

    cv_results = []

    for target in sources:
        cv = results_n2.filter(~pl.col("sources").list.contains(target))
        cv_sources = [source for source in sources if source != target]
        for metric in metrics:
            expr = pl.coalesce(
                pl.when(~pl.col("sources").list.contains(t)).then(
                    pl.col(f"{t}/train/{metric}")
                )
                for t in cv_sources
            )
            col = f"target/train/{metric}"

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
                        "target_value": model[f"{target}/train/{metric}"].item(),
                    },
                    **{p: best[p].item() for p in parameter_names},
                    **{
                        f"{source}/train/": model[f"{source}/train/{metric}"].item()
                        for source in sorted(DATASETS)
                        if f"{source}/train/{metric}" in model.columns
                    },
                }
            )

    for metric in metrics:
        print("metric:", metric)
        with pl.Config() as cfg:
            cfg.set_tbl_cols(20)
            print(
                pl.DataFrame(cv_results)
                .filter(pl.col("metric").eq(metric))
                .sort("target")
            )


if __name__ == "__main__":
    main()

import re
import tempfile

import click
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import DATASETS

GREATER_IS_BETTER = ["accuracy", "roc", "auprc", "r2"]


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
        with tempfile.TemporaryDirectory() as f:
            client.download_artifacts(run_id, "results.csv", f)
            results = pl.read_csv(f"{f}/results.csv")

        results = results.with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.lit(run.data.tags["sources"]).alias("sources"),
            pl.lit(run.data.tags["outcome"]).alias("outcome"),
        )
        all_results.append(results)

    results = pl.concat(all_results)
    results = results.with_columns(
        pl.col("sources").str.replace_all("'", '"').str.json_decode()
    )
    sources = results["sources"].explode().unique().to_list()

    metrics = map(re.compile(r"^[a-z]+\/train\/([a-z]+)$").match, results.columns)
    metrics = np.unique([m.groups()[0] for m in metrics if m is not None])

    results_n2 = results.filter(pl.col("sources").list.len() == len(sources) - 2)
    results_n1 = results.filter(pl.col("sources").list.len() == len(sources) - 1)

    cv_results = []

    for target in sources:
        cv = results_n2.filter(~pl.col("sources").list.contains(target))

        for metric in metrics:
            expr = pl.coalesce(
                pl.when(~pl.col("sources").list.contains(t)).then(
                    pl.col(f"{t}/train/{metric}")
                )
                for t in sources
            )
            col = f"target/train/{metric}"

            cv = cv.with_columns(expr.alias(col))
            cv_grouped = cv.group_by(["l1_ratio_idx", "alpha_idx"]).agg(pl.mean(col))

            if metric in GREATER_IS_BETTER:
                best = cv_grouped[cv_grouped[col].arg_max()]
            else:
                best = cv_grouped[cv_grouped[col].arg_min()]

            model = results_n1.filter(
                ~pl.col("sources").list.contains(target)
                & pl.col("alpha_idx").eq(best["alpha_idx"])
                & pl.col("l1_ratio_idx").eq(best["l1_ratio_idx"])
            )
            cv_results.append(
                {
                    **{
                        "target": target,
                        "metric": metric,
                        "l1_ratio_idx": best["l1_ratio_idx"].item(),
                        "alpha_idx": best["alpha_idx"].item(),
                        "cv_value": best[col].item(),
                        "target_value": model.filter(
                            ~pl.col("sources").list.contains(target)
                            & pl.col("alpha_idx").eq(best["alpha_idx"])
                            & pl.col("l1_ratio_idx").eq(best["l1_ratio_idx"])
                        )[f"{target}/train/{metric}"].item(),
                    },
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

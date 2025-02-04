import logging

import click
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.benchmarks import severinghaus_spo2_to_po2, mortality_from_apache_ii
from icu_benchmarks.constants import DATASETS, TASKS
from icu_benchmarks.load import load
from icu_benchmarks.metrics import metrics
from icu_benchmarks.mlflow_utils import log_df
import mlflow
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
@click.option("--data_dir", type=str, default="/cluster/work/math/lmalte/data")
@click.option("--outcome", type=str, default=None)
@click.option("--artifact_location", type=str, default="file:///cluster/work/math/lmalte/mlflow/artifacts")
def main(experiment_name, tracking_uri, data_dir, outcome, artifact_location):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name, artifact_location=artifact_location)
    else:
        experiment_id = experiment.experiment_id

    target_run = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.sources = '' and tags.outcome = '{outcome}'",
    )
    if len(target_run) > 0:
        target_run = target_run[0]
    else:
        target_run = client.create_run(
            experiment_id=experiment_id, tags={"sources": "", "outcome": outcome}
        )

    results = []
    for target in DATASETS:
        if outcome == "log_po2":
            df, y, _ = load(
                sources=[target],
                outcome="log_po2",
                split="test",
                variables=["spo2", "sao2"],
                data_dir=data_dir,
            )

            if len(y) == 0:
                continue
        
            sao2 = df.select(
                pl.when(pl.col("sao2_all_missing_h8"))
                .then(pl.col("spo2_mean_h8"))
                .otherwise(pl.col("sao2_ffilled"))
                .fill_null(pl.col("sao2_ffilled").mean())
            ).to_numpy()

            yhat = np.log(severinghaus_spo2_to_po2(sao2 / 100))
        elif outcome == "mortality_at_24h":
            if target == "miived":
                continue
            df, y, _, apache_ii = load(
                sources=[target],
                outcome="mortality_at_24h",
                split="test",
                variables=["adm"],
                data_dir=data_dir,
                other_columns=["apache_ii"]
            )
            adm = df["adm"].to_numpy()
            apache_ii = apache_ii.to_numpy().copy()
            apache_ii[np.isnan(apache_ii)] = 0.1
            yhat = mortality_from_apache_ii(apache_ii, adm)
        else:
            raise ValueError(f"No benchmark for outcome {outcome}.")
    
        results += [
            {"target": target, "target_value": value, "metric": key}
            for key, value in metrics(y, yhat, "", TASKS[outcome]["task"]).items()
        ]

    print(f"logging to {target_run.info.run_id}")
    log_df(
        pl.DataFrame(results),
        "benchmark_results.csv",
        client,
        target_run.info.run_id,
    )


if __name__ == "__main__":
    main()

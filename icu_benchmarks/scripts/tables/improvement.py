import logging
import tempfile

import click
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.metrics import get_equivalent_number_of_samples
from icu_benchmarks.mlflow_utils import log_markdown

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_result(client, experiment_name, result_name):
    """Get the result from a summary run in an experiment."""
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string="tags.sources = ''"
    )
    if len(runs) != 1:
        raise ValueError(f"Expected 1 summary run, got {len(runs)}")

    run_id = runs[0].info.run_id
    result_file = f"{result_name}_results.csv"
    with tempfile.TemporaryDirectory() as f:
        if result_file not in [x.path for x in client.list_artifacts(run_id)]:
            logger.warning(f"Run {run_id} in {experiment_name} has no {result_file}.")
            return None

        client.download_artifacts(run_id, result_file, f)
        return pl.read_csv(f"{f}/{result_file}")


SOURCES = ["aumc", "eicu", "mimic-carevue", "miiv", "sic", "hirid"]


@click.command()
@click.option("--target_experiment", type=str)
@click.option("--n_samples_experiment", type=str)
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns3.db",
)
@click.option("--from", "from_", type=str)
@click.option("--to", type=str)
def main(target_experiment, n_samples_experiment, tracking_uri, from_, to):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)
    target_experiment = client.get_experiment_by_name(target_experiment)
    target_run = client.search_runs(
        experiment_ids=[target_experiment.experiment_id],
        filter_string="tags.sources = ''",
    )[0]

    n_samples_result = get_result(client, n_samples_experiment, "n_samples")
    from_result = get_result(client, from_, "cv")
    to_result = get_result(client, to, "cv")

    results = []

    # Approach 1: For each seed, calculate the equivalent number of samples from -> to.
    # Take the log of the quotient, and average.
    for metric in n_samples_result["metric"].unique().to_list():
        for target in SOURCES:
            filter = pl.col("target").eq(target) & pl.col("metric").eq(metric)
            from_value = from_result.filter(filter)["test_value"].item()
            to_value = to_result.filter(filter)["test_value"].item()
            for seed in range(20):
                from_equiv, to_equiv = get_equivalent_number_of_samples(
                    n_samples_result.filter(filter & pl.col("seed").eq(seed)),
                    [from_value, to_value],
                    metric=metric,
                )
                results.append(
                    {
                        "target": target,
                        "metric": metric,
                        "seed": seed,
                        "from_value": from_value,
                        "to_value": to_value,
                        "from_equiv": from_equiv,
                        "to_equiv": to_equiv,
                    }
                )
    result = pl.DataFrame(results)

    result = (
        result.with_columns(
            (pl.col("to_equiv") / pl.col("from_equiv")).alias("improvement")
        )
        .group_by(["target", "metric"])
        .agg(
            [
                (100 * pl.col("improvement") - 100.0).quantile(0.2).alias("20%"),
                (100 * pl.col("improvement") - 100.0)
                .quantile(0.5, interpolation="linear")
                .alias("median"),
                (100 * pl.col("improvement") - 100.0).quantile(0.8).alias("80%"),
            ]
        )
        .sort(["metric", "target"])
    )
    print(result)
    log_markdown(result, "improvement.md", client, target_run.info.run_id)


if __name__ == "__main__":
    main()

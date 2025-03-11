import mlflow
from icu_benchmarks.constants import OUTCOMES, DATASETS
from mlflow.tracking import MlflowClient
from icu_benchmarks.plotting import get_method_name, METRIC_NAMES, SHORT_DATASET_NAMES
from icu_benchmarks.mlflow_utils import log_fig
from icu_benchmarks.metrics import get_equivalent_number_of_samples

import matplotlib.pyplot as plt
import numpy as np
import tempfile
import polars as pl
import click
from icu_benchmarks.constants import GREATER_IS_BETTER


@click.command()
@click.option("--experiment_name", type=str)
@click.option("--n_samples_name", type=str)
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns2.db",
)
def main(experiment_name: str, n_samples_name: str, tracking_uri: str):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    n_experiment = client.get_experiment_by_name(n_samples_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string="tags.sources = ''"
    )
    if len(runs) != 1:
        raise ValueError(f"Expected exactly one run. Got {runs:}")
    run = runs[0]

    nsamples_runs = client.search_runs(
        experiment_ids=[n_experiment.experiment_id], filter_string="tags.sources = ''"
    )
    if len(nsamples_runs) != 1:
        raise ValueError(f"Expected exactly one run. Got {nsamples_runs:}")
    nsamples_run = nsamples_runs[0]

    with tempfile.TemporaryDirectory() as f:
        client.download_artifacts(run.info.run_id, f"1v1_results.csv", f)
        df = pl.read_csv(f"{f}/1v1_results.csv")

    with tempfile.TemporaryDirectory() as f:
        client.download_artifacts(nsamples_run.info.run_id, f"n_samples_results.csv", f)
        df_nsamples = pl.read_csv(f"{f}/n_samples_results.csv")

    metrics = df["metric"].unique().to_numpy()
    sources = sorted(df["source"].unique())
    for metric in metrics:
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))
        df_ = df.filter(pl.col("metric").eq(metric)).sort("source")
        grid = df_.select([f"{s}/test/{metric}" for s in sources]).to_numpy()

        n_samples_grid1 = np.zeros(shape=(len(sources), len(sources), 20))
        for i, target in enumerate(sources):
            for seed in range(20):
                filter = (
                    pl.col("target").eq(target)
                    & pl.col("metric").eq(metric)
                    & pl.col("seed").eq(seed)
                )
                n_samples_grid1[:, i, seed] = get_equivalent_number_of_samples(
                    df_nsamples.filter(filter), grid[:, i], metric
                )
                # n_samples_grid[i, :, seed] = df_nsamples.filter(filter)["test_value"].to_numpy()
        n_samples_grid1 = np.mean(np.log10(n_samples_grid1), axis=-1)

        n_samples_grid2 = np.zeros_like(grid)
        for i, target in enumerate(sources):
            filter = pl.col("target").eq(target) & pl.col("metric").eq(metric)
            n_samples_grid2[:, i] = np.log10(
                get_equivalent_number_of_samples(
                    df_nsamples.filter(filter)
                    .group_by("n_target")
                    .agg(pl.all().median()),
                    grid[i, :],
                    metric,
                )
            )

        _ = ax[0].imshow(grid, cmap="viridis", aspect="auto")
        _ = ax[1].imshow(n_samples_grid1, cmap="viridis", aspect="auto")
        _ = ax[2].imshow(n_samples_grid2, cmap="viridis", aspect="auto")

        for (i, j), z in np.ndenumerate(grid):
            ax[0].text(j, i, f"{z:.3f}", ha="center", va="center")

        for (i, j), z in np.ndenumerate(n_samples_grid1):
            ax[1].text(j, i, f"{z:.3f}", ha="center", va="center")

        for (i, j), z in np.ndenumerate(n_samples_grid2):
            ax[2].text(j, i, f"{z:.3f}", ha="center", va="center")

        ax[0].set_xlabel("target dataset")
        ax[0].set_ylabel("source dataset")
        ax[0].set_xticks(np.arange(grid.shape[1]))
        ax[0].set_xticklabels([SHORT_DATASET_NAMES[x] for x in sources], rotation=45)
        ax[0].set_yticks(np.arange(grid.shape[0]))
        ax[0].set_yticklabels([SHORT_DATASET_NAMES[x] for x in sources])

        ax[1].set_xlabel("target dataset")
        ax[1].set_ylabel("source dataset")
        ax[1].set_xticks(np.arange(grid.shape[1]))
        ax[1].set_xticklabels([SHORT_DATASET_NAMES[x] for x in sources], rotation=45)
        ax[1].set_yticks(np.arange(grid.shape[0]))
        ax[1].set_yticklabels([SHORT_DATASET_NAMES[x] for x in sources])

        ax[2].set_xlabel("target dataset")
        ax[2].set_ylabel("source dataset")
        ax[2].set_xticks(np.arange(grid.shape[1]))
        ax[2].set_xticklabels([SHORT_DATASET_NAMES[x] for x in sources], rotation=45)
        ax[2].set_yticks(np.arange(grid.shape[0]))
        ax[2].set_yticklabels([SHORT_DATASET_NAMES[x] for x in sources])

        ax[1].set_title(f"{get_method_name(experiment_name)} ({METRIC_NAMES[metric]})")
        log_fig(
            fig,
            f"1v1/{metric}.png",
            client=client,
            run_id=run.info.run_id,
            bbox_inches="tight",
            pad_inches=0.1,
        )


if __name__ == "__main__":
    main()

from pathlib import Path

import click
import matplotlib.pyplot as plt
import polars as pl

from icu_benchmarks.constants import DATA_DIR, DATASETS
from icu_benchmarks.plotting import plot_continuous, plot_discrete

OUTPUT_PATH = Path(__file__).parents[2] / "figures" / "density_plots"

OUTCOMES = [
    "remaining_los",
    "mortality_at_24h",
    "decompensation_at_24h",
    "respiratory_failure_at_24h",
    "circulatory_failure_at_8h",
    "kidney_failure_at_48h",
]


@click.command()
def main():  # noqa D
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 15))

    for ax, outcome in zip(axes.flat[: len(OUTCOMES)], OUTCOMES):
        data = {
            dataset: pl.read_parquet(
                DATA_DIR / dataset / "features.parquet", columns=[outcome]
            ).to_series()
            for dataset in DATASETS
        }
        if outcome != "remaining_los":
            plot_discrete(ax, data, outcome, missings=True)
        else:
            data = {
                # Values < e^-10 ~ 0.00004 don't make sense.
                k: v.log().clip(-10, None).rename("log(remaining_los)")
                for k, v in data.items()
                if v.count() > 0
            }
            plot_continuous(ax, data, "log(remaining_los)")
    fig.savefig(OUTPUT_PATH / "outcomes.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

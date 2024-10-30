from pathlib import Path

import click
import matplotlib.pyplot as plt
import polars as pl

from icu_benchmarks.constants import DATA_DIR, DATASETS, OUTCOMES
from icu_benchmarks.plotting import plot_continuous, plot_discrete

OUTPUT_PATH = Path(__file__).parents[2] / "figures" / "density_plots"


@click.command()
def main():  # noqa D
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(25, 15))

    for ax, outcome in zip(axes.flat[: len(OUTCOMES)], OUTCOMES):
        data = {
            dataset: pl.read_parquet(
                DATA_DIR / dataset / "features.parquet", columns=[outcome]
            ).to_series()
            for dataset in DATASETS
        }
        if outcome not in ["remaining_los", "los_at_24h"]:
            plot_discrete(ax, data, outcome, missings=True)
        else:
            data = {
                # Values < e^-10 ~ 0.00004 don't make sense.
                k: v.log().clip(-10, None).rename(f"log({outcome})")
                for k, v in data.items()
                if v.count() > 0
            }
            plot_continuous(ax, data, f"log({outcome})")

    fig.savefig(OUTPUT_PATH / "outcomes.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

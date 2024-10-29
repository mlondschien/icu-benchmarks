from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from icu_benchmarks.constants import DATA_DIR, VARIABLE_REFERENCE_PATH

DATASETS = [
    "mimic",
    "ehrshot",
    "miived",
    "miiv",
    "eicu",
    "hirid",
    "aumc",
    "sic",
    "zigong",
    "picdb",
]
OUTPUT_PATH = Path(__file__).parents[2] / "figures" / "density_plots"

SOURCE_COLORS = {
    "eicu": "black",
    "mimic": "red",
    "hirid": "blue",
    "miiv": "orange",
    "aumc": "green",
    "sic": "purple",
    "zigong": "brown",
    "picdb": "pink",
    "ehrshot": "gray",
    "miived": "cyan",
}

variable_reference = pl.read_csv(
    VARIABLE_REFERENCE_PATH, separator="\t", null_values=["None"]
)
variable_reference = variable_reference.filter(
    pl.col("DataType").is_in(["continuous", "treatment_cont"])
)


@click.command()
@click.option("--ncols", default=10, help="Number of columns in the plot grid.")
def main(ncols):  # noqa D
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    nrows = len(variable_reference) // ncols + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

    for variable, ax in zip(
        variable_reference.rows(named=True), axes.flat[: len(variable_reference)]
    ):
        data = {}
        null_fractions = {}
        for dataset in DATASETS:
            if variable["VariableType"] == "static":
                file = DATA_DIR / dataset / "sta.parquet"
            else:
                file = DATA_DIR / dataset / "dyn.parquet"

            df = (
                pl.scan_parquet(file)
                .select(
                    pl.col(variable["VariableTag"]).clip(
                        lower_bound=variable["LowerBound"],
                        upper_bound=variable["UpperBound"],
                    )
                )
                .collect()
                .to_series()
            )

            if variable["LogTransform"] and variable["LogTransformEps"] is not None:
                df = (
                    (df + variable["LogTransformEps"])
                    .log()
                    .replace([np.nan, -np.inf], None)
                )
            elif variable["LogTransform"]:
                df = df.log().replace([np.nan, -np.inf], None)

            null_fractions[dataset] = df.is_null().mean()
            df = df.drop_nulls()
            data[dataset] = df.to_numpy()

        max_ = np.max([np.max(x) for x in data.values() if len(x) > 0])
        min_ = np.min([np.min(x) for x in data.values() if len(x) > 0])

        for dataset in DATASETS:
            df = data[dataset]
            if len(df) <= 1:
                ax.plot(
                    [], [], label=f"{dataset} ({100 * null_fractions[dataset]:.1f}%)"
                )
            elif len(np.unique(df)) == 1:
                ax.plot(df[0], [0], label=f"{dataset} (100%)", marker="x")
            else:
                # https://stackoverflow.com/a/35874531/10586763
                # `gaussian_kde` uses bw = std * bw_method(). To ensure equal bandwidths,
                # divide by the std of the dataset.
                bandwidth = (max_ - min_) / 100 / df.std()
                density = gaussian_kde(df, bw_method=lambda x: bandwidth)

                linspace = np.linspace(df.min(), df.max(), num=100)

                ax.plot(
                    linspace,
                    density(linspace),
                    label=f"{dataset} ({100 * null_fractions[dataset]:.1f}%)",
                    color=SOURCE_COLORS[dataset],
                )

        if variable["LogTransform"]:
            title = f'log({variable["VariableTag"]}): {variable["VariableName"]}'
        else:
            title = f'{variable["VariableTag"]}: {variable["VariableName"]}'

        ax.set_title(title)
        ax.legend()

    fig.savefig(OUTPUT_PATH / "densities.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

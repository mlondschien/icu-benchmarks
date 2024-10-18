from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

DATA_DIR = Path(__file__).parents[1] / "data"
DATASETS = ["mimic", "miiv", "eicu", "hirid", "aumc", "sic", "zigong", "picdb"]
OUTPUT_PATH = Path(__file__).parents[1] / "figures" / "density_plots"

SOURCE_COLORS = {
    "eicu": "black",
    "mimic": "red",
    "hirid": "blue",
    "miiv": "orange",
    "aumc": "green",
    "sic": "purple",
    "zigong": "brown",
    "picdb": "pink",
}

VARIABLE_REFERENCE_PATH = (
    Path(__file__).parents[1] / "resources" / "variable_reference.tsv"
)
variable_reference = pl.read_csv(VARIABLE_REFERENCE_PATH, separator="\t")
variable_reference = variable_reference.with_columns(
    pl.col("LowerBound").replace("None", None).cast(pl.Float64),
    pl.col("UpperBound").replace("None", None).cast(pl.Float64),
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
        for dataset in DATASETS:
            if variable["VariableType"] == "static":
                data = pl.scan_parquet(DATA_DIR / dataset / "sta.parquet")
            else:
                data = pl.scan_parquet(DATA_DIR / dataset / "dyn.parquet")

            data = data.select(variable["VariableTag"]).collect().to_series()
            data = data.clip(
                lower_bound=variable["LowerBound"], upper_bound=variable["UpperBound"]
            )

            if variable["LogTransform"] == "true":
                # We want to avoid mapping zeros to missings. A simple solution is to
                # use log1p (log(1 + x)) instead of log(x). The offshift `1` is
                # arbitrary. Ideally, one would use something "in the order of
                # measurement error".
                # If the variable has a lower bound > 0, we can simply log it.
                if variable["LowerBound"] is not None and variable["LowerBound"] > 0:
                    data = data.log()
                # If the variable has no bounds, we don't know which constant to choose.
                # One could alternatively use log1p here.
                elif variable["LowerBound"] is None or variable["UpperBound"] is None:
                    # log(0) = -np.inf, log(-1) = np.nan
                    data = data.log().replace([np.nan, -np.inf], None)
                # If the variable has a lower bound of 0 and an upper bound, we can
                # just use 1e-3 * upper bound as the offshift.
                elif variable["LowerBound"] == 0 and variable["UpperBound"] is not None:
                    data = (data + 0.001 * variable["UpperBound"]).log()

            null_fraction = data.is_null().mean()
            data = data.drop_nulls()

            if len(data) <= 1:
                ax.plot([], [], label=f"{dataset} (100%)", color=SOURCE_COLORS[dataset])
            else:
                density = gaussian_kde(data.to_numpy())
                linspace = np.linspace(data.min(), data.max(), num=100)

                ax.plot(
                    linspace,
                    density(linspace),
                    label=f"{dataset} ({100 * null_fraction:.1f}%)",
                    color=SOURCE_COLORS[dataset],
                )

            if variable["LogTransform"] == "true":
                title = f'log({variable["VariableTag"]}): {variable["VariableName"]}'
            else:
                title = f'{variable["VariableTag"]}: {variable["VariableName"]}'

            ax.set_title(title)
            ax.legend()

    fig.savefig(OUTPUT_PATH / "densities.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

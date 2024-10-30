from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from icu_benchmarks.constants import DATA_DIR, DATASETS, VARIABLE_REFERENCE_PATH
from icu_benchmarks.plotting import plot_continuous, plot_discrete

OUTPUT_PATH = Path(__file__).parents[2] / "figures" / "density_plots"

variable_reference = pl.read_csv(
    VARIABLE_REFERENCE_PATH, separator="\t", null_values=["None"]
)


@click.command()
@click.option("--ncols", default=10, help="Number of columns in the plot grid.")
def main(ncols):  # noqa D
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    nrows = (len(variable_reference) - 1) // ncols + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

    for variable, ax in zip(
        variable_reference.rows(named=True), axes.flat[: len(variable_reference)]
    ):
        data = {}
        file = "sta.parquet" if variable["VariableType"] == "static" else "dyn.parquet"

        if variable["DataType"] in ["continuous", "treatment_cont"]:
            for dataset in DATASETS:
                df = (
                    pl.scan_parquet(DATA_DIR / dataset / file)
                    .select(
                        pl.col(variable["VariableTag"]).clip(
                            lower_bound=variable["LowerBound"],
                            upper_bound=variable["UpperBound"],
                        )
                    )
                    .collect()
                    .to_series()
                )

                if variable["DataType"] == "treatment_cont":
                    df = df.replace(0, None)

                if variable["LogTransform"] and variable["LogTransformEps"] is not None:
                    df = (
                        (df + variable["LogTransformEps"])
                        .log()
                        .replace([np.nan, -np.inf], None)
                    )
                elif variable["LogTransform"]:
                    df = df.log().replace([np.nan, -np.inf], None)

                data[dataset] = df

            if variable["LogTransform"]:
                title = f'log({variable["VariableTag"]})'
            else:
                title = f'{variable["VariableTag"]}'

            _ = plot_continuous(ax, data, title)

        elif variable["DataType"] in ["categorical", "treatment_ind"]:
            for dataset in DATASETS:
                data[dataset] = pl.read_parquet(
                    DATA_DIR / dataset / file, columns=[variable["VariableTag"]]
                ).to_series()

            if variable["DataType"] == "treatment_ind":
                _ = plot_discrete(
                    ax,
                    {k: v.fill_null(False) for k, v in data.items()},
                    variable["VariableTag"],
                    False,
                )
            else:
                _ = plot_discrete(ax, data, variable["VariableTag"], True)

    fig.savefig(OUTPUT_PATH / "densities.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

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
).filter(pl.col("DatasetVersion").is_not_null())



@click.command()
@click.option("--ncols", default=6, help="Number of columns in the plot grid.")
@click.option("--data_dir", type=click.Path(exists=True))
@click.option("--extra_datasets", is_flag=True)
def main(data_dir=None, ncols=6, extra_datasets=False):  # noqa D
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    nrows = (len(variable_reference) - 1) // ncols + 1

    if not extra_datasets:
        datasets = [d for d in DATASETS if "-" not in d]
    else:
        datasets = DATASETS

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

    for variable, ax in zip(
        variable_reference.rows(named=True), axes.flat[: len(variable_reference)]
    ):
        data = {}
        file = "sta.parquet" if variable["VariableType"] == "static" else "dyn.parquet"

        if variable["DataType"] in ["continuous", "treatment_cont"]:
            for dataset in datasets:
                df = (
                    pl.scan_parquet(data_dir / dataset / file)
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
            for dataset in datasets:
                data[dataset] = pl.read_parquet(
                    data_dir / dataset / file, columns=[variable["VariableTag"]]
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

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH / f"densities_{extra_datasets}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

from pathlib import Path

import click
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from icu_benchmarks.constants import DATA_DIR, DATASETS
from icu_benchmarks.plotting import (
    SHORT_DATASET_NAMES,
    SOURCE_COLORS,
    plot_continuous,
    plot_discrete,
)

OUTPUT_PATH = Path(__file__).parents[3] / "figures" / "density_plots"


@click.command()
@click.option("--data_dir", type=click.Path(exists=True))
@click.option("--prevalence", type=click.Choice(["time-step", "patient"]))
@click.option("--extra_datasets", is_flag=True)
def main(data_dir=None, prevalence="time-step", extra_datasets=False):  # noqa D
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    datasets = DATASETS
    datasets.reverse()

    fig = plt.figure(figsize=(12, 8))
    plt.rcParams.update({"ytick.labelsize": 12, "xtick.labelsize": 12})
    gs = gridspec.GridSpec(4, 3, height_ratios=[1, 0.05, 1, 0.05], wspace=0.05)

    top_outcomes = [
        "mortality_at_24h",
        "circulatory_failure_at_8h",
        "respiratory_failure_at_24h",
        # "respiratory_failure_at_24h",
        # "severe_meld_at_48h"
    ]
    top_axes = [fig.add_subplot(gs[0, i]) for i in range(len(top_outcomes))]

    for ax, outcome in zip(top_axes, top_outcomes):
        data = {
            dataset: pl.read_parquet(
                data_dir / dataset / "features.parquet", columns=["stay_id", outcome]
            )
            for dataset in datasets
        }
        if prevalence == "patient":
            data = {
                k: v.group_by("stay_id")
                .agg(
                    pl.when(pl.col(outcome).is_not_null().any()).then(
                        pl.col(outcome).any()
                    )
                )
                .select(outcome)
                .to_series()
                for k, v in data.items()
            }
        else:
            data = {k: v.select(outcome).to_series() for k, v in data.items()}

        plot_discrete(
            ax,
            data,
            outcome,
            missings=True,
            legend=False,
            yticklabels=ax.get_subplotspec().is_first_col(),
        )

    bottom_outcomes = [
        # "log_pf_ratio_in_12h",
        # "log_lactate_in_4h",
        # "meld_score_in_24h_3",
        # "meld_score_in_24h",
        # "meld_score_in_24h"
        "log_lactate_in_4h",
        "log_bili_in_24h",
    ]
    ax1 = fig.add_subplot(gs[2, 1])
    ax2 = fig.add_subplot(gs[2, 2], sharey=ax1)
    bottom_axes = [ax1, ax2]

    for ax, outcome in zip(bottom_axes, bottom_outcomes):
        data = {
            # Values < e^-10 ~ 0.00004 don't make sense.
            dataset: pl.read_parquet(
                data_dir / dataset / "features.parquet", columns=[outcome]
            ).to_series()
            for dataset in datasets
        }
        plot_continuous(
            ax,
            data,
            outcome,
            legend=False,
            missing_rate=False,
            label=False,
        )

    # ax0 = fig.add_subplot(gs[2, 0])

    ax1.tick_params(axis="y")
    ax2.tick_params(axis="y", labelleft=False)  # Hide y-tick labels on ax2

    tab_blue = mcolors.to_rgba("tab:blue", alpha=0.9)
    patches = [
        Patch(facecolor=tab_blue, edgecolor=None, label="false"),
        Patch(facecolor="tab:orange", edgecolor=None, label="true"),
        Patch(facecolor="grey", edgecolor=None, label="missing"),
    ]
    fig.legend(
        patches,
        ["false", "true", "missing"],
        loc="center",
        bbox_to_anchor=(0.5, 0.59 / 1.1),
        ncol=3,
        fontsize=13,
    )

    datasets.reverse()
    white = Line2D([], [], color="none", lw=0)
    lines = [white, white]
    text = ["", ""]

    # lines = [Line2D([0], [0], color="white", lw=0), Line2D([0], [0], color="white", lw=0)], Line2D([0], [0], color="white", lw=0), Line2D([0], [0], color="white", lw=0), Line2D([0], [0], color="white", lw=0), Line2D([0], [0], color="white", lw=0)]
    # text = ["", "missings", "", "Po2/Fio2", "lactate"]
    for d in datasets:
        lines.append(Line2D([0], [0], color=SOURCE_COLORS[d], lw=2))
        text.append(SHORT_DATASET_NAMES[d])

    lines += [white, white]
    text += ["missing ", "lactate"]

    for d in datasets:
        lines.append(white)
        missingness = (
            pl.scan_parquet(data_dir / d / "features.parquet")
            .select(pl.col("log_lactate_in_4h").is_null().mean())
            .collect()
            .item()
        )
        text.append(f"{missingness:.1%}")

    lines += [white, white]
    text += ["values", "bilirubin"]

    for d in datasets:
        lines.append(white)
        missingness = (
            pl.scan_parquet(data_dir / d / "features.parquet")
            .select(pl.col("log_bili_in_24h").is_null().mean())
            .collect()
            .item()
        )
        text.append(f"{missingness:.1%}")

    fig.legend(
        lines,
        text,
        loc="center",
        bbox_to_anchor=(0.19, 0.33),  # (x, y), y from the bottom
        ncol=3,
        fontsize=13,
        handlelength=1.5,  # default is ~2
        labelspacing=0.3,
        columnspacing=-1,
        handletextpad=0.4,
    )

    fig.savefig(OUTPUT_PATH / "outcomes_m.png", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUTPUT_PATH / "outcomes_m.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":
    main()

from pathlib import Path

import click
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from icu_benchmarks.constants import DATA_DIR, DATASETS, VERY_SHORT_DATASET_NAMES, SOURCE_COLORS, OUTCOME_NAMES
from icu_features.load import load
from scipy.stats import gaussian_kde
from matplotlib.transforms import Bbox
OUTPUT_PATH = Path(__file__).parents[3] / "figures" / "density_plots"


@click.command()
@click.option("--data_dir", type=click.Path(exists=True), default="/cluster/work/math/lmalte/data")
@click.option("--prevalence", type=click.Choice(["time-step", "patient"]))
@click.option("--extra_datasets", is_flag=True)
def main(data_dir=None, prevalence="time-step", extra_datasets=False):  # noqa D
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    datasets = DATASETS
    datasets.reverse()

    fig = plt.figure(figsize=(9, 2.6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.04, hspace=0.39)

    binary_outcomes = ["circulatory_failure_at_8h", "kidney_failure_at_48h"]
    binary_axes = [fig.add_subplot(gs[0, i]) for i in range(len(binary_outcomes))]

    for ax, outcome in zip(binary_axes, binary_outcomes):
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, 1)

        y_pos = np.arange(len(datasets))
        left = np.zeros(len(datasets))
        
        _, y, other = load(
            datasets,
            outcome,
            split=None,
            data_dir="/cluster/work/math/lmalte/data",
            other_columns=["dataset"],
            variables=[],
            horizons=[],
        )
        df = other.with_columns(y=y).group_by("dataset").agg(pl.col("y").mean())
        df = df.sort("dataset")
        
        bars = ax.barh(y_pos, 1 -  df["y"], left=left, label=df["dataset"], height=0.8, color="tab:blue")
        for bar in bars:
            width = bar.get_width()
            ax.text(
                0.2,
                bar.get_y() + bar.get_height() / 2 - 0.05,
                f"{width * 100:.1f}" if width < 0.9995 else "99.9",
                ha="right",
                va="center",
                fontsize=10,
                color="black",
            )

        bars = ax.barh(y_pos, df["y"], left=1 - df["y"], label=df["dataset"], height=0.8, color="tab:orange")
        for bar in bars:
            width = bar.get_width()
            ax.text(
                0.99 if width > 0.2 else 1 - width - 0.01,
                bar.get_y() + bar.get_height() / 2 - 0.05,
                f"{width * 100:.1f}" if width < 0.9995 else "99.9",
                ha="right",
                va="center",
                fontsize=10,
                color="black",
            )
        
    binary_axes[0].set_yticks(y_pos, labels=[VERY_SHORT_DATASET_NAMES[ds] for ds in datasets], fontsize=10)
    binary_axes[1].set_yticks([])

    binary_axes[0].set_title("circ. failure within 8h", fontsize=10)
    binary_axes[1].set_title("acute kidney inj. within 48h", fontsize=10)

    cont_outcomes = [
        "log_lactate_in_4h",
        "log_creatinine_in_24h",
    ]
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3], sharey=ax3)
    cont_axes = [ax3, ax4]

    for ax, outcome in zip(cont_axes, cont_outcomes):
        _, y, other = load(
            datasets,
            outcome,
            split=None,
            data_dir="/cluster/work/math/lmalte/data",
            other_columns=["dataset"],
            variables=[],
            horizons=[],
        )
        bw = (np.max(y, axis=0) - np.min(y, axis=0)) / 80 /  np.std(y, axis=0)
        for dataset in datasets:
            filter_ = pl.col("dataset").eq(dataset)
            y_ = other.with_columns(y=y).filter(filter_).select("y").to_series()
            density = gaussian_kde(y_, bw_method=bw)
            linspace = np.linspace(y_.min(), y_.max(), num=300)

            ax.plot(linspace, density(linspace), color=SOURCE_COLORS[dataset], lw=2, alpha=0.8)
            ax.set_title(OUTCOME_NAMES[outcome], fontsize=10)


    ax3.yaxis.tick_right()
    ax4.yaxis.tick_right()
    ax3.tick_params(axis="y", labelright=False)

    patches = [
        Patch(facecolor="tab:blue", edgecolor=None, label="False"),
        Patch(facecolor="tab:orange", edgecolor=None, label="True"),
    ]
    fig.legend(
        patches,
        ["False", "True"],
        loc="center",
        bbox_to_anchor=(0.32, 0.06),
        ncol=2,
        fontsize=10,
        frameon=False,
    )

    for ax in cont_axes:
        ax.set_ylim(-0.03, 1.3)
        ax.set_yticks([0.0, 0.5, 1.0], labels = ["0.0", "0.5", "1.0"], fontsize=10)
        ax.tick_params(axis='x', labelsize=10)

    inv_fig = fig.transFigure.inverted()
    for i, label in enumerate(binary_axes[0].yaxis.get_majorticklabels()):
        bbox = label.get_window_extent()

        pixel_coords = (bbox.x0, bbox.y0 + bbox.height / 2)
        label_x_fig, label_y_fig = inv_fig.transform(pixel_coords)

        line_x_coords = [label_x_fig - 0.018, label_x_fig-0.005]
        line_y_coords = [label_y_fig+0.005, label_y_fig+0.005]
        line = Line2D(
            line_x_coords,
            line_y_coords,
            transform=fig.transFigure,  # Use the figure transform
            color=SOURCE_COLORS[datasets[i]],
            linewidth=3,
            clip_on=False
        )

        # Add the line to the axes' artists
        binary_axes[0].add_artist(line)


    fig.savefig(OUTPUT_PATH / "outcomes_m.png", bbox_inches="tight")
    fig.savefig(OUTPUT_PATH / "outcomes_m.pdf", bbox_inches=Bbox([[0.2, 0.05], [8.5, 2.5]]))
    plt.close(fig)


if __name__ == "__main__":
    main()

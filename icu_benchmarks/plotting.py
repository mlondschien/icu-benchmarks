import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

SOURCE_COLORS = {
    "eicu": "black",
    "mimic": "red",
    "mimic-carevue": "red",
    "mimic-metavision": "red",
    "hirid": "blue",
    "miiv": "orange",
    "miiv-late": "orange",
    "aumc": "green",
    "aumc-early": "green",
    "aumc-late": "green",
    "sic": "purple",
    "zigong": "brown",
    "picdb": "pink",
    "ehrshot": "gray",
    "miived": "cyan",
}

LINESTYLES = {
    "miiv-late": "dashed",
    "aumc-early": "dotted",
    "aumc-late": "dashed",
    "mimic-metavision": "dotted",
    "mimic-carevue": "dashed",
}


def plot_discrete(ax, data, name, missings=True):
    """
    Visualize the distribution of discrete variables with stacked horizontal bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    data : dict[str, polars.Series]
        Dictionary mapping dataset names to series. The series can contain missings and
        should have name `name`.
    name : str
        Name of the variable.
    missings : bool
        Whether to plot a gray bar for the fraction of missing values.
    """
    y_pos = np.arange(len(data))

    df = (
        pl.concat(
            [
                v.drop_nulls().value_counts(normalize=True, name=k)
                for k, v in data.items()
            ],
            how="align",
        )
        .fill_null(0)
        .sort(name)
    )

    ax.xaxis.set_visible(False)
    ax.set_xlim(0, 1)

    left = np.zeros(len(data))
    for row in df.rows():
        bars = ax.barh(y_pos, row[1:], left=left, label=row[0], height=0.6)
        ax.bar_label(
            bars,
            # No labels for blocks smaller than 0.5%. Too much clutter.
            labels=[f"{x * 100:.1f}" if x >= 0.005 else "" for x in bars.datavalues],
            label_type="center",
            color="black",
        )
        left += row[1:]

    if missings:
        _ = ax.barh(
            y_pos - 0.4,
            [v.is_null().mean() for v in data.values()],
            height=0.2,
            color="gray",
        )

    ax.legend(ncols=min(3, len(df)), bbox_to_anchor=(0.5, -0.05), loc="lower center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([d.replace("-", "-\n") for d in data.keys()])

    ax.set_title(name)
    return ax


def plot_continuous(ax, data, name, legend=True, missing_rate=True):
    """
    Visualize the distribution of continuous variables with kernel density estimates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    data : dict[str, polars.Series]
        Dictionary mapping dataset names to series. The series can contain missings and
        should have name `name`.
    name : str
        Title.
    legend : bool
        Whether to show a legend.
    missing_rate : bool
        Whether to include the missing rate in the legend.
    """
    null_fractions = {k: v.is_null().mean() for k, v in data.items()}
    data = {k: v.drop_nulls().to_numpy() for k, v in data.items()}

    max_ = np.max([np.max(x) for x in data.values() if len(x) > 0])
    min_ = np.min([np.min(x) for x in data.values() if len(x) > 0])

    for dataset, df in data.items():
        label = (
            f"{dataset} ({100 * null_fractions[dataset]:.1f}%)"
            if missing_rate
            else dataset
        )
        if len(df) <= 1:
            ax.plot([], [], label=label)
        elif len(np.unique(df)) == 1:
            ax.plot(df[0], [0], label=label, marker="x")
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
                label=label,
                color=SOURCE_COLORS[dataset],
                linestyle=LINESTYLES.get(dataset, "solid"),
            )

    ax.set_title(name)
    if legend:
        ax.legend()

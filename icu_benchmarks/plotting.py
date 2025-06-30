import warnings

import matplotlib.colors as mcolors
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from icu_benchmarks.constants import (
    OUTCOME_NAMES,
    PARAMETERS,
    SOURCE_COLORS,
    VERY_SHORT_DATASET_NAMES,
)


def plot_discrete(ax, data, name, missings=True, legend=True, yticklabels=True):
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
        if row[0] is True:
            color = "tab:orange"
        elif row[0] is False:
            color = mcolors.to_rgba("tab:blue", alpha=0.9)
        else:
            color = None
        bars = ax.barh(y_pos, row[1:], left=left, label=row[0], height=0.7, color=color)

        if len(df) == 2 and row[0] is False:
            for bar in bars:
                width = bar.get_width()
                if width < 0.005:
                    continue
                x_pos = 0.2 if width > 0.1 else width + 0.15
                text = f"{width * 100:.1f}" if width < 0.9995 else "99.9"
                ax.text(
                    x_pos,
                    bar.get_y() + bar.get_height() / 2 - 0.05,
                    text,
                    ha="right",
                    va="center",
                    fontsize=10,
                    color="black",
                )

        elif len(df) == 2 and row[0] is True:
            for bar in bars:
                width = bar.get_width()
                if width < 0.005:
                    continue
                x_pos = 0.99 if width > 0.2 else 1 - width - 0.01
                text = f"{width * 100:.1f}" if width < 0.9995 else "99.9"
                ax.text(
                    x_pos,
                    bar.get_y() + bar.get_height() / 2 - 0.05,
                    text,
                    ha="right",
                    va="center",
                    fontsize=10,
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

    if legend:
        ax.legend(
            ncols=min(3, len(df)), bbox_to_anchor=(0.5, -0.05), loc="lower center"
        )

    if yticklabels:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [VERY_SHORT_DATASET_NAMES.get(d, d) for d in data.keys()], fontsize=10
        )
    else:
        ax.set_yticklabels([])
        ax.set_yticks([])

    ax.set_title(OUTCOME_NAMES.get(name, name), fontsize=10)
    return ax


def plot_continuous(ax, data, name, label=True, legend=True, missing_rate=True):
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

    df = np.concatenate([v for v in data.values() if len(v) > 0])
    std = df.std().item()
    max_ = np.max([np.max(x) for x in data.values() if len(x) > 0])
    min_ = np.min([np.min(x) for x in data.values() if len(x) > 0])
    # `gaussian_kde` uses bw = std * bw_method(). To ensure equal bandwidths,
    # divide by the std of the dataset.
    # https://stackoverflow.com/a/35874531/10586763

    for dataset, df in data.items():
        label = None
        if legend and missing_rate:
            label = f"{dataset} ({100 * null_fractions[dataset]:.1f}%)"
        elif legend:
            label = dataset

        kwargs = {
            "label": label,
            "color": SOURCE_COLORS[dataset.split(":")[0]],
        }

        if len(df) <= 1:
            ax.plot([], [], **kwargs)
        elif len(np.unique(df)) == 1:
            ax.plot(df[0], [0], marker="x", **kwargs)
        else:
            density = gaussian_kde(df, bw_method=lambda x: (max_ - min_) / 80 / std)

            linspace = np.linspace(df.min(), df.max(), num=300)

            ax.plot(linspace, density(linspace), **kwargs)

    if legend:
        ax.legend()
    ax.set_title(OUTCOME_NAMES.get(name, name), fontsize=10)


def cv_results(results, metrics):  # noqa D
    params = [p for p in PARAMETERS if p in results.columns]

    if len(params) == 0:
        results = results.with_columns(pl.lit(0).alias("__param"))
        params = ["__param"]

    sources = results["sources"].explode().unique().to_list()
    sources = [s for s in sources if f"{s}/train_val/{metrics[0]}" in results.columns]
    cv = results.filter(pl.col("sources").list.len() == len(sources) - 1)

    if len(cv) == 0:
        return results.with_columns(
            pl.lit(0).alias(f"__cv_{metric}") for metric in metrics
        )

    for metric in metrics:
        cv = cv.with_columns(
            pl.coalesce(
                pl.when(~pl.col("sources").list.contains(s)).then(
                    pl.col(f"{s}/train_val/{metric}")
                )
                for s in sources
            ).alias(f"__cv_{metric}")
        )
    if cv.group_by(params).len().select(pl.col("len").ne(len(sources)).any()).item():
        warnings.warn(
            f"Not all sources present for CV for sources: {sources}.\n{cv.group_by(params).len()}"
        )
        # breakpoint()
    cv = cv.group_by(params).agg(pl.mean(f"__cv_{metric}") for metric in metrics)

    target = results.filter(pl.col("sources").list.len() == len(sources))
    cv = cv.join(target, on=params, how="full", coalesce=True)

    return cv

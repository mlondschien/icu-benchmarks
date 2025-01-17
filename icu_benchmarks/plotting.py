import re

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from icu_benchmarks.constants import TASKS

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
    "nwicu": "yellow",
}

LINESTYLES = {
    "miiv-late": "dashed",
    "aumc-early": "dotted",
    "aumc-late": "dashed",
    "mimic-metavision": "dotted",
    "mimic-carevue": "dashed",
}

PARAMETER_NAMES = [
    "alpha",
    "ratio",
    "l1_ratio",
    "gamma",
    "num_boost_round",
    "num_iteration",
    "learning_rate",
    "num_leaves",
]

GREATER_IS_BETTER = ["auc", "auprc", "accuracy", "prc", "r2"]


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


def plot_by_x(results, x, metric, aggregation="mean"):
    """
    Plot the value of `metric` for each dataset in `results` as a function of `x`.

    Parameters
    ----------
    x : str
        Variable to plot on the x-axis.
    results : pl.DataFrame
        DataFrame with variables `x` (str), `"{s}/train/{metric}` (float) and
        `"{s}/test/{metric}"` for `s` in `sources`, `"sources"` (list[str] or str
        representation of list), and columns from `PARAMETER_NAMES`.
    metric : str
        Metric to plot. E.g., `"mse"`, `"mae"`, `"quantile_0.9"` for regression or
        `"auc"`, `"accuracy"`, `"prc"`, `"log_loss"` for classification.
    aggregation : str, optional, default="mean"
        Aggregation method for cross-validation results. One of "mean", "median",
        "worst", "mean_05", "mean_1".
    """
    param_names = [p for p in PARAMETER_NAMES if p in results.columns and p != x]

    task = TASKS[results["outcome"].unique().to_list()[0]]

    if results["sources"].dtype == pl.String:
        expr = pl.col("sources").str.replace_all("'", '"').str.json_decode()
        results = results.with_columns(expr)

    sources = results["sources"].explode().unique().to_list()

    metrics = map(re.compile(r"^[a-z]+\/train\/(.+)$").match, results.columns)
    metrics = np.unique([m.groups()[0] for m in metrics if m is not None])

    results_n2 = results.filter(pl.col("sources").list.len() == len(sources) - 2)
    results_n1 = results.filter(pl.col("sources").list.len() == len(sources) - 1)

    cv_results = []

    fig, axes = plt.subplots(2, len(sources) // 2, figsize=(2.5 * len(sources), 10))

    def argbest(df, col, metric):
        if metric in GREATER_IS_BETTER:
            return df[df[col].arg_max()]
        return df[df[col].arg_min()]

    for target, ax in zip(sorted(sources), axes.flat[: len(sources)]):
        # cv_results are results where the current target and one additional dataset
        # were held out. For these, we will aggregate the value of `metric` on the
        # additional held-out dataset to select "optimal" parameters.
        cv_results = results_n2.filter(~pl.col("sources").list.contains(target))
        cv_sources = [source for source in sources if source != target]

        # target/train/{metric} is the value of `metric` for the held-out-dataset
        expr = pl.coalesce(
            pl.when(~pl.col("sources").list.contains(t)).then(
                pl.col(f"{t}/train/{metric}")
            )
            for t in cv_sources
        )
        cv_col = f"cv/train/{metric}"
        cv_results = cv_results.with_columns(expr.alias(cv_col))

        expr = pl.coalesce(
            pl.when(~pl.col("sources").list.contains(t)).then(
                pl.lit(task["n_samples"][t])
            )
            for t in cv_sources
        )
        cv_results = cv_results.with_columns(expr.alias("n_samples"))

        if aggregation in ["mean", "mean_0"]:
            agg = pl.mean(cv_col)
        elif aggregation == "mean_05":
            weights = pl.col("n_samples").sqrt() / pl.col("n_samples").sqrt().sum()
            agg = (weights * pl.col(cv_col)).sum()
        elif aggregation == "mean_1":
            weights = pl.col("n_samples") / pl.col("n_samples").sum()
            agg = (weights * pl.col(cv_col)).sum()
        elif aggregation == "median":
            agg = pl.median(cv_col)
        elif aggregation == "worst":
            agg = pl.min(cv_col) if metric in GREATER_IS_BETTER else pl.max(cv_col)
        else:
            raise ValueError(f"Unknown aggregation {aggregation}")

        cv_grouped = cv_results.group_by(param_names + [x]).agg(agg.alias(cv_col))
        # cv_best is the row in cv_grouped with the best value of `metric`.
        cv_best = argbest(cv_grouped, cv_col, metric)

        # cur_results_n1 are results where only target was held out (cur for this loop)
        cur_results_n1 = results_n1.filter(~pl.col("sources").list.contains(target))
        # We filter cur_results_n1 to only include rows with the best parameters
        # (according to cv), except for x, along which we plot. sort by x for plotting.
        _filter = pl.all_horizontal(pl.col(p).eq(cv_best[p]) for p in param_names)
        cur_results_n1_cv = cur_results_n1.filter(_filter).sort(x)

        ax.plot(
            cur_results_n1_cv[x],
            cur_results_n1_cv[f"{target}/test/{metric}"],
            label="model chosen by cv",
            color="blue",
            zorder=3,
        )
        _filter = pl.col(x) == cv_best[x].item()
        ax.scatter(
            cur_results_n1_cv.filter(_filter)[x],
            cur_results_n1_cv.filter(_filter)[f"{target}/test/{metric}"],
            color="blue",
            zorder=3,
            marker="*",
        )

        oracle_best = argbest(cur_results_n1, f"{target}/train/{metric}", metric)
        _filter = pl.all_horizontal(pl.col(p).eq(oracle_best[p]) for p in param_names)
        oracle_results = cur_results_n1.filter(_filter).sort(x)

        ax.plot(
            oracle_results[x],
            oracle_results[f"{target}/test/{metric}"],
            label="oracle model",
            color="black",
            zorder=2,
        )
        ax.scatter(
            oracle_best[x],
            oracle_best[f"{target}/test/{metric}"],
            color="black",
            zorder=2,
            marker="*",
        )

        ymin, ymax = ax.get_ylim()
        for _, group in cur_results_n1.group_by(param_names):
            group = group.sort(x)
            ax.plot(group[x], group[f"{target}/test/{metric}"], color="gray", alpha=0.1)

        ax.set_title(target)
        ax.set_ylim(ymin, ymax)
        if x in ["gamma", "alpha", "ratio", "learning_rate"]:
            ax.set_xscale("log")

        ax.legend()
        ax.set_xlabel(x)

    return fig

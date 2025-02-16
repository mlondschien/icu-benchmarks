import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from icu_benchmarks.constants import GREATER_IS_BETTER

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

# https://personal.sron.nl/~pault/#sec:qualitative
COLORS = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
    "black": "#000000",
    "indigo": "#332288",
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


METRIC_NAMES = {
    "brier": "brier score",
    "roc": "AuROC",
    "auprc": "AuPRC",
    "log_loss": "binomial neg. log-likelihood",
    "accuracy": "accuracy",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "abs_quantile_0.8": "80\\%-quantile of abs. errors",
    "abs_quantile_0.9": "90\\%-quantile of abs. errors",
    "abs_quantile_0.95": "95\\%-quantile of abs. errors",
}

DATASET_NAMES = {
    "sic": "SICdb",
    "aumc": "AmsterdamUMCdb",
    "eicu": "eICU",
    "miiv": "MIMIC-IV",
    "mimic-carevue": "MIMIC (CareVue subset)",
    "hirid": "HiRID",
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

    max_ = np.max([np.max(x) for x in data.values() if len(x) > 0])
    min_ = np.min([np.min(x) for x in data.values() if len(x) > 0])

    for dataset, df in data.items():
        label = None
        if legend and missing_rate:
            label = f"{dataset} ({100 * null_fractions[dataset]:.1f}%)"
        elif legend:
            label = dataset

        kwargs = {
            "label": label,
            "color": SOURCE_COLORS[dataset],
            "linestyle": LINESTYLES.get(dataset, "solid"),
        }

        if len(df) <= 1:
            ax.plot([], [], **kwargs)
        elif len(np.unique(df)) == 1:
            ax.plot(df[0], [0], marker="x", **kwargs)
        else:
            # https://stackoverflow.com/a/35874531/10586763
            # `gaussian_kde` uses bw = std * bw_method(). To ensure equal bandwidths,
            # divide by the std of the dataset.
            bandwidth = (max_ - min_) / 100 / df.std()
            density = gaussian_kde(df, bw_method=lambda x: bandwidth)

            linspace = np.linspace(df.min(), df.max(), num=100)

            ax.plot(linspace, density(linspace), **kwargs)

    if legend:
        ax.legend()
    ax.set_title(name)


def plot_by_x(results, x, metric):
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
    """
    COLORS = {
        "mean": "red",
        "worst": "blue",
        "median": "green",
    }

    param_names = [p for p in PARAMETER_NAMES if p in results.columns and p != x]

    aggregations = ["mean", "worst", "median"]

    if results["sources"].dtype == pl.String:
        expr = pl.col("sources").str.replace_all("'", '"').str.json_decode()
        results = results.with_columns(expr)

    sources = results["sources"].explode().unique().to_list()

    results_n2 = results.filter(pl.col("sources").list.len() == len(sources) - 2)
    results_n1 = results.filter(pl.col("sources").list.len() == len(sources) - 1)
    # results_1v1 = results.filter(pl.col("sources").list.len() == 1)

    cv_results = []

    fig, axes = plt.subplots(
        2, 3, figsize=(12, 8), constrained_layout=True, gridspec_kw={"hspace": 0.02}
    )

    for idx, (target, ax) in enumerate(zip(sorted(sources), axes.flat[: len(sources)])):
        mult = -1 if metric in GREATER_IS_BETTER else 1

        # cv_results are results where the current target and one additional dataset
        # were held out. For these, we will aggregate the value of `metric` on the
        # additional held-out dataset to select "optimal" parameters.
        cv_results = results_n2.filter(~pl.col("sources").list.contains(target))
        cv_sources = [source for source in sources if source != target]

        # target/train/{metric} is the value of `metric` for the held-out-dataset
        expr = pl.coalesce(
            pl.when(~pl.col("sources").list.contains(t)).then(
                pl.col(f"{t}/train_val/{metric}")
            )
            for t in cv_sources
        ) * pl.lit(mult)
        cv_col = f"cv/train_val/{metric}"
        cv_results = cv_results.with_columns(expr.alias(cv_col))

        # expr = pl.coalesce(
        #     pl.when(~pl.col("sources").list.contains(t)).then(
        #         pl.lit(task["n_samples"][t])
        #     )
        #     for t in cv_sources
        # )
        # cv_results = cv_results.with_columns(expr.alias("n_samples"))

        for aggregation in aggregations:
            if aggregation in ["mean", "mean_0"]:
                agg = pl.mean(cv_col)
            elif aggregation == "median":
                agg = pl.median(cv_col)
            elif aggregation == "worst":
                agg = pl.min(cv_col) if metric in GREATER_IS_BETTER else pl.max(cv_col)
            else:
                raise ValueError(f"Unknown aggregation {aggregation}")

            cv_grouped = cv_results.group_by(param_names + [x]).agg(agg.alias(cv_col))

            # cv_top_1 is the row in cv_grouped with the best value of `metric`.
            cv_top_1 = cv_grouped.top_k(1, by=cv_col, reverse=True)[0]

            # cur_results_n1 are results where only target was held out (cur for this loop)
            cur_results_n1 = results_n1.filter(~pl.col("sources").list.contains(target))

            # We filter cur_results_n1 to only include rows with the best parameters
            # (according to cv), except for x, along which we plot. sort by x for plotting.
            _filter = pl.all_horizontal(pl.col(p).eq(cv_top_1[p]) for p in param_names)
            cur_results_n1_cv = cur_results_n1.filter(_filter).sort(x)

            ax.plot(
                cur_results_n1_cv[x],
                cur_results_n1_cv[f"{target}/test/{metric}"],
                label=aggregation if idx == 0 else None,
                color=COLORS[aggregation],
                zorder=3,
            )
            _filter = pl.col(x) == cv_top_1[x].item()
            ax.scatter(
                cur_results_n1_cv.filter(_filter)[x],
                cur_results_n1_cv.filter(_filter)[f"{target}/test/{metric}"],
                color=COLORS[aggregation],
                zorder=3,
                marker="*",
            )
            # ax.text(
            #     cur_results_n1_cv.filter(_filter)[x].item(),
            #     cur_results_n1_cv.filter(_filter)[f"{target}/test/{metric}"].item(),
            #     aggregation,
            #     color="black",
            #     zorder=2,
            # )

        ymin, ymax = ax.get_ylim()
        # oracle_best = argbest(cur_results_n1, f"{target}/train/{metric}", metric)
        # _filter = pl.all_horizontal(pl.col(p).eq(oracle_best[p]) for p in param_names)
        # oracle_results = cur_results_n1.filter(_filter).sort(x)

        # ax.plot(
        #     oracle_results[x],
        #     oracle_results[f"{target}/test/{metric}"],
        #     label="oracle model",
        #     color="black",
        #     zorder=2,
        # )
        # ax.scatter(
        #     oracle_best[x],
        #     oracle_best[f"{target}/test/{metric}"],
        #     color="black",
        #     zorder=2,
        #     marker="*",
        # )

        # variable = "alpha" if "alpha" in results.columns else "num_iteration"
        # if metric in GREATER_IS_BETTER:
        #     ymax = max(cur_results_n1[f"{target}/test/{metric}"].max(), ymax)
        #     var_min = cur_results_n1.filter(pl.col(f"{target}/test/{metric}") >= ymin)[
        #         variable
        #     ].min()
        #     var_max = cur_results_n1.filter(pl.col(f"{target}/test/{metric}") >= ymin)[
        #         variable
        #     ].max()
        # else:
        #     ymin = min(cur_results_n1[f"{target}/test/{metric}"].min(), ymin)
        #     var_min = cur_results_n1.filter(pl.col(f"{target}/test/{metric}") <= ymax)[
        #         variable
        #     ].min()
        #     var_max = cur_results_n1.filter(pl.col(f"{target}/test/{metric}") <= ymax)[
        #         variable
        #     ].max()

        for _, group in cur_results_n1.group_by(param_names):
            group = group.sort(x)
            # var = group[variable].first() / cur_results_n1[variable].max()
            # color = np.clip((var - var_min) / max(0.01, (var_max - var_min)), 0, 1)
            # if 0 <= color <= 1:
            ax.plot(
                group[x],
                group[f"{target}/test/{metric}"],
                color="grey",
                alpha=0.1,
            )
        ax.set_ylabel(f"test {METRIC_NAMES[metric]}")
        if x in ["gamma", "alpha", "ratio", "learning_rate"]:
            ax.set_xscale("log")

        ax.set_xlabel(x)
        ax.label_outer()
        if metric in GREATER_IS_BETTER:
            ymax = cur_results_n1[f"{target}/test/{metric}"].max()
            ymax = ymax + 0.05 * (ymax - ymin)
            ax.set_ylim(ymin, ymax)
        else:
            ymin = cur_results_n1[f"{target}/test/{metric}"].min()
            ymin = ymin - 0.05 * (ymax - ymin)
            ax.set_ylim(ymin, ymax)

        ax.set_title(DATASET_NAMES[target])
        ax.yaxis.set_tick_params(labelleft=True)  # manually add x & y ticks again
        ax.xaxis.set_tick_params(labelbottom=True)

    fig.legend(loc="outside lower center", ncols=4)
    return fig

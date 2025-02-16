from pathlib import Path

import click
import matplotlib.pyplot as plt
import polars as pl

from icu_benchmarks.constants import DATA_DIR, DATASETS, TASKS
from icu_benchmarks.plotting import plot_continuous, plot_discrete

OUTPUT_PATH = Path(__file__).parents[3] / "figures" / "density_plots"

OUTCOMES = [
    "mortality_at_24h",
    # "decompensation_at_24h",
    "respiratory_failure_at_24h",
    "circulatory_failure_at_8h",
    "kidney_failure_at_48h",
    # "remaining_los",
    "los_at_24h",
    # "log_creatine_in_1h",
    # "log_lactate_in_1h",
    # "log_lactate_in_8h",
    "log_lactate_in_4h",
    "log_pf_ratio_in_12h",
    "log_rel_urine_rate_in_2h",
    # "log_rel_urine_rate_in_8h",
]


@click.command()
@click.option("--data_dir", type=click.Path(exists=True))
@click.option("--prevalence", type=click.Choice(["time-step", "patient"]))
@click.option("--extra_datasets", is_flag=True)
def main(data_dir=None, prevalence="time-step", extra_datasets=False):  # noqa D
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    if not extra_datasets:
        datasets = [d for d in DATASETS if "-" not in d]
    else:
        datasets = DATASETS

    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(14, 8),
        constrained_layout=True,
        gridspec_kw={"hspace": 0.02},
    )

    for ax, outcome in zip(axes.flat[: len(OUTCOMES)], OUTCOMES):
        task = TASKS[outcome]
        data = {
            dataset: pl.read_parquet(
                data_dir / dataset / "features.parquet", columns=["stay_id", outcome]
            )
            for dataset in datasets
        }
        if (prevalence == "patient") and (task["task"] == "binary"):
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

        if task["task"] == "binary":
            plot_discrete(ax, data, outcome, missings=True)
        elif (task["task"] == "regression") and (task["family"] == "gamma"):
            data = {
                # Values < e^-10 ~ 0.00004 don't make sense.
                k: v.log().clip(-10, None).rename(f"log({outcome})")
                for k, v in data.items()
                if v.count() > 0
            }
            plot_continuous(
                ax,
                data,
                f"log({outcome})",
                legend=False,
                missing_rate=False,
                label=outcome == "log_rel_urine_rate_in_2h",
            )
        elif task["task"] == "regression":
            plot_continuous(
                ax,
                data,
                outcome,
                legend=False,
                missing_rate=False,
                label=outcome == "log_rel_urine_rate_in_2h",
            )

    if ax.is_first_row() and not ax.is_first_column():
        ax.yaxis.set_ticklabels([])

    fig.savefig(OUTPUT_PATH / "outcomes.png")
    fig.savefig(OUTPUT_PATH / "outcomes.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

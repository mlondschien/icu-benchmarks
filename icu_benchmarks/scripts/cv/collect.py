import re
import tempfile

import click
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient

from icu_benchmarks.constants import DATASETS

GREATER_IS_BETTER = ["accuracy", "roc", "auprc", "r2"]


@click.command()
@click.option("--experiment_name", type=str)
@click.option(
    "--tracking_uri",
    type=str,
    default="sqlite:////cluster/work/math/lmalte/mlflow/mlruns.db",
)
def main(experiment_name: str, tracking_uri: str):  # noqa D
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    if "mlflow.note.content" in experiment.tags:
        print(experiment.tags["mlflow.note.content"])

    experiment_id = experiment.experiment_id

    runs = client.search_runs(experiment_ids=[experiment_id], max_results=10_000)

    all_results = []
    for run in runs:
        run_id = run.info.run_id
        with tempfile.TemporaryDirectory() as f:
            client.download_artifacts(run_id, "results.csv", f)
            results = pl.read_csv(f"{f}/results.csv")

        results = results.with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.lit(run.data.tags["sources"]).alias("sources"),
            pl.lit(run.data.tags["outcome"]).alias("outcome"),
        )
        all_results.append(results)

    results = pl.concat(all_results)
    results = results.with_columns(
        pl.col("sources").str.replace_all("'", '"').str.json_decode()
    )
    sources = results["sources"].explode().unique().to_list()

    metrics = map(re.compile(r"^[a-z]+\/train\/([a-z]+)$").match, results.columns)
    metrics = np.unique([m.groups()[0] for m in metrics if m is not None])

    results_n2 = results.filter(pl.col("sources").list.len() == len(sources) - 2)
    results_n1 = results.filter(pl.col("sources").list.len() == len(sources) - 1)

    cv_results = []

    for target in sources:
        cv = results_n2.filter(~pl.col("sources").list.contains(target))

        for metric in metrics:
            expr = pl.coalesce(
                pl.when(~pl.col("sources").list.contains(t)).then(
                    pl.col(f"{t}/train/{metric}")
                )
                for t in sources
            )
            col = f"target/train/{metric}"

            cv = cv.with_columns(expr.alias(col))
            cv_grouped = cv.group_by(["l1_ratio_idx", "alpha_idx"]).agg(pl.mean(col))

            if metric in GREATER_IS_BETTER:
                best = cv_grouped[cv_grouped[col].arg_max()]
            else:
                best = cv_grouped[cv_grouped[col].arg_min()]

            model = results_n1.filter(
                ~pl.col("sources").list.contains(target)
                & pl.col("alpha_idx").eq(best["alpha_idx"])
                & pl.col("l1_ratio_idx").eq(best["l1_ratio_idx"])
            )
            cv_results.append(
                {
                    **{
                        "target": target,
                        "metric": metric,
                        "l1_ratio_idx": best["l1_ratio_idx"].item(),
                        "alpha_idx": best["alpha_idx"].item(),
                        "cv_value": best[col].item(),
                        "target_value": model.filter(
                            ~pl.col("sources").list.contains(target)
                            & pl.col("alpha_idx").eq(best["alpha_idx"])
                            & pl.col("l1_ratio_idx").eq(best["l1_ratio_idx"])
                        )[f"{target}/train/{metric}"].item(),
                    },
                    **{
                        f"{source}/train/": model[f"{source}/train/{metric}"].item()
                        for source in sorted(DATASETS)
                        if f"{source}/train/{metric}" in model.columns
                    },
                }
            )

    for metric in metrics:
        print("metric:", metric)
        with pl.Config() as cfg:
            cfg.set_tbl_cols(20)
            print(
                pl.DataFrame(cv_results)
                .filter(pl.col("metric").eq(metric))
                .sort("target")
            )


if __name__ == "__main__":
    main()


# # df = pd.DataFrame.from_records([{f"metrics.{k}": v for k, v in run.data.metrics.items()} | {f"params.{k}": v for k, v in run.data.params.items()} | {f"tags.{k}": v for k, v in run.data.tags.items()} | {"run_id": run.info.run_id} for run in runs])
# # df = df[lambda x: x["tags.parent_run"].ne("None")]
# # metrics = ["roc", "brier", "log_loss", "auprc"]
# # splits = ["train", "val", "test"]

# # Invert log_loss to get something where higher is better

# # for dataset in DATASETS:
# #     mask = df["params.sources"] == dataset
# #     for metric in metrics:
# #         if dataset in ["mimic", "ehrshot", "miived"]:
# #             continue
# #         df.loc[mask, f"metrics.val/{metric}"] = df.loc[mask, f"metrics.{dataset}/val/{metric}"]
# for outcome in OUTCOMES:
#     runs = client.search_runs(experiment_ids=[experiment_id], filter_string=f"tags.outcome='{outcome}'", max_results=10_000)
#     df = pd.DataFrame.from_records([{f"metrics.{k}": v for k, v in run.data.metrics.items()} | {f"params.{k}": v for k, v in run.data.params.items()} | {f"tags.{k}": v for k, v in run.data.tags.items()} | {"run_id": run.info.run_id} for run in runs])
#     df = df[lambda x: x["tags.parent_run"].ne("None")]
#     df[[x for x in df.columns if "log_loss" in x or "bier" in x or "mse" in x or "mae" in x]] *= -1

#     metrics = ["mse", "mae", "r2"] if outcome in ["log_lactate_in_1h", "log_rel_urine_rate_in_1h"] else ["roc", "brier", "log_loss", "auprc"]
#     datasets = [d for d in DATASETS if f"['{d}']" in df["tags.sources"].unique()]

#     df = df[lambda x: x["tags.sources"].isin([f"['{d}']" for d in datasets])].reset_index(drop=True)
#     for sources in df["tags.sources"].unique():
#         mask = df["tags.sources"] == sources
#         sources = json.loads(sources.replace("\'", "\""))
#         for metric in metrics:
#             df.loc[mask, f"metrics.val/{metric}"] = np.nanmean([df.loc[mask, f"metrics.{source}/val/{metric}"] for source in sources], axis=0)

#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 30))
#     for ax, metric in zip(axes.flat, metrics):
#         # df[f"metrics.val/{metric}_max"] = df.groupby(["params.sources", "tags.outcome", "params.treatment_detail_level"])[f"metrics.val/{metric}"].transform("max")
#         # df_best = df[df[f"metrics.val/{metric}"] == df[f"metrics.val/{metric}_max"]]
#         df_best = df.iloc[df.groupby(["params.sources", "tags.outcome"])[f"metrics.val/{metric}"].idxmax()]
#         df_best.sort_values("params.sources", inplace=True)
#         mat = df_best[[f"metrics.{dataset}/test/{metric}" for dataset in datasets]].to_numpy()

#         im = ax.imshow(mat, cmap="viridis", aspect="auto")
#         ax.set_title(metric)
#         ax.set_xlabel("test dataset")
#         ax.set_ylabel("train dataset")
#         ax.set_xticks(np.arange(len(datasets))); ax.set_xticklabels(datasets)
#         ax.set_yticks(np.arange(len(datasets))); ax.set_yticklabels(datasets)

#         for (i, j), z in np.ndenumerate(mat):
#             ax.text(j, i, f"{z:.2f}", ha='center', va='center')

#     fig.savefig(f"compare_{outcome}_True.png")
#     print(f"Saved {outcome}")
#     # coefficients = dict()
#     # for row in df.iterrows():
#     #     run_id = row[1]["run_id"]
#     #     with tempfile.TemporaryDirectory() as f:
#     #         client.download_artifacts(run_id, "coefficients.csv", f)
#     #         coefficients[run_id] = pd.read_csv(f"{f}/coefficients.csv")

#     # variable_reference = pl.read_csv(VARIABLE_REFERENCE_PATH)

#     # breakpoint()

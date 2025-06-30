from icu_features.load import load
from icu_benchmarks.constants import TASKS, DATASETS, OUTCOMES
import numpy as np
import polars as pl

table = []

OUTCOMES = [
    "circulatory_failure_at_8h",
    "log_creatinine_in_24h",
    "kidney_failure_at_48h",
    "log_lactate_in_4h",
]

DATASETS = [
    "aumc",
    "eicu",
    "hirid",
    "mimic-carevue",
    "miiv",
    "sic",
    "nwicu",
    "picdb",
    "zigong",
]

table = [
    {
        "dataset_name": "AUMCdb",
        "dataset": "aumc",
        "country": "Netherlands",
        "years": "2003--2016",
        "comment": "",
    },
    {
        "dataset_name": "eICU",
        "dataset": "eicu",
        "country": "USA",
        "years": "2015--2016",
        "comment": "multi-center",
    },
    {
        "dataset_name": "HIRID",
        "dataset": "hirid",
        "country": "Switzerland",
        "years": "2008--2016",
        "comment": "",
    },
    {
        "dataset_name": "MIMIC-III (CV)",
        "dataset": "mimic-carevue",
        "country": "USA",
        "years": "2001--2008",
        "comment": "contains neonatal stays",
    },
    {
        "dataset_name": "MIMIC-IV",
        "dataset": "miiv",
        "country": "USA",
        "years": "2008--2022",
        "comment": "icludes Covid-19 cohort",
    },
    {
        "dataset_name": "SICdb",
        "dataset": "sic",
        "country": "Austria",
        "years": "2013--2021",
        "comment": "",
    },
    {
        "dataset_name": "NWICU",
        "dataset": "nwicu",
        "country": "USA",
        "years": "2020--2022",
        "comment": "multi-center",
    },
    {
        "dataset_name": "PICdb",
        "dataset": "picdb",
        "country": "China",
        "years": "2010--2018",
        "comment": "pediatric hospital",
    },
    {
        "dataset_name": "Zigong",
        "dataset": "zigong",
        "country": "China",
        "years": "2019--2020",
        "comment": "infection cohort",
    }
]

for idx in range(len(table)):
    _, y, other = load(
        [table[idx]["dataset"]],
        outcome="patient_id",
        split=None,
        data_dir="/cluster/work/math/lmalte/data",
        variables=[],
        other_columns=["stay_id", "patient_id"],
    )
    table[idx]["num_patients"] = f"{other['patient_id'].n_unique():,}"
    table[idx]["num_stays"] = f"{other['stay_id'].n_unique():,}"
    table[idx]["average_los"] = f"{len(y) / other['stay_id'].n_unique() / 24:.1f}"

    for outcome in ["circulatory_failure_at_8h", "kidney_failure_at_48h"]:
        _, y, other = load(
            [table[idx]["dataset"]],
            outcome=outcome,
            split=None,
            data_dir="/cluster/work/math/lmalte/data",
            variables=[],
            other_columns=["patient_id", "stay_id"],
        )
        table[idx][f"{outcome}/num_patients"] = f"{other['patient_id'].n_unique():,}"
        table[idx][f"{outcome}/num_stays"] = f"{other['stay_id'].n_unique():,}"
        table[idx][f"{outcome}/num_samples"] = f"{len(y):,}"
        table[idx][f"{outcome}/prevalence"] = f"{100 * y.mean():.1f}%"

    for outcome in [ "log_lactate_in_4h", "log_creatinine_in_24h"]:

        _, y, other = load(
            [table[idx]["dataset"]],
            outcome=outcome,
            split=None,
            data_dir="/cluster/work/math/lmalte/data",
            variables=[],
            other_columns=["patient_id", "stay_id"],
        )
        table[idx][f"{outcome}/num_patients"] = f"{other['patient_id'].n_unique():,}"
        table[idx][f"{outcome}/num_stays"] = f"{other['stay_id'].n_unique():,}"
        table[idx][f"{outcome}/num_samples"] = f"{len(y):,}"
        table[idx][f"{outcome}/mean (sd)"] = f"{y.mean():.2f} ({y.std():.2f})"

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(100)
print(pl.DataFrame(table).transpose(include_header=True))
breakpoint()
# for dataset in DATASETS:
#     sta = pl.scan_parquet(f"/cluster/work/math/lmalte/data/{dataset}/sta.parquet")
#     print(dataset)
#     print(sta.select([pl.col("death_icu").value_counts()]).collect())


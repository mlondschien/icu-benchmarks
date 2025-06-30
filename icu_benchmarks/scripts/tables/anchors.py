import polars as pl
from icu_features.load import load

from icu_benchmarks.constants import DATASETS

table = []

for dataset in DATASETS:
    _, _, other = load(
        [dataset],
        "stay_id",
        split="train_val",
        data_dir="/cluster/work/math/lmalte/data",
        variables=[],
        other_columns=[
            "patient_id",
            "dataset",
            "icd10_blocks",
            "ward",
            "adm",
            "insurance",
            "year",
        ],
    )
    table.append(
        {
            "dataset": dataset,
            "icd10_blocks": other.select(
                pl.col("icd10_blocks").list.explode().unique()
            ).shape[0],
            "years": other["year"].unique().shape[0],
            "adm": other["adm"].unique().shape[0],
            "wards": other["ward"].unique().shape[0],
            "insurance": other["insurance"].unique().shape[0],
            "patients": other["patient_id"].unique().shape[0],
        }
    )

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(100)
print(pl.DataFrame(table))

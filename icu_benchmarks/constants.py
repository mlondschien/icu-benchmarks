from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / "data"
VARIABLE_REFERENCE_PATH = (
    Path(__file__).parents[1] / "resources" / "variable_reference.tsv"
)


DATASETS = [
    "mimic",
    "ehrshot",
    "miived",
    "miiv",
    "eicu",
    "hirid",
    "aumc",
    "sic",
    "zigong",
    "picdb",
]


OUTCOMES = [
    "remaining_los",
    "mortality_at_24h",
    "los_at_24h",
    "decompensation_at_24h",
    "respiratory_failure_at_24h",
    "circulatory_failure_at_8h",
    "kidney_failure_at_48h",
]

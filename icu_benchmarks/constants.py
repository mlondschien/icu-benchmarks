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

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

TASKS = {
    "mortality_at_24h": {
        "task": "classification",
        "family": "binomial",
        # alpha such that fit on train split with l1_ratio=1 is zero. This differs a
        # bit by weighting scheme.
        "alpha_max": 0.07,
        "n_train": 194251,  # Number of observations in the training split
    },
    "decompensation_at_24h": {
        "task": "classification",
        "family": "binomial",
        "alpha_max": 0.025,
        "n_train": 17486953,
    },
    "respiratory_failure_at_24h": {
        "task": "classification",
        "family": "binomial",
        "alpha_max": 0.25,
        "n_train": 3959393
    },
    "circulatory_failure_at_8h": {
        "task": "classification",
        "family": "binomial",
        "alpha_max": 0.25,
        "n_train": 2036495
    },
    "kidney_failure_at_48h": {
        "task": "classification",
        "family": "binomial",
        "alpha_max": 0.15,
        "n_train": 9991660
    },
    "remaining_los": {
        "task": "regression",
        "family": "gamma",
        "n_train": 19495398
    },
    "los_at_24h": {
        "task": "regression",
        "family": "gamma",
    },
}



SIZES = {
    "mimic": {
        "remaining_los": [5179853, 33.3],
        "mortality_at_24h": [48560, 0.3],
        "los_at_24h": [48560, 0.3],
        "decompensation_at_24h": [4020520, 25.8],
        "respiratory_failure_at_24h": [997507, 6.4],
        "circulatory_failure_at_8h": [309693, 2.0],
        "kidney_failure_at_48h": [2948173, 18.9],
    },
    "ehrshot": {
        "remaining_los": [0, 0.0],
        "mortality_at_24h": [32740, 0.2],
        "los_at_24h": [0, 0.0],
        "decompensation_at_24h": [2902373, 18.6],
        "respiratory_failure_at_24h": [70040, 0.4],
        "circulatory_failure_at_8h": [11453, 0.1],
        "kidney_failure_at_48h": [60993, 0.4],
    },
    "miived": {
        "remaining_los": [1550213, 10.0],
        "mortality_at_24h": [11733, 0.1],
        "los_at_24h": [11733, 0.1],
        "decompensation_at_24h": [1550213, 10.0],
        "respiratory_failure_at_24h": [0, 0.0],
        "circulatory_failure_at_8h": [0, 0.0],
        "kidney_failure_at_48h": [0, 0.0],
    },
    "miiv": {
        "remaining_los": [2163733, 13.9],
        "mortality_at_24h": [22747, 0.1],
        "los_at_24h": [22747, 0.1],
        "decompensation_at_24h": [1599860, 10.3],
        "respiratory_failure_at_24h": [367053, 2.4],
        "circulatory_failure_at_8h": [217247, 1.4],
        "kidney_failure_at_48h": [1628933, 10.5],
    },
    "miived": {
        "remaining_los": [1550213,10.0],
        "mortality_at_24h": [11733,0.1],
        "los_at_24h": [11733,0.1],
        "decompensation_at_24h": [1550213,10.0],
        "respiratory_failure_at_24h": [0, 0.0],
        "circulatory_failure_at_8h": [0, 0.0],
        "kidney_failure_at_48h": [0, 0.0],
    },
    "eicu": {
        "remaining_los": [11623907, 74.7],
        "mortality_at_24h": [132760, 0.9],
        "los_at_24h": [132760, 0.9],
        "decompensation_at_24h": [11623907, 74.7],
        "respiratory_failure_at_24h": [2105320, 13.5],
        "circulatory_failure_at_8h": [331540, 2.1],
        "kidney_failure_at_48h": [5857953, 37.6],
    },
    "hirid": {
        "remaining_los": [1716120, 11.0],
        "mortality_at_24h": [16900, 0.1],
        "los_at_24h": [16900, 0.1],
        "decompensation_at_24h": [1075833, 6.9],
        "respiratory_failure_at_24h": [599713, 3.9],
        "circulatory_failure_at_8h": [646907, 4.2],
        "kidney_failure_at_48h": [980813, 6.3],
    },
    "aumc": {
        "remaining_los": [1789927, 11.5],
        "mortality_at_24h": [13133, 0.1],
        "los_at_24h": [13133, 0.1],
        "decompensation_at_24h": [1789927, 11.5],
        "respiratory_failure_at_24h": [854267, 5.5],
        "circulatory_failure_at_8h": [275340, 1.8],
        "kidney_failure_at_48h": [1475073, 9.5],
    },
    "sic": {
        "remaining_los": [1716120, 11.0],
        "mortality_at_24h": [19100, 0.1],
        "los_at_24h": [19100, 0.1],
        "decompensation_at_24h": [1254313, 8.1],
        "respiratory_failure_at_24h": [421500, 2.7],
        "circulatory_failure_at_8h": [1072500, 6.9],
        "kidney_failure_at_48h": [1277720, 8.2],
    },
    "zigong": {
        "remaining_los": [367333, 2.4],
        "mortality_at_24h": [2380, 0.0],
        "los_at_24h": [2380, 0.0],
        "decompensation_at_24h": [362853, 2.3],
        "respiratory_failure_at_24h": [0, 0.0],
        "circulatory_failure_at_8h": [2633, 0.0],
        "kidney_failure_at_48h": [0, 0.0],
    },
    "picdb": {
        "remaining_los": [1614980, 10.4],
        "mortality_at_24h": [9200, 0.1],
        "los_at_24h": [9200, 0.1],
        "decompensation_at_24h": [1614980, 10.4],
        "respiratory_failure_at_24h": [216947, 1.4],
        "circulatory_failure_at_8h": [9727, 0.1],
        "kidney_failure_at_48h": [40920, 0.3],
    },
}


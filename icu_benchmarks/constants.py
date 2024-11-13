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
    "log_creatine_in_1h",
    "log_lactate_in_1h",
]

# Approx. number of rows per GB of memory. If all ~1000 columns were float64, this would
# be ~125_000. We get a bit less due to boolean and categorical columns.
OBSERVATIONS_PER_GB = 160_000

TASKS = {
    "remaining_los": {
        "task": "regression",
        # outcome distribution
        "family": "gamma",
        # alpha such that fit on train split with l1_ratio=1 is zero. This differs a
        # bit by weighting scheme.
        "alpha_max": 0.6,
        # Total number of samples in all splits
        "n_samples": {
            "mimic": 5433113,
            "ehrshot": 0,
            "miived": 2113875,
            "miiv": 2329887,
            "eicu": 12119416,
            "hirid": 1781843,
            "aumc": 1832148,
            "sic": 1881990,
            "zigong": 392612,
            "picdb": 1651313,
        },
    },
    "mortality_at_24h": {
        "task": "classification",
        "family": "binomial",
        "alpha_max": 0.07,
        "n_samples": {
            "mimic": 48868,
            "ehrshot": 32890,
            "miived": 11860,
            "miiv": 23675,
            "eicu": 132649,
            "hirid": 16611,
            "aumc": 12762,
            "sic": 19543,
            "zigong": 2460,
            "picdb": 9225,
        },
    },
    "los_at_24h": {
        "task": "regression",
        "family": "gamma",
        "alpha_max": 0.5,
        "n_samples": {
            "mimic": 48868,
            "ehrshot": 0,
            "miived": 11860,
            "miiv": 23675,
            "eicu": 132649,
            "hirid": 16611,
            "aumc": 12762,
            "sic": 19543,
            "zigong": 2460,
            "picdb": 9225,
        },
    },
    "decompensation_at_24h": {
        "task": "classification",
        "family": "binomial",
        "alpha_max": 0.025,
        "n_samples": {
            "mimic": 4227266,
            "ehrshot": 3049961,
            "miived": 2113875,
            "miiv": 1729782,
            "eicu": 12119416,
            "hirid": 1105836,
            "aumc": 1832148,
            "sic": 1386615,
            "zigong": 387322,
            "picdb": 1651313,
        },
    },
    "respiratory_failure_at_24h": {
        "task": "classification",
        "family": "binomial",
        "alpha_max": 0.25,
        "n_samples": {
            "mimic": 1099866,
            "ehrshot": 72164,
            "miived": 0,
            "miiv": 396882,
            "eicu": 2199085,
            "hirid": 646056,
            "aumc": 884307,
            "sic": 489607,
            "zigong": 0,
            "picdb": 230158,
        },
    },
    "circulatory_failure_at_8h": {
        "task": "classification",
        "family": "binomial",
        "alpha_max": 0.25,
        "n_samples": {
            "mimic": 334339,
            "ehrshot": 14266,
            "miived": 0,
            "miiv": 244304,
            "eicu": 369894,
            "hirid": 685413,
            "aumc": 281161,
            "sic": 1176290,
            "zigong": 2762,
            "picdb": 12573,
        },
    },
    "kidney_failure_at_48h": {
        "task": "classification",
        "family": "binomial",
        "alpha_max": 0.15,
        "n_samples": {
            "mimic": 3111492,
            "ehrshot": 62150,
            "miived": 0,
            "miiv": 1759023,
            "eicu": 6007453,
            "hirid": 996732,
            "aumc": 1516864,
            "sic": 1399874,
            "zigong": 0,
            "picdb": 44773,
        },
    },
    "log_creatine_in_1h": {
        "task": "regression",
        "family": "gaussian",
        "n_samples": {
            "mimic": 280609,
            "ehrshot": 70821,
            "miived": 0,
            "miiv": 159329,
            "eicu": 605427,
            "hirid": 69649,
            "aumc": 109604,
            "sic": 92324,
            "zigong": 10091,
            "picdb": 23961,
        },
        "alpha_max": 0.53,
    },
    "log_lactate_in_1h": {
        "task": "regression",
        "family": "gaussian",
        "n_samples": {
            "mimic": 111451,
            "ehrshot": 10809,
            "miived": 0,
            "miiv": 93007,
            "eicu": 116289,
            "hirid": 205320,
            "aumc": 114515,
            "sic": 491539,
            "zigong": 11127,
            "picdb": 117142,
        },
        "alpha_max": 0.54,
    },
}

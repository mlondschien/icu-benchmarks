from pathlib import Path
from typing import Any, Dict

DATA_DIR = Path(__file__).parents[1] / "data"


SOURCE_COLORS = {
    "eicu": "#000000",
    "mimic": "#EE6677",
    "mimic-carevue": "#EE6677",
    "hirid": "#66CCEE",
    "miiv": "#AA3377",
    "aumc": "#4477AA",
    "sic": "#332288",
    "zigong": "#228833",
    "picdb": "#CCBB44",
    "nwicu": "#BBBBBB",
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
    "quantile_0.1": "10\\%-quantile of residuals",
    "quantile_0.25": "25\\%-quantile of residuals",
    "quantile_0.5": "median of residuals",
    "quantile_0.75": "75\\%-quantile of residuals",
    "quantile_0.9": "90\\%-quantile of residuals",
    "mean_residual": "mean of residuals",
}

DATASET_NAMES = {
    "sic": "SICdb",
    "aumc": "AmsterdamUMCdb",
    "eicu": "eICU",
    "miiv": "MIMIC-IV",
    "mimic-carevue": "MIMIC-III (CareVue subset)",
    "hirid": "HiRID",
}

SHORT_DATASET_NAMES = {
    "sic": "SICdb",
    "aumc": "AUMCdb",
    "eicu": "eICU",
    "miiv": "MIMIC-IV",
    "mimic-carevue": "MIMIC-III (CV)",
    "hirid": "HiRID",
    "nwicu": "NWICU",
    "zigong": "Zigong EHR",
    "picdb": "PICdb",
}

VERY_SHORT_DATASET_NAMES = {
    "sic": "SICdb",
    "aumc": "AUMCdb",
    "eicu": "eICU",
    "miiv": "MIMIC-IV",
    "mimic-carevue": "MIMIC-III",
    "hirid": "HiRID",
    "nwicu": "NWICU",
    "zigong": "Zigong",
    "picdb": "PICdb",
}

OUTCOME_NAMES = {
    "log_creatinine_in_24h": "log(creatinine) in 24h",
    "log_lactate_in_4h": "log(lactate) in 4h",
    "circulatory_failure_at_8h": "circ. failure within 8h",
    "kidney_failure_at_48h": "kidney failure within 48h",
}

VARIABLE_REFERENCE_PATH = (
    Path(__file__).parents[1] / "resources" / "variable_reference.tsv"
)

HORIZONS = [8, 24, 72]

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

CAT_MISSING_NAME = "(MISSING)"

ANCHORS = [
    "dataset",
    "hospital_id",
    "ward",
    "year",
    "adm",
    "insurance",
    "icd10_blocks",
    "icd10_ccsr",
    "apache_group",
    "patient_id",
]

OUTCOMES = [
    "remaining_los",
    "mortality_at_24h",
    "los_at_24h",
    "decompensation_at_24h",
    "respiratory_failure_at_24h",
    "circulatory_failure_at_8h",
    "kidney_failure_at_48h",
    "log_pf_ratio_in_12h",
    "log_rel_urine_rate_in_2h",
    "log_lactate_in_4h",
]

# Top variables according to fig 8a of Lyu et al 2024: An empirical study on
# KDIGO-defined acute kidney injury prediction in the intensive care unit.
KIDNEY_VARIABLES = [
    "time_hours",  # Time in hours since ICU admission
    "ufilt",  # Ultrafiltration on cont. RRT
    "ufilt_ind",  # Ultrafiltration on cont. RRT
    "rel_urine_rate",  # Urine rate per weight (ml/kg/h)
    "weight",
    "crea",  # Creatinine
    "etco2",  # End-tidal CO2
    "crp",  # C-reactive protein
    "anti_coag_ind",  # Indicator for antocoagulants treatment
    "hep",  # Heparin
    "hep_ind",  # Heparin
    "loop_diur",  # Loop diuretics
    "loop_diur_ind",  # Loop diuretics
    "resp",  # Respiratory rate
    "fluid_ind",  # Fluids
    "airway",  # Ventilation type
    "vent_ind",  # Indicator for any ventilation
    "bili",  # Bilirubin
    "anti_delir_ind",  # Indicator for antidelirium treatment
    "mg",  # Magnesium
    "op_pain_ind",  # Opioid pain medication
    "abx_ind",  # Antibiotics indicator
    "k",  # Potassium
]

# Top 20 variables of Hyland et al.: Early prediction of circulatory failure in the
# intensive care unit using machine learning. Table 1.
CIRC_VARIABLES = [
    "age",
    "cf_treat_ind",  # circ. failure treatments incl. dobu, norepi, milrin, theo, levo
    "cout",  # Cardiac output
    "crp",  # C-reactive protein
    "dbp",  # Diastolic blood pressure
    "dobu_ind",  # Dobutamine
    "dobu",  # Dobutamine
    "glu",  # Serum glucose
    "hr",  # Heart rate
    "inr_pt",  # Prothrombin
    "lact",  # Lactate
    "levo_ind",  # Levosimendan
    "levo",  # Levosimendan
    "map",  # mean arterial pressure
    "milrin_ind",  # Milrinone
    "milrin",  # Milrinone
    "nonop_pain_ind",  # Non-opioid pain medication
    "peak",  # Peak airway pressure
    "rass",  # Richmond Agitation Sedation Scale
    "sbp",  # Systolic blood pressure
    "spo2",  # Oxygen saturation (finger) SpO2
    "supp_o2_vent",  # Oxygen supplementation
    "teophyllin_ind",  # Theophylline
    "teophyllin",  # Theophylline
    "time_hours",  # Time in hours since ICU admission
]

TASKS: Dict[str, Dict[str, Any]] = {
    "circulatory_failure_at_8h": {
        "task": "binary",
        "family": "binomial",
        "alpha_max": 0.09,
        "n_samples": {
            "mimic": 270080,
            "miiv": 565139,
            "eicu": 287585,
            "hirid": 606160,
            "aumc": 217223,
            "sic": 1105034,
            "zigong": 1963,
            "picdb": 0,
            "mimic-metavision": 123609,
            "mimic-carevue": 144977,
            "miiv-late": 303327,
            "aumc-early": 40541,
            "aumc-late": 176682,
            "nwicu": 43563,
        },
        "variables": CIRC_VARIABLES,
        "size": 6308,
        "horizons": [8],
    },
    "circulatory_failure_at_8h_imputed": {
        "task": "binary",
        "family": "binomial",
        "alpha_max": 0.09,
        "n_samples": {
            "mimic": 270080,
            "miiv": 565139,
            "eicu": 287585,
            "hirid": 606160,
            "aumc": 217223,
            "sic": 1105034,
            "zigong": 1963,
            "picdb": 0,
            "mimic-metavision": 123609,
            "mimic-carevue": 144977,
            "miiv-late": 303327,
            "aumc-early": 40541,
            "aumc-late": 176682,
        },
        "variables": CIRC_VARIABLES,
        "size": 6308,
        "horizons": [8],
    },
    "kidney_failure_at_48h": {
        "task": "binary",
        "family": "binomial",
        "alpha_max": 0.1,
        "n_samples": {
            "mimic": 2961026,
            "miiv": 5279681,
            "eicu": 5668932,
            "hirid": 980458,
            "aumc": 1487426,
            "sic": 1367201,
            "zigong": 0,
            "picdb": 0,
            "mimic-metavision": 1313601,
            "mimic-carevue": 1632587,
            "miiv-late": 2694756,
            "aumc-early": 701300,
            "aumc-late": 786126,
            "nwicu": 29505,
        },
        "variables": KIDNEY_VARIABLES,
        "horizons": [24],
        "size": 9763,
    },
    "log_creatinine_in_24h": {
        "task": "regression",
        "family": "gaussian",
        "alpha_max": 0.6,
        "variables": KIDNEY_VARIABLES,
        "horizons": [24],
    },
    "log_lactate_in_4h": {
        "task": "regression",
        "family": "gaussian",
        "n_samples": {
            "mimic": 111135,
            "ehrshot": 10809,
            "miived": 0,
            "miiv": 270623,
            "eicu": 116096,
            "hirid": 205320,
            "aumc": 114515,
            "sic": 489852,
            "zigong": 10976,
            "picdb": 0,
            "mimic-metavision": 52425,
            "mimic-carevue": 57445,
            "miiv-late": 149767,
            "aumc-early": 21505,
            "aumc-late": 93010,
            "nwicu": 47710,
        },
        "alpha_max": 1.8,
        "variables": CIRC_VARIABLES,
        "horizons": [8],
        "size": 951,
    },
}

GREATER_IS_BETTER = ["roc", "auroc", "auc", "auprc", "accuracy", "prc", "r2"]

METRICS = [
    "mse",
    "rmse",
    "mae",
    "abs_quantile_0.8",
    "abs_quantile_0.9",
    "abs_quantile_0.95",
    "roc",
    "accuracy",
    "log_loss",
    "auprc",
    "brier",
    "grouped_mse_quantile_0.5",
    "grouped_mse_quantile_0.6",
    "grouped_mse_quantile_0.7",
    "grouped_mse_quantile_0.8",
    "grouped_mse_quantile_0.9",
    "log_losses_0.8",
    "log_losses_0.9",
    "log_losses_0.95",
    "log_losses_balanced_0.8",
    "log_losses_balanced_0.9",
    "log_losses_balanced_0.95",
]


PARAMETERS = [
    "alpha",
    "ratio",
    "l1_ratio",
    "gamma",
    "num_iteration",
    "learning_rate",
    "max_depth",
]

VERY_SHORT_DATASET_NAMES = {
    "sic": "SICdb",
    "aumc": "AUMCdb",
    "eicu": "eICU",
    "miiv": "MIMIC-IV",
    "mimic-carevue": "MIMIC-III",
    "hirid": "HiRID",
    "nwicu": "NWICU",
    "zigong": "Zigong",
    "picdb": "PICdb",
}

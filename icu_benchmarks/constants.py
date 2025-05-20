from pathlib import Path
from typing import Any, Dict

DATA_DIR = Path(__file__).parents[1] / "data"

VARIABLE_REFERENCE_PATH = (
    Path(__file__).parents[1] / "resources" / "variable_reference.tsv"
)

HORIZONS = [8, 24, 72]

DATASETS = [
    # "mimic",
    # "mimic-metavision",
    "aumc",
    "eicu",
    "hirid",
    "mimic-carevue",
    "miiv",
    "sic",
    # "miiv-late",
    # "aumc-early",
    # "aumc-late",
    "nwicu",
    # "miived"
    "picdb",
    "zigong",
]

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
    "miived": "MIMIC-IV ED",
}

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

# "preliminary selected variables" according to
# https://www.medrxiv.org/content/10.1101/2024.01.23.24301516v1 supp table 3
RESP_VARIABLES = [
    "fio2",
    "norepi",  # Norepinephrine
    "norepi_ind",  # Norepinephrine
    "dobu",  # Dobutamine
    "dobu_ind",  # Dobutamine
    "loop_diur",  # Loop diuretics
    "loop_diur_ind",  # Loop diuretics
    "benzdia",  # Benzodiazepines
    "benzdia_ind",  # Benzodiazepines
    "prop",  # Propofol
    "prop_ind",  # Propofol
    "ins_ind",  # Insulin
    "hep",  # Heparin
    "hep_ind",  # Heparin
    "cf_treat_ind",  # circulatory failure treatments incl. dobu, norepi.
    "sed_ind",  # sedation medication indicator incl. benzdia, prop.
    "age",
    # no emergency admission
    "vent_ind",  # Indicator for any ventilation
    "airway",  # Ventilation type
    "pco2",  # Partial pressure of carbon dioxide PaCO2
    "po2",  # Partial pressure of oxygen PaO2
    "sao2",  # Oxygen saturation (lab value) SaO2
    "spo2",  # Oxygen saturation (finger) SpO2
    "ps",  # Pressure support
    # No MV exp / MV spont. These are available in HiRID only
    "resp",  # Respiratory rate
    "supp_o2_vent",  # Oxygen supplementation
    "tgcs",  # Total Glasgow Coma Scale (Response)
    "mgcs",  # Motor Glasgow Coma Scale
    "peep",  # Positive end-expiratory pressure
    "map",  # Mean arterial pressure. ABPm is window-mean of map
    "peak",  # Peak airway pressure
    "ph",  # Used to determine po2 from sao2 according to the serveringhaus equation
    "temp",  # Temperature, used to determine po2 from sao2 according to serveringhaus
    "pf_ratio",  # ratio of po2 to fio2
]

# Top 20 variables of Hyland et al.: Early prediction of circulatory failure in the
# intensive care unit using machine learning. Table 1.
CIRC_VARIABLES = [
    "lact",  # Lactate
    "map",  # mean arterial pressure
    "time_hours",  # Time in hours since ICU admission
    "age",
    "hr",  # Heart rate
    "dobu",  # Dobutamine
    "dobu_ind",  # Dobutamine
    "milrin",  # Milrinone
    "milrin_ind",  # Milrinone
    "levo",  # Levosimendan
    "levo_ind",  # Levosimendan
    "teophyllin",  # Theophylline
    "teophyllin_ind",  # Theophylline
    "cf_treat_ind",  # circ. failure treatments incl. dobu, norepi, milrin, theo, levo
    "cout",  # Cardiac output
    "rass",  # Richmond Agitation Sedation Scale
    "inr_pt",  # Prothrombin
    "glu",  # Serum glucose
    "crp",  # C-reactive protein
    "dbp",  # Diastolic blood pressure
    "sbp",  # Systolic blood pressure
    "peak",  # Peak airway pressure
    "spo2",  # Oxygen saturation (finger) SpO2
    "nonop_pain_ind",  # Non-opioid pain medication
    "supp_o2_vent",  # Oxygen supplementation
]

GLU_VARIABLES = [
    "glu",
    "ins_ind",
    "log_time_hours",
    "age",
    "weight",
    "k",
]

MELD_VARIABLES = [  # from Manuel
    "crea",
    "alb",
    "alp",
    "alt",
    "ast",
    "bili",
    "bili_dir",
    "inr_pt",
    "plt",
    "hct",
    "ygt",
    "amm",
    "amyl",
    "lip",
    "fgn",
    "hep",
    "hep_ind",
    "op_pain_ind",
    "nonop_pain_ind",
    "plat_ind",
    "inf_alb_ind",
    "anti_coag_ind",
    "age",
    "height",
    "weight",
    "sex",
]


# Variables used to determine apache II
APACHE_II_VARIABLES = [
    "age",
    "crea",
    "fio2",
    "hct",
    "hr",
    "k",
    "na",
    "pco2",
    "po2",
    "resp",
    "temp",
    "tgcs",
    "wbc",
    # "mgcs",
    # "vgcs",
    # "egcs",
    # "tgcs",
    # "sofa",
    # "sofa4",
    # "urine_rate",
    # "spo2",
    # "glu",
    "map",
    # "lact",
]

SOFA_VARIABLES = [
    "po2",  # resp
    "fio2",  # resp
    "vent_ind",  # resp
    "crea",  # renal
    "urine_rate",  # renal
    "bili",  # liver
    "plt",  # coagulation
    "egcs",
    "mgcs",
    "vgcs",
    "tgcs",
    # "ett_gcs"  # neurological
    "map",  # cardiovascular
    "norepi",  # cardiovascular
    "dobu",  # cardiovascular
    "epi",  # cardiovascular
    "dopa",  # cardiovascular
    "weight",
]

MORT_VARIABLES = sorted(set(APACHE_II_VARIABLES + SOFA_VARIABLES + ["sex"]))


# Approx. number of rows per GB of memory. If all ~1000 columns were float64, this would
# be ~125_000. We get a bit less due to boolean and categorical columns.
OBSERVATION_PER_GB = 160_000

TASKS: Dict[str, Dict[str, Any]] = {
    "log_po2": {
        "task": "regression",
        "family": "gaussian",
        "alpha_max": 0.16,
        "n_samples": {
            "mimic": 338189,
            "mimic-metavision": 109443,
            "mimic-carevue": 225484,
            "miived": 0,
            "miiv": 415590,
            "miiv-late": 211551,
            "eicu": 277943,
            "hirid": 203075,
            "aumc": 439735,
            "aumc-early": 208448,
            "aumc-late": 231287,
            "sic": 482286,
            "zigong": 13336,
            "picdb": 0,
            "nwicu": 0,
        },
        "variables": [x for x in RESP_VARIABLES if x not in ["po2", "pf_ratio"]],
        "horizons": [8, 24],
        "size": 3578,
    },
    "remaining_los": {
        "task": "regression",
        "family": "gamma",
        "alpha_max": 0.6,
        "n_samples": {
            "mimic": 4586276,
            "ehrshot": 0,
            "miived": 0,
            "miiv": 7382185,
            "eicu": 12096141,
            "hirid": 1781542,
            "aumc": 1832031,
            "sic": 1858134,
            "zigong": 387493,
            "picdb": 0,
            "mimic-metavision": 1846108,
            "mimic-carevue": 2689494,
            "miiv-late": 3736973,
            "aumc-early": 863355,
            "aumc-late": 968676,
            "nwicu": 2179335,
        },
    },
    "mortality_at_24h": {
        "task": "binary",
        "family": "binomial",
        "alpha_max": 0.05,
        "n_samples": {
            "mimic": 45018,
            "ehrshot": 32890,
            "miived": 0,
            "miiv": 74814,
            "eicu": 132388,
            "hirid": 16611,
            "aumc": 12762,
            "sic": 19486,
            "zigong": 2422,
            "picdb": 0,
            "mimic-metavision": 19544,
            "mimic-carevue": 25007,
            "miiv-late": 36402,
            "aumc-early": 5544,
            "aumc-late": 7218,
            "nwicu": 21566,
        },
        "variables": APACHE_II_VARIABLES,
        "horizons": [24],
        "size": 215,
    },
    "los_at_24h": {
        "task": "regression",
        "family": "gamma",
        "alpha_max": 0.5,
        "n_samples": {
            "mimic": 45018,
            "ehrshot": 0,
            "miived": 0,
            "miiv": 74814,
            "eicu": 132388,
            "hirid": 16611,
            "aumc": 12762,
            "sic": 19486,
            "zigong": 2422,
            "picdb": 0,
            "mimic-metavision": 19544,
            "mimic-carevue": 25007,
            "miiv-late": 36402,
            "aumc-early": 5544,
            "aumc-late": 7218,
            "nwicu": 21566,
        },
    },
    "decompensation_at_24h": {
        "task": "binary",
        "family": "binomial",
        "alpha_max": 0.025,
        "n_samples": {
            "mimic": 3484186,
            "ehrshot": 3049961,
            "miived": 0,
            "miiv": 5460238,
            "eicu": 12097602,
            "hirid": 1105836,
            "aumc": 1832148,
            "sic": 1381888,
            "zigong": 382308,
            "picdb": 0,
            "mimic-metavision": 1355081,
            "mimic-carevue": 2089759,
            "miiv-late": 2822499,
            "aumc-early": 863397,
            "aumc-late": 968751,
            "nwicu": 2179674,
        },
    },
    "respiratory_failure_at_24h": {
        "task": "binary",
        "family": "binomial",
        "alpha_max": 0.1,
        "n_samples": {
            "mimic": 795094,
            "ehrshot": 72164,
            "miived": 0,
            "miiv": 1013481,
            "eicu": 1584555,
            "hirid": 443957,
            "aumc": 646446,
            "sic": 353134,
            "zigong": 0,
            "picdb": 0,
            "mimic-metavision": 319568,
            "mimic-carevue": 468680,
            "miiv-late": 465963,
            "aumc-early": 373011,
            "aumc-late": 273435,
            "nwicu": 0,
        },
        "variables": RESP_VARIABLES,
        "horizons": [8, 24],
        "size": 12670,
    },
    "severe_respiratory_failure_at_24h": {
        "task": "binary",
        "family": "binomial",
        "alpha_max": 0.1,
        "n_samples": {
            "mimic": 795094,
            "ehrshot": 72164,
            "miived": 0,
            "miiv": 1013481,
            "eicu": 1584555,
            "hirid": 443957,
            "aumc": 646446,
            "sic": 353134,
            "zigong": 0,
            "picdb": 0,
            "mimic-metavision": 319568,
            "mimic-carevue": 468680,
            "miiv-late": 465963,
            "aumc-early": 373011,
            "aumc-late": 273435,
            "nwicu": 0,
        },
        "variables": RESP_VARIABLES,
        "horizons": [8, 24],
        "size": 12670,
    },
    "circulatory_failure_at_8h": {
        "task": "binary",
        "family": "binomial",
        "alpha_max": 0.09,
        "n_samples": {
            "mimic": 270080,
            "ehrshot": 14266,
            "miived": 0,
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
            "ehrshot": 14266,
            "miived": 0,
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
    "kidney_failure_at_48h": {
        "task": "binary",
        "family": "binomial",
        "alpha_max": 0.1,
        "n_samples": {
            "mimic": 2961026,
            "ehrshot": 62150,
            "miived": 0,
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
    "log_rel_urine_rate_in_2h": {
        "task": "regression",
        "family": "gaussian",
        "n_samples": {
            "mimic": 278792,
            "ehrshot": 70821,
            "miived": 0,
            "miiv": 500728,
            "eicu": 604236,
            "hirid": 69649,
            "aumc": 109604,
            "sic": 92011,
            "zigong": 9951,
            "picdb": 0,
            "mimic-metavision": 118577,
            "mimic-carevue": 157241,
            "miiv-late": 256965,
            "aumc-early": 53878,
            "aumc-late": 55726,
            "nwicu": 146530,
        },
        "alpha_max": 0.53,
        "variables": KIDNEY_VARIABLES,
        "horizons": [8],
        "size": 6399,
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
    "log_pf_ratio_in_12h": {
        "task": "regression",
        "family": "gaussian",
        "n_samples": {},
        "alpha_max": 0.3,
        "variables": RESP_VARIABLES,
        "horizons": [8, 24],
        "size": 1469,
    },
    "severe_meld_at_48h": {
        "task": "binary",
        "family": "binomial",
        "n_samples": {},
        "alpha_max": 0.05,
        "variables": MELD_VARIABLES,
        "horizons": [24],
    },
    "severe_meld_at_48h_2": {
        "task": "binary",
        "family": "binomial",
        "n_samples": {},
        "alpha_max": 0.26,
        "variables": MELD_VARIABLES,
        "horizons": [24],
    },
    "meld_score_in_24h": {
        "task": "regression",
        "family": "gaussian",
        "n_samples": {},
        "alpha_max": 13.3,
        "variables": MELD_VARIABLES,
        "horizons": [24],
    },
    "log_bili_in_24h": {
        "task": "regression",
        "family": "gaussian",
        "n_samples": {},
        "alpha_max": 0.16,
        "variables": MELD_VARIABLES,
        "horizons": [24],
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
    "num_leaves",
    "l1_ratio",
    "gamma",
    "colsample_bytree",
    "bagging_fraction",
    "min_data_in_leaf",
    # "num_boost_round",
    "num_iteration",
    "l2_ratio",
    "learning_rate",
    # "num_leaves",
    "lambda_l2",
    "n_components",
    # "random_state"
]

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
    "miived": "MIMIC-IV ED",
}

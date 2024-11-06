# Script to extract harmonized and gridded data from all the datasets
# using the ricu package.
#
# Minimally adjusted from https://github.com/rvandewater/YAIB-cohorts/
# and https://github.com/prockenschaub/icuDG-preprocessing/
#
# On mac, it is advisable to set the R_MAX_VSIZE environment variable before
# running this. E.g., run `R_MAX_VSIZE=64000000000 Rscript base_cohort.R`
library(argparser)
library(assertthat)
library(rlang)
library(data.table)
library(vctrs)
library(yaml)

source("data_extraction/r/src/misc.R")
source("data_extraction/r/src/steps.R")
source("data_extraction/r/src/sequential.R")
source("data_extraction/r/src/obs_time.R")

# Argparser
p <- arg_parser("Extract patient stays for a given dataset.")

p <- add_argument(p, "--ricu_path", help="path to the ricu data", default="")
p <- add_argument(p, "--src", help="source database", default="mimic_demo")
p <- add_argument(p, "--out_path", help="output path", default="")
p <- add_argument(p, "--var_ref_path", help="path to the variable reference file", default="../resources/variable_reference.tsv")
p <- add_argument(p, "--max_len", help="maximum length of stay in days, default 2 weeks", default=14)

argv <- parse_args(p)

# Configure environment and script
Sys.setenv(RICU_DATA_PATH = argv$ricu_path)
print(glue("Ricu path: {Sys.getenv('RICU_DATA_PATH')}"))
library(ricu)

src <- argv$src
out_path <- paste0(argv$out_path, src)
print(glue("Output path: {out_path}"))

# Load the variable reference tsv
var_ref_df <- read.table(file = argv$var_ref_path, sep = "\t", header = TRUE)

if (!dir.exists(out_path)) {
  dir.create(out_path, recursive = TRUE)
}

cncpt_env <- new.env()

time_unit <- hours
freq <- 1L

# Maximum length of stay, cast to number of hours
max_len_days <- as.numeric(argv$max_len)
max_len_hours <- max_len_days * 24
print(glue("Max length of stay: {max_len_days} days = {max_len_hours} hours"))

var_ref_df <- var_ref_df[var_ref_df$DatasetVersion != "None", ]
print(glue("Variable reference table has {nrow(var_ref_df)} rows"))

# ------------------------------------------------
# Define variables to load
# ------------------------------------------------

# --------- Static variables ---------
var_static_df = var_ref_df[var_ref_df$VariableType == "static", ]
print(glue("Number of static variables: {nrow(var_static_df)}"))
base_static_vars <- c("patient_id", "death_icu", "los_hosp", "los_icu", "ed_disposition", "hospital_id", "anchor_year")
static_vars <- c(base_static_vars, var_static_df$VariableTag)

# --------- Dynamic variables ---------
var_dynamic_df = var_ref_df[var_ref_df$VariableType != "static", ]
dynamic_vars <- c(var_dynamic_df$VariableTag)
print(glue("Number of dynamic variables: {nrow(var_dynamic_df)}"))

# ------------------------------------------------
# Define stay identifer (can vary per dataset)
# ------------------------------------------------
if (src == "mimic" || src == "mimic_demo" || src == "picdb") {
  stay_id = "icustay_id"
} else if (src == "miiv") {
  stay_id = "stay_id"
} else if (src == "eicu" || src == "eicu_demo") {
  stay_id = "patientunitstayid"
}

if (src == "mimic" || src == "miiv" || src == "eicu" || src == "eicu_demo" || src == "mimic_demo" || src == "picdb") {
  patients <- change_interval(change_id(stay_windows(src, id_type="icustay"), stay_id, src = src), time_unit(freq))
  patients[, start := 0]
} else {
  patients <- stay_windows(src, id_type="icustay", interval = time_unit(freq))
}

patients <- as_win_tbl(patients, index_var = "start", dur_var = "end", interval = time_unit(freq))
arrow::write_parquet(patients, paste0(out_path, "/patients.parquet"))

# Define observation times ------------------------------------------------
stop_obs_at(patients, offset = ricu:::re_time(hours(max_len_hours), time_unit(freq)), by_ref = TRUE)

# Exclusion criteria
#  1. Invalid LoS (as in YAIB)
print("[Exclusion] Dropping patients with invalid LoS")
excl1 <- patients[end < 0, id_vars(patients), with = FALSE]

# 2. Stay <4h (YAIB was: 6h)
MINIMUM_STAY_LENGTH_HOURS <- 4
print(paste("[Exclusion] Minimum stay length in hours:", MINIMUM_STAY_LENGTH_HOURS))

x <- load_step("los_icu")
arrow::write_parquet(x, paste0(out_path, "/los_icu.parquet"))
x <- filter_step(x, ~ . < MINIMUM_STAY_LENGTH_HOURS / 24)

excl2 <- unique(x[, id_vars(x), with = FALSE])

# 3. Less than 4 measurements (as in YAIB)
MINIMUM_NUMBER_OF_MEASUREMENTS <- 4
print(paste("[Exclusion] Minimum number of measurements:", MINIMUM_NUMBER_OF_MEASUREMENTS))

n_obs_per_row <- function(x, ...) {
  # TODO: make sure this does not change by reference if a single concept is provided
  obs <- data_vars(x)
  x[, n := as.vector(rowSums(!is.na(.SD))), .SDcols = obs]
  x[, .SD, .SDcols = !c(obs)]
}

x <- load_step(dynamic_vars, interval=time_unit(freq), cache = TRUE)
x <- summary_step(x, "count", drop_index = TRUE)
x <- filter_step(x, ~ . < MINIMUM_NUMBER_OF_MEASUREMENTS)

excl3 <- unique(x[, id_vars(x), with = FALSE])

# 4. More than 48 hour gaps between measurements (YAIB was: 12h)
MAXIMUM_GAP_BETWEEN_MEASUREMENTS_HOURS <- 48
print(paste("[Exclusion] Maximum gap between measurements in hours:", MAXIMUM_GAP_BETWEEN_MEASUREMENTS_HOURS))

map_to_grid <- function(x) {
  grid <- ricu::expand(patients)
  merge(grid, x, all.x = TRUE)
}

longest_rle <- function(x, val) {
  x <- x[, rle(.SD[[data_var(x)]]), by = c(id_vars(x))]
  x <- x[values != val, lengths := 0]
  x[, .(lengths = max(lengths)), , by = c(id_vars(x))]
}

x <- load_step(dynamic_vars, interval=time_unit(freq), cache = TRUE)
x <- function_step(x, map_to_grid)
x <- function_step(x, n_obs_per_row)
x <- mutate_step(x, ~ . > 0)
x <- function_step(x, longest_rle, val = FALSE)
x <- filter_step(x, ~ . > as.numeric(ricu:::re_time(hours(MAXIMUM_GAP_BETWEEN_MEASUREMENTS_HOURS), time_unit(1)) / freq))

excl4 <- unique(x[, id_vars(x), with = FALSE])

# Apply Exclusion Criteria
patients <- exclude(patients, mget(paste0("excl", 1:4)))
attrition <- as.data.table(patients[c("incl_n", "excl_n_total", "excl_n")])
patients <- patients[['incl']]
patient_ids <- patients[, .SD, .SDcols = id_var(patients)]

# Load dynamic variables
print("Loading dynamic variables")
dyn <- load_step(dynamic_vars, interval=time_unit(freq), cache = TRUE)
print("Done.")

# Extract patient set with actual dynamic data. Patients without a single
# dynamic observation could slip through the above functions, but we want to
# exclude them. Should not happen for most datasets with good variable coverage.
dyn_valid_ids <- unique(dyn[[id_var(dyn)]])
nrow_patients_before <- nrow(patients)

patients <- patients[get(id_var(patients)) %in% dyn_valid_ids, ]
patient_ids <- patients[, .SD, .SDcols = id_var(patients)]
nrow_patients_valid <- nrow(patients)
if (nrow_patients_before != nrow_patients_valid) {
  print(glue("WARNING: Removed {nrow_patients_before - nrow_patients_valid} patients without any dynamic data"))
}

# Load static variables
print("Loading static predictors")
sta <- load_step(static_vars, cache = TRUE)
print("Done.")

# Transform all variables into the target format
dyn_fmt <- function_step(dyn, map_to_grid)
rename_cols(dyn_fmt, c("stay_id", "time"), meta_vars(dyn_fmt), by_ref = TRUE)

sta_fmt <- sta[patient_ids]
rename_cols(sta_fmt, c("stay_id"), id_vars(sta), by_ref = TRUE)

# `time` needs to be in seconds for arrow::write_parquet.
dyn_fmt[, time := as.difftime(as.numeric(time, units = "secs"), units = "secs")]

# Write to disk
arrow::write_parquet(dyn_fmt, paste0(out_path, "/dyn.parquet"))
arrow::write_parquet(sta_fmt, paste0(out_path, "/sta.parquet"))
fwrite(attrition, paste0(out_path, "/attrition.csv"))

# Write hospital data for multi-center datasets
if (src == "eicu") {
  arrow::write_parquet(as.data.frame(ricu::eicu$hospital), paste0(out_path, "/hospital.parquet"))
}
if (src == "eicu_demo") {
  arrow::write_parquet(as.data.frame(ricu::eicu_demo$hospital), paste0(out_path, "/hospital.parquet"))
}
print("Done.")

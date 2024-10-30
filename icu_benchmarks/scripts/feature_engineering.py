import logging
from pathlib import Path
from time import perf_counter

import click
import numpy as np
import polars as pl

from icu_benchmarks.constants import DATA_DIR, OUTCOMES, VARIABLE_REFERENCE_PATH
from icu_benchmarks.load import features

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

variable_reference = (
    pl.read_csv(VARIABLE_REFERENCE_PATH, separator="\t", null_values=["None"])
    .filter(pl.col("DatasetVersion").is_not_null())
    .with_columns(pl.col("PossibleValues").str.json_decode())
)


def continuous_features(column_name: str, time_col: str, horizons: list[int] = [8, 24]):
    """
    Compute continuous features for a column.

    These are:
    - ffilled: The column with missing values filled forward.
    - mean: The mean of the column within the last `horizon` hours. This is nan if there
      are only missing values within the last `horizon` hours.
    - std: The standard deviation of the column within the last `horizon` hours. This is
      nan if there are less than 2 non-missing values within the last `horizon` hours.
    - slope: The slope of the column within the last `horizon` hours. Uses hours as the
      x-axis. This is nan if there are less than 2 non-missing values within the last
      `horizon` hours.
    - fraction_nonnull: The fraction of non-missing values within the last `horizon`.
    - all_missing: True if all values within the last `horizon` are missing.

    Parameters
    ----------
    column_name : str
        Name of column for which to compute features. E.g., `hr`.
    time_col : str
        Name of the time column. E.g., `time_hours`.
    horizons : list[int]
        Horizons for which to compute the features. E.g., [8, 24].
    """
    # All features can be computed using cumulative sums. To "ignore" missing values,
    # we fill them with 0.
    col = pl.col(column_name).fill_null(0)
    # time is never null. But we want to ignore entries corresponding to missing values
    # in `column_name`.
    time = pl.when(pl.col(column_name).is_null()).then(0).otherwise(pl.col(time_col))
    nonnulls = pl.col(column_name).is_not_null().cast(pl.Int32).cum_sum()

    time_cs = time.cum_sum()
    time_sq_cs = (time * time).cum_sum()

    col_cs = col.cum_sum()
    col_sq_cs = (col * col).cum_sum()
    timexcol_cs = (time * col).cum_sum()

    expressions = [pl.col(column_name).forward_fill().alias(f"{column_name}_ffilled")]

    for horizon in horizons:
        window_size = (pl.col(time_col) + 1).clip(None, horizon)
        col_sum = col_cs - col_cs.shift(horizon, fill_value=0)
        col_sq_sum = col_sq_cs - col_sq_cs.shift(horizon, fill_value=0)
        time_sum = time_cs - time_cs.shift(horizon, fill_value=0)
        time_sq_sum = time_sq_cs - time_sq_cs.shift(horizon, fill_value=0)
        timexcol_sum = timexcol_cs - timexcol_cs.shift(horizon, fill_value=0)
        nonnulls_sum = nonnulls - nonnulls.shift(horizon, fill_value=0)

        # Use when to avoid division by zero & s.t. column contains Nones, not nans.
        col_mean = pl.when(nonnulls_sum > 0).then(col_sum / nonnulls_sum)

        # std = sum_i (x_i - mean)^2 / n = sum_i x_i^2 / n - mean^2
        col_std = (col_sq_sum / nonnulls_sum - col_mean * col_mean).clip(0, None).sqrt()
        col_std = pl.when(nonnulls_sum >= 2).then(col_std)

        # See https://en.wikipedia.org/wiki/Simple_linear_regression#Expanded_formulas
        numerator = nonnulls_sum * timexcol_sum - time_sum * col_sum
        denom = nonnulls_sum * time_sq_sum - time_sum * time_sum
        slope = pl.when(nonnulls_sum >= 2).then(numerator / denom)

        fraction_nonnull = nonnulls_sum / window_size

        all_missing = nonnulls_sum == 0

        expressions += [
            col_mean.alias(f"{column_name}_mean_h{horizon}"),
            col_std.alias(f"{column_name}_std_h{horizon}"),
            slope.alias(f"{column_name}_slope_h{horizon}"),
            fraction_nonnull.alias(f"{column_name}_fraction_nonnull_h{horizon}"),
            all_missing.alias(f"{column_name}_all_missing_h{horizon}"),
        ]

    return expressions


def discrete_features(column_name: str, time_col: str, horizons: list[int] = [8, 24]):
    """
    Compute discrete features for a column.

    These are:
    - mode: The mode of the column within the last `horizon` hours. Ignores missing
      values.
    - num_nonmissing: The number of non-missing values within the last `horizon` hours.

    Parameters
    ----------
    column_name : str
        Name of column for which to compute features. E.g., `hr`.
    time_col : str
        Name of the time column. E.g., `time_hours`.
    horizons : list[int]
        Horizons for which to compute the features. E.g., [8, 24].
    """

    def get_rolling_mode(series, horizon):
        """Compute rolling mode for a series of (value, time) tuples."""
        # If the column contains only missings, `mode` returns an empty list. Else, it
        # returns a list with the mode(s) of the column. This list has multiple entries
        # if there's ties. We take the first entry (after sorting) as the mode.
        return (
            pl.select(col=series[0], _time=series[1])
            # Use period="xi" (index-based), as polars does not support time-based with
            # the period column a `Duration`. It only supports this for "absolute" time.
            .rolling(index_column="_time", period=f"{horizon}i")
            .agg(pl.col("col").drop_nulls().mode())
            .select(
                pl.when(pl.col("col").list.len() >= 1)
                .then(pl.col("col").list.sort().list.first())
                .fill_null("(MISSING)")
            )
            .to_series()
        )

    nonnulls = (pl.col(column_name).is_not_null()).cum_sum()

    expressions = list()

    for horizon in horizons:
        col_mode = pl.map_groups(
            exprs=(pl.col(column_name), pl.col(time_col)),
            function=lambda series: get_rolling_mode(series, horizon),
            return_dtype=pl.List(pl.String),
        )
        nonnulls_sum = nonnulls - nonnulls.shift(horizon, fill_value=0).cast(pl.Float64)

        expressions += [
            col_mode.alias(f"{column_name}_mode_h{horizon}"),
            nonnulls_sum.alias(f"{column_name}_num_nonmissing_h{horizon}"),
        ]

    return expressions


def treatment_indicator_features(
    column_name: str, time_col: str, horizons: list[int] = [8, 24]
):
    """
    Compute features for a treatment indicator column.

    These are:
    - num_nonmissing: The number of non-missing values within the last `horizon` hours.
    - any_nonmissing: True if there is a non-missing value within the last `horizon`
    hours.

    Parameters
    ----------
    column_name : str
        Name of column for which to compute features. E.g., `hep_ind`.
    time_col : str
        Not used.
    horizons : list[int]
        Horizons for which to compute the features. E.g., [8, 24].
    """
    col_cs = pl.col(column_name).cast(pl.Int32).fill_null(0).cum_sum()

    expressions = list()

    for horizon in horizons:
        col_sum = col_cs - col_cs.shift(horizon, fill_value=0)
        expressions += [
            col_sum.alias(f"{column_name}_num_nonmissing_h{horizon}"),
            (col_sum > 0).alias(f"{column_name}_any_nonmissing_h{horizon}"),
        ]
    return expressions


def treatment_continuous_features(
    column_name: str,
    time_col: str,
    horizons: list[int] = [8, 24],
    log_transform: bool = True,
    log_eps: float = 0.0,
):
    """
    Compute features for a variable with continuous treatment values.

    These are:
    - rate: The log of the mean of the column within the last `horizon` hours after
      imputing with zero.

    Parameters
    ----------
    column_name : str
        Name of column for which to compute features. E.g., `hr`.
    time_col : str
        Not used.
    horizons : list[int]
        Horizons for which to compute the features. E.g., [8, 24].
    log_transform : bool, optional, default = True
        Whether to log transform the mean (rate).
    log_eps : float, optional, default = 0.0
        Epsilon to add before taking the log. This is to avoid taking the log of 0.
    """
    col_cs = pl.col(column_name).fill_null(0.0).cum_sum()
    expressions = list()

    for horizon in horizons:
        col_sum = col_cs - col_cs.shift(horizon, fill_value=0)
        col_mean = col_sum / (pl.col(time_col) + 1).clip(None, horizon)
        # This ensures that zeros get mapped to zeros.
        if log_transform:
            expressions += [
                (col_mean + log_eps).log().alias(f"log_{column_name}_rate_h{horizon}"),
            ]
        else:
            expressions += [col_mean.alias(f"{column_name}_rate_h{horizon}")]

    return expressions


def eep_label(events: pl.Expr, horizon: int):
    """
    From an event series, create a label for the early event prediction (eep) task.

     - If there is a positive event at the current time step, the label is missing.
     - Else, if there is a positive event within the next `horizon` hours, the label is
       true. This holds even if there is a negative event at the current time step.
     - Else, if there is a negative event within the next `horizon` hours (and no
       positive event within the next `horizon` hours or at the current time step), the
       label is false.

    The "next `horizon` hours" exclude the current time step.

    E.g., if `-` are missings, for a 4 hour horizon:

    event: - 0 0 - - - - - 0 0 0 - - 1 1 1 - 0 0
    label: 0 0 - - 0 0 0 0 0 1 1 1 1 - - - 0 0 -

    Note that at the time step of a positive event, the label is always missing. At the
    time step of a negative event, the label could be true, false, or missing.

    Parameters
    ----------
    events : pl.Expr
        An expression for an event series. Boolean with possibly missing values.
    horizon : int
        The horizon for the early event prediction task.
    """
    positive_labels = events.replace(False, None).backward_fill(horizon)
    # shift(-1) and backward_fill(horizon - 1) excludes the last zero.
    negative_labels = events.replace(True, None).shift(-1).backward_fill(horizon - 1)
    return pl.when(events.is_null() | events.ne(True)).then(
        pl.coalesce(positive_labels, negative_labels)
    )


def polars_nan_or(*args: pl.Expr):
    """
    Nan or operation for polars Series.

    If any of the arguments is nonmissing and positive, return True. If all arguments
    are nonmissing and negative, return False. Else, return None.

    Examples
    --------
    >>> import polars as pl
    >>> a = pl.Series("a", [1, 2, None])
    >>> b = pl.Series("b", [0, None, 2])
    >>> polars_nan_or(a < 0, b == 2)
    [ False, None, True ]
    """
    return (
        pl.when(pl.max_horizontal(*args))  # This ignores nans
        .then(True)
        .when(pl.all_horizontal([a.is_not_null() for a in args]))
        .then(False)
    )


def outcomes():
    """
    Compute outcomes.

    These are:
    - mortality_at_24h: A single label at time 24h after entry to the ICU whether the
      patient dies in the ICU. THis is a "once per patient" prediction task.
    - decompensation_at_24h: Whether the patient decompensates within the next 24 hours.
      This has label is true if the patient dies within the next 24 hours. Else, this is
      false. This does not have missing values.
    - respiratory_failure_at_24h: Whether the patient has a respiratory failure within
      the next 24 hours. If the PaO2/FiO2 ratio is below 200, the patient is considered
      to have a respiratory failure (event).
    - remaining_los: The remaining length of stay in the ICU.
    - circulatory_failure_at_8h: Whether the patient has a circulatory failure within
      the next 8 hours. Circulatory failure is defined via blood pressure and lactate.
      Blood pressure is considered low if the mean arterial pressure is below 65 mmHg or if
      the patient is on any blood pressure increasing drug. Lactate is high if it is
      above 2 mmol/l. If the two criteria (map and lactate) don't agree, or if one of them is
      missing, the event label is missing.
    - kidney_failure_at_48h: Whether the patient has a kidney failure within the next 48
      hours. The patient has a kidney failure if they are in stage 3 according to the
      KDIGO guidelines:
      https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf
    - los_at_24h: The length of stay in the ICU at 24 hours after entry.
    """
    # mortality_at_24h
    # This is a "once per patient" prediction task. At time step 24h, a label is
    # assigned. All other timesteps are missing.
    mortality_at_24h = (
        pl.when(pl.col("time_hours").eq(24))
        .then(pl.col("death_icu").first().fill_null(False))
        .alias("mortality_at_24h")
    )

    # decompensation_at_24h
    # This has label is true if the patient dies within the next 24 hours. Else, this
    # is false.
    decompensation_at_24h = (
        # Need <24 (or <=23) s.t. we get 24x the label `death_icu`.
        pl.when(pl.col("time_hours").max() - pl.col("time_hours") < 24)
        .then(pl.col("death_icu"))
        .otherwise(False)
    ).alias("decompensation_at_24h")

    # respiratory_failure_at_24h
    # If the PaO2/FiO2 ratio is below 200, the patient is considered to have a
    # respiratory failure (event). We predict whether the patient has one such event
    # in the next 24 hours. That is, if there is a positive event within the next 24
    # hours, the label is true. Else, if there is a negative event within the next 24
    # hours, the label is false. If there are no events within the next 24 hours, the
    # label is missing.
    RESP_PF_DEF_TSH = 200.0
    events = (pl.col("po2") / pl.col("fio2") * 100.0) < RESP_PF_DEF_TSH
    resp_failure_at_24h = eep_label(events, 24).alias("respiratory_failure_at_24h")

    # remaining_los
    remaining_los = (
        (pl.col("los_icu") - pl.col("time_hours") / 24.0)
        .clip(0, None)
        .alias("remaining_los")
    )

    # circulatory_failure_at_8h
    # A patient is considered to have a circulatory failure if the mean arterial
    # is low (below 65, or being raised by a drug) and the lactate is high (above 2).
    # If the two criteria (map and lactate) don't agree, or if one of them is missing,
    # the event label is missing.
    circulatory_drug_indicators = [  # These raise the map
        "dobu_ind",  # Dobutamine
        "levo_ind",  # Levosimendan
        "norepi_ind",  # Norepinephrine
        "epi_ind",  # Epinephrine
        "milrin_ind",  # Milrinone
        "teophyllin_ind",  # Teophylline
        "dopa_ind",  # Dopamine
        "adh_ind",  # Vasopressin
    ]
    drugs_indicator = (
        pl.sum_horizontal(
            [pl.col(c).fill_null(False) for c in circulatory_drug_indicators]
        )
        > 0
    )
    # Bad map is True if the mean arterial pressure is below 65 or if the patient is on
    # any map increasing drug. If the mean arterial pressure is above 65 and the patient
    # is not on any map increasing drug, bad_map is False. If the patient does not have
    # a map value and is not on any map increasing drug, bad_map is None.
    LOW_MAP_TSH = 65.0
    low_map = pl.col("map") <= LOW_MAP_TSH
    bad_map = pl.coalesce(drugs_indicator.replace(False, None), low_map)

    HIGH_LACT_TSH = 2.0
    bad_lact = pl.col("lact") >= HIGH_LACT_TSH
    # An event occurs if the map is "bad" (low or on drugs) and the lactate is high.
    # If the map is "good" (not low and not on drugs) and the lactate is not high, the
    # event label is negative. If the map and lact don't agree, or if one of them is
    # missing, the event label is missing.
    event = (
        pl.when(bad_map & bad_lact).then(True).when(~bad_map & ~bad_lact).then(False)
    )
    circulatory_failure_at_8h = eep_label(event, 8).alias("circulatory_failure_at_8h")

    # kidney_failure_at_48h
    # The patient has a kidney failure if they are in stage 3 according to
    # https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf
    relative_creatine = pl.col("crea") / pl.col("crea").shift(1).rolling_min(
        window_size=7 * 24, min_periods=1
    )

    # AKI 1 is
    # - max absolute creatinine increase of 0.3 within 48h or
    # - a relative creatinine increase of 1.5.
    creatine_min_48 = pl.col("crea").rolling_min(window_size=48, min_periods=1)
    creatine_max_48 = pl.col("crea").rolling_max(window_size=48, min_periods=1)
    creatine_change_48 = creatine_max_48 - creatine_min_48
    aki_1 = polars_nan_or(creatine_change_48 >= 0.3, relative_creatine >= 1.5)

    # AKI 3 is any of
    # - a relative creatine increase of 3.0 x baseline
    # - AKI 1 and creatinine >= 4.0
    # - not more than 0.3ml/kg/h urine rate for 24h
    # - no urine for 12h
    low_urine_rate = ((pl.col("urine_rate") / pl.col("weight")) < 0.3).cast(pl.Int32)
    low_urine_rate_24 = low_urine_rate.rolling_sum(window_size=24, min_periods=1).eq(24)

    # True if aki_1 is True and creatine >= 4 (neither missing). False if either aki_1
    # is False or creatine < 4. Else, missing.
    high_creatine = ~polars_nan_or(~aki_1, ~pl.col("crea").gt(4))
    # True if the urine_rate is consistently equal to 0 for 12 hours. False if it is
    # ever above 0. If all values are missing, the result is missing.
    anuria = (
        pl.col("urine_rate")
        .eq(0)
        .cast(pl.Int32)
        .rolling_sum(window_size=12, min_periods=1)
        .eq(12)
    )

    aki_3 = polars_nan_or(
        relative_creatine >= 3.0,
        high_creatine,
        low_urine_rate_24,
        anuria,
    )
    # If the weight is missing, the patient could only ever have a positive label, as
    # urine related conditions are always missing. We thus set the label to missing.
    aki_3 = pl.when(pl.col("weight").is_null()).then(None).otherwise(aki_3)
    kidney_failure_at_48h = eep_label(aki_3, 48).alias("kidney_failure_at_48h")

    los_at_24h = pl.when(pl.col("time_hours").eq(24)).then(pl.col("los_icu"))
    los_at_24h = los_at_24h.clip(0, None).alias("los_at_24h")

    return [
        mortality_at_24h,
        decompensation_at_24h,
        resp_failure_at_24h,
        remaining_los,
        circulatory_failure_at_8h,
        kidney_failure_at_48h,
        los_at_24h,
    ]


@click.command()
@click.option("--dataset", type=str, required=True)
@click.option("--data_dir", type=str, default=None)
def main(dataset: str, data_dir: str | Path | None):  # noqa D
    logger.info(f"dataset: {dataset}")
    data_dir = Path(data_dir) if data_dir is not None else Path(DATA_DIR)

    dyn = pl.scan_parquet(data_dir / dataset / "dyn.parquet")
    sta = pl.scan_parquet(data_dir / dataset / "sta.parquet")

    dyn = dyn.join(sta, on="stay_id", how="full", coalesce=True, validate="m:1")
    dyn = dyn.with_columns(
        (pl.col("time").dt.total_hours()).cast(pl.Int32).alias("time_hours")
    )

    # The code below assumes that all missing values are encoded as nulls, not nans.
    # nans behave differently in comparisons (1.0 == nan is False)
    tic = perf_counter()
    nan_columns = dyn.select(pl.col([pl.Float32, pl.Float64]).is_nan().any()).collect()
    nan_columns = [col.name for col in nan_columns if col.any()]
    if nan_columns:
        raise ValueError(f"The following columns contain nans: {nan_columns}")
    toc = perf_counter()
    logger.info(f"Time to check for nans: {toc - tic:.2f}s")

    # Lazy polars.DataFrame.upsample for time_column with dtype int.
    # The result is a DataFrame with a row for each hour between 0 and the maximum
    # time in the dataset for each stay_id.
    time_ranges = (
        dyn.group_by("stay_id")
        .agg(pl.col("time_hours").max().alias("max_time"))
        .with_columns(
            pl.col("max_time")
            .map_elements(
                # +1 to include max_time
                lambda x: pl.int_range(0, x + 1, eager=True, dtype=pl.Int32),
                return_dtype=pl.List(pl.Int32),
            )
            .alias("time_hours")
        )
        .drop("max_time")
        .explode("time_hours")
    )

    # equivalent to a pandas outer join
    dyn = dyn.join(time_ranges, on=["stay_id", "time_hours"], how="full", coalesce=True)
    dyn = dyn.sort(["stay_id", "time_hours"])

    for row in variable_reference.rows(named=True):
        col = pl.col(row["VariableTag"]).clip(row["LowerBound"], row["UpperBound"])

        # Log transform some of the continuous variables. We add a small epsilon to
        # avoid taking the log of 0.
        if row["LogTransform"]:
            eps = row["LogTransformEps"] or 0.0
            col = (col + eps).log().alias(f"log_{row['VariableTag']}")
            col = col.replace([np.nan, -np.inf], None)

        # Cast categorical variables to Enum. `samp` is binary. For simplicity, we cast
        # it to string first.
        if row["DataType"] == "categorical":
            enum = pl.Enum(row["PossibleValues"] + ["(MISSING)"])
            col = col.cast(pl.String).fill_null("(MISSING)").cast(enum)

        dyn = dyn.with_columns(col)

    expressions = ["time_hours"]

    for row in variable_reference.rows(named=True):
        tag = row["VariableTag"]

        if row["VariableType"] == "static" and row["LogTransform"]:
            expressions += [pl.col(f"log_{tag}")]
        elif row["VariableType"] == "static":
            expressions += [pl.col(tag)]

        elif row["DataType"] == "continuous" and row["LogTransform"]:
            expressions += continuous_features(
                f"log_{tag}", "time_hours", horizons=[8, 24]
            )
        elif row["DataType"] == "continuous":
            expressions += continuous_features(tag, "time_hours", horizons=[8, 24])

        elif row["DataType"] == "categorical":
            expressions += discrete_features(tag, "time_hours", horizons=[8, 24])

        elif row["DataType"] == "treatment_ind":
            expressions += treatment_indicator_features(
                tag, "time_hours", horizons=[8, 24]
            )

        elif row["DataType"] == "treatment_cont":
            expressions += treatment_continuous_features(
                tag,
                "time_hours",
                horizons=[8, 24],
                log_eps=row["LogTransformEps"],
                log_transform=row["LogTransform"],
            )
        else:
            raise ValueError(f"Unknown DataType: {row['DataType']}")

    expressions += outcomes()

    q = dyn.group_by("stay_id").agg(expressions).explode(pl.exclude("stay_id"))

    q = q.with_columns(
        (pl.col("stay_id").hash() / 2.0**64).alias("hash")  # useful for subsetting
    ).with_columns(
        pl.when(pl.col("hash") < 0.7)
        .then(pl.lit("train"))
        .when(pl.col("hash") < 0.85)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias("split"),
        pl.lit(dataset).alias("dataset"),
    )

    feature_names = set(features())
    schema_names = set(q.collect_schema().keys())
    missing_features = feature_names - schema_names

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    other_variables = {"stay_id", "hash", "split", "dataset", "time_hours"}
    extra_features = schema_names - feature_names - other_variables - set(OUTCOMES)
    if extra_features:
        logger.warning(f"Extra features: {extra_features}")

    tic = perf_counter()
    out = q.collect()
    toc = perf_counter()
    logger.info(f"Time to compute features: {toc - tic:.2f}s")
    logger.info(f"out.shape: {out.shape}")

    tic = perf_counter()
    out.write_parquet(data_dir / dataset / "features.parquet")
    toc = perf_counter()
    logger.info(f"Time to write features: {toc - tic:.2f}s")


if __name__ == "__main__":
    main()

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal

from icu_benchmarks.data import eep_label, outcomes, polars_nan_or


def to_bool(x):
    return pl.Series(x.split(" ")).replace("-", None).cast(pl.Int32).cast(pl.Boolean)


@pytest.mark.parametrize(
    "events, expected, horizon",
    [
        (
            "- 0 0 - - - - - 0 0 0 - - 1 1 1 - 0 0",
            "0 0 - - 0 0 0 0 0 1 1 1 1 - - - 0 0 -",
            4,
        )
    ],
)
def test_eep_labels(events, expected, horizon):
    df = pl.DataFrame(
        {
            "events": to_bool(events),
            "expected": to_bool(expected),
        }
    ).with_columns(eep_label(pl.col("events"), horizon).alias("labels"))
    assert_series_equal(df["labels"], df["expected"], check_names=False)


@pytest.mark.parametrize(
    "args, expected", [(("0 0 0 - - 1", "0 0 0 - - -", "0 1 - 1 0 0"), "0 1 - 1 - 1")]
)
def test_polars_nan_or(args, expected):
    df = pl.DataFrame({str(i): to_bool(x) for i, x in enumerate(args)}).with_columns(
        polars_nan_or(*[pl.col(str(i)) for i in range(len(args))]).alias("result")
    )
    assert_series_equal(df["result"], to_bool(expected), check_names=False)


@pytest.mark.parametrize(
    "outcome_name, input, expected",
    [
        (
            "mortality_at_24h",
            pl.DataFrame(
                {
                    "death_icu": True,
                    "time_hours": np.arange(0, 36),
                }
            ),
            pl.Series("mortality_at_24h", [None] * 23 + [True] + [None] * 12),
        ),
        (
            "mortality_at_24h",
            pl.DataFrame(
                {
                    "death_icu": False,
                    "time_hours": np.arange(0, 36),
                }
            ),
            pl.Series([None] * 23 + [False] + [None] * 12),
        ),
        (
            "decompensation_at_24h",
            pl.DataFrame(
                {
                    "death_icu": True,
                    "time_hours": np.arange(0, 36),
                }
            ),
            pl.Series([False] * 12 + [True] * 24),
        ),
        (
            "decompensation_at_24h",
            pl.DataFrame(
                {
                    "death_icu": False,
                    "time_hours": np.arange(0, 36),
                }
            ),
            pl.Series([False] * 36),
        ),
        (
            "respiratory_failure_at_24h",
            # events (in blocks of 12): - - - 1 - - 0
            pl.DataFrame(
                {
                    "po2": [None] * 12
                    + [200] * 12
                    + [None] * 12
                    + [50] * 12
                    + [None] * 24
                    + [150] * 12,
                    "fio2": [1.0] * 12
                    + [None] * 12
                    + [None] * 12
                    + [50] * 12
                    + [50] * 24
                    + [50] * 12,
                }
            ),
            pl.Series([None] * 12 + [True] * 24 + [None] * 12 + [False] * 35 + [None]),
        ),
        (
            "remaining_los",
            pl.DataFrame(
                {
                    "los_icu": 4.0,
                    "time_hours": np.arange(0, 4 * 24),
                }
            ),
            pl.Series(np.arange(4 * 24, 0, -1) / 24),
        ),
        (
            "circulatory_failure_at_8h",
            pl.DataFrame(
                {
                    "dobu_ind": None,  # these are either None or True
                    "levo_ind": None,
                    "norepi_ind": None,
                    "epi_ind": None,
                    "milrin_ind": None,
                    "teophyllin_ind": None,
                    "dopa_ind": None,
                    "adh_ind": None,
                    # events (in blocks of 4): - - 0 - 1
                    "map": [70] * 4 + [None] * 4 + [70] * 4 + [60] * 4 + [60] * 4,
                    "lact": [1] * 4 + [1] * 4 + [1] * 4 + [1] * 4 + [3] * 4,
                }
            ),
            pl.Series([False] * 8 + [True] * 8 + [None] * 4),
        ),
        (
            "circulatory_failure_at_8h",
            pl.DataFrame(
                {
                    "dobu_ind": None,
                    "levo_ind": None,
                    "norepi_ind": None,
                    "epi_ind": None,
                    "milrin_ind": None,
                    "teophyllin_ind": None,
                    "dopa_ind": [None] * 14 + [True] * 6,
                    "adh_ind": [None] * 12 + [True] * 4 + [None] * 4,
                    # events (in blocks of 4): 0 - 0 - 1
                    "map": [70] * 4 + [60] * 4 + [70] * 12,
                    "lact": [1] * 4 + [1] * 4 + [1] * 4 + [1] * 4 + [3] * 4,
                }
            ),
            pl.Series([False] * 8 + [True] * 8 + [None] * 4),
        ),
    ],
)
def test_outcomes(outcome_name, input, expected):
    expr = [e for e in outcomes() if e.meta.output_name() == outcome_name][0]
    assert_series_equal(
        input.with_columns(expr).select(outcome_name).to_series(),
        expected.rename(outcome_name),
    )

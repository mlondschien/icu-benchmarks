import polars as pl
import pytest
from polars.testing import assert_series_equal

from icu_benchmarks.data import eep_label, polars_nan_or


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

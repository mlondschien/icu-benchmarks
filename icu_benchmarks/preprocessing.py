import numpy as np
import polars as pl
from glum import GeneralizedLinearRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    SplineTransformer,
    StandardScaler,
)

from icu_benchmarks.constants import CAT_MISSING_NAME


def _sao2_imputation(df):
    return df.select(
        pl.when(pl.col("sao2_all_missing_h8"))
        .then(pl.col("spo2_mean_h8"))
        .otherwise(pl.col("sao2_ffilled"))
        .clip(lower_bound=0, upper_bound=99.7)
        .alias("sao2_imputed")
    )


def get_preprocessing(model, df):  # noqa D
    # time_hours, gcs columns are continuous, integer columns.
    continuous_variables = [col for col, dtype in df.schema.items() if dtype.is_float() or dtype.is_integer()]
    bool_variables = [col for col in df.columns if df[col].dtype == pl.Boolean]
    other = [
        col for col in df.columns if col not in continuous_variables + bool_variables
    ]
    if "GeneralizedLinear" in str(model) or "AnchorRegression" in str(model) or "DataSharedLasso" in str(model):
        imputer = SimpleImputer(strategy="mean", copy=False, keep_empty_features=True)
        scaler = StandardScaler(copy=False)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        transformer = ColumnTransformer(transformers=[
            ("continuous", imputer, continuous_variables),
            ("bool", "passthrough", bool_variables),
            ("other", encoder, other),
        ], sparse_threshold=0)
        preprocessor = Pipeline([("transformer", transformer), ("scaler", scaler)])
        preprocessor.set_output(transform="polars")
    elif "LGBM" in str(model):
        preprocessor = ColumnTransformer(
            transformers=[
                ("continuous", "passthrough", continuous_variables),
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        # LGBM from pyarrow allows only int, bool, float types. So we
                        # have to transform `airway` from str to int. Unknown value must
                        # be an int. 99 works since we should never have so many
                        # categories.
                        unknown_value=99,
                    ),
                    other + bool_variables,
                ),
            ]
        ).set_output(transform="polars")
    else:
        raise ValueError(f"Unknown model {model}")

    return preprocessor

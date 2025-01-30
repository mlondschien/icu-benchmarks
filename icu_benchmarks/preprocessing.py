import numpy as np
import polars as pl
from glum import GeneralizedLinearRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def get_preprocessing(model, df):  # noqa D
    continuous_variables = [col for col, dtype in df.schema.items() if dtype.is_float()]
    bool_variables = [col for col in df.columns if df[col].dtype == pl.Boolean]
    other = [
        col for col in df.columns if col not in continuous_variables + bool_variables
    ]

    if isinstance(model, GeneralizedLinearRegressor):
        scaler = SimpleImputer(strategy="mean", copy=False, keep_empty_features=True)
        imputer = StandardScaler(copy=False)
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "continuous",
                    Pipeline([("impute", imputer), ("scale", scaler)]),
                    continuous_variables,
                ),
                ("bool", "passthrough", bool_variables),
                (
                    "other",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    other,
                ),
            ],
            sparse_threshold=0,
            verbose=1,
        ).set_output(transform="polars")
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("continuous", "passthrough", continuous_variables),
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=np.nan
                    ),
                    other,
                ),
            ]
        ).set_output(transform="polars")

    return preprocessor

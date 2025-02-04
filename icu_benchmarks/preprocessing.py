import numpy as np
import polars as pl
from glum import GeneralizedLinearRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer, SplineTransformer
from icu_benchmarks.constants import CAT_MISSING_NAME

def _sao2_imputation(df):
    return df.select(
        pl.when(pl.col("sao2_all_missing_h8"))
        .then(pl.col("spo2_mean_h8"))
        .otherwise(pl.col("sao2_ffilled"))
        .clip(lower_bound=0, upper_bound=99.7)
        .alias("sao2_imputed")
    )



def get_preprocessing(model, df, outcome):  # noqa D
    continuous_variables = [col for col, dtype in df.schema.items() if dtype.is_float()]
    # continuous_variables = ["temp_ffilled", "ph_ffilled"]
    bool_variables = [col for col in df.columns if df[col].dtype == pl.Boolean]
    # bool_variables = []
    other = [
        col for col in df.columns if col not in continuous_variables + bool_variables
    ]
    # other = []
    if isinstance(model, GeneralizedLinearRegressor):
        imputer = SimpleImputer(strategy="mean", copy=False, keep_empty_features=True)
        scaler = StandardScaler(copy=False)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers = [
            ("continuous", imputer, continuous_variables),
            ("bool", "passthrough", bool_variables),
            ("other", encoder, other),
        ]

        if outcome == "log_po2":
            sao2_imputer = FunctionTransformer(_sao2_imputation)
            splines = SplineTransformer(
                knots=np.array([50, 60, 70, 80, 85, 90, 95, 100]).reshape(-1, 1),
                degree=3,
            )
            na_imputer = SimpleImputer(strategy="mean", copy=False, keep_empty_features=True)
            sao2_splines = Pipeline([("sao2_impute", sao2_imputer), ("na_imputer", na_imputer), ("splines", splines)])
            transformers.append(("sao2_splines", sao2_splines, ["sao2_all_missing_h8", "spo2_mean_h8", "sao2_ffilled"]))
        
        transformer = ColumnTransformer(transformers=transformers, sparse_threshold=0)
        preprocessor = Pipeline([("transformer", transformer), ("scaler", scaler)])
        preprocessor.set_output(transform="polars")
    else:
        # continuous_variables = ["spo2_mean_h8", "sao2_ffilled", "temp_ffilled", "ph_ffilled"]
        # other = ["sao2_all_missing_h8"]
        preprocessor = ColumnTransformer(
            transformers=[
                ("continuous", "passthrough", continuous_variables),
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=CAT_MISSING_NAME
                    ),
                    other + bool_variables,
                ),
            ]
        ).set_output(transform="polars")

    return preprocessor

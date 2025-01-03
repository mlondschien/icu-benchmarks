import numpy as np
import polars as pl
from glum import GeneralizedLinearRegressor


class DataSharedLasso(GeneralizedLinearRegressor):
    """Data Shared Lasso Estimator from S. Gross and R. Tibshirani."""

    def __post_init__(self):  # noqa: D
        if self.P1 is not None:
            raise ValueError("P1 must be None")
        if self.scale_predictors:
            raise ValueError("scale_predictors must be False")

    def fit(self, X, y, sample_weight=None, datasets=None):  # noqa: D
        self.fit_datasets_ = np.sort(np.unique(datasets))

        if isinstance(X, np.ndarray):
            means = [X[datasets == d].mean(axis=0) for d in self.fit_datasets_]
            X_interacted = np.hstack(
                [X]
                + [
                    (X - means[idx][np.newaxis, :]) * (datasets == d)[:, np.newaxis]
                    for idx, d in enumerate(self.fit_datasets_)
                ]
            )
        elif isinstance(X, pl.DataFrame):
            means = [X.filter(datasets == d).mean() for d in self.fit_datasets_]
            X_interacted = X.with_columns(_datasets=datasets)
            X_interacted = X_interacted.with_columns(
                [
                    pl.when(pl.col("_datasets").eq(d))
                    .then(X[col] - means[idx][col])
                    .otherwise(0)
                    .alias(f"_dataset={d}_x_{col}")
                    for col in X.columns
                    for idx, d in enumerate(self.fit_datasets_)
                ]
            ).drop("_datasets")
        else:
            raise ValueError("X must be a numpy array or polars DataFrame")

        self.P1 = np.repeat(
            [1] + [1 / np.sqrt(np.sum(datasets == d)) for d in self.fit_datasets_],
            X.shape[1],
        )

        super().fit(X_interacted, y, sample_weight=sample_weight)

        return self

    def predict(self, X):  # noqa: D
        if isinstance(X, np.ndarray):
            X_interacted = np.hstack(
                [X, np.zeros((X.shape[0], len(self.fit_datasets_) * X.shape[1]))]
            )
        elif isinstance(X, pl.DataFrame):
            X_interacted = X.with_columns(
                [
                    pl.lit(0).alias(f"_dataset={d}_x_{col}")
                    for col in X.columns
                    for d in self.fit_datasets_
                ]
            )
        else:
            raise ValueError("X must be a numpy array or polars DataFrame")

        return super().predict(X_interacted)

import numpy as np
import polars as pl
from glum import GeneralizedLinearRegressor
import gin
import tabmat

@gin.configurable
class DataSharedLasso(GeneralizedLinearRegressor):
    """Data Shared Lasso Estimator from S. Gross and R. Tibshirani."""


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
            [1] + [np.sqrt(np.sum(datasets == d) / len(datasets)) for d in self.fit_datasets_],
            X.shape[1],
        )
        
        # Need to convert to tabmat here. Else, the feature names are not set correctly.
        X_interacted = tabmat.from_df(X_interacted)

        super().fit(X_interacted, y) # , sample_weight=sample_weight)

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
        
        # convert to tabmat here as we did so in fit
        X_interacted = tabmat.from_df(X_interacted)
        return super().predict(X_interacted)



@gin.configurable
class AnchorRegression(GeneralizedLinearRegressor):
    """Anchor Regression Estimator from D. Rothenhäusler et al."""

    def __init__(self, gamma=1.0,
    alpha=None, l1_ratio=0, P1='identity', P2='identity', fit_intercept=True, family='normal', link='auto', solver='auto', max_iter=100, max_inner_iter=100000, gradient_tol=None, step_size_tol=None, hessian_approx=0.0, warm_start=False, alpha_search=False, alphas=None, n_alphas=100, min_alpha_ratio=None, min_alpha=None, start_params=None, selection='cyclic', random_state=None, copy_X=None, check_input=True, verbose=0, scale_predictors=False, lower_bounds=None, upper_bounds=None, A_ineq=None, b_ineq=None, force_all_finite=True, drop_first=False, robust=True, expected_information=False, formula=None, interaction_separator=':', categorical_format='{name}[{category}]', cat_missing_method='fail', cat_missing_name='(MISSING)'
    ):
        if gamma!=1 and  family not in ["gaussian", "normal"]:
            raise ValueError(f"Family {family} not supported for AnchorRegression.")
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, P1=P1, P2=P2, fit_intercept=fit_intercept, family=family, link=link, solver=solver, max_iter=max_iter, max_inner_iter=max_inner_iter, gradient_tol=gradient_tol, step_size_tol=step_size_tol, hessian_approx=hessian_approx, warm_start=warm_start, alpha_search=alpha_search, alphas=alphas, n_alphas=n_alphas, min_alpha_ratio=min_alpha_ratio, min_alpha=min_alpha, start_params=start_params, selection=selection, random_state=random_state, copy_X=copy_X, check_input=check_input, verbose=verbose, scale_predictors=scale_predictors, lower_bounds=lower_bounds, upper_bounds=upper_bounds, A_ineq=A_ineq, b_ineq=b_ineq, force_all_finite=force_all_finite, drop_first=drop_first, robust=robust, expected_information=expected_information, formula=formula, interaction_separator=interaction_separator, categorical_format=categorical_format, cat_missing_method=cat_missing_method, cat_missing_name=cat_missing_name)
        self.gamma = gamma

    def fit(self, X, y, sample_weight=None, datasets=None):  # noqa: D
        _kappa = (self.gamma - 1) / self.gamma

        self.fit_datasets_ = np.sort(np.unique(datasets))

        if not isinstance(y, np.ndarray):
            y = y.to_numpy()

        if isinstance(X, pl.DataFrame):
            X_tilde = X.to_numpy()
        else:
            X_tilde = X
        
        y_mean = y.mean()
        y = y - y_mean
        
        x_mean = X_tilde.mean(axis=0)
        X_tilde = X_tilde - x_mean

        for d in self.fit_datasets_:
            mask = datasets == d

            y_tilde[mask] = np.sqrt(1 - _kappa) * y_tilde[mask] + (1 - np.sqrt(1 - _kappa)) * y_tilde[mask, :].mean()
            X_tilde[mask, :] = np.sqrt(1 - _kappa) * X_tilde[mask, :] + (1 - np.sqrt(1 - _kappa)) * X_tilde[mask, :].mean(axis=0)

        if isinstance(X, pl.DataFrame):
            X_tilde = pl.DataFrame(X_tilde, columns=X.columns)
                
        X_tilde = tabmat.from_df(X_tilde)
        super().fit(X_tilde, y, sample_weight=sample_weight)

        return self
    
    def predict(self, X):  # noqa: D
        # convert to tabmat here as we did so in fit
        X = tabmat.from_df(X)
        return super().predict(X)
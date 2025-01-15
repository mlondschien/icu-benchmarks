import gin
import lightgbm as lgb
import numpy as np
import polars as pl
import tabmat
from glum import GeneralizedLinearRegressor


@gin.configurable
class DataSharedLasso(GeneralizedLinearRegressor):
    """Data Shared Lasso Estimator from S. Gross and R. Tibshirani."""
    # All parameters are copied from the GLR estimator. They need to be explicit to
    # adhere to sklearn's API. GLR inherits from BaseEstimator.
    def __init__(
        self,
        ratio=1.0,
        alpha=None,
        l1_ratio=0,
        P1="identity",
        P2="identity",
        fit_intercept=True,
        family="normal",
        link="auto",
        solver="auto",
        max_iter=100,
        max_inner_iter=100000,
        gradient_tol=None,
        step_size_tol=None,
        hessian_approx=0.0,
        warm_start=False,
        alpha_search=False,
        alphas=None,
        n_alphas=100,
        min_alpha_ratio=None,
        min_alpha=None,
        start_params=None,
        selection="cyclic",
        random_state=None,
        copy_X=None,
        check_input=True,
        verbose=0,
        scale_predictors=False,
        lower_bounds=None,
        upper_bounds=None,
        A_ineq=None,
        b_ineq=None,
        force_all_finite=True,
        drop_first=False,
        robust=True,
        expected_information=False,
        formula=None,
        interaction_separator=":",
        categorical_format="{name}[{category}]",
        cat_missing_method="fail",
        cat_missing_name="(MISSING)",
    ):
        super().__init__(
            alpha=alpha,
            l1_ratio=l1_ratio,
            P1=P1,
            P2=P2,
            fit_intercept=fit_intercept,
            family=family,
            link=link,
            solver=solver,
            max_iter=max_iter,
            max_inner_iter=max_inner_iter,
            gradient_tol=gradient_tol,
            step_size_tol=step_size_tol,
            hessian_approx=hessian_approx,
            warm_start=warm_start,
            alpha_search=alpha_search,
            alphas=alphas,
            n_alphas=n_alphas,
            min_alpha_ratio=min_alpha_ratio,
            min_alpha=min_alpha,
            start_params=start_params,
            selection=selection,
            random_state=random_state,
            copy_X=copy_X,
            check_input=check_input,
            verbose=verbose,
            scale_predictors=scale_predictors,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            force_all_finite=force_all_finite,
            drop_first=drop_first,
            robust=robust,
            expected_information=expected_information,
            formula=formula,
            interaction_separator=interaction_separator,
            categorical_format=categorical_format,
            cat_missing_method=cat_missing_method,
            cat_missing_name=cat_missing_name,
        )
        self.ratio = ratio

    def fit(self, X, y, sample_weight=None, dataset=None):  # noqa: D
        self.fit_datasets_ = np.sort(np.unique(dataset))

        if isinstance(X, np.ndarray):
            means = [X[dataset == d].mean(axis=0) for d in self.fit_datasets_]
            X_interacted = np.hstack(
                [X]
                + [(dataset == d).reshape(-1,1) for d in self.fit_datasets_]
                + [
                    (X - means[idx][np.newaxis, :]) * (dataset == d)[:, np.newaxis]
                    for idx, d in enumerate(self.fit_datasets_)
                ]
            )
        elif isinstance(X, pl.DataFrame):
            means = [X.filter(dataset == d).mean() for d in self.fit_datasets_]
            X_interacted = X.with_columns(_dataset=dataset)
            X_interacted = X_interacted.with_columns(
                [pl.col("_dataset").eq(d).alias(f"_dataset={d}") for d in self.fit_datasets_]
                + [
                    pl.when(pl.col("_dataset").eq(d))
                    .then(X[col] - means[idx][col])
                    .otherwise(0)
                    .alias(f"_dataset={d}_x_{col}")
                    for idx, d in enumerate(self.fit_datasets_)
                    for col in X.columns
                ]
            ).drop("_dataset")
        else:
            raise ValueError("X must be a numpy array or polars DataFrame")

        # The DSL uses the loss
        # sum_i (y_i - yhat_i)^2 / n + r_0 lambda || beta_0 ||_1 + 
        #    lambda sum_g r_g || beta_g ||_1.
        # Here, beta_0 are the "shared" coefficients, and beta_g are the group specific
        # coefficients.
        # Given G groups of equal size, the authors suggest to use a scaling
        # r_g ~ 1/sqrt(G).
        # We have G groups with n_g observations and weights w_g. Let g(i) be the i-th
        # observation's group. Assume all weights sum to 1. The loss is:
        # sum_i w_g(i) (y_i - yhat_i)^2 + lambda || beta_0 ||_1 +
        #    lambda sum_g r_g || beta_g ||_1
        # = sum_g (w_g * n_g) * [ sum_{i in G} (y_i - yhat_i)^2 / n_g + 
        #    lambda r_g / (w_g * n_g) || beta_g ||_1 ] + lambda_0 || beta_0 ||_1
        # Let n_eff = [ sum_i w_g(i) ]^2 / sum_i w_g(i)^2.
        # Asymptotic Lasso theory suggest scaling l1-penalty with sqrt(1 / sample size).
        # This would imply
        # - lambda ~ sqrt(1 / n_eff)
        # - lambda r_g / (w_g * n_g) ~ sqrt(1 / n_g)
        #    => r_g ~ (w_g * n_g) * sqrt(n_eff / n_g)
        # This implies:
        # - weighting_exponent: 0 => w_g = 1/n, n_eff = n, r_g = sqrt(n_g / n)
        # - weighting_exponent: -0.5 => w_g = 1/sqrt(n_g)/const, n_eff = const^2 / G,
        #   r_g = sqrt(1/G)
        # - weighting_exponent: 1.0 => w_g = 1 / n_g / G
        #   n_eff = G^2 / sum_g (1/n_g), r_g = 1 / [ sqrt(sum_g (1/n_g)) * sqrt(n_g) ]
        # 
        # Here, we allow the w_i to be arbitrary. We thus replace (w_g * n_g) with
        # sum_{i in g} w_g.
        #
        # If all groups have the same size and are equally weighted, then
        # r_g = 1 / sqrt(G).
        #
        # We also allow an extra term scaling r_g *= ratio, where ratio is a tuning
        # parameter around 1.
        if sample_weight is None:
            rg = [
                np.sqrt(np.sum(dataset == d) / len(dataset)) for d in self.fit_datasets_
            ]
        else:
            eff_sample_size = np.sum(sample_weight) ** 2 / np.sum(sample_weight ** 2)
            rg = [
                np.sqrt(eff_sample_size / np.sum(dataset == d)) * np.sum(sample_weight[dataset == d]) 
                for d in self.fit_datasets_
            ]

        # extra 1 for dataset-specific intercepts
        self.P1 = np.repeat([1] + [self.ratio * r for r in rg], X.shape[1] + 1)[1:]
        self.P2 = np.repeat([1] + [self.ratio * r for r in rg], X.shape[1] + 1)[1:]

        # Need to convert to tabmat here. Else, the feature names are not set correctly.
        if isinstance(X_interacted, pl.DataFrame):
            X_interacted = tabmat.from_df(X_interacted)

        super().fit(X_interacted, y, sample_weight=sample_weight)

        return self

    def predict(self, X):  # noqa: D
        if isinstance(X, np.ndarray):
            X_interacted = np.hstack(
                [X, np.zeros((X.shape[0], len(self.fit_datasets_) * (X.shape[1] + 1)))]
            )
        elif isinstance(X, pl.DataFrame):
            X_interacted = X.with_columns(
                [pl.lit(0).alias(f"_dataset={d}") for d in self.fit_datasets_] + [
                    pl.lit(0).alias(f"_dataset={d}_x_{col}")
                    for col in X.columns
                    for d in self.fit_datasets_
                ]
            )
        else:
            raise ValueError("X must be a numpy array or polars DataFrame")

        # convert to tabmat here as we did so in fit
        if isinstance(X_interacted, pl.DataFrame):
            X_interacted = tabmat.from_df(X_interacted)

        return super().predict(X_interacted)


@gin.configurable
class AnchorRegression(GeneralizedLinearRegressor):
    """Anchor Regression Estimator from D. Rothenhäusler et al."""

    # All parameters are copied from the GLM estimator. They need to be explicit to
    # adhere to sklearn's API. GLR inherits from BaseEstimator.
    def __init__(
        self,
        gamma=1.0,
        alpha=None,
        l1_ratio=0,
        P1="identity",
        P2="identity",
        fit_intercept=True,
        family="normal",
        link="auto",
        solver="auto",
        max_iter=100,
        max_inner_iter=100000,
        gradient_tol=None,
        step_size_tol=None,
        hessian_approx=0.0,
        warm_start=False,
        alpha_search=False,
        alphas=None,
        n_alphas=100,
        min_alpha_ratio=None,
        min_alpha=None,
        start_params=None,
        selection="cyclic",
        random_state=None,
        copy_X=None,
        check_input=True,
        verbose=0,
        scale_predictors=False,
        lower_bounds=None,
        upper_bounds=None,
        A_ineq=None,
        b_ineq=None,
        force_all_finite=True,
        drop_first=False,
        robust=True,
        expected_information=False,
        formula=None,
        interaction_separator=":",
        categorical_format="{name}[{category}]",
        cat_missing_method="fail",
        cat_missing_name="(MISSING)",
    ):
        if gamma != 1 and family not in ["gaussian", "normal"]:
            raise ValueError(f"Family {family} not supported for AnchorRegression.")

        super().__init__(
            alpha=alpha,
            l1_ratio=l1_ratio,
            P1=P1,
            P2=P2,
            fit_intercept=fit_intercept,
            family=family,
            link=link,
            solver=solver,
            max_iter=max_iter,
            max_inner_iter=max_inner_iter,
            gradient_tol=gradient_tol,
            step_size_tol=step_size_tol,
            hessian_approx=hessian_approx,
            warm_start=warm_start,
            alpha_search=alpha_search,
            alphas=alphas,
            n_alphas=n_alphas,
            min_alpha_ratio=min_alpha_ratio,
            min_alpha=min_alpha,
            start_params=start_params,
            selection=selection,
            random_state=random_state,
            copy_X=copy_X,
            check_input=check_input,
            verbose=verbose,
            scale_predictors=scale_predictors,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            force_all_finite=force_all_finite,
            drop_first=drop_first,
            robust=robust,
            expected_information=expected_information,
            formula=formula,
            interaction_separator=interaction_separator,
            categorical_format=categorical_format,
            cat_missing_method=cat_missing_method,
            cat_missing_name=cat_missing_name,
        )
        self.gamma = gamma

    def fit(self, X, y, sample_weight=None, dataset=None):  # noqa: D
        """
        Fit the Anchor Regression model.

        Parameters
        ----------
        X : np.ndarray or polars.DataFrame
            The input data.
        y : np.ndarray
            The outcome.
        sample_weight : np.ndarray, optional
            The sample weights.
        dataset : np.ndarray
            Array of dataset indicators. Each unique value is assumed to correspond to a
            single environment. The anchor then is a one-hot encoding of this.
        """
        if self.gamma == 1:
            if isinstance(X, pl.DataFrame):
                X = tabmat.from_df(X)

            super().fit(X, y, sample_weight=sample_weight)
            return self

        # 1 - kappa = 1 - (gamma - 1) / gamma = 1 / gamma
        mult = np.sqrt(1 / self.gamma)

        self.fit_datasets_ = np.sort(np.unique(dataset))

        if not isinstance(y, np.ndarray):
            y = y.to_numpy()

        if isinstance(X, pl.DataFrame):
            X_tilde = X.to_numpy()
        else:
            X_tilde = X

        y_mean = y.mean()
        y_tilde = y - y_mean

        x_mean = X_tilde.mean(axis=0)
        X_tilde = X_tilde - x_mean

        for d in self.fit_datasets_:
            mask = dataset == d
            y_tilde[mask] = mult * y_tilde[mask] + (1 - mult) * y_tilde[mask].mean()
            X_tilde[mask, :] = mult * X_tilde[mask, :] + (1 - mult) * X_tilde[
                mask, :
            ].mean(axis=0)

        X_tilde = X_tilde + x_mean
        y_tilde = y_tilde + y_mean

        if isinstance(X, pl.DataFrame):  # glum does not support polars yet
            X_tilde = tabmat.from_df(pl.DataFrame(X_tilde, schema=X.columns))

        super().fit(X_tilde, y_tilde, sample_weight=sample_weight)

        return self

    def predict(self, X):  # noqa: D
        # convert to tabmat here as we did so in fit
        if isinstance(X, pl.DataFrame):
            X = tabmat.from_df(X)

        return super().predict(X)


@gin.configurable
class LGBMAnchorModel:  # noqa: D
    def __init__(self, objective, params, num_boost_round=100):
        self.objective = objective
        self.params = params
        self.num_boost_round = num_boost_round
        self.model = None

    def fit(self, X, y, sample_weight=None, dataset=None):  # noqa: D
        categorical_features = [
            c
            for c, dtype in X.schema.items()
            if not dtype.is_float() and not dtype == pl.Boolean
        ]

        if "gamma" in self.params:
            self.objective = self.objective(
                self.params.pop("gamma"), categories=np.unique(dataset)
            )
        else:
            self.objective = self.objective()

        data = lgb.Dataset(
            X.to_arrow(),
            label=y,
            weight=sample_weight,
            categorical_feature=categorical_features,
            free_raw_data=False,
            feature_name=X.columns,
        )
        data.anchor = dataset

        self.params["objective"] = self.objective.objective

        self.model = lgb.train(
            params=self.params,
            train_set=data,
            num_boost_round=self.num_boost_round,
        )
        return self

    def predict(self, X, num_iteration=-1):  # noqa: D
        if self.model is None:
            raise ValueError("Model not fitted")
        scores = self.model.predict(X.to_arrow(), num_iteration=num_iteration)
        return self.objective.predictions(scores)

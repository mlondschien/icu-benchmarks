import copy

import gin
import lightgbm as lgb
import numpy as np
import polars as pl
import scipy
import tabmat
from glum import GeneralizedLinearRegressor
from ivmodels.utils import proj
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


@gin.configurable
class GeneralizedLinearModel(GeneralizedLinearRegressor):
    """GeneralizedLinearRegressor that can be fit on pl.DataFrames and constant y's."""

    # All parameters are copied from the GLR estimator. They need to be explicit to
    # adhere to sklearn's API. GLR inherits from BaseEstimator.
    def __init__(
        self,
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

    def fit(
        self, X, y, sample_weight=None, Z=None, offset=None, X_proj=None, y_proj=None
    ):
        """
        Fit method that can handle constant y's.

        If y is constant and self.family='binomial', the "optimal" parameters would be
        given via intercept=+-inf. Instead, we treat this problem as if there was an
        additional observation with the opposite label.
        """
        if isinstance(X, pl.DataFrame):
            X = tabmat.from_df(X)

        if self.family == "binomial" and len(np.unique(y)) == 1:
            # If there is no variation in y, "fit" intercept as if there were len(y)
            # observations with label y[0] and 1 observation with opposite label.
            # The model then predicts the probability len(y) / (len(y) + 1) for label
            # y[0] and 1 / (len(y) + 1) for the opposite label.
            self.intercept_ = -(1 - 2 * y[0]) * np.log(len(y))
            self.intercept_path_ = [self.intercept_] * len(self.alpha)
            self.coef_path_ = [np.zeros(X.shape[1]) for _ in range(len(self.alpha))]
            self.coef_ = self.coef_path_[-1]  # coef_ is checked for by _is_fitted
            self._alphas = self.alpha  # for when using predict(alpha=...)
            self.n_features_in_ = X.shape[1]  # this is used in GLR.linear_predictor
            return self

        return super().fit(X, y, sample_weight=sample_weight, offset=offset)

    def predict(self, X, **kwargs):
        """Predict, allowing for a polars.DataFrame as `X` input."""
        if isinstance(X, pl.DataFrame):
            X = tabmat.from_df(X)
        return super().predict(X, **kwargs)


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
        if isinstance(dataset, pl.Series):
            dataset = dataset.to_numpy()

        self.fit_datasets_ = np.sort(np.unique(dataset))

        if isinstance(X, np.ndarray):
            means = [X[dataset == d].mean(axis=0) for d in self.fit_datasets_]
            X_interacted = np.hstack(
                [X]
                + [(dataset == d).reshape(-1, 1) for d in self.fit_datasets_]
                + [
                    (X - means[idx][np.newaxis, :]) * (dataset == d)[:, np.newaxis]
                    for idx, d in enumerate(self.fit_datasets_)
                ]
            )
        elif isinstance(X, pl.DataFrame):
            means = [X.filter(dataset == d).mean() for d in self.fit_datasets_]
            X_interacted = X.with_columns(_dataset=dataset)
            X_interacted = X_interacted.with_columns(
                [
                    pl.col("_dataset").eq(d).alias(f"_dataset={d}")
                    for d in self.fit_datasets_
                ]
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
            eff_sample_size = np.sum(sample_weight) ** 2 / np.sum(sample_weight**2)
            rg = [
                np.sqrt(eff_sample_size / np.sum(dataset == d))
                * np.sum(sample_weight[dataset == d])
                for d in self.fit_datasets_
            ]

        # extra 1 for dataset-specific intercepts
        self.P1 = np.repeat([1] + [self.ratio * r for r in rg], X.shape[1] + 1)[1:]
        self.P2 = np.repeat([1] + [self.ratio * r for r in rg], X.shape[1] + 1)[1:]

        # Need to convert to tabmat here. Else, the feature names are not set correctly.
        if isinstance(X_interacted, pl.DataFrame):
            X_interacted = tabmat.from_df(X_interacted)

        super().fit(X_interacted, y, sample_weight=sample_weight)

        self.coef_path_ = [x[: X.shape[1]] for x in self.coef_path_]
        self.coef_ = self.coef_path_[: X.shape[1]]
        self.n_features_in_ = X.shape[1]

        return self

    def linear_predictor(self, X, **kwargs):  # noqa D
        # convert to tabmat here as we did so in fit
        if isinstance(X, pl.DataFrame):
            X = tabmat.from_df(X)

        return super().linear_predictor(X, **kwargs)


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
        exogenous_regex=None,
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
        self.exogenous_regex = exogenous_regex

    def fit(self, X, y, sample_weight=None, Z=None, X_proj=None, y_proj=None, **kwargs):
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
        anchor : np.ndarray
            Anchors.
        """
        if self.gamma == 1:
            if isinstance(X, pl.DataFrame):
                X = tabmat.from_df(X)

            super().fit(X, y, sample_weight=sample_weight)
            return self

        # 1 - kappa = 1 - (gamma - 1) / gamma = 1 / gamma
        mult = np.sqrt(1 / self.gamma)

        if not isinstance(y, np.ndarray):
            y = y.to_numpy()

        if isinstance(X, pl.DataFrame):
            Xt = X.to_numpy()
        else:
            Xt = X

        y_mean = y.mean()
        yt = y - y_mean

        x_mean = Xt.mean(axis=0)
        Xt = Xt - x_mean

        if X_proj is None:
            X_proj = proj(Z, Xt)
        if y_proj is None:
            y_proj = proj(Z, yt)

        Xt = mult * Xt + (1 - mult) * X_proj
        yt = mult * yt + (1 - mult) * y_proj

        Xt = Xt + x_mean
        yt = yt + y_mean

        if isinstance(X, pl.DataFrame):  # glum does not support polars yet, tabmat does
            Xt = tabmat.from_df(pl.DataFrame(Xt, schema=X.columns))

        super().fit(Xt, yt, sample_weight=sample_weight)

        return self

    def predict(self, X, **kwargs):  # noqa: D
        # convert to tabmat here as we did so in fit
        if isinstance(X, pl.DataFrame):
            X = tabmat.from_df(X)

        return super().predict(X, **kwargs)


@gin.configurable
class AnticausalAnchorRegression(GeneralizedLinearRegressor):  # noqa D
    # All parameters are copied from the GLM estimator. They need to be explicit to
    # adhere to sklearn's API. GLR inherits from BaseEstimator.
    def __init__(
        self,
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
        l2_ratio=0.0,
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
        self.l2_ratio = l2_ratio

    def fit(self, X, y, sample_weight=None, dataset=None):  # noqa: D
        """
        Fit the anticausal Anchor Regression model.

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
        if sample_weight is not None:
            raise ValueError

        if isinstance(dataset, np.ndarray):
            dataset = pl.from_numpy(dataset, schema=["__group"])
        elif isinstance(dataset, pl.Series):
            dataset = dataset.alias("__group")
        else:
            raise ValueError

        X_mean = X.mean()
        X = X.with_columns(pl.col(col).sub(X_mean[col]) for col in X.columns)
        X_proj = X.group_by(dataset).agg(pl.all().mean(), pl.len().alias("__weight"))
        cov = np.cov(
            X_proj.drop(["__group", "__weight"]).to_numpy().T,
            fweights=X_proj["__weight"],
        )
        del X_proj

        D, V = scipy.linalg.eigh(cov)
        D[D < 1e-10] = (
            1e-10  # D might have some negative values. 1e-10 for num. accuracy
        )
        D += 100 * self.l2_ratio

        X = X.to_numpy() @ V

        self.P2 = 100 * D

        super().fit(X, y)

        X_mean = X_mean.to_numpy()
        self.coef_path_ = [V @ c for c in self.coef_path_]
        self.coef_ = self.coef_path_[-1]
        self.intercept_path_ = [
            i - X_mean @ c for i, c in zip(self.intercept_path_, self.coef_path_)
        ]
        self.intercept_ = self.intercept_path_[-1]

        return self

    def predict(self, X, **kwargs):  # noqa: D
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()

        return super().predict(X, **kwargs)


@gin.configurable
class LGBMAnchorModel(BaseEstimator):
    """
    LightGBM Model with Anchor Loss.

    Parameters
    ----------
    objective : objective of anchorboosting
        Needs to have an `objective` and a `score` method.
    params : dict
        Parameters for the LightGBM model.
    num_boost_round : int
        Number of boosting rounds.
    """

    def __init__(
        self,
        objective,
        num_leaves=31,
        learning_rate=0.1,
        gamma=1,
        seed=0,
        deterministic=True,
        num_boost_round=100,
        n_components=None,
        **kwargs,
    ):
        self.objective = objective
        self.gamma = gamma
        self.params = {
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "seed": seed,
            "deterministic": deterministic,
            **kwargs,
        }
        self.num_boost_round = num_boost_round
        self.booster = None
        self.n_components = n_components

    def fit(self, X, y, sample_weight=None, Z=None, categories=None):
        """
        Fit the model.

        Parameters
        ----------
        X : polars.DataFrame
            The input data.
        y : np.ndarray
            The outcome.
        sample_weight : np.ndarray, optional
            The sample weights.
        anchor : np.ndarray
            Array of dataset indicators. Each unique value is assumed to correspond to a
            single environment. The anchor then is a one-hot encoding of this.
        """
        # if isinstance(dataset, pl.Series):
        #     dataset = dataset.to_numpy()

        categorical_features = [c for c in X.columns if "categorical" in c]

        dataset_params = {
            "data": X.to_arrow(),
            "label": y,
            "weight": sample_weight,
            "categorical_feature": categorical_features,
            "free_raw_data": False,
            "feature_name": X.columns,
        }

        if isinstance(self.objective, str):
            self.params["objective"] = self.objective
            if self.objective == "regression":
                self.init_score_ = np.mean(y)
            elif self.objective == "binary":
                p = min(1 - 1e-6, max(1e-6, np.sum(y) / len(y)))
                self.init_score_ = np.log(p / (1 - p))
            else:
                raise ValueError(f"Unknown objective: {self.objective}")
            dataset_params["init_score"] = np.ones(len(y)) * self.init_score_
        else:
            if self.n_components is not None:
                self.objective = self.objective(
                    self.gamma,
                    categories=categories,
                    n_components=self.n_components,
                )
            else:
                self.objective = self.objective(
                    self.gamma,
                    categories=categories,
                )
            self.params["objective"] = self.objective.objective
            dataset_params["init_score"] = self.objective.init_score(y)
            self.init_score_ = dataset_params["init_score"][0]

        data = lgb.Dataset(**dataset_params)
        data.anchor = Z

        self.booster = lgb.train(
            params=self.params,
            train_set=data,
            num_boost_round=self.num_boost_round,
        )
        return self

    def predict(self, X, num_iteration=-1):
        """
        Predict the outcome.

        Parameters
        ----------
        X : polars.DataFrame or pyarrow.Table
            The input data.
        num_iteration : int
            Number of boosting iterations to use. If -1, all are used. Else, needs to be
            in [0, num_boost_round].
        """
        if self.booster is None:
            raise ValueError("Booster not fitted")

        if isinstance(X, pl.DataFrame):
            X = X.to_arrow()

        scores = self.booster.predict(X, num_iteration=num_iteration, raw_score=True)

        if hasattr(self.objective, "predictions"):
            return self.objective.predictions(scores + self.init_score_)
        elif self.objective == "binary":
            return 1 / (1 + np.exp(-scores - self.init_score_))
        else:
            return scores + self.init_score_


class CVMixin:
    """Mixin adding a `fit_predict_cv` method."""

    def __init__(self, cv=5, **kwargs):
        super().__init__(**kwargs)
        self.cv = cv

    def fit_predict_cv(self, X, y, groups=None, predict_kwargs=None):
        """
        Fit the model on the training data and predict on the validation data.

        Parameters
        ----------
        X : tabmat.BaseMatrix
            Data to train and predict on.
        y : np.ndarray
            Outcome.
        groups : np.ndarray, optional
            Group indicators for cross-validation.
        predict_kwargs : List of dict, optional
            Additional arguments for the prediction. `predict(X, key1=value1, ...)` will
            be called for each dict `{key1: value1, ...}` in the list `predict_kwargs`.
            `predict_kwargs=None` is thus equivalent to `predict_kwargs=[{}]`.

        Returns
        -------
        yhat : np.ndarray of shape n_samples, len(predict_kwargs)
            Predictions for each set of arguments in `predict_kwargs`.
        """
        if predict_kwargs is None:
            predict_kwargs = [{}]

        yhat = np.zeros((len(y), len(predict_kwargs)), dtype=np.float64)

        for train_idx, val_idx in GroupKFold(n_splits=self.cv).split(X, y, groups):
            X_train, y_train, X_val = X[train_idx], y[train_idx], X[val_idx]
            self.fit(X_train, y_train)

            yhat[val_idx, :] = self.predict_with_kwargs(X_val, predict_kwargs)

        self.fit(X, y)

        return yhat

    def predict_with_kwargs(self, X, predict_kwargs=None):
        """
        Predict the outcome for each kwargs in predict_kwargs.

        Parameters
        ----------
        X : tabmat.BaseMatrix
            Data to predict on.
        predict_kwargs : List of dict, optional
            Additional arguments for the prediction. `predict(X, key1=value1, ...)` will
            be called for each dict `{key1: value1, ...}` in the list `predict_kwargs`.
            `predict_kwargs=None` is thus equivalent to `predict_kwargs=[{}]`.

        Returns
        -------
        yhat : np.ndarray of shape n_samples, len(predict_kwargs)
            Predictions for each set of arguments in `predict_kwargs`.
        """
        if predict_kwargs is None:
            predict_kwargs = [{}]

        yhat = np.zeros((X.shape[0], len(predict_kwargs)), dtype=np.float64)
        for idx, predict_kwarg in enumerate(predict_kwargs):
            yhat[:, idx] = self.predict(X, **predict_kwarg)

        return yhat


@gin.configurable
class EmpiricalBayesCV(CVMixin, GeneralizedLinearModel):
    """
    Empirical Bayes elastic net Regression with prior around beta_prior.

    This optimizes
    1 / n_train ||y - X beta||^2 + l1_ratio * alpha || beta - beta_prior ||_2^2 +
    (1 - l1_ratio) * alpha || beta - beta_prior ||_1
    over `beta`.

    Parameters
    ----------
    alpha : float
        Regularization parameter.
    prior : glum.GeneralizedLinearRegressor
        Prior.
    P2 : np.array
        Standard deviation of beta for each feature.
    fit_intercept : bool
        Whether to fit an intercept.
    cv : int
        Number of folds for cross-validation.

    """

    def __init__(
        self,
        prior=None,
        prior_alpha=None,
        cv=5,
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
            cv=cv,
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
        self.prior = prior
        if prior_alpha is not None:
            # https://github.com/Quantco/glum/blob/b471522c611b263c00ae841fd0f46660c31a\
            # 6d5f/src/glum/_glm.py#L1297
            isclose = np.isclose(self.prior._alphas, prior_alpha, atol=1e-12)
            if np.sum(isclose) == 1:
                prior_alpha_idx = np.argmax(isclose)  # cf. stackoverflow.com/a/61117770
            else:
                raise ValueError(f"Could not get index for prior_alpha {prior_alpha}.")

            self.prior.intercept_ = self.prior.intercept_path_[prior_alpha_idx]
            self.prior.coef_ = self.prior.coef_path_[prior_alpha_idx]

    def fit(self, X, y, sample_weights=None, dataset=None):  # noqa D
        offset = self.prior.linear_predictor(X)

        super().fit(X, y, offset=offset)

        return self

    def predict(self, X, **kwargs):  # noqa D
        if "offset" not in kwargs:
            kwargs["offset"] = self.prior.linear_predictor(X)

        return super().predict(X, **kwargs)

    def predict_with_kwargs(self, X, predict_kwargs=None):
        """
        Predict the outcome for each kwarg in predict_kwargs.

        This overwrites the `predict_with_kwargs` method from `CVMixin` to only compute
        offset once.

        Parameters
        ----------
        X : tabmat.BaseMatrix
            Data to predict on.
        predict_kwargs : List of dict, optional
            Additional arguments for the prediction. `predict(X, key1=value1, ...)` will
            be called for each dict `{key1: value1, ...}` in the list `predict_kwargs`.
            `predict_kwargs=None` is thus equivalent to `predict_kwargs=[{}]`.

        Returns
        -------
        yhat : np.ndarray of shape n_samples, len(predict_kwargs)
            Predictions for each set of arguments in `predict_kwargs`.
        """
        if predict_kwargs is None:
            predict_kwargs = [{}]

        offset = self.prior.linear_predictor(X)
        yhat = np.zeros((X.shape[0], len(predict_kwargs)), dtype=np.float64)
        for idx, predict_kwarg in enumerate(predict_kwargs):
            yhat[:, idx] = self.predict(X, offset=offset, **predict_kwarg)

        return yhat


@gin.configurable
class RefitLGBMModelCV(CVMixin, BaseEstimator):
    """
    LGBM Model that gets refit on new data.

    Parameters
    ----------
    prior : LGBMAnchorModel
        Prior model with attribute `booster`.
    decay_rate : float
        Decay rate for refitting. If `decay_rate=1`, the new data is ignored.
    objective : str
        Objective for the refit. Either "regression" or "binary".
    cv : int
        Number of folds for cross-validation.
    """

    def __init__(self, prior=None, decay_rate=0.5, objective=None, cv=5):
        super().__init__(cv=cv)
        self.prior = prior
        self.decay_rate = decay_rate
        self.objective = objective

    def fit(self, X, y):  # noqa D
        self.model = copy.deepcopy(self.prior)
        # For some reason, the model params are not copied over.
        # https://github.com/microsoft/LightGBM/issues/6821
        self.model.params = copy.deepcopy(self.prior.params)

        self.model.booster.params["num_threads"] = 1  # outer loop go brrrr
        self.model.booster.params["force_col_wise"] = True  # should be redundant
        self.model.booster.params["objective"] = self.objective

        if self.decay_rate < 1:
            self.model = self.model.refit(
                X=X,
                y=y,
                decay_rate=self.decay_rate,
            )
        return self

    def predict(self, X, **kwargs):  # noqa D
        return self.model.predict(X, **kwargs)


@gin.configurable
class RefitInterceptModelCV(CVMixin):
    """Model that refits the intercept on new data."""

    def __init__(self, prior=None, cv=5):
        super().__init__(cv=cv)
        self.prior = prior
        self.offset = 0

    def fit(self, X, y, sample_weight=None):
        """Compute by how much the prior's intercept needs to be adjusted."""
        self.offset = y.mean() - self.prior.predict(X).mean()
        return self

    def predict(self, X, **kwargs):
        """Return prior's predictions, adjusted by the offset."""
        return self.prior.predict(X, **kwargs) + self.offset


@gin.configurable
class PriorPassthroughCV(CVMixin):
    """Model that simply uses the prior for predictions."""

    def __init__(self, prior, cv=5):
        super().__init__(cv=cv)
        self.prior = prior

    def fit(self, X, y, sample_weight=None):
        """Do nothing."""
        return self

    def predict(self, X, **kwargs):
        """Predict using the prior."""
        return self.prior.predict(X, **kwargs)

    def fit_predict_cv(self, X, y, groups=None, predict_kwargs=None):
        """
        Return predictions of `prior` for X.

        Parameters
        ----------
        X : tabmat.BaseMatrix
            Data to train and predict on.
        y : np.ndarray
            Not used.
        groups : np.ndarray, optional
            Not used.
        predict_kwargs : List of dict, optional
            Additional arguments for the prediction. `predict(X, key1=value1, ...)` will
            be called for each dict `{key1: value1, ...}` in the list `predict_kwargs`.
            `predict_kwargs=None` is thus equivalent to `predict_kwargs=[{}]`.

        Returns
        -------
        yhat : np.ndarray of shape n_samples, len(predict_kwargs)
            Predictions for each set of arguments in `predict_kwargs`.
        """
        if predict_kwargs is None:
            predict_kwargs = [{}]

        return self.predict_with_kwargs(X, predict_kwargs)


@gin.configurable
class PipelineCV(CVMixin, Pipeline):  # noqa D
    def __init__(self, steps, cv=5):
        super().__init__(cv=cv, steps=steps)


@gin.configurable
class FFill:
    """Predict an outcome by forward filling the last value."""

    def __init__(self, outcome=None):  # noqa D
        self.outcome = outcome
        self.mean = None

    def fit(self, X, y=None, **kwargs):  # noqa: D
        self.mean = np.mean(y)
        return self

    def predict(self, X, **kwargs):  # noqa: D
        if self.outcome == "log_lactate_in_4h":
            yhat = X["continuous__log_lact_ffilled"].to_numpy()

        yhat[np.isnan(yhat)] = self.mean
        return yhat

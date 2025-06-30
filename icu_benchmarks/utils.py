from __future__ import annotations

import warnings

import formulaic
import numpy as np
import scipy
from scipy.interpolate import BSpline
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder


class MaxIterationWarning(UserWarning): ...  # noqa D


def fit_monotonic_spline(
    x,
    y,
    x_new,
    bspline_degree=2,
    knot_frequency=0.5,
    lambda_smoothing=0.0001,
    kappa_penalty=10**6,
    maxiter: int = 30,
    increasing: bool = True,
) -> np.ndarray:
    """
    Fit a monotonic spline.

    From https://github.com/fohrloop/penalized-splines/blob/main/penalized_splines.py

    Parameters
    ----------
    x: array-like
        The x-coordinates of the training data.
    y: array-like
        The y-coordinates of the training data.
    x_new: array-like
        The x-coordinates of the test data.
    bspline_degree: int
        The degree of the B-spline (which is also the degree of the fitted spline
        function). The order of the splines is degree + 1.
    knot_frequency: float
        Number of knots per training data point.
    lambda_smoothing: float
        The smoothing parameter. Higher values will result in smoother curves.
    kappa_penalty: float
        The penalty parameter for enforcing monotonicity. Higher values will result in
        more monotonic curves. kappa_penalty of 0 means that monotonicity is not
        enforced at all.
    maxiter: int
        Maximum number of iterations for the algorithm. If the algorithm does not
        converge within this number of iterations, a warning is issued.
    increasing: bool
        Whether the fitted spline should be increasing or decreasing. If True, the
        fitted spline will be increasing. If False, the fitted spline will be
        decreasing.
    """
    if not increasing:
        return -fit_monotonic_spline(
            x,
            -y,
            x_new,
            bspline_degree,
            knot_frequency,
            lambda_smoothing,
            kappa_penalty,
            maxiter,
            True,
        )

    xmin, xmax = min(x), max(x)
    num_knots = int(len(x) * knot_frequency)
    knot_interval = (xmax - xmin) / num_knots

    # You need to add deg knots on each side of the interval. See, for example,
    # De Leeuw (2017) Computing and Fitting Monotone Splines
    # The basic interval is [min(x_train), max(x_train)], and
    # the extended interval is [min(knots), max(knots)].
    # You may only ask for values within the basic interval, as there are always m
    # (=deg+1) non-zero B-splines. Outside the basic interval, there are less B-splines
    # with non-zero values and the model is extrapolating.
    knots = np.linspace(
        xmin - (bspline_degree + 1) * knot_interval,
        xmax + (bspline_degree + 1) * knot_interval,
        bspline_degree * 2 + num_knots + 1,
    )
    alphas = np.ones(len(x))

    # Introduced in scipy 1.8.0
    B = BSpline.design_matrix(x=x, t=knots, k=bspline_degree).toarray()
    n_base_funcs = B.shape[1]
    I = np.eye(n_base_funcs)  # noqa: E741
    D3 = np.diff(I, n=3, axis=0)
    D1 = np.diff(I, n=1, axis=0)

    # Monotone smoothing
    V = np.zeros(n_base_funcs - 1)

    B_gram = B.T @ B
    A = B_gram + lambda_smoothing * D3.T @ D3
    BTy = B.T @ y
    for _ in range(maxiter):
        W = np.diag(V * kappa_penalty)
        # The equation
        # (B'B + λD3'D3 + κD1'VD1)α = B'y
        alphas = np.linalg.solve(A + D1.T @ W @ D1, BTy)
        V_new = (D1 @ alphas < 0) * 1
        dv = np.sum(V != V_new)
        V = V_new
        if dv == 0:
            break
    else:
        warnings.warn(
            "Max iteration reached. The results are not reliable.", MaxIterationWarning
        )

    spl = BSpline(knots, alphas, bspline_degree, extrapolate=False)

    y_new = np.empty_like(x_new)
    mask = (x_new >= xmin) & (x_new <= xmax)
    y_new[mask] = spl(x_new[mask])

    # Inspired by https://github.com/scikit-learn/scikit-learn/blob/00d3ef9f4d7e224e59f\
    # 9e01f678abb918231858f/sklearn/preprocessing/_polynomial.py#L1079-L1113
    f_min, f_max = spl(xmin), spl(xmax)  # function value
    df_min, df_max = spl(xmin, nu=1), spl(xmax, nu=1)  # first derivative

    y_new[x_new < xmin] = f_min + (x_new[x_new < xmin] - xmin) * df_min
    y_new[x_new > xmax] = f_max + (x_new[x_new > xmax] - xmax) * df_max

    return y_new


def get_model_matrix(df, formula):  # noqa: D
    if formula == "icd10_blocks only":
        return MultiLabelBinarizer().fit_transform(df["icd10_blocks"]).astype("float")

    if formula == "random10":
        rng = np.random.default_rng(0)
        return rng.choice(10, size=df.shape[0])
    elif formula == "random100":
        rng = np.random.default_rng(0)
        return rng.choice(100, size=df.shape[0])
    elif formula == "random1000":
        rng = np.random.default_rng(0)
        return rng.choice(1000, size=df.shape[0])
    elif formula == "random10000":
        rng = np.random.default_rng(0)
        return rng.choice(10000, size=df.shape[0])

    if formula == "":
        return OrdinalEncoder().fit_transform(df[["dataset"]]).astype("int32").flatten()
    elif formula == "patient_id":
        return (
            OrdinalEncoder()
            .fit_transform(
                (df["patient_id"].astype("string") + df["dataset"]).to_frame()
            )
            .astype("int32")
            .flatten()
        )

    Zs = []
    datasets = df["dataset"].unique()
    for dataset in datasets:
        mask = df["dataset"] == dataset
        if formula == "icd10_blocks":  # interact icd10 blocks with dataset
            df.loc[mask, "icd10_blocks"] = df.loc[mask, "icd10_blocks"].apply(
                lambda x: x.tolist() + [dataset]
            )
            Zs.append(
                MultiLabelBinarizer()
                .fit_transform(df[mask]["icd10_blocks"])
                .cast("float")
            )
        # icd10 blocks plus other stuff interacted with dataset
        elif "icd10_blocks" in formula:
            formula_ = formula.replace("icd10_blocks + ", "")
            df.loc[mask, "icd10_blocks"] = df.loc[mask, "icd10_blocks"].apply(
                lambda x: x.tolist() + [dataset]
            )
            Zs.append(
                np.hstack(
                    [
                        MultiLabelBinarizer().fit_transform(df[mask]["icd10_blocks"]),
                        formulaic.Formula(formula_)
                        .get_model_matrix(df[mask], na_action="ignore")
                        .to_numpy()
                        .astype("float"),
                    ]
                )
            )
        else:
            Zs.append(
                formulaic.Formula(formula)
                .get_model_matrix(df[mask], na_action="ignore")
                .to_numpy()
            )

    csum = np.cumsum([0] + [Z.shape[1] for Z in Zs])
    Z = np.zeros((df.shape[0], csum[-1]), dtype=np.float64)
    for i, (dataset, Zi) in enumerate(zip(datasets, Zs)):
        Z[df["dataset"] == dataset, csum[i] : csum[i + 1]] = Zi

    return Z


def find_intersection(x, t, increasing=True):
    """Interpolate t where first x > 0 (if increasing)."""
    mult = 1 if increasing else -1
    where = np.where(mult * x > 0)[0]
    if len(where) == 0:  # No intersection -> after last point
        return np.inf
    elif where[0] == 0:  # Intersection before first point
        return -np.inf

    interp = scipy.interpolate.interp1d(
        x=x[[where[0] - 1, where[0]]],
        y=t[[where[0] - 1, where[0]]],
        kind="linear",
    )
    return interp(0.0)

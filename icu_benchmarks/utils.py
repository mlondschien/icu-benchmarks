from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import BSpline
import scipy

class MaxIterationWarning(UserWarning): ...


def fit_monotonic_spline(
    x,
    y,
    x_new,
    bspline_degree=3,
    knot_segments=10,
    lambda_smoothing=0.1,
    kappa_penalty=10**6,
    maxiter: int = 30,
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
    knot_segments: int
        number of inter-knot segments between min(x_train) and max(x_train). Defines the
        number of internal knots, which is knot_segments + 1. The total amount of knots
        is calculated as `bspline_degree * 2 + knot_segments + 1`; Few knots will be
        added outside of the `x_train` range to fix boundary effects.
    lambda_smoothing: float
        The smoothing parameter. Higher values will result in smoother curves.
    kappa_penalty: float
        The penalty parameter for enforcing monotonicity. Higher values will result in
        more monotonic curves. kappa_penalty of 0 means that monotonicity is not
        enforced at all.
    maxiter: int
        Maximum number of iterations for the algorithm. If the algorithm does not
        converge within this number of iterations, a warning is issued.
    """
    xmin, xmax = min(x), max(x)
    knot_interval = (xmax - xmin) / knot_segments

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
        bspline_degree * 2 + knot_segments + 1,
    )
    alphas = np.ones(len(x))

    # Introduced in scipy 1.8.0
    B = BSpline.design_matrix(x=x, t=knots, k=bspline_degree).toarray()
    n_base_funcs = B.shape[1]
    I = np.eye(n_base_funcs)
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

    y_new = np.empty_like(x)
    mask = (x_new >= xmin) & (x_new <= xmax)
    y_new[mask] = spl(x_new[mask])

    # Inspired by https://github.com/scikit-learn/scikit-learn/blob/00d3ef9f4d7e224e59f\
    # 9e01f678abb918231858f/sklearn/preprocessing/_polynomial.py#L1079-L1113
    f_min, f_max = spl(xmin), spl(xmax)  # function value
    df_min, df_max = spl(xmin, nu=1), spl(xmax, nu=1)  # first derivative

    y_new[x_new < xmin] = f_min + (x_new[x_new < xmin] - xmin) * df_min
    y_new[x_new > xmax] = f_max + (x_new[x_new > xmax] - xmax) * df_max
    
    return y_new



def find_intersection(x, values, increasing):
    """Find the point where x where values intercepts the x-axis."""
    argsort = np.argsort(x)
    values = values[argsort]
    x = x[argsort]
    iso_values = scipy.optimize.isotonic_regression(values, increasing=increasing).x

    if (increasing and iso_values[0] > 0) or (not increasing and iso_values[0] < 0):
        return np.min(x)

    if (increasing and iso_values[-1] < 0) or (not increasing and iso_values[-1] > 0):
        fit = np.polynomial.polynomial.Polynomial.fit(x[:-6], values[:-6], 1)
        x = x.tolist() + [x[-1] + 10]
        iso_values = iso_values.tolist() + [fit(x[-1])]

    if (iso_values[-1] < 0 and increasing) or (iso_values[-1] > 0 and not increasing):
        return np.nan

    interp = scipy.interpolate.interp1d(x=iso_values, y=x, kind="linear")
    return interp(0.0)


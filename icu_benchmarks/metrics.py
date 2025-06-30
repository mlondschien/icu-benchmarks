import numpy as np
import scipy
from anchorboosting.models import Proj
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from icu_benchmarks.constants import GREATER_IS_BETTER


def metrics(y, yhat, prefix, task, Z=None, n_categories=None):  # noqa D
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()
    if not isinstance(yhat, np.ndarray):
        yhat = yhat.to_numpy()

    y = y.flatten()
    yhat = yhat.flatten()

    if task == "binary" and (yhat.max() > 1 or yhat.min()) < 0:
        if False:
            raise ValueError("Predictions need to be in [0, 1] for binary task")
        else:
            yhat = yhat.clip(1e-14, 1 - 1e-14)

    score_residuals = y - yhat
    q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    quantiles = np.quantile(score_residuals, q)

    out = {}
    for i, q in enumerate(q):
        out[f"{prefix}quantile_{q}"] = quantiles[i]
    out[f"{prefix}mean_residual"] = np.mean(score_residuals)

    if task == "binary":
        yhat = yhat.clip(1e-14, 1 - 1e-14)
        log_losses = np.where(y == 1, -np.log(yhat), -np.log(1 - yhat))
        if np.unique(y).size > 1:
            out[f"{prefix}roc"] = roc_auc_score(y, yhat)
            out[f"{prefix}accuracy"] = accuracy_score(y, yhat >= 0.5)
            out[f"{prefix}log_loss"] = np.mean(log_losses)
            out[f"{prefix}auprc"] = average_precision_score(y, yhat)
            out[f"{prefix}brier"] = np.mean(score_residuals**2)
        else:
            out[f"{prefix}roc"] = np.nan
            out[f"{prefix}accuracy"] = np.nan
            out[f"{prefix}log_loss"] = np.nan
            out[f"{prefix}auprc"] = np.nan
            out[f"{prefix}brier"] = np.nan

        q = [0.8, 0.9, 0.95]
        losses_q = np.quantile(log_losses, q)
        for i, q in enumerate(q):
            out[f"{prefix}log_loss_quantile_{q}"] = losses_q[i]

    elif task == "regression":
        abs_residuals = np.abs(score_residuals)
        q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        abs_quantiles = np.quantile(abs_residuals, q)
        se = score_residuals**2
        mse = np.mean(se)

        out[f"{prefix}mse"] = mse
        out[f"{prefix}rmse"] = np.sqrt(mse)
        out[f"{prefix}mae"] = np.mean(abs_residuals)

        for i, q in enumerate(q):
            out[f"{prefix}abs_quantile_{q}"] = abs_quantiles[i]
            out[f"{prefix}mse_quantile_{q}"] = abs_quantiles[i] ** 2

    if Z is not None:
        proj = Proj(Z)
        if task == "binary":
            logloss_proj = proj(log_losses)
            q = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            logloss_quantiles = np.quantile(logloss_proj, q)
            for i, q in enumerate(q):
                out[f"{prefix}proj_logloss_quantile_{q}"] = logloss_quantiles[i]
            out[f"{prefix}proj_logloss"] = np.mean(logloss_proj)

        residuals_proj = proj(score_residuals)
        errors = residuals_proj**2
        q = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        error_quantiles = np.quantile(errors, q)
        for i, q in enumerate(q):
            out[f"{prefix}proj_residuals_sq_quantile_{q}"] = error_quantiles[i]

        out[f"{prefix}proj_residuals_sq"] = np.mean(errors)

    return out


def get_equivalent_number_of_samples(n_samples, values, metric):
    """
    Get the number of target samples needed to achieve a certain value of metric.

    First, fit an isotonic (monotone) regression through the points
    `test_value ~ n_target`. Then, linearly interpolate `fitted values ~ log(n_target)`.
    Extrapolate beyond `max(fitted_values)` and `min(fitted_values)` with
    `max(n_target)` and `min(n_target)`, respectively (opposite if greater is not
    better).

    Parameters
    ----------
    n_samples: pl.DataFrame
        With columns `"n_target"` and `"test_value"`.
    values: np.ndarray
        Values for which to estimate the number of samples of the test distribution to
        achieve the same value of the metric.
    metric: str
        Name of the metric. Used to determine whether greater is better.
    """
    n_samples = n_samples.sort(by="n_target")
    isotonic_regression = scipy.optimize.isotonic_regression(
        n_samples["test_value"], increasing=metric in GREATER_IS_BETTER
    ).x

    log_n_target = np.log(n_samples["n_target"].to_numpy())
    fit1 = np.polyfit(isotonic_regression[-6:], log_n_target[-6:], 1)
    fit2 = np.polyfit(isotonic_regression[:6], log_n_target[:6], 1)
    y = [log_n_target[0] - 4] + log_n_target.tolist() + [log_n_target[-1] + 4]
    x = (
        [y[0] * fit2[0] + fit2[1]]
        + isotonic_regression.tolist()
        + [y[-1] * fit1[0] + fit1[1]]
    )

    interp = scipy.interpolate.interp1d(
        x=x,
        y=y,
        kind="linear",
    )
    return np.exp(interp(values))

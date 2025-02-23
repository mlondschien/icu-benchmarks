import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    roc_auc_score,
)
import scipy
from icu_benchmarks.constants import GREATER_IS_BETTER

def metrics(y, yhat, prefix, task, groups=None):  # noqa D
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()
    if not isinstance(yhat, np.ndarray):
        yhat = yhat.to_numpy()

    y = y.flatten()
    yhat = yhat.flatten()

    if task == "binary":
        return {
            f"{prefix}roc": roc_auc_score(y, yhat) if np.unique(y).size > 1 else 0.0,
            f"{prefix}accuracy": (
                accuracy_score(y, yhat >= 0.5) if np.unique(y).size > 1 else 0.0
            ),
            f"{prefix}log_loss": log_loss(y, yhat) if np.unique(y).size > 1 else np.inf,
            f"{prefix}auprc": (
                average_precision_score(y, yhat) if np.unique(y).size > 1 else 0.0
            ),
            f"{prefix}brier": (
                np.mean((y - yhat) ** 2) if np.unique(y).size > 1 else np.inf
            ),
        }
    elif task == "regression":
        residuals = y - yhat
        quantiles = np.quantile(residuals, [0.1, 0.25, 0.5, 0.75, 0.9])
        abs_residuals = np.abs(residuals)
        abs_quantiles = np.quantile(abs_residuals, [0.8, 0.9, 0.95])
        mse = np.mean(abs_residuals**2)
        out = {
            f"{prefix}mse": mse,
            f"{prefix}rmse": np.sqrt(mse),
            f"{prefix}mae": np.mean(abs_residuals),
            f"{prefix}abs_quantile_0.8": abs_quantiles[0],
            f"{prefix}abs_quantile_0.9": abs_quantiles[1],
            f"{prefix}abs_quantile_0.95": abs_quantiles[2],
            f"{prefix}quantile_0.1": quantiles[0],
            f"{prefix}quantile_0.25": quantiles[1],
            f"{prefix}quantile_0.5": quantiles[2],
            f"{prefix}quantile_0.75": quantiles[3],
            f"{prefix}quantile_0.9": quantiles[4],
            f"{prefix}mean_residual": np.mean(residuals),
        }

        if groups is not None:
            grouped_mses = map(np.mean, np.split(mse, np.unique(groups, return_index=True)[1][1:]))
            mse_quantiles = np.quantile(grouped_mses, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
            out["grouped_mse_quantile_0.5"] = mse_quantiles[0]
            out["grouped_mse_quantile_0.6"] = mse_quantiles[1]
            out["grouped_mse_quantile_0.7"] = mse_quantiles[2]
            out["grouped_mse_quantile_0.8"] = mse_quantiles[3]
            out["grouped_mse_quantile_0.9"] = mse_quantiles[4]
            out["grouped_mse_quantile_0.95"] = mse_quantiles[5]
        
        return out

    else:
        raise ValueError(f"Unknown task {task}")


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
    isotonic_regression = scipy.optimize.isotonic_regression(n_samples["test_value"], increasing=metric in GREATER_IS_BETTER).x
    min_, max_ = np.log(n_samples["n_target"].min()), np.log(n_samples["n_target"].max())
    interp = scipy.interpolate.interp1d(
        x=isotonic_regression,
        y=np.log(n_samples["n_target"]),
        kind="linear",
        fill_value=(min_, max_) if metric in GREATER_IS_BETTER else (max_, min_),
        bounds_error=False
    )
    return np.exp(interp(values))
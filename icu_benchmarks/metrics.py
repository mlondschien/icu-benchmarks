import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    roc_auc_score,
)


def metrics(y, yhat, prefix, task):  # noqa D
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
        return {
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
    else:
        raise ValueError(f"Unknown task {task}")

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    r2_score,
    roc_auc_score,
)


def metrics(y, yhat, prefix, task):  # noqa D
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()
    if not isinstance(yhat, np.ndarray):
        yhat = yhat.to_numpy()

    y = y.flatten()
    yhat = yhat.flatten()

    if task == "classification":
        return {
            f"{prefix}roc": roc_auc_score(y, yhat) if np.unique(y).size > 1 else 0,
            f"{prefix}accuracy": (
                accuracy_score(y, yhat >= 0.5) if np.unique(y).size > 1 else 0
            ),
            f"{prefix}log_loss": log_loss(y, yhat) if np.unique(y).size > 1 else 0,
            f"{prefix}auprc": (
                average_precision_score(y, yhat) if np.unique(y).size > 1 else 0
            ),
            f"{prefix}brier": np.mean((y - yhat) ** 2) if np.unique(y).size > 1 else 0,
        }
    elif task == "regression":
        return {
            f"{prefix}r2": r2_score(y, yhat),
            f"{prefix}mse": np.mean((y - yhat) ** 2),
            f"{prefix}mae": np.mean(np.abs(y - yhat)),
            f"{prefix}quantile_0.8": np.quantile(np.abs(y - yhat), 0.8),
            f"{prefix}quantile_0.9": np.quantile(np.abs(y - yhat), 0.9),
            f"{prefix}quantile_0.95": np.quantile(np.abs(y - yhat), 0.95),
        }
    else:
        raise ValueError(f"Unknown task {task}")

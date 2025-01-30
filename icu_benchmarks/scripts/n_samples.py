
import logging
from time import perf_counter
import polars as pl
import click
import gin
import mlflow
import numpy as np
import pandas as pd
import tabmat
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from icu_benchmarks.constants import TASKS
from line_profiler import profile
from icu_benchmarks.gin import GeneralizedLinearRegressor
from icu_benchmarks.load import load
from icu_benchmarks.mlflow_utils import log_df, setup_mlflow

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(thread)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@gin.configurable
def sources(sources=gin.REQUIRED):  # noqa D
    return sources


@gin.configurable
def outcome(outcome=gin.REQUIRED):  # noqa D
    return outcome


@gin.configurable
def targets(targets=gin.REQUIRED):  # noqa D
    return targets


@gin.configurable
def l1_ratios(l1_ratios=gin.REQUIRED):  # noqa D
    return l1_ratios

@gin.configurable
def treatment_detail_level(treatment_detail_level=gin.REQUIRED):  # noqa D
    return treatment_detail_level


@profile
def metrics(y, yhat, prefix, task):  # noqa D
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()
    if not isinstance(yhat, np.ndarray):
        yhat = yhat.to_numpy()
    
    y = y.flatten()
    yhat = yhat.flatten()

    if task == "classification":
        return {
            f"{prefix}/roc": roc_auc_score(y, yhat) if np.unique(y).size > 1 else 0,
            f"{prefix}/accuracy": (
                accuracy_score(y, yhat >= 0.5) if np.unique(y).size > 1 else 0
            ),
            f"{prefix}/log_loss": log_loss(y, yhat) if np.unique(y).size > 1 else 0,
            f"{prefix}/average_prc": (
                average_precision_score(y, yhat) if np.unique(y).size > 1 else 0
            ),
            f"{prefix}/auprc": (
                auc(*precision_recall_curve(y, yhat)[1::-1]) if np.unique(y).size > 1 else 0
            ),
            f"{prefix}/brier": np.mean((y - yhat) ** 2) if np.unique(y).size > 1 else 0,
        }
    elif task == "regression":
        return {
            f"{prefix}/r2": np.corrcoef(y, yhat)[0, 1] ** 2,
            f"{prefix}/rmse": np.sqrt(np.mean((y - yhat) ** 2)),
            f"{prefix}/mae": np.mean(np.abs(y - yhat)),
        }
    else:
        raise ValueError(f"Unknown task {task}")

@click.command()
@click.option("--config", type=click.Path(exists=True))
@profile
def main(config: str):  # noqa D
    gin.parse_config_file(config)
    task = TASKS[outcome()]

    tags = {
        "outcome": outcome(),
        "sources": sources(),
        "targets": targets(),
        "l1_ratios": l1_ratios(),
        "parent_run": None,
    }

    n_seeds = 10
    n_n_patients = 16

    parent_run = setup_mlflow(tags=tags)

    df_train, y_train, _ = load(sources=sources(), outcome=outcome(), split="train", treatment_detail_level=treatment_detail_level())
    
    continuous_variables = [col for col, dtype in df_train.schema.items() if dtype.is_float()]
    other = [col for col in df_train.columns if col not in continuous_variables]
    
    df_train = df_train.with_columns(outcome=y_train)

    df_val, y_val, _ = load(sources=sources(), outcome=outcome(), split="val", treatment_detail_level=treatment_detail_level())
    df_val = df_val.with_columns(outcome=y_val)

    df_test, y_test, _ = load(sources=sources(), outcome=outcome(), split="test", treatment_detail_level=treatment_detail_level())

    train_stay_ids = df_train["stay_id"]
    val_stay_ids = df_val["stay_id"]

    imputer = SimpleImputer(strategy="mean", copy=True, keep_empty_features=True)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous",
                imputer,
                continuous_variables,
            ),
            ("other", "passthrough", other),
        ],
        sparse_threshold=0,
        verbose=1,
    ).set_output(transform="polars")

    metric_names = list(metrics(y_train.astype("float"), y_train, "", task=task["task"]).keys())

    results = pd.DataFrame(index=range(n_n_patients * n_seeds), columns=["n_patients", "seed"] + metric_names)
    row = 0

    for n_patients in np.logspace(2, 5, n_n_patients):
        logger.info(f"n_patients: {n_patients}\n")
        if n_patients > len(train_stay_ids):
            continue

        n_train = int(n_patients * 0.825)
        n_val = int(n_patients * 0.175)

        for seed in range(n_seeds):
            logger.info(f"seed: {seed}\n")
            test_results = []
            rng = np.random.default_rng(seed)
            train_patients = rng.choice(train_stay_ids, n_train, replace=False)
            val_patients = rng.choice(val_stay_ids, n_val, replace=False)

            df_train_sample = df_train.filter(pl.col("stay_id").is_in(train_patients))
            y_train_sample = df_train_sample["outcome"]
            df_train_sample = tabmat.from_df(preprocessor.fit_transform(df_train_sample))

            df_val_sample = df_val.filter(pl.col("stay_id").is_in(val_patients))
            y_val_sample = df_val_sample["outcome"]
            df_val_sample = tabmat.from_df(preprocessor.transform(df_val_sample))

            df_test_sample = tabmat.from_df(preprocessor.transform(df_test))

            glms = []
            for l1_ratio in l1_ratios():
                glm = GeneralizedLinearRegressor(l1_ratio=l1_ratio, family=task["family"])
                glm.fit(df_train_sample, y_train_sample)
                glms.append(glm)
            
            logger.info("validating")
            val_results = []
            for l1_ratio_idx, l1_ratio in enumerate(l1_ratios()):
                glm = glms[l1_ratio_idx]
                for alpha_idx, alpha in enumerate(glm._alphas):
                    glm.intercept_ = glm.intercept_path_[alpha_idx]
                    glm.coef_ = glm.coef_path_[alpha_idx]
                    yhat_val_sample = glm.predict(df_val_sample)
                    print("scoring")
                    val_results.append(metrics(y_val_sample, yhat_val_sample, "", task=task["task"]))
                    print("done scoring")
            
            logger.info("testing")
            for metric_name in metric_names:
                best_idx = np.argmax([result[metric_name] for result in val_results])
                best_l1_ratio_idx = best_idx // len(glm._alphas)
                best_alpha_idx = best_idx % len(glm._alphas)
                glm = glms[best_l1_ratio_idx]
                glm.intercept_ = glm.intercept_path_[best_alpha_idx]
                glm.coef_ = glm.coef_path_[best_alpha_idx]

                yhat_test = glm.predict(df_test_sample)
                print("scoring")
                test_results.append(metrics(y_test, yhat_test, "", task=task["task"])[metric_name])
                print("done scoring")

            results.iloc[row] = [n_patients, seed] + test_results
            pd.options.display.max_rows = 200
            print(results.head(row))
            row += 1
     
    log_df(results, "results.csv")

    # This needs to be at the end of the script to log all relevant information
    mlflow.log_text(gin.operative_config_str(), "config.gin")


if __name__ == "__main__":
    main()

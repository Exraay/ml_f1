from __future__ import annotations

from typing import Dict, Iterable, Tuple
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
    }


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_features: int,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_pred - y_true
    abs_err = np.abs(resid)
    n = len(y_true)
    eps = 1e-6

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = (
        1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1)
        if n > n_features + 1 else np.nan
    )

    mape = float(np.mean(abs_err / np.clip(np.abs(y_true), eps, None)) * 100)
    smape = float(np.mean(2 * abs_err / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100)
    rmspe = float(np.sqrt(np.mean((resid / np.clip(y_true, eps, None)) ** 2)) * 100)

    try:
        msle = mean_squared_log_error(
            np.clip(y_true, eps, None),
            np.clip(y_pred, eps, None),
        )
        rmsle = float(np.sqrt(msle))
    except Exception:
        msle = np.nan
        rmsle = np.nan

    bias = float(resid.mean())
    std_err = float(resid.std())
    med_err = float(np.median(resid))
    mad = float(np.median(np.abs(resid - np.median(resid))))

    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if n > 1 else np.nan
    spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")) if n > 1 else np.nan

    return {
        "n": n,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "adj_r2": adj_r2,
        "explained_variance": explained_variance_score(y_true, y_pred),
        "mape_pct": mape,
        "smape_pct": smape,
        "rmspe_pct": rmspe,
        "medae": median_absolute_error(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "msle": msle,
        "rmsle": rmsle,
        "bias_mean_error": bias,
        "median_error": med_err,
        "std_error": std_err,
        "mad_error": mad,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "p50_abs_error": float(np.percentile(abs_err, 50)),
        "p90_abs_error": float(np.percentile(abs_err, 90)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
        "p99_abs_error": float(np.percentile(abs_err, 99)),
        "within_1s_pct": float(np.mean(abs_err <= 1.0) * 100),
        "within_2s_pct": float(np.mean(abs_err <= 2.0) * 100),
        "within_5s_pct": float(np.mean(abs_err <= 5.0) * 100),
        "mean_true": float(np.mean(y_true)),
        "std_true": float(np.std(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "std_pred": float(np.std(y_pred)),
    }


def evaluate_models(
    models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    *,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    metrics_rows = []
    predictions: Dict[str, np.ndarray] = {}
    fitted_models: Dict[str, object] = {}

    iterable = list(models.items())
    log = print
    if verbose:
        try:
            from tqdm import tqdm
            iterable = tqdm(iterable, total=len(iterable), desc="Models")
            log = tqdm.write
        except Exception:  # noqa: BLE001
            pass

        log(f"Train size: {len(X_train):,} | Eval size: {len(X_eval):,}")
        log(f"Features: {X_train.shape[1]}")

    for name, model in iterable:
        if verbose:
            log(f"Training {name}...")
        start = time()
        fitted = model.fit(X_train, y_train)
        preds = fitted.predict(X_eval)
        scores = compute_metrics(y_eval, preds)
        scores["model"] = name
        metrics_rows.append(scores)
        predictions[name] = preds
        fitted_models[name] = fitted
        elapsed = time() - start

        if verbose:
            msg = f"{name} -> MAE: {scores['mae']:.4f}, R2: {scores['r2']:.4f} ({elapsed:.1f}s)"
            log(msg)
            if hasattr(fitted, "best_params_"):
                log(f"{name} best_params: {fitted.best_params_}")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("mae").reset_index(drop=True)
    return metrics_df, predictions, fitted_models

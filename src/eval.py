from __future__ import annotations

from typing import Dict, Iterable, Tuple
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
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

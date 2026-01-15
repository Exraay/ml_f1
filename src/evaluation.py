from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


def time_based_split(
    feature_df: pd.DataFrame, test_season: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = feature_df[feature_df["Season"] < test_season].copy()
    test_df = feature_df[feature_df["Season"] == test_season].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Adjust seasons or check data.")
    return train_df, test_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
    }


def train_and_score(
    models: Dict[str, Pipeline],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str = "LapTimeSeconds",
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    metrics_rows = []
    predictions: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        fitted = model.fit(X_train, y_train)
        preds = fitted.predict(X_test)
        scores = compute_metrics(y_test, preds)
        scores["model"] = name
        metrics_rows.append(scores)
        predictions[name] = preds

    metrics_df = pd.DataFrame(metrics_rows).sort_values("mae").reset_index(drop=True)
    return metrics_df, predictions


def aggregate_seed_runs(results: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate metrics over seeds. Expects dict mapping seed -> metrics_df.
    """
    combined = []
    for seed, df in results.items():
        temp = df.copy()
        temp["seed"] = seed
        combined.append(temp)
    out = pd.concat(combined, ignore_index=True)
    summary = (
        out.groupby("model")[["mae", "rmse"]]
        .agg(["mean", "std"])
        .swaplevel(axis=1)
        .sort_index(axis=1)
        .reset_index()
    )
    summary.columns = ["model", "mae_mean", "mae_std", "rmse_mean", "rmse_std"]
    return summary.sort_values("mae_mean").reset_index(drop=True)


def error_breakdown(
    test_df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    by: str,
    target_col: str = "LapTimeSeconds",
) -> pd.DataFrame:
    """
    Compute MAE per category (e.g., by='Driver' or 'EventName') for each model.
    """
    rows = []
    for model_name, preds in predictions.items():
        df = test_df.copy()
        df["pred"] = preds
        df["abs_err"] = (df[target_col] - df["pred"]).abs()
        grouped = df.groupby(by)["abs_err"].mean().reset_index()
        grouped["model"] = model_name
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

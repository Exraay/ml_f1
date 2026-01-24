from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.visualization import setup_f1_style


def chronological_split(
    feature_df: pd.DataFrame, split_season: int = 2023
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train: all seasons < split_season plus first half of split_season.
    Test: second half of split_season.
    """
    if "Season" not in feature_df.columns or "RoundNumber" not in feature_df.columns:
        raise ValueError("feature_df must include Season and RoundNumber for splitting.")

    df = feature_df.sort_values(["Season", "RoundNumber"]).reset_index(drop=True)
    season_df = df[df["Season"] == split_season]
    rounds = sorted(season_df["RoundNumber"].dropna().unique().tolist())
    if not rounds:
        raise ValueError(f"No rounds found for season {split_season}.")
    cutoff = len(rounds) // 2
    if cutoff == 0:
        raise ValueError("Not enough rounds to create a half-season split.")

    train_rounds = set(rounds[:cutoff])
    test_rounds = set(rounds[cutoff:])

    train_df = df[
        (df["Season"] < split_season)
        | ((df["Season"] == split_season) & (df["RoundNumber"].isin(train_rounds)))
    ].copy()
    test_df = df[
        (df["Season"] == split_season) & (df["RoundNumber"].isin(test_rounds))
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Check input data.")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y), 1e-6)
    return float(np.mean(np.abs((y - p) / denom)) * 100.0)


def actionable_accuracy(y_true: np.ndarray, y_pred: np.ndarray, *, threshold_s: float = 0.3) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(p - y) <= float(threshold_s)))


def directional_accuracy(
    df: pd.DataFrame,
    *,
    actual_col: str = "LapTimeSeconds",
    pred_col: str = "pred",
    group_cols: Iterable[str] = ("Season", "RoundNumber", "Driver"),
    order_col: str = "LapNumber",
) -> float:
    """
    % of times we correctly predict whether the *next lap* is faster or slower.
    """
    required = set(group_cols) | {actual_col, pred_col, order_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data = data.sort_values(list(group_cols) + [order_col]).reset_index(drop=True)

    correct = []
    for _, g in data.groupby(list(group_cols), dropna=False):
        a = g[actual_col].astype(float).to_numpy()
        p = g[pred_col].astype(float).to_numpy()
        if len(a) < 2:
            continue
        da = np.diff(a)
        dp = np.diff(p)
        sa = np.sign(da)
        sp = np.sign(dp)
        mask = sa != 0  # ignore flat laps (rare)
        if mask.any():
            correct.append((sa[mask] == sp[mask]).astype(float))
    if not correct:
        return float("nan")
    return float(np.concatenate(correct).mean())


def within_stint_trend_correlation(
    df: pd.DataFrame,
    *,
    actual_col: str = "LapTimeSeconds",
    pred_col: str = "pred",
    group_cols: Iterable[str] = ("Season", "RoundNumber", "Driver", "Stint"),
    order_col: str = "LapNumber",
    use_deltas: bool = True,
) -> Tuple[float, pd.DataFrame]:
    """
    Pearson correlation between predicted and actual within-stint trends.

    If `use_deltas=True`, correlates first differences (lap-to-lap changes) within each stint.
    Returns (weighted_mean_corr, per_group_df).
    """
    required = set(group_cols) | {actual_col, pred_col, order_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data = data.sort_values(list(group_cols) + [order_col]).reset_index(drop=True)

    rows = []
    for key, g in data.groupby(list(group_cols), dropna=False):
        a = g[actual_col].astype(float).to_numpy()
        p = g[pred_col].astype(float).to_numpy()
        if len(a) < 3:
            continue
        if use_deltas:
            a = np.diff(a)
            p = np.diff(p)
        if len(a) < 2 or np.std(a) == 0 or np.std(p) == 0:
            corr = np.nan
        else:
            corr = float(np.corrcoef(a, p)[0, 1])
        rows.append(
            {
                "group": key,
                "n_laps": int(len(g)),
                "corr": corr,
                "use_deltas": bool(use_deltas),
            }
        )

    per = pd.DataFrame(rows)
    if per.empty:
        return float("nan"), per

    # Weight by number of deltas (len-1) when using deltas, else by len
    weights = per["n_laps"] - 1 if use_deltas else per["n_laps"]
    valid = per["corr"].notna() & (weights > 0)
    if not valid.any():
        return float("nan"), per

    w = weights[valid].to_numpy(dtype=float)
    c = per.loc[valid, "corr"].to_numpy(dtype=float)
    return float(np.average(c, weights=w)), per


def cumulative_race_time_error(
    df: pd.DataFrame,
    *,
    actual_col: str = "LapTimeSeconds",
    pred_col: str = "pred",
    group_cols: Iterable[str] = ("Season", "RoundNumber", "Driver"),
) -> pd.DataFrame:
    """
    Per driver-race cumulative error: sum(pred) - sum(actual).
    """
    required = set(group_cols) | {actual_col, pred_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data["cum_error_s"] = data[pred_col].astype(float) - data[actual_col].astype(float)
    per = (
        data.groupby(list(group_cols), dropna=False)["cum_error_s"]
        .sum()
        .reset_index()
    )
    per["abs_cum_error_s"] = per["cum_error_s"].abs()
    return per.sort_values("abs_cum_error_s", ascending=False).reset_index(drop=True)


def compute_time_series_quality_metrics(
    df: pd.DataFrame,
    *,
    actual_col: str = "LapTimeSeconds",
    pred_col: str = "pred",
    threshold_s: float = 0.3,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Compute time-series/sequential quality metrics on a lap-level dataframe.
    Returns (metrics_dict, per_stint_corr_df, per_race_cum_df).
    """
    required = {actual_col, pred_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    y_true = df[actual_col].astype(float).to_numpy()
    y_pred = df[pred_col].astype(float).to_numpy()

    da = directional_accuracy(df, actual_col=actual_col, pred_col=pred_col)
    trend_corr, per_stint = within_stint_trend_correlation(df, actual_col=actual_col, pred_col=pred_col)
    per_race = cumulative_race_time_error(df, actual_col=actual_col, pred_col=pred_col)

    metrics = {
        "mape_pct": mean_absolute_percentage_error(y_true, y_pred),
        "directional_accuracy": float(da),
        "within_stint_trend_corr": float(trend_corr),
        "cumulative_race_time_abs_error_s_median": float(per_race["abs_cum_error_s"].median()),
        "cumulative_race_time_abs_error_s_mean": float(per_race["abs_cum_error_s"].mean()),
        "actionable_accuracy@0.3s": actionable_accuracy(y_true, y_pred, threshold_s=threshold_s),
    }
    return metrics, per_stint, per_race


def train_and_evaluate(
    models: Dict[str, object],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str = "LapTimeSeconds",
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    X_train = train_df[list(feature_cols)]
    y_train = train_df[target_col]
    X_test = test_df[list(feature_cols)]
    y_test = test_df[target_col]

    metrics_rows = []
    predictions: Dict[str, np.ndarray] = {}
    fitted_models: Dict[str, object] = {}

    for name, model in models.items():
        fitted = model.fit(X_train, y_train)
        preds = fitted.predict(X_test)
        scores = compute_metrics(y_test, preds)
        scores["model"] = name
        metrics_rows.append(scores)
        predictions[name] = preds
        fitted_models[name] = fitted

    metrics_df = pd.DataFrame(metrics_rows).sort_values("mae").reset_index(drop=True)
    return metrics_df, predictions, fitted_models


def format_metrics_markdown(metrics_df: pd.DataFrame) -> str:
    """
    Return a Markdown table with MAE, RMSE, and R2.
    """
    headers = ["Model", "MAE", "RMSE", "R2"]
    lines = ["| " + " | ".join(headers) + " |", "|---|---|---|---|"]
    for _, row in metrics_df.iterrows():
        lines.append(
            "| {model} | {mae:.4f} | {rmse:.4f} | {r2:.4f} |".format(
                model=row["model"],
                mae=row["mae"],
                rmse=row["rmse"],
                r2=row["r2"],
            )
        )
    return "\n".join(lines)


def select_best_ensemble(metrics_df: pd.DataFrame, ensemble_names: Iterable[str]) -> str:
    candidates = metrics_df[metrics_df["model"].isin(list(ensemble_names))].copy()
    if candidates.empty:
        raise ValueError("No ensemble models found in metrics_df.")
    return candidates.sort_values("mae").iloc[0]["model"]


def _resolve_pipeline(estimator: object) -> Pipeline:
    if hasattr(estimator, "best_estimator_"):
        estimator = estimator.best_estimator_
    if isinstance(estimator, Pipeline):
        return estimator
    raise ValueError("Estimator is not a sklearn Pipeline.")


def _get_feature_names(preprocessor) -> List[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())
    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    return []


def plot_feature_importance(
    fitted_estimator: object,
    output_path: Path | str,
    top_n: int = 10,
) -> None:
    setup_f1_style()
    pipeline = _resolve_pipeline(fitted_estimator)
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_.")

    importances = model.feature_importances_
    feature_names = _get_feature_names(preprocessor)
    if feature_names:
        n = min(len(importances), len(feature_names))
        importances = importances[:n]
        feature_names = feature_names[:n]
    else:
        feature_names = [f"f{i}" for i in range(len(importances))]

    order = np.argsort(importances)[::-1][:top_n]
    sorted_importances = importances[order]
    sorted_names = [feature_names[i] for i in order]

    plt.figure(figsize=(8, 5))
    plt.barh(sorted_names[::-1], sorted_importances[::-1])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_predicted_vs_actual(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    event_name: str,
    season: int,
    output_path: Path | str,
    driver: str | None = None,
    target_col: str = "LapTimeSeconds",
) -> None:
    setup_f1_style()
    df = test_df.copy()
    df["pred"] = predictions
    if "TyreLife" not in df.columns:
        raise ValueError("TyreLife is required for the predicted vs. actual plot.")

    mask = df["Season"] == season
    if "EventName" in df.columns:
        mask &= df["EventName"].str.contains(event_name, case=False, na=False)
    elif "Circuit" in df.columns:
        mask &= df["Circuit"].str.contains(event_name, case=False, na=False)
    if driver is not None and "Driver" in df.columns:
        mask &= df["Driver"].astype(str) == str(driver)

    race_df = df[mask].copy()
    if race_df.empty:
        raise ValueError("No data found for the specified race filter.")

    race_df.sort_values("TyreLife", inplace=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(race_df["TyreLife"], race_df[target_col], label="Actual", alpha=0.6)
    plt.scatter(race_df["TyreLife"], race_df["pred"], label="Predicted", alpha=0.6)
    title = f"Predicted vs. Actual ({event_name} {season})"
    if driver:
        title += f" - {driver}"
    plt.title(title)
    plt.xlabel("TyreLife")
    plt.ylabel("Lap Time (s)")
    plt.legend()
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_dataframe(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

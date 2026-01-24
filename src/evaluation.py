from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def time_based_split(
    feature_df: pd.DataFrame, test_season: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by season for time-series validation."""
    train_df = feature_df[feature_df["Season"] < test_season].copy()
    test_df = feature_df[feature_df["Season"] == test_season].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Adjust seasons or check data.")
    return train_df, test_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.

    Returns:
        Dictionary with MAE, RMSE, R², MAPE, and prediction statistics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # Prediction statistics
    residuals = y_true - y_pred
    std_residuals = np.std(residuals)
    max_error = np.max(np.abs(residuals))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "std_residuals": std_residuals,
        "max_error": max_error,
    }


def train_and_score(
    models: Dict[str, Pipeline],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str = "LapTimeSeconds",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Train all models and compute evaluation metrics.

    Args:
        models: Dictionary of model name -> estimator.
        train_df: Training data.
        test_df: Test data.
        feature_cols: Feature column names.
        target_col: Target column name.
        verbose: Print progress.

    Returns:
        Tuple of (metrics_df, predictions_dict, fitted_models_dict).
    """
    feature_cols = list(feature_cols)
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values

    metrics_rows = []
    predictions: Dict[str, np.ndarray] = {}
    fitted_models: Dict[str, Any] = {}

    for name, model in models.items():
        if verbose:
            print(f"Training {name}...", end=" ", flush=True)

        try:
            fitted = model.fit(X_train, y_train)
            preds = fitted.predict(X_test)
            scores = compute_metrics(y_test, preds)
            scores["model"] = name
            metrics_rows.append(scores)
            predictions[name] = preds
            fitted_models[name] = fitted

            if verbose:
                print(f"MAE: {scores['mae']:.4f}, R²: {scores['r2']:.4f}")

        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            continue

    metrics_df = pd.DataFrame(metrics_rows).sort_values("mae").reset_index(drop=True)

    # Reorder columns for clarity
    col_order = ["model", "mae", "rmse", "r2", "mape", "std_residuals", "max_error"]
    metrics_df = metrics_df[[c for c in col_order if c in metrics_df.columns]]

    return metrics_df, predictions, fitted_models


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


# =============================================================================
# SCIENTIFIC MODEL COMPARISON
# =============================================================================


def create_comparison_table(
    metrics_df: pd.DataFrame,
    baseline_model: str = "Linear Regression",
    latex: bool = False,
) -> str:
    """
    Create formatted comparison table for scientific reporting.

    Args:
        metrics_df: DataFrame from train_and_score with model metrics.
        baseline_model: Name of baseline model for improvement calculation.
        latex: If True, output LaTeX table format.

    Returns:
        Formatted string table.
    """
    df = metrics_df.copy()

    # Calculate improvement over baseline
    baseline_mae = df.loc[df["model"] == baseline_model, "mae"].values
    if len(baseline_mae) > 0:
        baseline_mae = baseline_mae[0]
        df["mae_improvement"] = ((baseline_mae - df["mae"]) / baseline_mae * 100).round(2)
    else:
        df["mae_improvement"] = 0.0

    # Sort by MAE
    df = df.sort_values("mae").reset_index(drop=True)

    if latex:
        return _format_latex_table(df)
    else:
        return _format_text_table(df)


def _format_text_table(df: pd.DataFrame) -> str:
    """Format comparison table as text."""
    lines = []
    lines.append("=" * 90)
    lines.append("MODEL COMPARISON - Lap Time Prediction")
    lines.append("=" * 90)
    lines.append("")

    # Header
    header = f"{'Model':<20} {'MAE (s)':>10} {'RMSE (s)':>10} {'R²':>8} {'MAPE (%)':>10} {'vs Baseline':>12}"
    lines.append(header)
    lines.append("-" * 90)

    # Best model indicator
    best_model = df.iloc[0]["model"]

    for _, row in df.iterrows():
        model_name = row["model"]
        indicator = " *" if model_name == best_model else ""

        mae = f"{row['mae']:.4f}"
        rmse = f"{row['rmse']:.4f}"
        r2 = f"{row['r2']:.4f}"
        mape = f"{row.get('mape', 0):.2f}"

        improvement = row.get("mae_improvement", 0)
        if improvement > 0:
            imp_str = f"+{improvement:.1f}%"
        elif improvement < 0:
            imp_str = f"{improvement:.1f}%"
        else:
            imp_str = "baseline"

        line = f"{model_name:<20} {mae:>10} {rmse:>10} {r2:>8} {mape:>10} {imp_str:>12}{indicator}"
        lines.append(line)

    lines.append("-" * 90)
    lines.append(f"* Best performing model: {best_model}")
    lines.append("=" * 90)

    return "\n".join(lines)


def _format_latex_table(df: pd.DataFrame) -> str:
    """Format comparison table as LaTeX."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Model Comparison for Lap Time Prediction}")
    lines.append("\\label{tab:model_comparison}")
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append("Model & MAE (s) & RMSE (s) & $R^2$ & MAPE (\\%) & Improvement \\\\")
    lines.append("\\midrule")

    best_model = df.iloc[0]["model"]

    for _, row in df.iterrows():
        model_name = row["model"]
        if model_name == best_model:
            model_name = f"\\textbf{{{model_name}}}"

        mae = f"{row['mae']:.4f}"
        rmse = f"{row['rmse']:.4f}"
        r2 = f"{row['r2']:.4f}"
        mape = f"{row.get('mape', 0):.2f}"

        improvement = row.get("mae_improvement", 0)
        if improvement > 0:
            imp_str = f"+{improvement:.1f}\\%"
        elif improvement < 0:
            imp_str = f"{improvement:.1f}\\%"
        else:
            imp_str = "---"

        line = f"{model_name} & {mae} & {rmse} & {r2} & {mape} & {imp_str} \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def print_comparison_summary(
    metrics_df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    baseline_model: str = "Linear Regression",
) -> None:
    """
    Print comprehensive comparison summary.

    Args:
        metrics_df: DataFrame from train_and_score.
        predictions: Dictionary of model predictions.
        y_test: Actual test values.
        baseline_model: Name of baseline model.
    """
    print(create_comparison_table(metrics_df, baseline_model))
    print()

    # Additional statistics
    print("DETAILED ANALYSIS")
    print("-" * 50)

    for model_name, preds in predictions.items():
        residuals = y_test - preds
        print(f"\n{model_name}:")
        print(f"  Mean residual:     {np.mean(residuals):+.4f}s")
        print(f"  Std residual:      {np.std(residuals):.4f}s")
        print(f"  95th percentile:   {np.percentile(np.abs(residuals), 95):.4f}s")
        print(f"  Max error:         {np.max(np.abs(residuals)):.4f}s")

        # Within 1 second accuracy
        within_1s = np.mean(np.abs(residuals) < 1.0) * 100
        within_05s = np.mean(np.abs(residuals) < 0.5) * 100
        print(f"  Within 0.5s:       {within_05s:.1f}%")
        print(f"  Within 1.0s:       {within_1s:.1f}%")


def compare_models(
    models: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "LapTimeSeconds",
    baseline_model: str = "Linear Regression",
    save_path: Path | None = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], str]:
    """
    Complete model comparison workflow.

    This is the main function for scientific model comparison.

    Args:
        models: Dictionary of model name -> estimator.
        train_df: Training data.
        test_df: Test data.
        feature_cols: Feature column names.
        target_col: Target column name.
        baseline_model: Name of baseline for improvement calculation.
        save_path: Optional path to save results CSV.
        verbose: Print progress and results.

    Returns:
        Tuple of (metrics_df, predictions_dict, formatted_table_string).
    """
    if verbose:
        print("\n" + "=" * 60)
        print("SCIENTIFIC MODEL COMPARISON")
        print("=" * 60)
        print(f"Training samples: {len(train_df):,}")
        print(f"Test samples:     {len(test_df):,}")
        print(f"Features:         {len(feature_cols)}")
        print("=" * 60 + "\n")

    # Train and evaluate
    metrics_df, predictions, fitted_models = train_and_score(
        models, train_df, test_df, feature_cols, target_col, verbose=verbose
    )

    # Create comparison table
    table_str = create_comparison_table(metrics_df, baseline_model)

    if verbose:
        print("\n" + table_str)

        # Print detailed analysis
        y_test = test_df[target_col].values
        print("\nDETAILED RESIDUAL ANALYSIS")
        print("-" * 50)
        for model_name, preds in predictions.items():
            residuals = y_test - preds
            within_05s = np.mean(np.abs(residuals) < 0.5) * 100
            within_1s = np.mean(np.abs(residuals) < 1.0) * 100
            print(f"{model_name}: {within_05s:.1f}% within 0.5s, {within_1s:.1f}% within 1.0s")

    # Save results
    if save_path:
        save_dataframe(metrics_df, save_path)
        if verbose:
            print(f"\nResults saved to: {save_path}")

    return metrics_df, predictions, table_str

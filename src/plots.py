from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def save_plot(fig: "go.Figure", path: Path | str, *, write_png: bool = True) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path.with_suffix(".html")))

    if write_png:
        try:
            fig.write_image(str(output_path.with_suffix(".png")), scale=2)
        except Exception as exc:  # noqa: BLE001
            print(f"PNG export skipped ({exc}). Install kaleido to enable PNG export.")

    return output_path


def plot_model_comparison(metrics_df: pd.DataFrame) -> "go.Figure":
    df = metrics_df.copy()
    fig = go.Figure()
    fig.add_bar(x=df["model"], y=df["mae"], name="MAE (s)")
    fig.update_layout(
        title="Model Comparison (MAE)",
        xaxis_title="Model",
        yaxis_title="MAE (s)",
        template="plotly_white",
        height=420,
    )
    return fig


def plot_actual_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Predicted vs Actual",
    sample_size: int = 3000,
) -> "go.Figure":
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n > sample_size:
        idx = np.random.default_rng(42).choice(n, size=sample_size, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    fig = px.scatter(
        df,
        x="Actual",
        y="Predicted",
        title=title,
        opacity=0.6,
        template="plotly_white",
    )
    min_v = float(min(y_true.min(), y_pred.min()))
    max_v = float(max(y_true.max(), y_pred.max()))
    fig.add_trace(
        go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Perfect",
        )
    )
    fig.update_layout(height=500)
    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Residual Distribution",
    bins: int = 50,
) -> "go.Figure":
    residuals = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    fig = px.histogram(
        residuals,
        nbins=bins,
        title=title,
        labels={"value": "Residual (s)"},
        template="plotly_white",
    )
    fig.update_layout(height=420, showlegend=False)
    return fig


def plot_error_by_category(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    category_col: str,
    top_n: int = 12,
    title: Optional[str] = None,
) -> "go.Figure":
    data = df.copy()
    data["AbsError"] = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    grouped = (
        data.groupby(category_col)["AbsError"]
        .mean()
        .sort_values()
        .head(top_n)
        .reset_index()
    )
    title = title or f"MAE by {category_col}"
    fig = px.bar(
        grouped,
        x="AbsError",
        y=category_col,
        orientation="h",
        title=title,
        template="plotly_white",
    )
    fig.update_layout(height=450)
    return fig

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


_STYLE_READY = False


def setup_f1_style() -> None:
    """
    Apply a FastF1/F1-style matplotlib theme globally (dark background + FastF1 colors).
    Safe to call multiple times.
    """
    global _STYLE_READY  # noqa: PLW0603
    if _STYLE_READY:
        return

    from fastf1 import plotting

    plotting.setup_mpl(color_scheme="fastf1")

    # Ensure consistent dark dashboard look across seaborn/matplotlib.
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b0f14",
            "axes.facecolor": "#0b0f14",
            "savefig.facecolor": "#0b0f14",
            "axes.edgecolor": "#9aa4ad",
            "axes.labelcolor": "#d6dde6",
            "xtick.color": "#d6dde6",
            "ytick.color": "#d6dde6",
            "text.color": "#d6dde6",
            "grid.color": "#27313a",
            "grid.alpha": 0.35,
        }
    )
    sns.set_theme(style="darkgrid")
    _STYLE_READY = True


_FALLBACK_TEAM_COLORS = {
    "Red Bull": "#3671C6",
    "Ferrari": "#E80020",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "RB": "#6692FF",
    "AlphaTauri": "#6692FF",
    "Racing Bulls": "#6692FF",
    "Haas": "#B6BABD",
    "Sauber": "#52E252",
    "Alfa Romeo": "#52E252",
    "Kick Sauber": "#52E252",
}


def team_color(team: str, session: Optional[object] = None) -> str:
    """
    Return an official-ish team color.

    FastF1 3.7 exposes `fastf1.plotting.get_team_color(team, session, colormap=...)`
    (requires a `Session`). If no session is provided, we use a stable fallback palette.
    """
    setup_f1_style()
    team_str = str(team) if team is not None else "unknown"

    if session is not None:
        from fastf1 import plotting

        try:
            return plotting.get_team_color(team_str, session=session, colormap="official")
        except Exception:  # noqa: BLE001
            try:
                return plotting.get_team_color(team_str, session=session, colormap="fastf1")
            except Exception:  # noqa: BLE001
                pass

    # Fallback: normalize a bit and map common substrings.
    normalized = team_str.strip()
    for key, color in _FALLBACK_TEAM_COLORS.items():
        if key.lower() in normalized.lower():
            return color
    return "#9aa4ad"


def model_reliability_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    0-100% score based on variance of signed errors.

    Definition:
      score = 100 / (1 + Var(error))
    where error = (y_pred - y_true) in seconds.
    """
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    var = float(np.var(errors))
    score = 100.0 / (1.0 + var)
    return float(np.clip(score, 0.0, 100.0))


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 400,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap CI for the mean of `values`.
    Returns (low, high) quantiles for the given CI.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    if len(v) == 1:
        return (float(v[0]), float(v[0]))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boot_means = v[idx].mean(axis=1)
    alpha = 1.0 - ci
    low = float(np.quantile(boot_means, alpha / 2))
    high = float(np.quantile(boot_means, 1 - alpha / 2))
    return (low, high)


def stratified_mae_with_ci(
    df: pd.DataFrame,
    group_col: str,
    *,
    abs_err_col: str = "abs_err",
    min_n: int = 50,
    n_boot: int = 400,
    ci: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute MAE + sample count + bootstrap CI per group.
    Expects `abs_err_col` to contain absolute errors in seconds.
    """
    data = df[[group_col, abs_err_col]].copy()
    data[group_col] = data[group_col].astype(str)
    data[abs_err_col] = pd.to_numeric(data[abs_err_col], errors="coerce")
    data = data[np.isfinite(data[abs_err_col].to_numpy())]

    rows = []
    for group, gdf in data.groupby(group_col, dropna=False):
        values = gdf[abs_err_col].to_numpy(dtype=float)
        n = int(len(values))
        mae = float(np.mean(values)) if n else np.nan
        if n < min_n:
            lo, hi = (np.nan, np.nan)
        else:
            lo, hi = _bootstrap_mean_ci(values, n_boot=n_boot, ci=ci, seed=seed)
        rows.append(
            {
                "group": str(group),
                "n": n,
                "mae": mae,
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    out = pd.DataFrame(rows).sort_values(["mae", "n"], ascending=[True, False])
    out.reset_index(drop=True, inplace=True)
    return out


def plot_stratified_mae(
    summary_df: pd.DataFrame,
    *,
    title: str,
    ax: Optional[plt.Axes] = None,
    color: str = "#64C4FF",
) -> plt.Axes:
    """
    Dot plot with horizontal 95% CI error bars for MAE per group.
    Expects columns: group, mae, ci_low, ci_high, n.
    """
    setup_f1_style()
    required = {"group", "mae", "ci_low", "ci_high", "n"}
    missing = required - set(summary_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = summary_df.copy()
    df = df.sort_values("mae", ascending=True).reset_index(drop=True)
    df["label"] = df["group"].astype(str) + "  (n=" + df["n"].astype(int).astype(str) + ")"

    y = np.arange(len(df))
    x = df["mae"].to_numpy(dtype=float)
    lo = df["ci_low"].to_numpy(dtype=float)
    hi = df["ci_high"].to_numpy(dtype=float)
    has_ci = np.isfinite(lo) & np.isfinite(hi)
    xerr = np.vstack([x - lo, hi - x])
    xerr[:, ~has_ci] = 0.0

    if ax is None:
        _, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(df) + 1)))

    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        markersize=7,
        color=color,
        ecolor="#d6dde6",
        elinewidth=2.0,
        capsize=4,
        alpha=0.95,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("MAE (s)")
    ax.set_title(title)
    ax.grid(True, axis="x")

    # Annotate MAE values
    for yi, xi in zip(y, x):
        ax.text(xi + 0.02, yi, f"{xi:.3f}s", va="center", ha="left", fontsize=11)

    plt.tight_layout()
    return ax


def plot_model_quality_scorecard(
    metrics: dict,
    *,
    title: str = "Model Quality Scorecard",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Simple dashboard-style scorecard for scalar metrics.
    """
    setup_f1_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")

    ax.text(0.02, 0.92, title, fontsize=18, fontweight="bold", transform=ax.transAxes)

    items = list(metrics.items())
    # Render in two columns
    left = items[: (len(items) + 1) // 2]
    right = items[(len(items) + 1) // 2 :]

    def fmt(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "NA"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    y0 = 0.72
    dy = 0.16
    for i, (k, v) in enumerate(left):
        ax.text(0.02, y0 - i * dy, f"{k}", fontsize=12, transform=ax.transAxes)
        ax.text(0.38, y0 - i * dy, fmt(v), fontsize=14, fontweight="bold", transform=ax.transAxes)

    for i, (k, v) in enumerate(right):
        ax.text(0.55, y0 - i * dy, f"{k}", fontsize=12, transform=ax.transAxes)
        ax.text(0.92, y0 - i * dy, fmt(v), fontsize=14, fontweight="bold", ha="right", transform=ax.transAxes)

    return ax


def plot_calibration_curve(
    nominal_vs_observed: np.ndarray,
    *,
    title: str = "Reliability Diagram (Prediction Intervals)",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot nominal vs observed coverage for prediction intervals.
    Expects array shape (k, 2): [nominal, observed].
    """
    setup_f1_style()
    arr = np.asarray(nominal_vs_observed, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("nominal_vs_observed must have shape (k, 2).")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    x = arr[:, 0]
    y = arr[:, 1]
    ax.plot([0, 1], [0, 1], linestyle="--", color="#d6dde6", linewidth=2, alpha=0.7, label="Ideal")
    ax.plot(x, y, marker="o", linewidth=2.5, color="#64C4FF", label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, axis="both")
    plt.tight_layout()
    return ax


def plot_uncertainty_vs_tyrelife(
    df: pd.DataFrame,
    *,
    tyre_life_col: str = "TyreLife",
    std_col: str = "pred_std",
    actual_col: str = "LapTimeSeconds",
    lower_col: str = "ci_low_90",
    upper_col: str = "ci_high_90",
    bins: Tuple[float, float, float, float] = (0, 5, 15, 60),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Coverage plot: how uncertainty varies with TyreLife buckets.
    Expects `std_col` in seconds.
    """
    setup_f1_style()
    data = df.copy()
    required = {tyre_life_col, std_col}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data[tyre_life_col] = pd.to_numeric(data[tyre_life_col], errors="coerce")
    data[std_col] = pd.to_numeric(data[std_col], errors="coerce")
    data = data.dropna(subset=[tyre_life_col, std_col]).copy()

    labels = ["Fresh (1-5)", "Mid (6-15)", "Worn (16+)"]
    data["ty_bucket"] = pd.cut(
        data[tyre_life_col],
        bins=[-np.inf, bins[1], bins[2], np.inf],
        labels=labels,
    ).astype(str)

    summary = (
        data.groupby("ty_bucket")[std_col]
        .agg(["mean", "median", "count"])
        .reindex(labels)
        .reset_index()
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=summary, x="ty_bucket", y="mean", ax=ax, color="#64C4FF", alpha=0.9)
    ax.set_title("Prediction Uncertainty vs TyreLife")
    ax.set_xlabel("TyreLife bucket")
    ax.set_ylabel("Predicted std (s)")
    for i, row in summary.iterrows():
        ax.text(i, row["mean"] + 0.01, f"n={int(row['count'])}", ha="center", va="bottom", fontsize=11)

    # Optional: add 90% interval coverage per bucket if CI columns exist.
    if all(c in data.columns for c in (actual_col, lower_col, upper_col)):
        cov = []
        for label in labels:
            g = data[data["ty_bucket"] == label]
            if g.empty:
                cov.append(np.nan)
                continue
            y = pd.to_numeric(g[actual_col], errors="coerce").to_numpy(dtype=float)
            lo = pd.to_numeric(g[lower_col], errors="coerce").to_numpy(dtype=float)
            hi = pd.to_numeric(g[upper_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
            if not mask.any():
                cov.append(np.nan)
            else:
                cov.append(float(((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])).mean()))

        ax2 = ax.twinx()
        ax2.plot(labels, cov, color="#ffd166", marker="o", linewidth=2.5, label="90% CI coverage")
        ax2.set_ylabel("Observed coverage (90% CI)")
        ax2.set_ylim(0, 1)
        ax2.grid(False)
        ax2.legend(loc="upper right")

    plt.tight_layout()
    return ax


def plot_coverage_by_uncertainty_bins(
    binned: np.ndarray,
    *,
    title: str = "Calibration by Uncertainty (Std Bins)",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot observed coverage vs predicted std bins.
    Expects binned array columns: (std_center, n, coverage, mae_proxy).
    """
    setup_f1_style()
    arr = np.asarray(binned, dtype=float)
    if arr.size == 0:
        raise ValueError("Empty binned data.")
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("binned must have shape (k, 4).")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    std_center = arr[:, 0]
    n = arr[:, 1]
    coverage = arr[:, 2]

    ax.plot(std_center, coverage, marker="o", linewidth=2.5, color="#64C4FF")
    ax.axhline(0.9, linestyle="--", color="#d6dde6", linewidth=2, alpha=0.7, label="Target 90%")
    ax.set_xlabel("Predicted std (s) — bin mean")
    ax.set_ylabel("Observed coverage (90% CI)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, axis="both")

    for x, y, nn in zip(std_center, coverage, n):
        ax.text(x, y + 0.03, f"n={int(nn)}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    return ax


@dataclass(frozen=True)
class TimingTowerResult:
    table: pd.DataFrame
    ax: plt.Axes


def plot_predictive_timing_tower(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    top_n: int = 10,
    session: Optional[object] = None,
    ax: Optional[plt.Axes] = None,
) -> TimingTowerResult:
    """
    Horizontal bar chart: top N drivers by lowest MAE, colored by team.
    """
    setup_f1_style()
    df = test_df.copy().reset_index(drop=True)
    df["y_true"] = np.asarray(y_true, dtype=float)
    df["y_pred"] = np.asarray(y_pred, dtype=float)
    df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()

    required = {"Driver", "Team", "abs_err"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    per_driver = (
        df.groupby(["Driver", "Team"], dropna=False)["abs_err"]
        .mean()
        .reset_index()
        .rename(columns={"abs_err": "MAE"})
        .sort_values("MAE", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    colors = [team_color(team, session=session) for team in per_driver["Team"]]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.barh(per_driver["Driver"].astype(str), per_driver["MAE"], color=colors, alpha=0.95)
    ax.invert_yaxis()
    ax.set_title("Predictive Timing Tower — Top 10 Driver MAE")
    ax.set_xlabel("MAE (s)")
    ax.set_ylabel("")
    ax.grid(True, axis="x")

    for i, value in enumerate(per_driver["MAE"].to_numpy()):
        ax.text(value + 0.02, i, f"{value:.3f}s", va="center", ha="left", fontsize=11)

    plt.tight_layout()
    return TimingTowerResult(table=per_driver, ax=ax)


def plot_tyre_error_correlation_heatmap(
    df: pd.DataFrame,
    *,
    row: str = "Circuit",
    col: str = "Compound",
    session: Optional[object] = None,  # reserved for future use
    ax: Optional[plt.Axes] = None,
    min_group_size: int = 80,
) -> plt.Axes:
    """
    Heatmap of corr(TyreLife, error) grouped by (row x col).

    Expects columns: TyreLife, LapTimeSeconds (actual), pred (prediction).
    """
    setup_f1_style()
    data = df.copy()
    required = {"TyreLife", "LapTimeSeconds", "pred", row, col}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data["error"] = data["pred"].astype(float) - data["LapTimeSeconds"].astype(float)
    data["TyreLife"] = data["TyreLife"].astype(float)

    def _corr(group: pd.DataFrame) -> float:
        if len(group) < min_group_size:
            return np.nan
        if group["TyreLife"].nunique(dropna=True) < 3:
            return np.nan
        return float(group["TyreLife"].corr(group["error"]))

    corr_df = (
        data.groupby([row, col], dropna=False)
        .apply(_corr)
        .reset_index(name="corr")
        .pivot(index=row, columns=col, values="corr")
        .sort_index()
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 7))

    sns.heatmap(
        corr_df,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="#27313a",
        cbar_kws={"label": "corr(TyreLife, error)"},
    )
    ax.set_title("Where the model struggles: correlation TyreLife vs prediction error")
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    plt.tight_layout()
    return ax


def plot_race_evolution(
    driver_name: str,
    session_data: pd.DataFrame,
    *,
    x_col: str = "LapNumber",
    actual_col: str = "LapTimeSeconds",
    pred_col: str = "pred",
    tyre_life_col: str = "TyreLife",
    track_status_col: str = "TrackStatus",
    track_status_flag_col: str = "TrackStatusFlag",
    pit_in_col: str = "PitInTime",
    pit_out_col: str = "PitOutTime",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Live lap simulation plot for a single driver.

    Plots:
      - solid: actual lap time
      - dashed: predicted lap time
      - shaded: +/- |error| around prediction
      - vertical markers: pit stops and track status changes (yellow/SC)

    Note: If pit/track-status columns are missing, the corresponding markers are skipped.
    """
    setup_f1_style()
    df = session_data.copy()
    df = df.sort_values(x_col).reset_index(drop=True)

    required = {x_col, actual_col, pred_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    x = df[x_col].astype(float).to_numpy()
    y_actual = df[actual_col].astype(float).to_numpy()
    y_pred = df[pred_col].astype(float).to_numpy()
    abs_err = np.abs(y_pred - y_actual)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    ax.plot(x, y_actual, label="Actual", linewidth=2.8)
    ax.plot(x, y_pred, label="Predicted", linewidth=2.8, linestyle="--")
    ax.fill_between(
        x,
        y_pred - abs_err,
        y_pred + abs_err,
        alpha=0.18,
        label="Error margin (±|err|)",
    )

    # Pit stop markers
    pit_laps: list[float] = []
    if pit_in_col in df.columns:
        pit_laps.extend(df.loc[df[pit_in_col].notna(), x_col].astype(float).tolist())
    if pit_out_col in df.columns:
        pit_laps.extend(df.loc[df[pit_out_col].notna(), x_col].astype(float).tolist())
    pit_laps = sorted(set(pit_laps))
    if not pit_laps and "Stint" in df.columns:
        stint_change = df["Stint"].astype(str).ne(df["Stint"].astype(str).shift(1))
        pit_laps = (
            df.loc[stint_change.fillna(False), x_col]
            .astype(float)
            .tolist()
        )
        if pit_laps:
            pit_laps = pit_laps[1:]  # first change is start of race/stint
    for lap in pit_laps:
        ax.axvline(lap, color="#d6dde6", alpha=0.25, linewidth=1.2)

    # Track status markers (changes)
    status_series = None
    if track_status_col in df.columns:
        status_series = df[track_status_col].astype(str)
    elif track_status_flag_col in df.columns:
        status_series = df[track_status_flag_col].astype(str)

    if status_series is not None:
        changes = status_series.ne(status_series.shift(1))
        change_laps = df.loc[changes.fillna(False), [x_col]].copy()
        change_states = status_series.loc[changes.fillna(False)].tolist()

        for lap, state in zip(change_laps[x_col].astype(float).tolist(), change_states):
            state_l = str(state).lower()
            if any(s in state_l for s in ["yellow", "vsc", "sc", "neutralized", "4", "5", "6", "7"]):
                color = "#ffd166" if "yellow" in state_l or "2" in state_l or "3" in state_l else "#ef476f"
                ax.axvline(lap, color=color, alpha=0.35, linewidth=1.6)

    title = f"Live Lap Simulation — {driver_name}"
    if tyre_life_col in df.columns and df[tyre_life_col].notna().any():
        title += " (TyreLife context)"
    ax.set_title(title)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    plt.tight_layout()
    return ax

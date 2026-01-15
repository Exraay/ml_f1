from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def _track_status_bucket(status: str | float | int) -> str:
    """
    Map raw TrackStatus codes to coarse categories.
    1-3 -> green/yellow; 4-7 were filtered but kept as 'neutralized' fallback.
    """
    if pd.isna(status):
        return "unknown"
    status_str = str(status)
    if any(flag in status_str for flag in ["4", "5", "6", "7"]):
        return "neutralized"
    if "2" in status_str or "3" in status_str:
        return "yellow"
    return "green"


def add_lap_time_features(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Add target and history-based timing features.
    """
    df = laps.copy()
    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

    group_keys = ["SessionKey", "Driver"]
    missing = [col for col in group_keys + ["LapNumber"] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for lag features: {missing}")
    df.sort_values(by=group_keys + ["LapNumber"], inplace=True)

    df["LapTimeLag1"] = df.groupby(group_keys)["LapTimeSeconds"].shift(1)
    df["LapTimeLag2"] = df.groupby(group_keys)["LapTimeSeconds"].shift(2)
    df["LapTimeLag3"] = df.groupby(group_keys)["LapTimeSeconds"].shift(3)
    df["RollingMean3"] = df.groupby(group_keys)["LapTimeSeconds"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    return df


def build_feature_table(clean_laps: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Transform cleaned laps into a model-ready feature matrix.
    Returns (feature_df, numeric_cols, categorical_cols).
    """
    df = add_lap_time_features(clean_laps)

    for col in ("TyreLife", "Stint", "TrackStatus", "Compound", "EventName"):
        if col not in df.columns:
            df[col] = np.nan

    df["TyreLife"] = df["TyreLife"].astype(float)
    df["Stint"] = df["Stint"].astype(float)
    df["TrackStatusFlag"] = df["TrackStatus"].apply(_track_status_bucket)

    # Ensure categories are strings to work with OneHotEncoder/TargetEncoder
    df["Driver"] = df["Driver"].astype(str)
    if "Team" in df.columns:
        df["Team"] = df["Team"].astype(str)
    elif "TeamName" in df.columns:
        df["Team"] = df["TeamName"].astype(str)
    else:
        df["Team"] = "unknown"

    df["Compound"] = df["Compound"].fillna("UNKNOWN").astype(str)
    df["EventName"] = df["EventName"].astype(str)
    if "Circuit" in df.columns:
        df["Circuit"] = df["Circuit"].astype(str)
    else:
        df["Circuit"] = df["EventName"]

    numeric_features: List[str] = [
        "LapNumber",
        "Stint",
        "TyreLife",
        "LapTimeLag1",
        "LapTimeLag2",
        "LapTimeLag3",
        "RollingMean3",
    ]

    categorical_features: List[str] = [
        "Driver",
        "Team",
        "Compound",
        "TrackStatusFlag",
        "Circuit",
    ]

    metadata_cols = ["Season", "RoundNumber", "EventName"]
    for col in metadata_cols:
        if col not in df.columns:
            df[col] = np.nan

    feature_df = df[
        numeric_features + categorical_features + ["LapTimeSeconds"] + metadata_cols
    ].copy()
    feature_df.dropna(subset=["LapTimeSeconds"], inplace=True)
    feature_df.reset_index(drop=True, inplace=True)
    return feature_df, numeric_features, categorical_features

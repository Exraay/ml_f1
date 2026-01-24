from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_seasons: Tuple[int, ...] = (2022, 2023)
    val_seasons: Tuple[int, ...] = (2024,)
    test_season: int = 2025
    test_rounds: int | None = 6  # first N rounds of test_season


def split_by_season_round(
    df: pd.DataFrame,
    *,
    config: SplitConfig = SplitConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/val/test by season and (optionally) first N rounds of test season.
    """
    required = {"Season", "RoundNumber"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for split: {sorted(missing)}")

    data = df.copy()
    train_df = data[data["Season"].isin(config.train_seasons)].copy()
    val_df = data[data["Season"].isin(config.val_seasons)].copy()
    test_df = data[data["Season"] == config.test_season].copy()

    if config.test_rounds is not None:
        test_df = test_df[test_df["RoundNumber"] <= int(config.test_rounds)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "One of the splits is empty. Adjust seasons/rounds or check input data."
        )

    sort_cols = [c for c in ["Season", "RoundNumber", "DriverNumber", "Driver", "LapNumber"] if c in data.columns]
    if sort_cols:
        train_df = train_df.sort_values(sort_cols)
        val_df = val_df.sort_values(sort_cols)
        test_df = test_df.sort_values(sort_cols)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

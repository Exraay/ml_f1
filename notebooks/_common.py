from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data import build_base_dataset
from src.feature_enhancements import add_enhanced_features
from src.encoding import apply_target_encoding
from src.split import SplitConfig, split_by_season_round

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "laps_base.parquet"
METADATA_PATH = ROOT / "data" / "processed" / "laps_base.metadata.json"
TARGET_COL = "LapTimeSeconds"


def build_dataset_if_missing(years: List[int] | None = None) -> Tuple[pd.DataFrame, Dict]:
    if DATASET_PATH.exists() and METADATA_PATH.exists():
        return load_dataset()

    years = years or [2022, 2023, 2024, 2025]
    df, numeric_features, categorical_features = build_base_dataset(years)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH, index=False)
    metadata = {
        "dataset": str(DATASET_PATH.as_posix()),
        "seasons": years,
        "target": TARGET_COL,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return df, metadata


def load_dataset() -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_parquet(DATASET_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return df, metadata


def prepare_features(
    df: pd.DataFrame,
    metadata: Dict,
    *,
    split_config: SplitConfig = SplitConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    train_df, val_df, test_df = split_by_season_round(df, config=split_config)

    # Train-only enhancements for tuning phase
    train_enh, val_enh, added_cols = add_enhanced_features(train_df, val_df)

    # Train+val enhancements for final test
    trainval = pd.concat([train_df, val_df], ignore_index=True)
    trainval_enh, test_enh, _ = add_enhanced_features(trainval, test_df)

    cats = metadata["categorical_features"]

    # Target encoding (fit on train only)
    train_enc, val_enc = apply_target_encoding(
        train_enh, [val_enh], cols=cats, target_col=TARGET_COL
    )

    # Target encoding for final test (fit on train+val)
    trainval_enc, test_enc = apply_target_encoding(
        trainval_enh, [test_enh], cols=cats, target_col=TARGET_COL
    )

    base_numeric = metadata["numeric_features"]
    encoded_cols = [f"{c}__te" for c in cats]
    numeric_features = [c for c in base_numeric + added_cols + encoded_cols if c in train_enc.columns]

    for df_out in (val_enc, trainval_enc, test_enc):
        for col in numeric_features:
            if col not in df_out.columns:
                df_out[col] = np.nan

    return train_enc, val_enc, trainval_enc, test_enc, numeric_features

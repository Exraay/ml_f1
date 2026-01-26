from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data import build_base_dataset
from src.feature_enhancements import add_enhanced_features
from src.encoding import apply_target_encoding
from src.split import SplitConfig, split_by_season_round

ROOT = Path(__file__).resolve().parents[1]
DATA_FORMAT = "parquet"  # "parquet" or "csv"
DATASET_PARQUET_PATH = ROOT / "data" / "processed" / "laps_base.parquet"
DATASET_CSV_PATH = ROOT / "data" / "processed" / "laps_base.csv"
DATASET_PATH = DATASET_PARQUET_PATH if DATA_FORMAT == "parquet" else DATASET_CSV_PATH
METADATA_PATH = ROOT / "data" / "processed" / "laps_base.metadata.json"
TARGET_COL = "LapTimeSeconds"


def build_dataset_if_missing(
    years: List[int] | None = None,
    *,
    rebuild: bool = False,
    include_physics: bool = True,
    exclude_lap1: bool = False,
    remove_outliers: bool = True,
    outlier_z: float = 6.0,
    outlier_min_samples: int = 15,
    balance_categories: bool = True,
    balance_category_cols: List[str] | None = None,
    min_category_count: int = 50,
) -> Tuple[pd.DataFrame, Dict]:
    if not rebuild and DATASET_PATH.exists() and METADATA_PATH.exists():
        return load_dataset()

    years = years or [2022, 2023, 2024, 2025]
    df, numeric_features, categorical_features = build_base_dataset(
        years,
        include_physics=include_physics,
        exclude_lap1=exclude_lap1,
        remove_outliers=remove_outliers,
        outlier_z=outlier_z,
        outlier_min_samples=outlier_min_samples,
        balance_categories=balance_categories,
        balance_category_cols=balance_category_cols,
        min_category_count=min_category_count,
        verbose=True,
    )
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PARQUET_PATH, index=False)
    df.to_csv(DATASET_CSV_PATH, index=False)
    metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "dataset": str(DATASET_PARQUET_PATH.as_posix()),
        "dataset_csv": str(DATASET_CSV_PATH.as_posix()),
        "compound_filter": "valid_only",
        "seasons": years,
        "include_physics": include_physics,
        "exclude_lap1": exclude_lap1,
        "remove_outliers": remove_outliers,
        "outlier_z": outlier_z,
        "outlier_min_samples": outlier_min_samples,
        "balance_categories": balance_categories,
        "balance_category_cols": balance_category_cols,
        "min_category_count": min_category_count,
        "target": TARGET_COL,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return df, metadata


def load_dataset() -> Tuple[pd.DataFrame, Dict]:
    if DATA_FORMAT == "csv":
        df = pd.read_csv(DATASET_CSV_PATH)
    else:
        df = pd.read_parquet(DATASET_PARQUET_PATH)
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

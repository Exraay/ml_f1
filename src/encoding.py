from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class TargetEncoder:
    cols: List[str]
    smoothing: float = 10.0
    min_samples: int = 1
    mappings: Dict[str, Dict[str, float]] | None = None
    global_mean: float | None = None

    def fit(self, df: pd.DataFrame, *, target_col: str) -> "TargetEncoder":
        data = df.copy()
        y = pd.to_numeric(data[target_col], errors="coerce")
        self.global_mean = float(y.mean())
        self.mappings = {}

        for col in self.cols:
            stats = (
                data.groupby(col, dropna=False)[target_col]
                .agg(["mean", "count"])
                .reset_index()
            )
            stats["count"] = stats["count"].astype(float)
            stats["mean"] = stats["mean"].astype(float)

            smooth = (
                stats["count"] * stats["mean"] + self.smoothing * self.global_mean
            ) / (stats["count"] + self.smoothing)
            mapping = dict(zip(stats[col].astype(str), smooth.astype(float)))
            self.mappings[col] = mapping

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mappings is None or self.global_mean is None:
            raise ValueError("TargetEncoder must be fit before transform.")

        out = df.copy()
        for col in self.cols:
            mapping = self.mappings.get(col, {})
            encoded = out[col].astype(str).map(mapping)
            out[f"{col}__te"] = encoded.fillna(self.global_mean).astype(float)
        return out

    def fit_transform(self, df: pd.DataFrame, *, target_col: str) -> pd.DataFrame:
        return self.fit(df, target_col=target_col).transform(df)


def apply_target_encoding(
    train_df: pd.DataFrame,
    other_dfs: Iterable[pd.DataFrame],
    *,
    cols: List[str],
    target_col: str,
    smoothing: float = 10.0,
) -> List[pd.DataFrame]:
    encoder = TargetEncoder(cols=cols, smoothing=smoothing)
    train_encoded = encoder.fit_transform(train_df, target_col=target_col)
    outputs = [train_encoded]
    for df in other_dfs:
        outputs.append(encoder.transform(df))
    return outputs

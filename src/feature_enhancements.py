from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _require_cols(df: pd.DataFrame, cols: Tuple[str, ...], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: missing required columns: {missing}")


@dataclass(frozen=True)
class FeatureEnhancer:
    """
    Train-fit feature enhancer.

    Important: all historical aggregates are fitted on TRAIN only to avoid leakage.
    """

    typical_stint_len_by_compound: Dict[str, float]
    typical_stint_len_global: float
    driver_circuit_median_pace: pd.DataFrame  # columns: Driver, Circuit, driver_circuit_median_pace
    team_circuit_median_pace: pd.DataFrame  # columns: Team, Circuit, team_circuit_median_pace
    driver_median_pace: Dict[str, float]
    team_median_pace: Dict[str, float]
    global_median_pace: float
    driver_compound_median_degradation: pd.DataFrame  # Driver, Compound, driver_compound_median_degradation
    driver_median_degradation: Dict[str, float]
    global_median_degradation: float

    @staticmethod
    def fit(train_df: pd.DataFrame) -> "FeatureEnhancer":
        _require_cols(
            train_df,
            (
                "Season",
                "RoundNumber",
                "Driver",
                "Team",
                "Circuit",
                "Compound",
                "LapTimeSeconds",
                "TyreLife",
                "Stint",
                "LapNumber",
                "LapTimeLag1",
                "LapTimeLag2",
                "RollingMean3",
            ),
            where="FeatureEnhancer.fit(train_df)",
        )

        temp = train_df.copy()
        temp["Driver"] = temp["Driver"].astype(str)
        temp["Team"] = temp["Team"].astype(str)
        temp["Circuit"] = temp["Circuit"].astype(str)
        temp["Compound"] = temp["Compound"].astype(str)

        # Base degradation proxy (for fitting degradation aggregates)
        tyre_life = pd.to_numeric(temp["TyreLife"], errors="coerce").astype(float)
        denom = np.maximum(tyre_life - 1.0, 1.0)
        lag1 = pd.to_numeric(temp["LapTimeLag1"], errors="coerce").astype(float)
        lag2 = pd.to_numeric(temp["LapTimeLag2"], errors="coerce").astype(float)
        temp["degradation_rate"] = (lag1 - lag2) / denom

        # Typical stint length by compound (median of max TyreLife within each stint)
        stint_max = (
            temp.groupby(["Season", "RoundNumber", "Driver", "Stint", "Compound"], dropna=False)[
                "TyreLife"
            ]
            .max()
            .reset_index(name="stint_len")
        )
        stint_max["Compound"] = stint_max["Compound"].astype(str)
        typical_by_comp = (
            stint_max.groupby("Compound", dropna=False)["stint_len"].median().to_dict()
        )
        typical_global = float(stint_max["stint_len"].median())

        # Historical median pace at each circuit (driver/team)
        driver_circuit = (
            temp.groupby(["Driver", "Circuit"], dropna=False)["LapTimeSeconds"]
            .median()
            .reset_index(name="driver_circuit_median_pace")
        )
        team_circuit = (
            temp.groupby(["Team", "Circuit"], dropna=False)["LapTimeSeconds"]
            .median()
            .reset_index(name="team_circuit_median_pace")
        )
        driver_median = (
            temp.groupby("Driver", dropna=False)["LapTimeSeconds"].median().to_dict()
        )
        team_median = temp.groupby("Team", dropna=False)["LapTimeSeconds"].median().to_dict()
        global_median = float(temp["LapTimeSeconds"].median())

        # Historical degradation per compound
        degr = temp[["Driver", "Compound", "degradation_rate"]].copy()
        degr = degr.replace([np.inf, -np.inf], np.nan).dropna(subset=["degradation_rate"])
        driver_compound_degr = (
            degr.groupby(["Driver", "Compound"], dropna=False)["degradation_rate"]
            .median()
            .reset_index(name="driver_compound_median_degradation")
        )
        driver_degr = (
            degr.groupby("Driver", dropna=False)["degradation_rate"].median().to_dict()
            if not degr.empty
            else {}
        )
        global_degr = float(degr["degradation_rate"].median()) if not degr.empty else 0.0

        return FeatureEnhancer(
            typical_stint_len_by_compound={str(k): float(v) for k, v in typical_by_comp.items()},
            typical_stint_len_global=typical_global,
            driver_circuit_median_pace=driver_circuit,
            team_circuit_median_pace=team_circuit,
            driver_median_pace={str(k): float(v) for k, v in driver_median.items()},
            team_median_pace={str(k): float(v) for k, v in team_median.items()},
            global_median_pace=global_median,
            driver_compound_median_degradation=driver_compound_degr,
            driver_median_degradation={str(k): float(v) for k, v in driver_degr.items()},
            global_median_degradation=global_degr,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features. Safe to call on train and test.
        """
        _require_cols(
            df,
            (
                "Season",
                "RoundNumber",
                "Driver",
                "Team",
                "Circuit",
                "Compound",
                "LapTimeSeconds",
                "TyreLife",
                "Stint",
                "LapNumber",
                "LapTimeLag1",
                "LapTimeLag2",
                "RollingMean3",
            ),
            where="FeatureEnhancer.transform(df)",
        )

        out = df.copy()
        out["Driver"] = out["Driver"].astype(str)
        out["Team"] = out["Team"].astype(str)
        out["Circuit"] = out["Circuit"].astype(str)
        out["Compound"] = out["Compound"].astype(str)

        # 1) Cyclical encodings for LapNumber (race phase)
        max_lap = out.groupby(["Season", "RoundNumber"], dropna=False)["LapNumber"].transform("max")
        max_lap = pd.to_numeric(max_lap, errors="coerce").astype(float).replace(0.0, np.nan)
        lap = pd.to_numeric(out["LapNumber"], errors="coerce").astype(float)
        phase = 2.0 * np.pi * (lap / max_lap)
        out["LapNumber_sin"] = np.sin(phase)
        out["LapNumber_cos"] = np.cos(phase)

        # 2) Tyre degradation rate proxy
        tyre_life = pd.to_numeric(out["TyreLife"], errors="coerce").astype(float)
        denom = np.maximum(tyre_life - 1.0, 1.0)
        lag1 = pd.to_numeric(out["LapTimeLag1"], errors="coerce").astype(float)
        lag2 = pd.to_numeric(out["LapTimeLag2"], errors="coerce").astype(float)
        out["TyreDegradationRate"] = (lag1 - lag2) / denom
        out["TyreDegradationRate"] = out["TyreDegradationRate"].replace([np.inf, -np.inf], np.nan)

        # 3) Stint phase features (start/middle/end within each driver-stint)
        group = ["Season", "RoundNumber", "Driver", "Stint"]
        out = out.sort_values(group + ["LapNumber"]).reset_index(drop=True)
        lap_in_stint = out.groupby(group, dropna=False)["LapNumber"].rank(method="first")
        stint_len = out.groupby(group, dropna=False)["LapNumber"].transform("count").astype(float)
        out["lap_in_stint"] = lap_in_stint.astype(float)
        out["stint_len"] = stint_len
        out["stint_is_start"] = (out["lap_in_stint"] <= 3).astype(float)
        out["stint_is_end"] = (out["lap_in_stint"] >= (out["stint_len"] - 2)).astype(float)
        out["stint_is_middle"] = (
            (out["stint_is_start"] == 0.0) & (out["stint_is_end"] == 0.0)
        ).astype(float)

        # 4) Relative features (driver vs field context within the same race and lap number)
        field_mean_roll3 = out.groupby(["Season", "RoundNumber", "LapNumber"], dropna=False)[
            "RollingMean3"
        ].transform("mean")
        out["RollingMean3_field_mean"] = pd.to_numeric(field_mean_roll3, errors="coerce").astype(float)
        out["RollingMean3_vs_field"] = (
            pd.to_numeric(out["RollingMean3"], errors="coerce").astype(float)
            - out["RollingMean3_field_mean"]
        )

        typical_len = out["Compound"].map(self.typical_stint_len_by_compound)
        typical_len = pd.to_numeric(typical_len, errors="coerce").astype(float)
        typical_len = typical_len.fillna(self.typical_stint_len_global)
        out["TypicalStintLen_compound"] = typical_len
        out["TyreLife_frac_typical"] = tyre_life / np.maximum(typical_len, 1.0)

        # 5) Interaction terms (TyreLife * Compound one-hot)
        compound_upper = out["Compound"].astype(str).str.upper()
        out["is_SOFT"] = (compound_upper == "SOFT").astype(float)
        out["is_MEDIUM"] = (compound_upper == "MEDIUM").astype(float)
        out["is_HARD"] = (compound_upper == "HARD").astype(float)
        out["TyreLife_x_SOFT"] = tyre_life * out["is_SOFT"]
        out["TyreLife_x_MEDIUM"] = tyre_life * out["is_MEDIUM"]
        out["TyreLife_x_HARD"] = tyre_life * out["is_HARD"]

        # 6) Historical aggregates (fitted on train only)
        out = out.merge(
            self.driver_circuit_median_pace,
            on=["Driver", "Circuit"],
            how="left",
        )
        out = out.merge(
            self.team_circuit_median_pace,
            on=["Team", "Circuit"],
            how="left",
        )
        out = out.merge(
            self.driver_compound_median_degradation,
            on=["Driver", "Compound"],
            how="left",
        )

        out["driver_median_pace"] = out["Driver"].map(self.driver_median_pace)
        out["team_median_pace"] = out["Team"].map(self.team_median_pace)
        out["driver_circuit_median_pace"] = out["driver_circuit_median_pace"].fillna(
            out["driver_median_pace"]
        )
        out["team_circuit_median_pace"] = out["team_circuit_median_pace"].fillna(
            out["team_median_pace"]
        )
        out["driver_circuit_median_pace"] = out["driver_circuit_median_pace"].fillna(
            self.global_median_pace
        )
        out["team_circuit_median_pace"] = out["team_circuit_median_pace"].fillna(
            self.global_median_pace
        )

        out["driver_median_degradation"] = out["Driver"].map(self.driver_median_degradation)
        out["driver_compound_median_degradation"] = out[
            "driver_compound_median_degradation"
        ].fillna(out["driver_median_degradation"])
        out["driver_compound_median_degradation"] = out[
            "driver_compound_median_degradation"
        ].fillna(self.global_median_degradation)

        # Fill remaining engineered NaNs conservatively
        engineered = [
            "LapNumber_sin",
            "LapNumber_cos",
            "TyreDegradationRate",
            "lap_in_stint",
            "stint_len",
            "stint_is_start",
            "stint_is_middle",
            "stint_is_end",
            "RollingMean3_field_mean",
            "RollingMean3_vs_field",
            "TypicalStintLen_compound",
            "TyreLife_frac_typical",
            "is_SOFT",
            "is_MEDIUM",
            "is_HARD",
            "TyreLife_x_SOFT",
            "TyreLife_x_MEDIUM",
            "TyreLife_x_HARD",
            "driver_circuit_median_pace",
            "team_circuit_median_pace",
            "driver_compound_median_degradation",
        ]
        for col in engineered:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        out["TyreDegradationRate"] = out["TyreDegradationRate"].fillna(0.0)
        out["RollingMean3_vs_field"] = out["RollingMean3_vs_field"].fillna(0.0)

        return out


def add_enhanced_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Fit the enhancer on `train_df` and transform both train/test.
    Returns (train_enhanced, test_enhanced, added_numeric_cols).
    """
    enhancer = FeatureEnhancer.fit(train_df)
    train_out = enhancer.transform(train_df)
    test_out = enhancer.transform(test_df)

    added_numeric = [
        "LapNumber_sin",
        "LapNumber_cos",
        "TyreDegradationRate",
        "lap_in_stint",
        "stint_len",
        "stint_is_start",
        "stint_is_middle",
        "stint_is_end",
        "RollingMean3_field_mean",
        "RollingMean3_vs_field",
        "TypicalStintLen_compound",
        "TyreLife_frac_typical",
        "is_SOFT",
        "is_MEDIUM",
        "is_HARD",
        "TyreLife_x_SOFT",
        "TyreLife_x_MEDIUM",
        "TyreLife_x_HARD",
        "driver_circuit_median_pace",
        "team_circuit_median_pace",
        "driver_compound_median_degradation",
    ]
    return train_out, test_out, added_numeric


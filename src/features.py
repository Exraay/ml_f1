from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# FUEL CONSUMPTION RATES (kg per lap) - Research-based estimates
# =============================================================================
# F1 cars carry max 110kg fuel (since 2022 regulations)
# Consumption varies by track characteristics:
# - High power tracks (long straights): ~2.0-2.2 kg/lap
# - Medium tracks: ~1.8-2.0 kg/lap
# - Low consumption (short/twisty): ~1.5-1.7 kg/lap
# Sources: F1 Technical regulations, team telemetry estimates

FUEL_CONSUMPTION_BY_CIRCUIT: dict[str, float] = {
    # High consumption tracks (long straights, high speed)
    "Monza": 2.2,
    "Spa-Francorchamps": 2.1,
    "Spa": 2.1,
    "Jeddah": 2.1,
    "Baku": 2.0,
    "Sakhir": 2.0,
    "Bahrain": 2.0,
    "Las Vegas": 2.1,
    "Azerbaijan": 2.0,
    # Medium consumption tracks
    "Silverstone": 1.9,
    "Austin": 1.9,
    "COTA": 1.9,
    "Barcelona": 1.85,
    "Catalunya": 1.85,
    "Suzuka": 1.9,
    "Melbourne": 1.85,
    "Albert Park": 1.85,
    "Imola": 1.85,
    "Zandvoort": 1.8,
    "Miami": 1.85,
    "Lusail": 1.9,
    "Qatar": 1.9,
    "Shanghai": 1.85,
    "China": 1.85,
    "Interlagos": 1.8,
    "São Paulo": 1.8,
    "Sao Paulo": 1.8,
    "Mexico City": 1.75,  # High altitude = less air resistance
    "Mexico": 1.75,
    "Spielberg": 1.8,
    "Austria": 1.8,
    "Red Bull Ring": 1.8,
    "Budapest": 1.75,
    "Hungary": 1.75,
    "Hungaroring": 1.75,
    "Yas Marina": 1.85,
    "Abu Dhabi": 1.85,
    # Low consumption tracks (short/twisty)
    "Monaco": 1.6,
    "Monte Carlo": 1.6,
    "Singapore": 1.7,
    "Marina Bay": 1.7,
}

DEFAULT_FUEL_CONSUMPTION = 1.85  # kg/lap default
MAX_FUEL_LOAD = 110.0  # kg (F1 2022+ regulations)
TYPICAL_START_FUEL = 100.0  # kg (teams rarely use full 110kg)
TIRE_CLIFF_LAP = 18  # heuristic lap threshold where grip drop-off accelerates


# =============================================================================
# TIRE DEGRADATION COEFFICIENTS
# =============================================================================
# Degradation rate multiplier per compound (relative grip loss per lap)
# Based on Pirelli compound characteristics:
# - SOFT: High grip, fast degradation
# - MEDIUM: Balanced
# - HARD: Lower grip, slow degradation
# - INTERMEDIATE/WET: Different behavior, moisture dependent

TIRE_DEGRADATION_RATE: dict[str, float] = {
    "SOFT": 0.035,       # ~3.5% grip loss per lap of tire life
    "MEDIUM": 0.025,     # ~2.5% grip loss per lap
    "HARD": 0.018,       # ~1.8% grip loss per lap
    "INTERMEDIATE": 0.02,  # Variable, depends on track drying
    "WET": 0.015,        # Lower mechanical degradation
    "UNKNOWN": 0.025,    # Default to medium
}

# Base grip level per compound (1.0 = maximum theoretical grip)
TIRE_BASE_GRIP: dict[str, float] = {
    "SOFT": 1.00,
    "MEDIUM": 0.97,
    "HARD": 0.94,
    "INTERMEDIATE": 0.92,
    "WET": 0.88,
    "UNKNOWN": 0.97,
}


def _get_fuel_consumption_rate(circuit: str) -> float:
    """Get fuel consumption rate for a circuit, with fuzzy matching."""
    if pd.isna(circuit):
        return DEFAULT_FUEL_CONSUMPTION

    circuit_str = str(circuit).strip()

    # Direct match
    if circuit_str in FUEL_CONSUMPTION_BY_CIRCUIT:
        return FUEL_CONSUMPTION_BY_CIRCUIT[circuit_str]

    # Fuzzy match (check if circuit name contains key or vice versa)
    circuit_lower = circuit_str.lower()
    for key, value in FUEL_CONSUMPTION_BY_CIRCUIT.items():
        if key.lower() in circuit_lower or circuit_lower in key.lower():
            return value

    return DEFAULT_FUEL_CONSUMPTION


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


# =============================================================================
# PHYSICS-BASED FEATURE ENGINEERING
# =============================================================================


def add_fuel_load_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Estimate remaining fuel weight based on lap number and circuit.

    Physics rationale:
    - Fuel weight directly impacts car performance (~0.03s per kg)
    - Lighter car = faster lap times (better acceleration, braking, cornering)
    - Consumption rate varies by circuit power demands

    Args:
        df: DataFrame with LapNumber and Circuit columns.
        verbose: Print fuel consumption mapping info.

    Returns:
        DataFrame with EstimatedFuelWeight and FuelEffect columns.
    """
    result = df.copy()

    # Get circuit column (try multiple possible names)
    circuit_col = None
    for col in ["Circuit", "EventName", "Location"]:
        if col in result.columns:
            circuit_col = col
            break

    if circuit_col is None or "LapNumber" not in result.columns:
        result["EstimatedFuelWeight"] = np.nan
        result["FuelEffect"] = np.nan
        return result

    # Calculate fuel consumption rate per circuit
    result["_FuelRate"] = result[circuit_col].apply(_get_fuel_consumption_rate)

    # Estimate remaining fuel: StartFuel - (LapNumber * ConsumptionRate)
    # Clamp to minimum 5kg (cars must finish with measurable fuel)
    result["EstimatedFuelWeight"] = (
        TYPICAL_START_FUEL - (result["LapNumber"] * result["_FuelRate"])
    ).clip(lower=5.0)

    # Normalized fuel effect (0 = empty, 1 = full tank)
    result["FuelEffect"] = result["EstimatedFuelWeight"] / TYPICAL_START_FUEL

    # Clean up temp column
    result.drop(columns=["_FuelRate"], inplace=True)

    if verbose:
        circuits = result[circuit_col].unique()
        print(f"\nFuel consumption rates applied for {len(circuits)} circuits:")
        sample_circuits = list(circuits)[:5]
        for circ in sample_circuits:
            rate = _get_fuel_consumption_rate(circ)
            print(f"  - {circ}: {rate} kg/lap")
        if len(circuits) > 5:
            print(f"  ... and {len(circuits) - 5} more circuits")

    return result


def add_tire_degradation_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Create tire degradation proxy combining TireLife and Compound.

    Physics rationale:
    - Tires lose grip over their lifetime due to thermal/mechanical degradation
    - Softer compounds degrade faster but offer more initial grip
    - Feature represents estimated remaining grip level

    Args:
        df: DataFrame with TyreLife and Compound columns.
        verbose: Print degradation model info.

    Returns:
        DataFrame with TireDegradation and EstimatedGrip columns.
    """
    result = df.copy()

    # Ensure required columns exist
    if "TyreLife" not in result.columns:
        result["TyreLife"] = np.nan
    if "Compound" not in result.columns:
        result["Compound"] = "UNKNOWN"

    # Normalize compound names
    result["_Compound"] = result["Compound"].fillna("UNKNOWN").str.upper().str.strip()

    # Get degradation rate and base grip per compound
    result["_DegRate"] = result["_Compound"].map(TIRE_DEGRADATION_RATE).fillna(0.025)
    result["_BaseGrip"] = result["_Compound"].map(TIRE_BASE_GRIP).fillna(0.97)

    # Calculate tire degradation (cumulative grip loss)
    # TireDegradation = TireLife * DegradationRate
    result["TireDegradation"] = (result["TyreLife"].fillna(0) * result["_DegRate"]).clip(
        upper=0.5  # Cap at 50% degradation (cliff point)
    )

    # Estimated remaining grip = BaseGrip - Degradation
    # Clamp to minimum 0.5 (below this, tire is at "cliff")
    result["EstimatedGrip"] = (result["_BaseGrip"] - result["TireDegradation"]).clip(
        lower=0.5
    )

    # Tire age category for potential encoding
    result["TireAgeCategory"] = pd.cut(
        result["TyreLife"].fillna(0),
        bins=[0, 10, 20, 30, 100],
        labels=["fresh", "mid", "old", "worn"],
        include_lowest=True,
    )

    # Clean up temp columns
    result.drop(columns=["_Compound", "_DegRate", "_BaseGrip"], inplace=True)

    if verbose:
        compounds = result["Compound"].value_counts()
        print(f"\nTire degradation model applied:")
        for compound, count in compounds.head(5).items():
            rate = TIRE_DEGRADATION_RATE.get(str(compound).upper(), 0.025)
            base = TIRE_BASE_GRIP.get(str(compound).upper(), 0.97)
            print(f"  - {compound}: {count:,} laps (base grip: {base}, deg rate: {rate}/lap)")

    return result


def add_tire_dropoff_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add non-linear tire life features to approximate the drop-off ("cliff").

    Features:
    - TyreLifeLog: log1p(TyreLife)
    - TyreLifeSq: TyreLife^2
    - TyreLifeCliff: max(0, TyreLife - TIRE_CLIFF_LAP)
    """
    result = df.copy()

    if "TyreLife" not in result.columns:
        result["TyreLife"] = np.nan

    tyre_life = pd.to_numeric(result["TyreLife"], errors="coerce").astype(float)
    result["TyreLifeLog"] = np.log1p(tyre_life.clip(lower=0))
    result["TyreLifeSq"] = (tyre_life ** 2).astype(float)
    result["TyreLifeCliff"] = (tyre_life - float(TIRE_CLIFF_LAP)).clip(lower=0)

    if verbose:
        print("\nTire drop-off features added:")
        print(f"  - TyreLifeLog: log1p(TyreLife)")
        print(f"  - TyreLifeSq: TyreLife^2")
        print(f"  - TyreLifeCliff: max(0, TyreLife - {TIRE_CLIFF_LAP})")

    return result


def add_track_evolution_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Create track evolution and session progress features.

    Physics rationale:
    - Track surface improves during session (rubber laid down = more grip)
    - Temperature affects tire/track grip levels
    - Session progress indicates track evolution state

    Args:
        df: DataFrame with LapNumber, SessionKey, and optional weather columns.
        verbose: Print feature info.

    Returns:
        DataFrame with SessionProgress, TrackEvolution, and weather features.
    """
    result = df.copy()

    # Calculate total laps per session and cumulative laps across the field
    if "LapNumber" in result.columns and "SessionKey" in result.columns:
        result["TotalLaps"] = result.groupby("SessionKey")["LapNumber"].transform("max")
        result["SessionProgress"] = result["LapNumber"] / result["TotalLaps"].clip(lower=1)

        # Count laps per lap number and accumulate across the field
        lap_counts = (
            result.groupby(["SessionKey", "LapNumber"], dropna=False)
            .size()
            .reset_index(name="FieldLapCount")
        )
        lap_counts["CumulativeFieldLaps"] = lap_counts.groupby("SessionKey")[
            "FieldLapCount"
        ].cumsum()
        lap_counts["TotalFieldLaps"] = lap_counts.groupby("SessionKey")[
            "FieldLapCount"
        ].transform("sum")
        lap_counts["FieldLapProgress"] = (
            lap_counts["CumulativeFieldLaps"]
            / lap_counts["TotalFieldLaps"].clip(lower=1)
        )

        # Merge back to each lap
        result = result.merge(
            lap_counts[["SessionKey", "LapNumber", "CumulativeFieldLaps", "FieldLapProgress"]],
            on=["SessionKey", "LapNumber"],
            how="left",
        )

        # Track evolution proxy based on cumulative laps in the session
        result["TrackEvolution"] = np.log1p(result["CumulativeFieldLaps"]) / np.log1p(
            result["CumulativeFieldLaps"].groupby(result["SessionKey"]).transform("max").clip(lower=1)
        )
    else:
        result["TotalLaps"] = np.nan
        result["SessionProgress"] = np.nan
        result["CumulativeFieldLaps"] = np.nan
        result["FieldLapProgress"] = np.nan
        result["TrackEvolution"] = np.nan

    # Weather features (if available)
    weather_features_added = []

    if "AirTemp" in result.columns:
        result["AirTemp"] = pd.to_numeric(result["AirTemp"], errors="coerce")
        weather_features_added.append("AirTemp")

    if "TrackTemp" in result.columns:
        result["TrackTemp"] = pd.to_numeric(result["TrackTemp"], errors="coerce")
        weather_features_added.append("TrackTemp")

        # Temperature effect on grip (optimal around 30-40°C)
        # Below optimal = less grip, above optimal = thermal degradation
        optimal_temp = 35.0
        result["TempGripEffect"] = 1.0 - (
            np.abs(result["TrackTemp"] - optimal_temp) / 100.0
        ).clip(upper=0.2)
        weather_features_added.append("TempGripEffect")

    if "Humidity" in result.columns:
        result["Humidity"] = pd.to_numeric(result["Humidity"], errors="coerce")
        weather_features_added.append("Humidity")

    if "Rainfall" in result.columns:
        result["IsWet"] = result["Rainfall"].fillna(False).astype(bool)
        weather_features_added.append("IsWet")

    if verbose:
        print(f"\nTrack evolution features added:")
        print(f"  - SessionProgress: LapNumber / TotalLaps (0.0 to 1.0)")
        print(f"  - FieldLapProgress: cumulative laps across the field / total laps")
        print(f"  - TrackEvolution: log-scaled rubber-in effect (field laps)")
        if weather_features_added:
            print(f"  - Weather features: {', '.join(weather_features_added)}")
        else:
            print(f"  - Weather features: None available in data")

    return result


def add_physics_features(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply all physics-based feature engineering in sequence.

    This is a convenience function that applies:
    1. Fuel load estimation
    2. Tire degradation modeling
    3. Track evolution features

    Args:
        df: DataFrame with lap data.
        verbose: Print feature engineering statistics.

    Returns:
        DataFrame with all physics-based features added.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("PHYSICS-BASED FEATURE ENGINEERING")
        print("=" * 60)

    result = df.copy()
    result = add_fuel_load_features(result, verbose=verbose)
    result = add_tire_degradation_features(result, verbose=verbose)
    result = add_tire_dropoff_features(result, verbose=verbose)
    result = add_track_evolution_features(result, verbose=verbose)

    if verbose:
        print("=" * 60 + "\n")

    return result


def build_feature_table(
    clean_laps: pd.DataFrame,
    include_physics: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Transform cleaned laps into a model-ready feature matrix.

    Args:
        clean_laps: Cleaned lap data from clean_laps().
        include_physics: If True, add physics-based features (fuel, tires, track).
        verbose: Print feature engineering statistics.

    Returns:
        Tuple of (feature_df, numeric_cols, categorical_cols).
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

    # Apply physics-based features
    if include_physics:
        df = add_physics_features(df, verbose=verbose)

    # Base numeric features
    numeric_features: List[str] = [
        "LapNumber",
        "Stint",
        "TyreLife",
        "LapTimeLag1",
        "LapTimeLag2",
        "LapTimeLag3",
        "RollingMean3",
    ]

    # Add physics-based numeric features
    physics_numeric: List[str] = [
        "EstimatedFuelWeight",
        "FuelEffect",
        "TireDegradation",
        "EstimatedGrip",
        "TyreLifeLog",
        "TyreLifeSq",
        "TyreLifeCliff",
        "SessionProgress",
        "CumulativeFieldLaps",
        "FieldLapProgress",
        "TrackEvolution",
    ]

    # Add weather features if available
    weather_numeric: List[str] = []
    for col in ["AirTemp", "TrackTemp", "TempGripEffect", "Humidity"]:
        if col in df.columns and df[col].notna().any():
            weather_numeric.append(col)

    if include_physics:
        numeric_features.extend(physics_numeric)
        numeric_features.extend(weather_numeric)

    categorical_features: List[str] = [
        "Driver",
        "Team",
        "Compound",
        "TrackStatusFlag",
        "Circuit",
    ]

    # Add TireAgeCategory if physics features enabled
    if include_physics and "TireAgeCategory" in df.columns:
        df["TireAgeCategory"] = df["TireAgeCategory"].astype(str)
        categorical_features.append("TireAgeCategory")

    metadata_cols = ["Season", "RoundNumber", "EventName", "SessionKey"]
    for col in metadata_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Build final feature dataframe
    all_cols = numeric_features + categorical_features + ["LapTimeSeconds"] + metadata_cols
    available_cols = [col for col in all_cols if col in df.columns]

    feature_df = df[available_cols].copy()
    feature_df.dropna(subset=["LapTimeSeconds"], inplace=True)
    feature_df.reset_index(drop=True, inplace=True)

    # Print feature summary
    if verbose:
        print("\n" + "=" * 60)
        print("FEATURE MATRIX SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(feature_df):,}")
        print(f"\nNumeric features ({len(numeric_features)}):")
        for feat in numeric_features:
            if feat in feature_df.columns:
                non_null = feature_df[feat].notna().sum()
                print(f"  - {feat}: {non_null:,} non-null values")
        print(f"\nCategorical features ({len(categorical_features)}):")
        for feat in categorical_features:
            if feat in feature_df.columns:
                n_unique = feature_df[feat].nunique()
                print(f"  - {feat}: {n_unique} unique values")
        print("=" * 60 + "\n")

    return feature_df, numeric_features, categorical_features

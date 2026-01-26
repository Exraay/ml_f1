from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import fastf1
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.features import build_feature_table

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
SCHEDULE_BACKENDS: list[Optional[str]] = ["fastf1", "f1timing", "ergast", None]


@dataclass(frozen=True)
class SessionIdentifier:
    season: int
    round_number: int
    event_name: str
    location: str | None = None

    @property
    def key(self) -> str:
        return f"{self.season}_{self.round_number:02d}"


def enable_cache(cache_dir: Path | str | None = None) -> Path:
    """
    Enable FastF1 on-disk caching to avoid repeated API calls.
    """
    cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))
    return cache_path


def _iter_race_sessions_from_cache(year: int, cache_dir: Path) -> Iterable[SessionIdentifier]:
    year_dir = cache_dir / str(year)
    if not year_dir.exists():
        return []

    entries = []
    for session_info_path in year_dir.rglob("session_info.ff1pkl"):
        if not session_info_path.parent.name.endswith("_Race"):
            continue
        try:
            info = pd.read_pickle(session_info_path)
        except Exception:  # noqa: BLE001
            continue

        data = info.get("data", {}) if isinstance(info, dict) else {}
        meeting = data.get("Meeting", {})
        event_name = str(meeting.get("Name", "")).strip()
        if not event_name:
            continue

        round_number = meeting.get("Number", None)
        start_date = data.get("StartDate", None)
        if start_date is None:
            try:
                folder_date = session_info_path.parent.parent.name.split("_", maxsplit=1)[0]
                start_date = pd.to_datetime(folder_date)
            except Exception:  # noqa: BLE001
                start_date = pd.NaT

        location = None
        if "Location" in meeting:
            location = str(meeting.get("Location"))
        elif "Circuit" in meeting and isinstance(meeting["Circuit"], dict):
            location = str(meeting["Circuit"].get("ShortName", ""))

        entries.append(
            {
                "event_name": event_name,
                "round_number": round_number,
                "start_date": start_date,
                "location": location,
            }
        )

    if not entries:
        return []

    # Sort by date and assign round numbers; if any are missing, use sequence for all
    entries = sorted(entries, key=lambda x: (pd.to_datetime(x["start_date"], errors="coerce"), x["event_name"]))
    use_sequence = any(entry["round_number"] is None for entry in entries)
    sessions = []
    for idx, entry in enumerate(entries, start=1):
        rnd = idx if use_sequence else entry["round_number"]
        sessions.append(
            SessionIdentifier(
                season=year,
                round_number=int(rnd),
                event_name=entry["event_name"],
                location=entry["location"],
            )
        )

    return sessions


def iter_race_sessions(
    years: Iterable[int],
    cache_dir: Path | str | None = None,
    prefer_cache: bool = True,
) -> Iterable[SessionIdentifier]:
    cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    for year in years:
        cached = list(_iter_race_sessions_from_cache(year, cache_path))
        if cached:
            print(f"Using cached schedule for season {year} ({len(cached)} races).")
            for session_id in cached:
                yield session_id
            if prefer_cache:
                continue

        schedule = None
        last_exc: Exception | None = None
        for backend in SCHEDULE_BACKENDS:
            try:
                schedule = fastf1.get_event_schedule(
                    year, include_testing=False, backend=backend
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue

        if schedule is None:
            cached = list(_iter_race_sessions_from_cache(year, cache_path))
            if cached:
                print(f"Using cached schedule for season {year} ({len(cached)} races).")
                for session_id in cached:
                    yield session_id
            else:
                print(f"Skipping season {year}: {last_exc}")
            continue

        for _, event in schedule.iterrows():
            round_number = int(event["RoundNumber"])
            event_name = str(event["EventName"])
            location = None
            if "Location" in event.index:
                location = str(event["Location"])
            elif "CircuitName" in event.index:
                location = str(event["CircuitName"])
            yield SessionIdentifier(
                season=year,
                round_number=round_number,
                event_name=event_name,
                location=location,
            )


def _load_race_session(session_id: SessionIdentifier) -> pd.DataFrame:
    def _build_session_from_cache() -> "fastf1.core.Session":
        from fastf1.events import Event
        from fastf1.core import Session

        cache_dir = DEFAULT_CACHE_DIR / str(session_id.season)
        session_info_path = None
        for path in cache_dir.rglob("session_info.ff1pkl"):
            if not path.parent.name.endswith("_Race"):
                continue
            try:
                info = pd.read_pickle(path)
            except Exception:  # noqa: BLE001
                continue
            data = info.get("data", {}) if isinstance(info, dict) else {}
            meeting = data.get("Meeting", {})
            if str(meeting.get("Name", "")).strip() == session_id.event_name:
                session_info_path = path
                break

        if session_info_path is None:
            raise RuntimeError(f"Cached session_info not found for {session_id.event_name}")

        info = pd.read_pickle(session_info_path)
        data = info.get("data", {}) if isinstance(info, dict) else {}
        start_date = data.get("StartDate", None)
        start_dt = pd.to_datetime(start_date) if start_date is not None else pd.NaT
        event_date = start_dt.normalize() if not pd.isna(start_dt) else pd.NaT

        event_series = pd.Series(
            {
                "EventName": session_id.event_name,
                "EventDate": event_date,
                "Session1": "Race",
                "Session1Date": start_dt,
                "Session1DateUtc": start_dt,
                "Session2": None,
                "Session2Date": pd.NaT,
                "Session2DateUtc": pd.NaT,
                "Session3": None,
                "Session3Date": pd.NaT,
                "Session3DateUtc": pd.NaT,
                "Session4": None,
                "Session4Date": pd.NaT,
                "Session4DateUtc": pd.NaT,
                "Session5": None,
                "Session5Date": pd.NaT,
                "Session5DateUtc": pd.NaT,
                "EventFormat": "conventional",
            }
        )
        event = Event(event_series, year=session_id.season)
        return Session(event, "Race", f1_api_support=True)

    try:
        session = fastf1.get_session(session_id.season, session_id.round_number, "R")
    except Exception:
        try:
            session = fastf1.get_session(session_id.season, session_id.event_name, "R")
        except Exception:
            session = _build_session_from_cache()
    session.load(telemetry=False, weather=True, laps=True)
    laps = session.laps.copy()
    if laps.empty:
        return laps

    # Merge weather data onto laps by LapStartTime
    if hasattr(session, "weather_data") and not session.weather_data.empty:
        weather = session.weather_data.copy()
        if "Time" in weather.columns and "LapStartTime" in laps.columns:
            weather_cols = ["AirTemp", "TrackTemp", "Humidity", "Rainfall"]
            weather = weather[["Time"] + [c for c in weather_cols if c in weather.columns]]
            weather = weather.dropna(subset=["Time"]).sort_values("Time")
            laps = laps.drop(columns=[c for c in weather_cols if c in laps.columns], errors="ignore")
            laps = laps.sort_values("LapStartTime")
            laps = pd.merge_asof(
                laps,
                weather,
                left_on="LapStartTime",
                right_on="Time",
                direction="nearest",
            )
            laps = laps.drop(columns=["Time"], errors="ignore")

    laps["Season"] = session_id.season
    laps["RoundNumber"] = session_id.round_number
    laps["EventName"] = session_id.event_name
    circuit_name = session_id.location or session_id.event_name
    laps["Circuit"] = circuit_name
    laps["SessionDate"] = pd.to_datetime(session.date)
    laps["SessionKey"] = session_id.key
    return laps


def load_laps_for_seasons(
    years: List[int],
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Download laps for multiple seasons (Race sessions only) with caching.
    """
    enable_cache(cache_dir)

    all_laps: list[pd.DataFrame] = []
    session_ids = list(iter_race_sessions(years, cache_dir=cache_dir))
    if not session_ids:
        raise RuntimeError(
            "No race sessions found. FastF1 could not load any event schedules. "
            "Check your network/proxy settings and try "
            "`fastf1.get_event_schedule(2022, backend='fastf1')` in a notebook."
        )

    for session_id in tqdm(session_ids, desc="Race sessions"):
        try:
            laps = _load_race_session(session_id)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {session_id.key} ({session_id.event_name}): {exc}")
            continue
        if laps.empty:
            continue
        all_laps.append(laps)

    if not all_laps:
        raise RuntimeError("No lap data could be loaded. Check cache/network.")

    combined = pd.concat(all_laps, ignore_index=True)
    combined.sort_values(
        by=["Season", "RoundNumber", "DriverNumber", "LapNumber"], inplace=True
    )
    combined.reset_index(drop=True, inplace=True)
    return combined


def clean_laps(
    raw_laps: pd.DataFrame,
    exclude_lap1: bool = False,
    remove_outliers: bool = True,
    outlier_z: float = 6.0,
    outlier_min_samples: int = 15,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply domain-specific filters for reliable lap data.
    """
    laps = raw_laps.copy()
    initial_count = len(laps)
    filter_stats: dict[str, int] = {}

    def _apply_filter(df: pd.DataFrame, mask: pd.Series, name: str) -> pd.DataFrame:
        removed = (~mask).sum()
        if removed > 0:
            filter_stats[name] = removed
        return df[mask]

    if "LapTime" in laps.columns:
        laps = _apply_filter(laps, laps["LapTime"].notna(), "No LapTime")
    if "IsAccurate" in laps.columns:
        laps = _apply_filter(laps, laps["IsAccurate"] == True, "IsAccurate=False")  # noqa: E712
    if "PitOutTime" in laps.columns:
        laps = _apply_filter(laps, ~laps["PitOutTime"].notna(), "Pit-Out")
    if "PitInTime" in laps.columns:
        laps = _apply_filter(laps, ~laps["PitInTime"].notna(), "Pit-In")
    if "TrackStatus" in laps.columns:
        safety_mask = laps["TrackStatus"].astype(str).str.contains("[4567]", na=False)
        laps = _apply_filter(laps, ~safety_mask, "SC/VSC/RedFlag")
    if "Deleted" in laps.columns:
        laps = _apply_filter(laps, laps["Deleted"] != True, "Deleted")  # noqa: E712
    if "LapNumber" in laps.columns:
        laps = _apply_filter(laps, laps["LapNumber"] > 0, "Formation Lap")
    if exclude_lap1 and "LapNumber" in laps.columns:
        laps = _apply_filter(laps, laps["LapNumber"] > 1, "Lap 1 (Standing Start)")

    if remove_outliers and "LapTime" in laps.columns:
        # Robust, high-side outlier removal per session+driver
        laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
        group_cols = [c for c in ["SessionKey", "Driver"] if c in laps.columns]
        if not group_cols:
            group_cols = ["Season", "RoundNumber"]

        def _mad(s: pd.Series) -> float:
            if len(s) < int(outlier_min_samples):
                return float("nan")
            median = float(np.median(s))
            return float(np.median(np.abs(s - median)))

        med = laps.groupby(group_cols, dropna=False)["LapTimeSeconds"].transform("median")
        mad = laps.groupby(group_cols, dropna=False)["LapTimeSeconds"].transform(_mad)
        robust_z = 0.6745 * (laps["LapTimeSeconds"] - med) / mad

        mask = (
            mad.isna()
            | (laps["LapTimeSeconds"] <= med)
            | (robust_z <= float(outlier_z))
        )
        laps = _apply_filter(laps, mask, f"Outliers (MAD z>{outlier_z})")

    laps.reset_index(drop=True, inplace=True)

    if verbose and filter_stats:
        final_count = len(laps)
        total_removed = initial_count - final_count
        print("\n" + "=" * 50)
        print("Lap Cleaning Statistics")
        print("=" * 50)
        print(f"Initial laps:  {initial_count:,}")
        for reason, count in filter_stats.items():
            print(f"  - {reason}: {count:,} removed")
        print("=" * 50)
        print(
            f"Final laps:    {final_count:,} ({total_removed:,} removed, "
            f"{100*final_count/initial_count:.1f}% retained)"
        )
        print("=" * 50 + "\n")

    return laps


def build_base_dataset(
    years: List[int],
    *,
    cache_dir: Path | str | None = None,
    include_physics: bool = True,
    exclude_lap1: bool = False,
    remove_outliers: bool = True,
    outlier_z: float = 6.0,
    outlier_min_samples: int = 15,
    balance_categories: bool = True,
    balance_category_cols: List[str] | None = None,
    min_category_count: int = 50,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build the base, split-safe dataset with engineered features.
    """
    raw = load_laps_for_seasons(years, cache_dir=cache_dir)
    clean = clean_laps(
        raw,
        exclude_lap1=exclude_lap1,
        remove_outliers=remove_outliers,
        outlier_z=outlier_z,
        outlier_min_samples=outlier_min_samples,
        verbose=verbose,
    )
    feature_df, numeric_features, categorical_features = build_feature_table(
        clean,
        include_physics=include_physics,
        balance_categories=balance_categories,
        balance_category_cols=balance_category_cols,
        min_category_count=min_category_count,
        verbose=verbose,
    )
    return feature_df, numeric_features, categorical_features

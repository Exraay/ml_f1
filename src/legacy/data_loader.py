from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import fastf1
import pandas as pd
from tqdm import tqdm

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


def iter_race_sessions(years: Iterable[int]) -> Iterable[SessionIdentifier]:
    for year in years:
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
    session = fastf1.get_session(session_id.season, session_id.round_number, "R")
    session.load(telemetry=False, weather=False, laps=True)
    laps = session.laps.copy()
    if laps.empty:
        return laps

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
    session_ids = list(iter_race_sessions(years))
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
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply domain-specific filters for reliable lap data.

    Filters applied:
    - Remove laps without a valid LapTime
    - Remove laps flagged as inaccurate (IsAccurate == False)
    - Remove pit-in and pit-out laps
    - Remove laps under Safety Car / Virtual Safety Car (TrackStatus 4-7)
    - Remove deleted laps (if Deleted column exists)
    - Remove formation lap (LapNumber == 0)
    - Optionally remove lap 1 (standing start, often atypical)

    Args:
        raw_laps: Raw lap data from FastF1.
        exclude_lap1: If True, exclude lap 1 (standing start) from results.
        verbose: If True, print statistics about filtered laps.

    Returns:
        Cleaned DataFrame with only valid racing laps.
    """
    laps = raw_laps.copy()
    initial_count = len(laps)
    filter_stats: dict[str, int] = {}

    # Track count before each filter
    def _apply_filter(df: pd.DataFrame, mask: pd.Series, name: str) -> pd.DataFrame:
        removed = (~mask).sum()
        if removed > 0:
            filter_stats[name] = removed
        return df[mask]

    # 1. Remove laps without LapTime
    if "LapTime" in laps.columns:
        laps = _apply_filter(laps, laps["LapTime"].notna(), "No LapTime")

    # 2. Remove inaccurate laps
    if "IsAccurate" in laps.columns:
        laps = _apply_filter(laps, laps["IsAccurate"] == True, "IsAccurate=False")  # noqa: E712

    # 3. Remove pit-out laps
    if "PitOutTime" in laps.columns:
        laps = _apply_filter(laps, ~laps["PitOutTime"].notna(), "Pit-Out")

    # 4. Remove pit-in laps
    if "PitInTime" in laps.columns:
        laps = _apply_filter(laps, ~laps["PitInTime"].notna(), "Pit-In")

    # 5. Remove Safety Car / VSC laps (TrackStatus 4=SC, 5=Red, 6=VSC, 7=VSC Ending)
    if "TrackStatus" in laps.columns:
        safety_mask = laps["TrackStatus"].astype(str).str.contains("[4567]", na=False)
        laps = _apply_filter(laps, ~safety_mask, "SC/VSC/RedFlag")

    # 6. Remove deleted laps (if column exists)
    if "Deleted" in laps.columns:
        laps = _apply_filter(laps, laps["Deleted"] != True, "Deleted")  # noqa: E712

    # 7. Remove formation lap (lap 0)
    if "LapNumber" in laps.columns:
        laps = _apply_filter(laps, laps["LapNumber"] > 0, "Formation Lap")

    # 8. Optionally remove lap 1 (standing start)
    if exclude_lap1 and "LapNumber" in laps.columns:
        laps = _apply_filter(laps, laps["LapNumber"] > 1, "Lap 1 (Standing Start)")

    laps.reset_index(drop=True, inplace=True)

    # Print statistics
    if verbose and filter_stats:
        final_count = len(laps)
        total_removed = initial_count - final_count
        print(f"\n{'='*50}")
        print(f"Lap Cleaning Statistics")
        print(f"{'='*50}")
        print(f"Initial laps:  {initial_count:,}")
        for reason, count in filter_stats.items():
            print(f"  - {reason}: {count:,} removed")
        print(f"{'='*50}")
        print(f"Final laps:    {final_count:,} ({total_removed:,} removed, "
              f"{100*final_count/initial_count:.1f}% retained)")
        print(f"{'='*50}\n")

    return laps

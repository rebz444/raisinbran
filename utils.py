"""Shared utility functions for the photometry analysis pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config import (
    BEHAV_DIR,
    BEHAV_LOG_CSV,
    FIBER_CHANNEL_MAP,
    MATCHED_SESSIONS_CSV,
    MERGED_SESSIONS_CSV,
    PHOTO_ROOT,
    PHOTOMETRY_LOG_URL,
)


# ---------------------------------------------------------------------------
# Session log helpers
# ---------------------------------------------------------------------------

def load_merged_log() -> pd.DataFrame:
    """
    Merge matched_sessions.csv with the behavior log CSV and save the result.
    Returns the merged DataFrame.
    """
    data_log = pd.read_csv(MATCHED_SESSIONS_CSV)
    behav_log = pd.read_csv(BEHAV_LOG_CSV, index_col=0)
    merged_log = pd.merge(data_log, behav_log, on=["mouse", "date"], how="inner")
    merged_log.to_csv(MERGED_SESSIONS_CSV)
    return merged_log


def load_photometry_log() -> pd.DataFrame:
    """Load the photometry log from Google Sheets, normalizing dates to YYYY-MM-DD."""
    log = pd.read_csv(PHOTOMETRY_LOG_URL)
    log["date"] = pd.to_datetime(log["date"]).dt.strftime("%Y-%m-%d")
    return log


# ---------------------------------------------------------------------------
# Channel lookup helpers
# ---------------------------------------------------------------------------

def get_channels_for_session(mouse: str, date: str, photometry_log: pd.DataFrame) -> Dict[str, str]:
    """
    Return a dict of {channel: label} for a session based on the photometry log.
    Label format: {channel}_{side}_{area}_{sensor}  e.g. "G4_l_str_GRAB"

    - Striatum fibers: includes both red and green channels.
    - V1 fibers: includes red channel only.
    Returns empty dict if the session is not found in the log.
    """
    row = photometry_log[
        (photometry_log["mouse"] == mouse) & (photometry_log["date"] == date)
    ]
    if row.empty:
        return {}

    row = row.iloc[0]
    n_fibers = sum(
        1 for i in range(1, 5)
        if pd.notna(row.get(f"fiber_{i}_area")) and str(row.get(f"fiber_{i}_area")).strip() != ""
    )
    if n_fibers not in FIBER_CHANNEL_MAP:
        return {}

    channel_labels: Dict[str, str] = {}
    for fiber_num, (r_ch, g_ch) in FIBER_CHANNEL_MAP[n_fibers].items():
        area = str(row.get(f"fiber_{fiber_num}_area", "")).strip().lower()
        side = str(row.get(f"fiber_{fiber_num}_side", "")).strip().lower()
        sensor = str(row.get(f"fiber_{fiber_num}_sensor", "")).strip()
        if area in ("str", "striatum"):
            channel_labels[r_ch] = f"{r_ch}_{side}_{area}_{sensor}"
            channel_labels[g_ch] = f"{g_ch}_{side}_{area}_{sensor}"
        elif area == "v1":
            channel_labels[r_ch] = f"{r_ch}_{side}_{area}_{sensor}"

    return channel_labels


def get_grab_channel(mouse: str, date: str, photometry_log: pd.DataFrame) -> Optional[str]:
    """
    Return the green channel name (e.g. 'G4') for the striatum GRAB fiber
    for this session, or None if not found.
    """
    row = photometry_log[
        (photometry_log["mouse"] == mouse) & (photometry_log["date"] == date)
    ]
    if row.empty:
        return None

    row = row.iloc[0]
    n_fibers = sum(
        1 for i in range(1, 5)
        if pd.notna(row.get(f"fiber_{i}_area")) and str(row.get(f"fiber_{i}_area")).strip() != ""
    )
    if n_fibers not in FIBER_CHANNEL_MAP:
        return None

    for fiber_num, (_, g_ch) in FIBER_CHANNEL_MAP[n_fibers].items():
        area = str(row.get(f"fiber_{fiber_num}_area", "")).strip().lower()
        sensor = str(row.get(f"fiber_{fiber_num}_sensor", "")).strip()
        if area in ("str", "striatum") and "grab" in sensor.lower():
            return g_ch

    return None


# ---------------------------------------------------------------------------
# Photometry alignment
# ---------------------------------------------------------------------------

def assign_trials_to_photometry(phot: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """
    Align photometry samples to behavioral trials and compute per-sample trial time.

    Expects:
    - phot: DataFrame with 't_sec' column (photometry timestamps)
    - trials: DataFrame with 'start_time', 'end_time', 'bg_length', 'time_waited' columns

    Returns phot with added columns: session_trial_num, trial_time, decision_time,
    plus all trial columns merged in.
    """
    trials = trials.copy()
    session_first_start = float(trials.iloc[0]["start_time"])
    trials["start_time"] = trials["start_time"].astype(float) - session_first_start
    trials["end_time"] = trials["end_time"].astype(float) - session_first_start

    phot_sorted = phot.sort_values("t_sec").reset_index(drop=True)
    trials_sorted = trials.sort_values("start_time").reset_index(drop=True)

    merged = pd.merge_asof(
        phot_sorted,
        trials_sorted,
        left_on="t_sec",
        right_on="start_time",
        direction="backward",
    )

    merged = merged[merged["t_sec"] <= merged["end_time"]].reset_index(drop=True)
    merged["trial_time"] = merged["t_sec"] - merged["start_time"]
    merged["decision_time"] = merged["bg_length"] + merged["time_waited"]
    return merged

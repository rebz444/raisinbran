"""Shared utility functions for the photometry analysis pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config import (
    BEHAV_DIR,
    FIBER_CHANNEL_MAP,
    PHOTO_ROOT,
    PHOTOMETRY_LOG_URL,
    PIPELINE_SESSION_LOG,
)


# ---------------------------------------------------------------------------
# Pipeline session log
# ---------------------------------------------------------------------------

def update_session_log(session_id: str, updates: dict) -> None:
    """Read pipeline_session_log.csv, update/insert a row for session_id, write back.

    Columns are added on first use; existing columns are preserved unchanged.
    """
    if PIPELINE_SESSION_LOG.exists():
        log = pd.read_csv(PIPELINE_SESSION_LOG, index_col=0)
    else:
        log = pd.DataFrame()

    for col, val in updates.items():
        log.loc[session_id, col] = val

    log.to_csv(PIPELINE_SESSION_LOG)


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------

def get_session_groups(bg_length_threshold: float = 2.0) -> Dict[str, str]:
    """
    Return {mouse: group} where group is 'long' or 'short', loaded from
    pipeline_session_log.csv.

    Uses a 'group' column directly if present; otherwise derives from the
    median bg_length per mouse using bg_length_threshold as the split point.
    """
    if not PIPELINE_SESSION_LOG.exists():
        raise FileNotFoundError(f"Pipeline session log not found: {PIPELINE_SESSION_LOG}")
    log = pd.read_csv(PIPELINE_SESSION_LOG, index_col=0)

    if "group" in log.columns:
        _aliases = {"l": "long", "s": "short", "long": "long", "short": "short"}
        return {
            mouse: _aliases.get(str(grp), str(grp))
            for mouse, grp in log.groupby("mouse")["group"].first().items()
        }

    if "bg_length" not in log.columns:
        raise ValueError(
            "pipeline_session_log.csv has neither 'group' nor 'bg_length' column; "
            "cannot determine long/short BG assignment"
        )
    mouse_bg = log.groupby("mouse")["bg_length"].median()
    return {
        mouse: ("long" if length > bg_length_threshold else "short")
        for mouse, length in mouse_bg.items()
    }


# ---------------------------------------------------------------------------
# Session log helpers
# ---------------------------------------------------------------------------

def load_merged_log() -> pd.DataFrame:
    """
    Return pipeline sessions that have a matched behavior directory.

    Reads pipeline_session_log.csv and filters to rows where behav_dir is set
    (i.e. step 3a has run and found a matching behavior directory).
    """
    if not PIPELINE_SESSION_LOG.exists():
        return pd.DataFrame()
    log = pd.read_csv(PIPELINE_SESSION_LOG, index_col=0)
    if "behav_dir" not in log.columns:
        return pd.DataFrame()
    return log[log["behav_dir"].notna()].reset_index(drop=True)


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
            channel_labels[r_ch] = f"{side}_str_rCaMP"
            channel_labels[g_ch] = f"{side}_str_{sensor}"
        elif area == "v1":
            channel_labels[r_ch] = f"{side}_v1_rCaMP"

    return channel_labels


def get_grab_channel(mouse: str, date: str, photometry_log: pd.DataFrame) -> Optional[str]:
    """
    Return the biological label for the striatum GRAB fiber green channel
    (e.g. 'l_str_GRAB'), or None if not found.
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

    for fiber_num, (_, _g_ch) in FIBER_CHANNEL_MAP[n_fibers].items():
        area = str(row.get(f"fiber_{fiber_num}_area", "")).strip().lower()
        side = str(row.get(f"fiber_{fiber_num}_side", "")).strip().lower()
        sensor = str(row.get(f"fiber_{fiber_num}_sensor", "")).strip()
        if area in ("str", "striatum") and "grab" in sensor.lower():
            return f"{side}_str_{sensor}"

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

    Returns phot with added columns: trial_time, plus all trial columns merged in.
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
    return merged

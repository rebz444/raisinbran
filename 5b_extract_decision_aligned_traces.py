#!/usr/bin/env python3
"""
Decision-Aligned DA Trace Extraction
=====================================

This script loads raw photometry traces and generates decision-aligned averages
for Figure 3.1 of the committee presentation.

The key insight from the literature:
- Hamilos et al. aligned traces backward from first-lick (decision)
- This avoids temporal smearing across variable wait times
- Steeper ramps should precede earlier decisions

Usage:
    python 5b_extract_decision_aligned_traces.py
    
Outputs:
    - decision_aligned_traces.csv: Per-trial traces aligned to decision
    - decision_aligned_averages.csv: Per-mouse and overall averages
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import NON_IMPULSIVE_CUTOFF
from utils import get_grab_channel, get_session_groups, load_photometry_log

# =============================================================================
# CONFIGURATION - same as main script
# =============================================================================

class Config:
    PROCESSED_OUT: Path = Path("/Volumes/T7 Shield/photometry/processed_output")
    BEHAV_DIR: Path = Path("/Volumes/T7 Shield/photometry/behav_analyzed")
    OUTPUT_DIR: Path = Path("/Volumes/T7 Shield/photometry/committee_figures")
    
    # Time windows for alignment
    PRE_DECISION_WINDOW = 5.0   # seconds before decision to include
    POST_DECISION_WINDOW = 1.0  # seconds after decision to include
    TIMEBIN_MS = 50             # temporal resolution for averaging
    
    # Trial filters
    MIN_WAIT = 0.8              # minimum wait time to include
    
    COLORS = {
        "long": "#2E86AB",
        "short": "#A23B72", 
        "da_signal": "#1B9E77",
        "iso": "#999999",
    }
    
    MOUSE_COLORS = {
        "RZ083": "#1f77b4", "RZ084": "#ff7f0e", "RZ085": "#2ca02c",
        "RZ086": "#d62728", "RZ087": "#9467bd", "RZ088": "#8c564b",
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_session_data(config: Config, session_id: str, group_map: Dict[str, str], photometry_log: "pd.DataFrame | None" = None) -> Optional[Dict]:
    """
    Load all data for a single session.

    Returns dict with:
        - phot: photometry DataFrame
        - trials: trials DataFrame
        - events: events DataFrame
        - mouse: mouse ID
        - group: long/short (from sessions_dff_behav_merged via group_map)
    """
    session_dir = config.PROCESSED_OUT / session_id

    # Parse mouse from session_id
    parts = session_id.split("_")
    mouse = parts[0] if parts else None
    if mouse not in group_map:
        return None
    
    # Find photometry file via log lookup, fall back to scanning for any green channel CSV
    phot_file = None
    date = parts[1] if len(parts) > 1 else None
    if photometry_log is not None and mouse and date:
        grab_label = get_grab_channel(mouse, date, photometry_log)
        if grab_label:
            candidate = session_dir / f"{grab_label}.csv"
            if candidate.exists():
                phot_file = candidate
    if phot_file is None:
        # Fallback: first CSV that isn't a known summary file
        for candidate in sorted(session_dir.glob("*.csv")):
            if candidate.name not in ("photometry_long.csv", "channel_summary.csv", "correction_quality.csv"):
                phot_file = candidate
                break
    
    if phot_file is None:
        return None
    
    # Find behavior files
    behav_subdir = None
    for subdir in config.BEHAV_DIR.iterdir():
        if session_id in subdir.name or subdir.name in session_id:
            behav_subdir = subdir
            break
    
    if behav_subdir is None:
        # Try to match by date
        date_str = "_".join(parts[1:3]) if len(parts) >= 3 else ""
        for subdir in config.BEHAV_DIR.iterdir():
            if date_str in subdir.name and mouse in subdir.name:
                behav_subdir = subdir
                break
    
    if behav_subdir is None:
        return None
    
    trials_file = list(behav_subdir.glob("trials_analyzed_*.csv"))
    events_file = list(behav_subdir.glob("events_processed_*.csv"))
    
    if not trials_file or not events_file:
        return None
    
    try:
        phot = pd.read_csv(phot_file)
        trials = pd.read_csv(trials_file[0])
        events = pd.read_csv(events_file[0])
        
        return {
            "phot": phot,
            "trials": trials,
            "events": events,
            "mouse": mouse,
            "group": group_map[mouse],
            "session_id": session_id,
        }
    except Exception as e:
        print(f"Error loading {session_id}: {e}")
        return None


def get_trial_decision_time(events: pd.DataFrame, trial_num: int) -> Optional[float]:
    """Get the decision (first lick / consumption start) time for a trial."""
    trial_events = events[events["session_trial_num"] == trial_num]
    consumption = trial_events[trial_events["state"] == "in_consumption"]
    if len(consumption) > 0:
        return float(consumption.iloc[0]["trial_time"])
    return None


def get_trial_cue_times(events: pd.DataFrame, trial_num: int) -> Tuple[Optional[float], Optional[float]]:
    """Get cue_on and cue_off times for a trial."""
    trial_events = events[events["session_trial_num"] == trial_num]
    
    cue_on = trial_events[trial_events["state"] == "in_background"]
    cue_off = trial_events[trial_events["state"] == "in_wait"]
    
    cue_on_t = float(cue_on.iloc[0]["trial_time"]) if len(cue_on) > 0 else None
    cue_off_t = float(cue_off.iloc[0]["trial_time"]) if len(cue_off) > 0 else None
    
    return cue_on_t, cue_off_t


# =============================================================================
# TRACE EXTRACTION
# =============================================================================

def extract_decision_aligned_trace(
    phot: pd.DataFrame,
    trials: pd.DataFrame,
    events: pd.DataFrame,
    trial_num: int,
    pre_window: float,
    post_window: float,
    signal_col: str = "dff_zscored",
) -> Optional[Dict]:
    """
    Extract photometry trace aligned to decision time.
    
    Returns dict with:
        - time: time relative to decision (negative = before)
        - signal: DA signal values
        - decision_time: absolute decision time
        - wait_time: time waited from cue_off to decision
    """
    # Get decision time
    decision_time = get_trial_decision_time(events, trial_num)
    if decision_time is None:
        return None
    
    # Get cue times
    cue_on_t, cue_off_t = get_trial_cue_times(events, trial_num)
    if cue_off_t is None:
        return None
    
    wait_from_cue_off = decision_time - cue_off_t
    wait_from_cue_on = (decision_time - cue_on_t) if cue_on_t is not None else float("nan")

    # Get trial start time to convert to session time
    trial_row = trials[trials["session_trial_num"] == trial_num]
    if len(trial_row) == 0:
        return None
    
    trial_start = float(trial_row.iloc[0]["start_time"])
    first_trial_start = float(trials.iloc[0]["start_time"])
    trial_start_sec = trial_start - first_trial_start
    
    # Absolute decision time in session seconds
    abs_decision_time = trial_start_sec + decision_time
    
    # Extract window around decision
    start_time = abs_decision_time - pre_window
    end_time = abs_decision_time + post_window
    
    mask = (phot["t_sec"] >= start_time) & (phot["t_sec"] <= end_time)
    segment = phot[mask].copy()
    
    if len(segment) < 10:
        return None
    
    # Convert to decision-relative time
    segment["time_rel_decision"] = segment["t_sec"] - abs_decision_time
    
    return {
        "time": segment["time_rel_decision"].values,
        "signal": segment[signal_col].values if signal_col in segment.columns else segment["dff_zscored"].values,
        "iso": segment["iso"].values if "iso" in segment.columns else None,
        "decision_time": decision_time,
        "wait_from_cue_off": wait_from_cue_off,
        "wait_from_cue_on": wait_from_cue_on,
        "cue_on_t": cue_on_t,
        "cue_off_t": cue_off_t,
    }


def interpolate_to_common_timebase(
    traces: List[Dict],
    time_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate all traces to a common time grid.
    
    Returns:
        - signals: (n_trials, n_timepoints) array
        - valid_mask: (n_trials, n_timepoints) boolean array
    """
    n_trials = len(traces)
    n_times = len(time_grid)
    
    signals = np.full((n_trials, n_times), np.nan)
    
    for i, trace in enumerate(traces):
        if trace is None:
            continue
        
        t = trace["time"]
        y = trace["signal"]
        
        # Interpolate
        interp_signal = np.interp(time_grid, t, y, left=np.nan, right=np.nan)
        signals[i, :] = interp_signal
    
    valid_mask = ~np.isnan(signals)
    
    return signals, valid_mask


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def extract_all_sessions(config: Config) -> pd.DataFrame:
    """
    Extract decision-aligned traces from all sessions.
    
    Returns DataFrame with columns:
        - mouse, session_id, group, trial_num
        - wait_time
        - time_rel_decision, signal (one row per timepoint)
    """
    all_rows = []
    group_map = get_session_groups()
    photometry_log = load_photometry_log()

    # Find all session directories
    session_dirs = [d for d in config.PROCESSED_OUT.iterdir() if d.is_dir()]

    for session_dir in session_dirs:
        session_id = session_dir.name

        # Load session data
        data = load_session_data(config, session_id, group_map, photometry_log)
        if data is None:
            continue
        
        print(f"Processing {session_id} ({data['mouse']}, {data['group']})")
        
        phot = data["phot"]
        trials = data["trials"]
        events = data["events"]
        mouse = data["mouse"]
        group = data["group"]
        
        # Filter trials — same logic as file 4:
        #   bg_restart  = bg_repeats > 1
        #   miss_trial  = miss_trial column
        #   impulsive   = time_waited < NON_IMPULSIVE_CUTOFF (from config.py)
        impulsive_mask = trials.get("time_waited", pd.Series([float("nan")] * len(trials), dtype=float)).astype(float) < NON_IMPULSIVE_CUTOFF
        good_trials = trials[
            (trials.get("miss_trial", False) == False) &
            (trials.get("bg_repeats", 0).fillna(0) <= 1) &
            (~impulsive_mask)
        ]["session_trial_num"].unique()

        for trial_num in good_trials:
            trace = extract_decision_aligned_trace(
                phot, trials, events, trial_num,
                pre_window=config.PRE_DECISION_WINDOW,
                post_window=config.POST_DECISION_WINDOW,
            )

            if trace is None:
                continue

            # Use group-appropriate wait time for filtering and storage,
            # matching the alignment in file 4 (cue_on for long BG, cue_off for short BG).
            if group == "long":
                wait_time = trace["wait_from_cue_on"]
            else:
                wait_time = trace["wait_from_cue_off"]

            if not np.isfinite(wait_time) or wait_time < config.MIN_WAIT:
                continue

            # Add rows for each timepoint
            for t, s in zip(trace["time"], trace["signal"]):
                all_rows.append({
                    "mouse": mouse,
                    "session_id": session_id,
                    "group": group,
                    "trial_num": trial_num,
                    "wait_time": wait_time,
                    "wait_from_cue_off": trace["wait_from_cue_off"],
                    "wait_from_cue_on": trace["wait_from_cue_on"],
                    "time_rel_decision": t,
                    "signal": s,
                })
    
    return pd.DataFrame(all_rows)


def compute_averages(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Compute time-binned averages per mouse and overall.
    """
    # Create time bins
    time_bins = np.arange(
        -config.PRE_DECISION_WINDOW,
        config.POST_DECISION_WINDOW + config.TIMEBIN_MS/1000,
        config.TIMEBIN_MS / 1000
    )
    
    df["time_bin"] = pd.cut(df["time_rel_decision"], bins=time_bins, labels=time_bins[:-1])
    df["time_bin"] = df["time_bin"].astype(float)
    
    # Per-mouse averages
    mouse_avg = df.groupby(["mouse", "group", "time_bin"]).agg({
        "signal": ["mean", "std", "count"]
    }).reset_index()
    mouse_avg.columns = ["mouse", "group", "time_bin", "mean", "std", "n"]
    mouse_avg["sem"] = mouse_avg["std"] / np.sqrt(mouse_avg["n"])
    
    # Overall averages
    overall_avg = df.groupby(["time_bin"]).agg({
        "signal": ["mean", "std", "count"]
    }).reset_index()
    overall_avg.columns = ["time_bin", "mean", "std", "n"]
    overall_avg["sem"] = overall_avg["std"] / np.sqrt(overall_avg["n"])
    overall_avg["mouse"] = "all"
    overall_avg["group"] = "all"
    
    # Group averages
    group_avg = df.groupby(["group", "time_bin"]).agg({
        "signal": ["mean", "std", "count"]
    }).reset_index()
    group_avg.columns = ["group", "time_bin", "mean", "std", "n"]
    group_avg["sem"] = group_avg["std"] / np.sqrt(group_avg["n"])
    group_avg["mouse"] = group_avg["group"] + "_avg"
    
    return pd.concat([mouse_avg, group_avg, overall_avg], ignore_index=True)


def plot_decision_aligned_figure(avg_df: pd.DataFrame, config: Config):
    """
    Generate Figure 3.1: Decision-aligned DA traces.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Panel A: Overall average
    ax = axes[0]
    overall = avg_df[avg_df["mouse"] == "all"].sort_values("time_bin")
    
    ax.plot(overall["time_bin"], overall["mean"], 
            color=config.COLORS["da_signal"], linewidth=2)
    ax.fill_between(overall["time_bin"], 
                    overall["mean"] - overall["sem"],
                    overall["mean"] + overall["sem"],
                    color=config.COLORS["da_signal"], alpha=0.3)
    
    ax.axvline(0, color="k", linestyle="--", alpha=0.5, label="Decision")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Time from decision (s)")
    ax.set_ylabel("DA (z-scored dF/F)")
    ax.set_title(f"A. All mice (n={overall['n'].max()} trials max)")
    ax.legend()
    
    # Panel B: Group comparison
    ax = axes[1]
    for group in ["long", "short"]:
        grp = avg_df[avg_df["mouse"] == f"{group}_avg"].sort_values("time_bin")
        if len(grp) > 0:
            ax.plot(grp["time_bin"], grp["mean"], 
                    color=config.COLORS[group], linewidth=2, label=f"{group.upper()} BG")
            ax.fill_between(grp["time_bin"],
                            grp["mean"] - grp["sem"],
                            grp["mean"] + grp["sem"],
                            color=config.COLORS[group], alpha=0.2)
    
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Time from decision (s)")
    ax.set_ylabel("DA (z-scored dF/F)")
    ax.set_title("B. Long vs Short BG")
    ax.legend()
    
    # Panel C: Per-mouse traces
    ax = axes[2]
    for mouse in [m for m in avg_df["mouse"].unique() if m not in ("all", "long_avg", "short_avg")]:
        mouse_data = avg_df[avg_df["mouse"] == mouse].sort_values("time_bin")
        if len(mouse_data) > 0:
            ax.plot(mouse_data["time_bin"], mouse_data["mean"],
                    color=config.MOUSE_COLORS.get(mouse, "gray"),
                    linewidth=1.5, alpha=0.8, label=mouse)
    
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Time from decision (s)")
    ax.set_ylabel("DA (z-scored dF/F)")
    ax.set_title("C. Per-mouse traces")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    
    plt.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "fig3_1_ramp_dynamics.png", dpi=300)
    fig.savefig(config.OUTPUT_DIR / "fig3_1_ramp_dynamics.pdf")
    plt.close()
    
    print(f"Figure saved to {config.OUTPUT_DIR / 'fig3_1_ramp_dynamics.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Decision-Aligned Trace Extraction")
    print("="*60)
    
    # Extract traces
    print("\nExtracting traces from all sessions...")
    traces_df = extract_all_sessions(config)
    
    if len(traces_df) == 0:
        print("ERROR: No traces extracted. Check paths and data files.")
        return
    
    print(f"\nExtracted {len(traces_df)} datapoints from {traces_df['session_id'].nunique()} sessions")
    print(f"Mice: {traces_df['mouse'].unique()}")
    
    # Save raw traces
    traces_df.to_csv(config.OUTPUT_DIR / "decision_aligned_traces.csv", index=False)
    
    # Compute averages
    print("\nComputing time-binned averages...")
    avg_df = compute_averages(traces_df, config)
    avg_df.to_csv(config.OUTPUT_DIR / "decision_aligned_averages.csv", index=False)
    
    # Generate figure
    print("\nGenerating figure...")
    plot_decision_aligned_figure(avg_df, config)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

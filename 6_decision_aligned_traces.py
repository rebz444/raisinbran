#!/usr/bin/env python3
"""
Decision-Aligned DA Trace Visualization
========================================

Generate trial-averaged DA traces aligned BACKWARD from decision (first lick).
This avoids temporal smearing across variable wait times and reveals the true
shape of the DA ramp leading up to action.

Key questions this addresses:
1. Are DA signals actually ramping up before decision in all mice?
2. Do Long BG mice show declining DA (suggesting artifact) or late ramps?
3. Does the ramp shape differ between Short and Long BG groups?

Outputs:
- decision_aligned_averages.csv: Time-binned averages per mouse
- fig_decision_aligned_by_mouse.png: Per-mouse traces
- fig_decision_aligned_by_group.png: Group comparison
- fig_decision_aligned_heatmap.png: Trial-by-trial heatmap sorted by wait time

Usage:
    python 6_decision_aligned_traces.py
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from config import (
    BEHAV_DIR,
    DEFAULT_SIGNAL_COL,
    NON_IMPULSIVE_CUTOFF,
    PROCESSED_OUT,
)
from utils import get_grab_channel, get_session_groups, load_photometry_log


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # Paths (from config.py)
    PROCESSED_OUT: Path = PROCESSED_OUT
    BEHAV_DIR: Path = BEHAV_DIR
    OUTPUT_DIR: Path = Path("/Volumes/T7 Shield/photometry/committee_figures")

    # Time windows for alignment
    PRE_DECISION_SEC: float = 5.0    # seconds before decision to include
    POST_DECISION_SEC: float = 1.0   # seconds after decision to include
    BIN_SIZE_MS: int = 50            # bin size for averaging (ms)

    # Signal column (from config.py)
    SIGNAL_COL: str = DEFAULT_SIGNAL_COL

    # Plotting
    MOUSE_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "RZ074": "#1f77b4",
        "RZ075": "#ff7f0e",
        "RZ081": "#2ca02c",
        "RZ082": "#d62728",
        "RZ083": "#9467bd",
        "RZ085": "#8c564b",
    })
    GROUP_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "short": "#E85D24",
        "long": "#2E86AB",
    })


# =============================================================================
# DATA LOADING
# =============================================================================

def load_session_data(
    session_dir: Path,
    group_map: Dict[str, str],
    photometry_log: pd.DataFrame,
) -> Optional[Dict]:
    """
    Load photometry and behavior data for a single session directory.

    Returns dict with phot, trials, events, mouse, date, group, session_id,
    or None if any required file is missing.
    """
    session_id = session_dir.name
    parts = session_id.split("_")
    mouse = parts[0] if parts else None

    if mouse not in group_map:
        return None

    # session_id format: RZ074_20250808_121323
    # Convert YYYYMMDD → YYYY-MM-DD to match photometry log after normalization
    date_raw = parts[1] if len(parts) >= 2 else ""
    date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}" if len(date_raw) == 8 else date_raw

    # Find GRAB photometry channel
    grab_channel = get_grab_channel(mouse, date, photometry_log)
    if grab_channel is None:
        return None

    phot_file = session_dir / f"{grab_channel}.csv"
    if not phot_file.exists():
        return None

    # Locate matching behavior subdirectory
    # behav_analyzed dirs use format: 2025-08-08_12-13-28_RZ074
    behav_subdir = None
    for subdir in BEHAV_DIR.iterdir():
        if date in subdir.name and mouse in subdir.name:
            behav_subdir = subdir
            break
    if behav_subdir is None:
        return None

    trials_files = list(behav_subdir.glob("trials_analyzed_*.csv"))
    events_files = list(behav_subdir.glob("events_processed_*.csv"))
    if not trials_files or not events_files:
        return None

    try:
        return {
            "phot": pd.read_csv(phot_file),
            "trials": pd.read_csv(trials_files[0]),
            "events": pd.read_csv(events_files[0]),
            "mouse": mouse,
            "date": date,
            "group": group_map[mouse],
            "session_id": session_id,
        }
    except Exception as e:
        print(f"  Error loading {session_id}: {e}")
        return None


# =============================================================================
# TRACE EXTRACTION
# =============================================================================

def get_trial_event_times(
    events: pd.DataFrame, trial_num: int
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (cue_on, cue_off, decision) trial-relative times for one trial."""
    te = events[events["session_trial_num"] == trial_num]
    cue_on_rows = te[te["state"] == "in_background"]["trial_time"]
    cue_off_rows = te[te["state"] == "in_wait"]["trial_time"]
    decision_rows = te[te["state"] == "in_consumption"]["trial_time"]

    cue_on = float(cue_on_rows.iloc[0]) if len(cue_on_rows) else None
    cue_off = float(cue_off_rows.iloc[0]) if len(cue_off_rows) else None
    decision = float(decision_rows.iloc[0]) if len(decision_rows) else None
    return cue_on, cue_off, decision


def extract_decision_aligned_trace(
    phot: pd.DataFrame,
    trials: pd.DataFrame,
    events: pd.DataFrame,
    trial_num: int,
    group: str,
    config: Config,
) -> Optional[Dict]:
    """
    Extract photometry trace aligned to decision time for a single trial.

    Uses group-appropriate wait time: long BG → time from cue_on, short BG → from cue_off.
    Filtered against NON_IMPULSIVE_CUTOFF.

    Returns dict with time_rel, signal, iso, wait_time, or None to skip.
    """
    cue_on, cue_off, decision = get_trial_event_times(events, trial_num)
    if decision is None or cue_off is None:
        return None

    wait_from_cue_off = decision - cue_off
    wait_from_cue_on = (decision - cue_on) if cue_on is not None else float("nan")
    wait_time = wait_from_cue_on if group == "long" else wait_from_cue_off

    if not np.isfinite(wait_time) or wait_time < NON_IMPULSIVE_CUTOFF:
        return None

    # Convert trial-relative decision time to session-absolute seconds
    trial_row = trials[trials["session_trial_num"] == trial_num]
    if len(trial_row) == 0:
        return None

    first_trial_start = float(trials.iloc[0]["start_time"])
    trial_start_sec = float(trial_row.iloc[0]["start_time"]) - first_trial_start
    abs_decision = trial_start_sec + decision

    # Extract photometry window
    mask = (
        (phot["t_sec"] >= abs_decision - config.PRE_DECISION_SEC)
        & (phot["t_sec"] <= abs_decision + config.POST_DECISION_SEC)
    )
    segment = phot[mask]
    if len(segment) < 10:
        return None

    time_rel = segment["t_sec"].values - abs_decision
    signal = segment[config.SIGNAL_COL].values if config.SIGNAL_COL in segment.columns else None
    if signal is None:
        return None

    iso = segment["iso"].values if "iso" in segment.columns else None

    return {
        "time_rel": time_rel,
        "signal": signal,
        "iso": iso,
        "wait_time": wait_time,
        "wait_from_cue_off": wait_from_cue_off,
        "wait_from_cue_on": wait_from_cue_on,
    }


def process_session(data: Dict, config: Config) -> List[Dict]:
    """
    Extract all valid decision-aligned traces from one session.
    Applies miss_trial, bg_repeats, and NON_IMPULSIVE_CUTOFF filters.
    """
    trials = data["trials"]
    events = data["events"]
    phot = data["phot"]
    mouse = data["mouse"]
    group = data["group"]
    session_id = data["session_id"]

    # Good-trial mask (same logic as file 4 / 5b)
    miss_mask = trials.get("miss_trial", pd.Series(False, index=trials.index)).astype(bool)
    repeat_mask = trials.get("bg_repeats", pd.Series(0, index=trials.index)).fillna(0).astype(int) > 1
    good_trial_nums = trials[~miss_mask & ~repeat_mask]["session_trial_num"].unique()

    traces = []
    for trial_num in good_trial_nums:
        trace = extract_decision_aligned_trace(phot, trials, events, trial_num, group, config)
        if trace is not None:
            trace["mouse"] = mouse
            trace["session_id"] = session_id
            trace["group"] = group
            traces.append(trace)

    return traces


# =============================================================================
# AVERAGING AND BINNING
# =============================================================================

def interpolate_to_common_grid(
    traces: List[Dict],
    time_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate all traces to a common time grid.

    Returns:
        signals: (n_trials, n_timepoints) array
        isos:    (n_trials, n_timepoints) array
        wait_times: (n_trials,) array
    """
    n = len(traces)
    n_t = len(time_grid)

    signals = np.full((n, n_t), np.nan)
    isos = np.full((n, n_t), np.nan)
    wait_times = np.array([t["wait_time"] for t in traces])

    for i, trace in enumerate(traces):
        signals[i] = np.interp(time_grid, trace["time_rel"], trace["signal"], left=np.nan, right=np.nan)
        if trace["iso"] is not None:
            isos[i] = np.interp(time_grid, trace["time_rel"], trace["iso"], left=np.nan, right=np.nan)

    return signals, isos, wait_times


def compute_averages(traces: List[Dict], config: Config) -> pd.DataFrame:
    """Compute time-binned mean ± SEM per mouse."""
    time_grid = np.arange(
        -config.PRE_DECISION_SEC,
        config.POST_DECISION_SEC + config.BIN_SIZE_MS / 1000,
        config.BIN_SIZE_MS / 1000,
    )

    rows = []
    for mouse in sorted(set(t["mouse"] for t in traces)):
        mouse_traces = [t for t in traces if t["mouse"] == mouse]
        if len(mouse_traces) < 5:
            continue

        signals, _, wait_times = interpolate_to_common_grid(mouse_traces, time_grid)
        group = mouse_traces[0]["group"]

        for i, t in enumerate(time_grid):
            valid = signals[:, i][np.isfinite(signals[:, i])]
            if len(valid) < 3:
                continue
            rows.append({
                "mouse": mouse,
                "group": group,
                "time": t,
                "mean": np.mean(valid),
                "std": np.std(valid),
                "sem": np.std(valid) / np.sqrt(len(valid)),
                "n": len(valid),
                "median_wait": np.nanmedian(wait_times),
            })

    return pd.DataFrame(rows)


# =============================================================================
# PLOTTING
# =============================================================================

def setup_figure_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _add_decision_lines(ax):
    ax.axvline(0, color="k", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)


def plot_by_mouse(avg_df: pd.DataFrame, config: Config, output_dir: Path):
    """Per-mouse decision-aligned trace panels."""
    mice = sorted(avg_df["mouse"].unique())
    n_cols = min(3, len(mice))
    n_rows = int(np.ceil(len(mice) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False)

    for i, mouse in enumerate(mice):
        ax = axes[i // n_cols, i % n_cols]
        mdf = avg_df[avg_df["mouse"] == mouse].sort_values("time")
        color = config.MOUSE_COLORS.get(mouse, "gray")
        group = mdf["group"].iloc[0] if len(mdf) else "unknown"

        ax.plot(mdf["time"], mdf["mean"], color=color, linewidth=2)
        ax.fill_between(mdf["time"], mdf["mean"] - mdf["sem"], mdf["mean"] + mdf["sem"],
                        color=color, alpha=0.3)
        _add_decision_lines(ax)
        ax.set_xlabel("Time from decision (s)")
        ax.set_ylabel("DA (z-scored)")
        ax.set_title(
            f"{mouse} ({group.upper()} BG)\n"
            f"wait={mdf['median_wait'].iloc[0]:.1f}s, n={mdf['n'].max()}"
        )

    for i in range(len(mice), n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / "fig_decision_aligned_by_mouse.png")
    plt.close()
    print(f"Saved: {output_dir / 'fig_decision_aligned_by_mouse.png'}")


def plot_by_group(avg_df: pd.DataFrame, config: Config, output_dir: Path):
    """Group-comparison figure: group average, short mice, long mice."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: group averages
    ax = axes[0]
    for group in ("short", "long"):
        gdf = avg_df[avg_df["group"] == group]
        if len(gdf) == 0:
            continue
        grp = gdf.groupby("time").agg(
            mean=("mean", "mean"),
            sem=("sem", lambda x: np.sqrt(np.sum(x**2)) / len(x)),
        ).reset_index()
        color = config.GROUP_COLORS.get(group, "gray")
        ax.plot(grp["time"], grp["mean"], color=color, linewidth=2, label=f"{group.upper()} BG")
        ax.fill_between(grp["time"], grp["mean"] - grp["sem"], grp["mean"] + grp["sem"],
                        color=color, alpha=0.2)
    _add_decision_lines(ax)
    ax.set_xlabel("Time from decision (s)")
    ax.set_ylabel("DA (z-scored)")
    ax.set_title("A. Group comparison")
    ax.legend()

    # Panels B & C: individual mice per group
    for ax, group, label in zip(axes[1:], ("short", "long"), ("B. Short BG mice", "C. Long BG mice")):
        group_mice = [m for m in avg_df["mouse"].unique() if avg_df.loc[avg_df["mouse"] == m, "group"].iloc[0] == group]
        for mouse in group_mice:
            mdf = avg_df[avg_df["mouse"] == mouse].sort_values("time")
            ax.plot(mdf["time"], mdf["mean"], color=config.MOUSE_COLORS.get(mouse, "gray"),
                    linewidth=1.5, alpha=0.8, label=mouse)
        _add_decision_lines(ax)
        ax.set_xlabel("Time from decision (s)")
        ax.set_ylabel("DA (z-scored)")
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_decision_aligned_by_group.png")
    plt.close()
    print(f"Saved: {output_dir / 'fig_decision_aligned_by_group.png'}")


def plot_heatmap_by_wait(
    traces: List[Dict],
    config: Config,
    output_dir: Path,
    mice_to_plot: Optional[List[str]] = None,
):
    """Trial-by-trial heatmap sorted by wait time."""
    if mice_to_plot is None:
        all_mice = list(set(t["mouse"] for t in traces))
        short = [m for m in all_mice if config.GROUP_COLORS.get("short") and
                 next((t["group"] for t in traces if t["mouse"] == m), "") == "short"]
        long = [m for m in all_mice if
                next((t["group"] for t in traces if t["mouse"] == m), "") == "long"]
        mice_to_plot = ([short[0]] if short else []) + ([long[0]] if long else [])

    if not mice_to_plot:
        print("No valid mice for heatmap")
        return

    time_grid = np.arange(
        -config.PRE_DECISION_SEC,
        config.POST_DECISION_SEC + config.BIN_SIZE_MS / 1000,
        config.BIN_SIZE_MS / 1000,
    )

    fig, axes = plt.subplots(1, len(mice_to_plot), figsize=(6 * len(mice_to_plot), 5))
    if len(mice_to_plot) == 1:
        axes = [axes]

    for ax, mouse in zip(axes, mice_to_plot):
        mouse_traces = [t for t in traces if t["mouse"] == mouse]
        if len(mouse_traces) < 10:
            ax.set_title(f"{mouse}: insufficient data")
            continue

        signals, _, wait_times = interpolate_to_common_grid(mouse_traces, time_grid)
        sort_idx = np.argsort(wait_times)
        signals_sorted = signals[sort_idx]
        wait_sorted = wait_times[sort_idx]

        vmax = np.nanpercentile(np.abs(signals_sorted), 95)
        im = ax.imshow(
            signals_sorted, aspect="auto",
            extent=[time_grid[0], time_grid[-1], len(signals_sorted), 0],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        )
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Time from decision (s)")
        ax.set_ylabel("Trials (sorted by wait time)")
        group = next((t["group"] for t in traces if t["mouse"] == mouse), "unknown")
        ax.set_title(f"{mouse} ({group.upper()} BG)\n{len(mouse_traces)} trials")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("DA (z)")
        ax.text(time_grid[-1] + 0.1, 0, f"wait={wait_sorted[-1]:.1f}s", fontsize=8, va="top")
        ax.text(time_grid[-1] + 0.1, len(signals_sorted), f"wait={wait_sorted[0]:.1f}s",
                fontsize=8, va="bottom")

    plt.tight_layout()
    fig.savefig(output_dir / "fig_decision_aligned_heatmap.png")
    plt.close()
    print(f"Saved: {output_dir / 'fig_decision_aligned_heatmap.png'}")


def plot_slope_comparison(traces: List[Dict], config: Config, output_dir: Path):
    """Steep vs shallow ramp comparison within each group."""
    time_grid = np.arange(
        -config.PRE_DECISION_SEC,
        config.POST_DECISION_SEC + config.BIN_SIZE_MS / 1000,
        config.BIN_SIZE_MS / 1000,
    )

    for trace in traces:
        t, sig = trace["time_rel"], trace["signal"]
        mask = (t >= -3) & (t <= -0.2)
        trace["slope"] = np.polyfit(t[mask], sig[mask], 1)[0] if mask.sum() >= 5 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, group in zip(axes, ("short", "long")):
        group_traces = [t for t in traces if t["group"] == group and np.isfinite(t.get("slope", np.nan))]
        if len(group_traces) < 20:
            ax.set_title(f"{group.upper()} BG: insufficient data")
            continue

        slopes = np.array([t["slope"] for t in group_traces])
        median_slope = np.median(slopes)
        color = config.GROUP_COLORS.get(group, "blue")

        import warnings
        for subset, label, c in [
            ([t for t in group_traces if t["slope"] > median_slope],
             f"Steep (>{median_slope:.2f})", color),
            ([t for t in group_traces if t["slope"] <= median_slope],
             f"Shallow (<={median_slope:.2f})", "gray"),
        ]:
            sigs, _, _ = interpolate_to_common_grid(subset, time_grid)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_sig = np.nanmean(sigs, axis=0)
                sem_sig = np.nanstd(sigs, axis=0) / np.sqrt(np.sum(np.isfinite(sigs), axis=0))
            ax.plot(time_grid, mean_sig, linewidth=2, label=label, color=c)
            ax.fill_between(time_grid, mean_sig - sem_sig, mean_sig + sem_sig, alpha=0.2, color=c)

        _add_decision_lines(ax)
        ax.set_xlabel("Time from decision (s)")
        ax.set_ylabel("DA (z-scored)")
        ax.set_title(f"{group.upper()} BG: Steep vs Shallow ramps")
        ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "fig_slope_quartile_comparison.png")
    plt.close()
    print(f"Saved: {output_dir / 'fig_slope_quartile_comparison.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_figure_style()

    print("=" * 70)
    print("Decision-Aligned DA Trace Analysis")
    print("=" * 70)

    print("\nLoading session metadata...")
    group_map = get_session_groups()
    photometry_log = load_photometry_log()

    print(f"Group map: {len(group_map)} mice")

    print("\nExtracting decision-aligned traces...")
    all_traces = []

    for session_dir in sorted(config.PROCESSED_OUT.iterdir()):
        if not session_dir.is_dir():
            continue

        data = load_session_data(session_dir, group_map, photometry_log)
        if data is None:
            continue

        traces = process_session(data, config)
        if traces:
            all_traces.extend(traces)
            print(f"  {data['session_id']}: {len(traces)} trials")

    if not all_traces:
        print("ERROR: No traces extracted!")
        return

    print(f"\nTotal traces: {len(all_traces)}")
    print(f"Mice: {sorted(set(t['mouse'] for t in all_traces))}")

    print("\nComputing time-binned averages...")
    avg_df = compute_averages(all_traces, config)
    avg_df.to_csv(config.OUTPUT_DIR / "decision_aligned_averages.csv", index=False)

    print("\nGenerating figures...")
    print("  - Per-mouse traces...")
    plot_by_mouse(avg_df, config, config.OUTPUT_DIR)

    print("  - Group comparison...")
    plot_by_group(avg_df, config, config.OUTPUT_DIR)

    print("  - Trial heatmaps...")
    short_mice = [m for m in set(t["mouse"] for t in all_traces)
                  if group_map.get(m) == "short"]
    long_mice = [m for m in set(t["mouse"] for t in all_traces)
                 if group_map.get(m) == "long"]
    plot_heatmap_by_wait(
        all_traces, config, config.OUTPUT_DIR,
        ([short_mice[0]] if short_mice else []) + ([long_mice[0]] if long_mice else []),
    )

    print("  - Slope quartile comparison...")
    plot_slope_comparison(all_traces, config, config.OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    for group in ("short", "long"):
        group_traces = [t for t in all_traces if t["group"] == group]
        if not group_traces:
            continue
        wait_times = [t["wait_time"] for t in group_traces]
        mice = sorted(set(t["mouse"] for t in group_traces))
        print(f"\n{group.upper()} BG:")
        print(f"  Mice: {mice}")
        print(f"  Total trials: {len(group_traces)}")
        print(f"  Mean wait: {np.mean(wait_times):.2f}s")
        print(f"  Median wait: {np.median(wait_times):.2f}s")

    print(f"\nDone! Figures saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

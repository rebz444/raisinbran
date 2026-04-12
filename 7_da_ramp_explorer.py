#!/usr/bin/env python3
"""
DA Ramp Explorer + Quartile Analysis
=====================================

Systematic exploration of DA dynamics across all combinations of:
- Groups: all mice, short BG, long BG, per-mouse
- Direction: forward (anchor → decision), backward (decision ← anchor)
- Anchor: last_lick, cue_on, cue_off

For each combination:
- Average trace (mean ± SEM)
- Slope in the anchor→decision window
- Correlation: slope vs window duration
- Heatmap sorted by window duration

For backward-aligned traces, also runs quartile analysis:
- Trials split into quartiles by window duration
- Average trace per quartile
- Mean slope per quartile (bar plot)
- Heatmap with quartile boundaries

Outputs:
├── traces_forward/
├── traces_backward/
├── slope_scatter_forward/
├── slope_scatter_backward/
├── heatmaps_forward/
├── heatmaps_backward/
├── quartile_traces/
├── quartile_slopes/
├── quartile_heatmaps/
├── summary_statistics.csv
└── quartile_summary.csv

Usage:
    python 7_da_ramp_explorer.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from config import BEHAV_DIR, DEFAULT_SIGNAL_COL, PROCESSED_OUT
from utils import get_grab_channel, get_session_groups, load_photometry_log


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # Paths
    PROCESSED_OUT: Path = PROCESSED_OUT
    BEHAV_DIR: Path = BEHAV_DIR
    OUTPUT_DIR: Path = Path("/Volumes/T7 Shield/photometry/da_ramp_analysis")

    # Signal
    SIGNAL_COL: str = DEFAULT_SIGNAL_COL

    # Timing
    POST_DECISION_SEC: float = 1.0
    BIN_SIZE_MS: int = 50
    MIN_WINDOW_SEC: float = 0.5
    SLOPE_BUFFER_START: float = 0.3
    SLOPE_BUFFER_END: float = 0.2

    # Anchors
    ANCHORS: Tuple[str, ...] = ("last_lick", "cue_on", "cue_off")

    # Quartile analysis
    N_QUARTILES: int = 4

    # Colors
    GROUP_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "short": "#E85D24",
        "long":  "#2E86AB",
        "all":   "#555555",
    })
    MOUSE_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "RZ074": "#1f77b4",
        "RZ075": "#ff7f0e",
        "RZ081": "#2ca02c",
        "RZ082": "#d62728",
        "RZ083": "#9467bd",
        "RZ085": "#8c564b",
    })
    QUARTILE_COLORS: List[str] = field(default_factory=lambda: [
        "#a6cee3",  # Q1 (shortest)
        "#1f78b4",  # Q2
        "#b2df8a",  # Q3
        "#33a02c",  # Q4 (longest)
    ])


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_sessions(config: Config) -> List[Dict]:
    """Load all sessions with photometry and behavior data."""
    group_map = get_session_groups()
    photometry_log = load_photometry_log()
    all_sessions = []

    for session_dir in sorted(config.PROCESSED_OUT.iterdir()):
        if not session_dir.is_dir():
            continue

        session_id = session_dir.name
        parts = session_id.split("_")
        mouse = parts[0] if parts else None

        if mouse not in group_map:
            continue

        date_raw = parts[1] if len(parts) >= 2 else ""
        date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}" if len(date_raw) == 8 else date_raw

        grab_channel = get_grab_channel(mouse, date, photometry_log)
        if grab_channel is None:
            continue

        phot_file = session_dir / f"{grab_channel}.csv"
        if not phot_file.exists():
            continue

        behav_subdir = None
        for subdir in config.BEHAV_DIR.iterdir():
            if date in subdir.name and mouse in subdir.name:
                behav_subdir = subdir
                break
        if behav_subdir is None:
            continue

        trials_files = list(behav_subdir.glob("trials_analyzed_*.csv"))
        events_files = list(behav_subdir.glob("events_processed_*.csv"))
        if not trials_files or not events_files:
            continue

        try:
            all_sessions.append({
                "phot":       pd.read_csv(phot_file),
                "trials":     pd.read_csv(trials_files[0]),
                "events":     pd.read_csv(events_files[0]),
                "mouse":      mouse,
                "date":       date,
                "group":      group_map[mouse],
                "session_id": session_id,
            })
        except Exception as e:
            print(f"  Error loading {session_id}: {e}")

    return all_sessions


# =============================================================================
# TRIAL EXTRACTION
# =============================================================================

@dataclass
class TrialData:
    """All timing and signal data for a single trial."""
    trial_num: int
    mouse: str
    group: str
    session_id: str

    # Event times (trial-relative)
    cue_on: float
    cue_off: float
    decision: float
    last_lick: float

    # Window durations (anchor → decision)
    window_last_lick: float
    window_cue_on: float
    window_cue_off: float

    # Photometry
    phot_t_abs: np.ndarray
    phot_signal: np.ndarray
    trial_start_abs: float


def extract_trial_data(session: Dict, config: Config) -> List[TrialData]:
    """Extract all valid trials from a session."""
    phot       = session["phot"]
    trials     = session["trials"]
    events     = session["events"]
    mouse      = session["mouse"]
    group      = session["group"]
    session_id = session["session_id"]

    first_trial_start = float(trials.iloc[0]["start_time"])
    phot_t     = phot["t_sec"].to_numpy(dtype=float)
    phot_signal = phot[config.SIGNAL_COL].to_numpy(dtype=float)

    extracted = []

    for _, trial_row in trials.iterrows():
        trial_num = int(trial_row["session_trial_num"])

        te = events[events["session_trial_num"] == trial_num]
        cue_on_rows  = te[te["state"] == "in_background"]["trial_time"]
        cue_off_rows = te[te["state"] == "in_wait"]["trial_time"]

        cue_on  = float(cue_on_rows.iloc[0])  if len(cue_on_rows)  else np.nan
        cue_off = float(cue_off_rows.iloc[0]) if len(cue_off_rows) else np.nan

        # Decision time from trials table
        twsco = trial_row.get("time_waited_since_cue_on")
        if pd.isna(twsco) or not np.isfinite(float(cue_on)):
            continue
        decision = cue_on + float(twsco)

        if "time_waited_since_last_lick" in trial_row and pd.notna(trial_row["time_waited_since_last_lick"]):
            last_lick = decision - float(trial_row["time_waited_since_last_lick"])
        else:
            last_lick = np.nan

        window_last_lick = decision - last_lick if np.isfinite(last_lick) else np.nan
        window_cue_on    = float(twsco)  # == decision - cue_on by construction
        window_cue_off   = decision - cue_off if np.isfinite(cue_off) else np.nan

        if window_cue_off < config.MIN_WINDOW_SEC:
            continue

        trial_start_abs = float(trial_row["start_time"]) - first_trial_start

        extracted.append(TrialData(
            trial_num=trial_num,
            mouse=mouse,
            group=group,
            session_id=session_id,
            cue_on=cue_on,
            cue_off=cue_off,
            decision=decision,
            last_lick=last_lick,
            window_last_lick=window_last_lick,
            window_cue_on=window_cue_on,
            window_cue_off=window_cue_off,
            phot_t_abs=phot_t,
            phot_signal=phot_signal,
            trial_start_abs=trial_start_abs,
        ))

    return extracted


# =============================================================================
# TRACE EXTRACTION
# =============================================================================

def get_trace_forward(trial: TrialData, anchor: str, config: Config) -> Optional[Dict]:
    """Forward-aligned trace: anchor at t=0, extends toward decision."""
    anchor_time = getattr(trial, anchor)
    if not np.isfinite(anchor_time):
        return None

    window_duration = getattr(trial, f"window_{anchor}")
    if not np.isfinite(window_duration) or window_duration < config.MIN_WINDOW_SEC:
        return None

    anchor_abs   = trial.trial_start_abs + anchor_time
    decision_abs = trial.trial_start_abs + trial.decision
    end_abs      = decision_abs + config.POST_DECISION_SEC

    mask = (trial.phot_t_abs >= anchor_abs) & (trial.phot_t_abs <= end_abs)
    if mask.sum() < 10:
        return None

    time_rel = trial.phot_t_abs[mask] - anchor_abs
    signal   = trial.phot_signal[mask]

    return {
        "time_rel":        time_rel,
        "signal":          signal,
        "window_duration": window_duration,
        "mouse":           trial.mouse,
        "group":           trial.group,
        "trial_num":       trial.trial_num,
        "session_id":      trial.session_id,
    }


def get_trace_backward(trial: TrialData, anchor: str, config: Config) -> Optional[Dict]:
    """Backward-aligned trace: decision at t=0, extends back to anchor."""
    anchor_time = getattr(trial, anchor)
    if not np.isfinite(anchor_time):
        return None

    window_duration = getattr(trial, f"window_{anchor}")
    if not np.isfinite(window_duration) or window_duration < config.MIN_WINDOW_SEC:
        return None

    anchor_abs   = trial.trial_start_abs + anchor_time
    decision_abs = trial.trial_start_abs + trial.decision
    end_abs      = decision_abs + config.POST_DECISION_SEC

    mask = (trial.phot_t_abs >= anchor_abs) & (trial.phot_t_abs <= end_abs)
    if mask.sum() < 10:
        return None

    time_rel = trial.phot_t_abs[mask] - decision_abs
    signal   = trial.phot_signal[mask]

    # Baseline-subtract using first 200 ms at the anchor so all traces
    # start near 0 regardless of session-level z-score offset
    baseline_mask = (time_rel >= -window_duration) & (time_rel <= -window_duration + 0.2)
    if baseline_mask.sum() >= 2:
        signal = signal - np.mean(signal[baseline_mask])

    return {
        "time_rel":        time_rel,
        "signal":          signal,
        "window_duration": window_duration,
        "mouse":           trial.mouse,
        "group":           trial.group,
        "trial_num":       trial.trial_num,
        "session_id":      trial.session_id,
    }


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def compute_slope(
    time_rel: np.ndarray,
    signal: np.ndarray,
    window_duration: float,
    direction: str,
    config: Config,
) -> Tuple[float, float]:
    """Compute slope in the valid window. Returns (slope, r_squared)."""
    if direction == "forward":
        start = config.SLOPE_BUFFER_START
        end   = window_duration - config.SLOPE_BUFFER_END
    else:
        start = -window_duration + config.SLOPE_BUFFER_START
        end   = -config.SLOPE_BUFFER_END

    if end <= start + 0.2:
        return np.nan, np.nan

    mask = (time_rel >= start) & (time_rel <= end)
    t = time_rel[mask]
    y = signal[mask]

    if len(t) < 5:
        return np.nan, np.nan

    t_mean = np.mean(t)
    y_mean = np.mean(y)
    tt = t - t_mean
    yy = y - y_mean
    denom = np.sum(tt ** 2)
    if denom == 0:
        return np.nan, np.nan

    slope = np.sum(tt * yy) / denom
    yhat  = slope * (t - t_mean) + y_mean
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum(yy ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(slope), float(r2)


def interpolate_traces_to_grid(
    traces: List[Dict],
    time_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate traces to a common time grid. Returns (signals, window_durations)."""
    signals = np.full((len(traces), len(time_grid)), np.nan)
    windows = np.full(len(traces), np.nan)
    for i, trace in enumerate(traces):
        signals[i, :] = np.interp(time_grid, trace["time_rel"], trace["signal"],
                                   left=np.nan, right=np.nan)
        windows[i] = trace["window_duration"]
    return signals, windows


def compute_correlation(slopes: np.ndarray, windows: np.ndarray) -> Tuple[float, float, int]:
    """Pearson r between slopes and window durations. Returns (r, p, n)."""
    mask = np.isfinite(slopes) & np.isfinite(windows)
    n = mask.sum()
    if n < 5:
        return np.nan, np.nan, n

    s, w = slopes[mask], windows[mask]
    r = np.corrcoef(s, w)[0, 1]

    if abs(r) < 1:
        from scipy import stats
        t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
        p = 2 * stats.t.sf(abs(t_stat), n - 2)
    else:
        p = 0.0

    return float(r), float(p), int(n)


def assign_quartiles(traces: List[Dict], n_quartiles: int = 4) -> List[Dict]:
    """Assign quartile labels based on window_duration (modifies in place)."""
    windows = np.array([t["window_duration"] for t in traces])
    edges = np.percentile(windows, np.linspace(0, 100, n_quartiles + 1))
    for trace in traces:
        w = trace["window_duration"]
        for q in range(n_quartiles):
            if w >= edges[q] and (w < edges[q + 1] or q == n_quartiles - 1):
                trace["quartile"]       = q + 1
                trace["quartile_range"] = f"{edges[q]:.1f}-{edges[q+1]:.1f}s"
                break
    return traces


# =============================================================================
# PLOTTING — STANDARD EXPLORATION
# =============================================================================

def setup_style():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.dpi":        150,
        "savefig.bbox":       "tight",
    })


def plot_average_trace(ax, traces, time_grid, color, label, direction):
    """Plot mean ± SEM trace."""
    if len(traces) < 3:
        return
    signals, windows = interpolate_traces_to_grid(traces, time_grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_sig = np.nanmean(signals, axis=0)
        n_valid  = np.sum(np.isfinite(signals), axis=0)
        sem_sig  = np.nanstd(signals, axis=0) / np.sqrt(n_valid)

    ax.plot(time_grid, mean_sig, color=color, linewidth=2, label=label)
    ax.fill_between(time_grid, mean_sig - sem_sig, mean_sig + sem_sig, color=color, alpha=0.2)

    if direction == "backward":
        ax.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    else:
        ax.axvline(np.nanmedian(windows), color="k", linestyle="--", linewidth=1, alpha=0.5)


def plot_traces_grid(all_traces, config, direction, output_dir, level):
    """Grid of average traces (rows = groups/mice, cols = anchors)."""
    anchors = config.ANCHORS
    names   = sorted(all_traces.keys())
    n_rows, n_cols = len(names), len(anchors)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    if direction == "forward":
        max_window = max(
            max((t["window_duration"] for t in traces), default=0)
            for traces in all_traces.values() if traces
        )
        time_grid = np.arange(0, min(max_window + config.POST_DECISION_SEC, 15),
                              config.BIN_SIZE_MS / 1000)
    else:
        max_window = max(
            max((t["window_duration"] for t in traces), default=0)
            for traces in all_traces.values() if traces
        )
        time_grid = np.arange(-min(max_window, 12), config.POST_DECISION_SEC,
                              config.BIN_SIZE_MS / 1000)

    for row, name in enumerate(names):
        for col, anchor in enumerate(anchors):
            ax = axes[row, col]
            traces = [t for t in all_traces[name] if t.get("anchor") == anchor]
            if len(traces) < 3:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{name} | {anchor}")
                continue

            color = (config.GROUP_COLORS.get("all") if level == "all"
                     else config.GROUP_COLORS.get(name, "#555555") if level == "group"
                     else config.MOUSE_COLORS.get(name, "#555555"))

            plot_average_trace(ax, traces, time_grid, color, name, direction)

            slopes  = []
            windows = []
            for t in traces:
                slope, _ = compute_slope(t["time_rel"], t["signal"],
                                         t["window_duration"], direction, config)
                slopes.append(slope)
                windows.append(t["window_duration"])

            r, p, n = compute_correlation(np.array(slopes), np.array(windows))
            ax.set_title(f"{name} | {anchor}\nn={n}, slope={np.nanmean(slopes):.3f}, r={r:.2f}")
            ax.set_xlabel("Time (s)" if row == n_rows - 1 else "")
            ax.set_ylabel("DA (z)" if col == 0 else "")
            ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)

    direction_label = "Forward (anchor → decision)" if direction == "forward" else "Backward (decision ← anchor)"
    fig.suptitle(f"{direction_label} | Level: {level}", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / f"traces_{level}.png")
    plt.close()
    print(f"  Saved: {output_dir / f'traces_{level}.png'}")


def plot_slope_scatter(all_traces, config, direction, output_dir, level):
    """Slope vs window duration scatter for each group/mouse × anchor."""
    anchors = config.ANCHORS
    names   = sorted(all_traces.keys())
    n_rows, n_cols = len(names), len(anchors)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

    for row, name in enumerate(names):
        for col, anchor in enumerate(anchors):
            ax = axes[row, col]
            traces = [t for t in all_traces[name] if t.get("anchor") == anchor]
            if len(traces) < 10:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{name} | {anchor}")
                continue

            slopes  = np.array([compute_slope(t["time_rel"], t["signal"],
                                              t["window_duration"], direction, config)[0]
                                 for t in traces])
            windows = np.array([t["window_duration"] for t in traces])
            mask    = np.isfinite(slopes) & np.isfinite(windows)
            s, w    = slopes[mask], windows[mask]

            if len(s) < 10:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{name} | {anchor}")
                continue

            color = (config.GROUP_COLORS.get("all") if level == "all"
                     else config.GROUP_COLORS.get(name, "#555555") if level == "group"
                     else config.MOUSE_COLORS.get(name, "#555555"))

            ax.scatter(w, s, c=color, alpha=0.3, s=10, edgecolors="none")
            r, p, n = compute_correlation(slopes, windows)
            if np.isfinite(r):
                z = np.polyfit(w, s, 1)
                x_line = np.array([w.min(), w.max()])
                ax.plot(x_line, z[0] * x_line + z[1], color=color, linewidth=2)

            ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
            ax.set_xlabel("Window duration (s)" if row == n_rows - 1 else "")
            ax.set_ylabel("Slope (z/s)" if col == 0 else "")
            p_str = f"p={p:.3f}" if p >= 0.001 else f"p={p:.1e}"
            ax.set_title(f"{name} | {anchor}\nr={r:.3f}, {p_str}, n={n}")

    direction_label = "Forward" if direction == "forward" else "Backward"
    fig.suptitle(f"Slope vs Window Duration | {direction_label} | Level: {level}",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / f"slope_scatter_{level}.png")
    plt.close()
    print(f"  Saved: {output_dir / f'slope_scatter_{level}.png'}")


def plot_heatmap(traces, config, direction, output_dir, name, anchor):
    """Trial-by-trial heatmap sorted by window duration."""
    if len(traces) < 20:
        return

    max_window = max(t["window_duration"] for t in traces)
    if direction == "forward":
        time_grid = np.arange(0, min(max_window + config.POST_DECISION_SEC, 12),
                              config.BIN_SIZE_MS / 1000)
    else:
        time_grid = np.arange(-min(max_window, 10), config.POST_DECISION_SEC,
                              config.BIN_SIZE_MS / 1000)

    signals, windows = interpolate_traces_to_grid(traces, time_grid)
    sort_idx         = np.argsort(windows)
    signals_sorted   = signals[sort_idx]
    windows_sorted   = windows[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.nanpercentile(np.abs(signals_sorted), 95)
    im = ax.imshow(
        signals_sorted, aspect="auto",
        extent=[time_grid[0], time_grid[-1], len(signals_sorted), 0],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
    )
    if direction == "backward":
        ax.axvline(0, color="k", linestyle="--", linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trials (sorted by window duration)")
    ax.set_title(f"{name} | {anchor} | {'Forward' if direction == 'forward' else 'Backward'}\n{len(traces)} trials")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("DA (z)")
    ax.text(time_grid[-1] + 0.1, 0,              f"win={windows_sorted[-1]:.1f}s", fontsize=8, va="top")
    ax.text(time_grid[-1] + 0.1, len(signals_sorted), f"win={windows_sorted[0]:.1f}s",  fontsize=8, va="bottom")

    plt.tight_layout()
    fig.savefig(output_dir / f"heatmap_{name}_{anchor}.png")
    plt.close()


# =============================================================================
# PLOTTING — QUARTILE ANALYSIS (backward only)
# =============================================================================

def plot_quartile_traces(traces, config, output_path, title):
    """Average trace per quartile (backward-aligned)."""
    n_q    = config.N_QUARTILES
    traces = assign_quartiles(traces, n_q)

    max_window = max(t["window_duration"] for t in traces)
    time_grid  = np.arange(-min(max_window, 12), config.POST_DECISION_SEC,
                            config.BIN_SIZE_MS / 1000)

    fig, ax = plt.subplots(figsize=(10, 6))
    quartile_stats = []

    for q in range(1, n_q + 1):
        q_traces = [t for t in traces if t.get("quartile") == q]
        if len(q_traces) < 10:
            continue

        signals, _ = interpolate_traces_to_grid(q_traces, time_grid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_sig = np.nanmean(signals, axis=0)
            n_valid  = np.sum(np.isfinite(signals), axis=0)
            sem_sig  = np.nanstd(signals, axis=0) / np.sqrt(n_valid)

        q_range = q_traces[0].get("quartile_range", f"Q{q}")
        slopes  = np.array([compute_slope(t["time_rel"], t["signal"],
                                             t["window_duration"], "backward", config)[0]
                                for t in q_traces])
        mean_slope = np.nanmean(slopes)

        color = config.QUARTILE_COLORS[q - 1]
        label = f"Q{q}: {q_range} (n={len(q_traces)}, slope={mean_slope:.2f})"
        ax.plot(time_grid, mean_sig, color=color, linewidth=2, label=label)
        ax.fill_between(time_grid, mean_sig - sem_sig, mean_sig + sem_sig,
                        color=color, alpha=0.2)

        quartile_stats.append({
            "quartile":    q,
            "range":       q_range,
            "n_trials":    len(q_traces),
            "mean_window": np.mean([t["window_duration"] for t in q_traces]),
            "mean_slope":  mean_slope,
            "std_slope":   np.nanstd(slopes),
        })

    ax.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5, label="Decision")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Time from decision (s)")
    ax.set_ylabel("DA (z-scored, anchor-baselined)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close()

    return quartile_stats


def plot_slope_by_quartile(all_traces, config, output_dir, level, anchor):
    """Bar plot of mean slope per quartile for each group/mouse."""
    names = sorted(all_traces.keys())
    n_q   = config.N_QUARTILES

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 5), squeeze=False)
    summary_data = []

    for idx, name in enumerate(names):
        ax     = axes.flatten()[idx]
        traces = all_traces[name]

        if len(traces) < 50:
            ax.set_title(f"{name}: insufficient data")
            continue

        traces = assign_quartiles(traces, n_q)
        means, sems, ns, ranges = [], [], [], []

        for q in range(1, n_q + 1):
            q_traces = [t for t in traces if t.get("quartile") == q]
            if len(q_traces) < 10:
                means.append(np.nan); sems.append(np.nan)
                ns.append(0); ranges.append(f"Q{q}")
                continue

            all_slopes = np.array([compute_slope(t["time_rel"], t["signal"],
                                                  t["window_duration"], "backward", config)[0]
                                    for t in q_traces])
            windows = np.array([t["window_duration"] for t in q_traces])
            slopes  = all_slopes[np.isfinite(all_slopes)]

            means.append(np.mean(slopes))
            sems.append(np.std(slopes) / np.sqrt(len(slopes)))
            ns.append(len(slopes))
            ranges.append(q_traces[0].get("quartile_range", f"Q{q}"))

            mask = np.isfinite(all_slopes) & np.isfinite(windows)
            r    = np.corrcoef(all_slopes[mask], windows[mask])[0, 1] if mask.sum() >= 10 else np.nan

            summary_data.append({
                "name": name, "anchor": anchor,
                "quartile": q, "range": ranges[-1],
                "n": ns[-1], "mean_slope": means[-1],
                "sem_slope": sems[-1], "within_quartile_r": r,
            })

        x      = np.arange(n_q)
        colors = config.QUARTILE_COLORS
        ax.bar(x, means, yerr=sems, capsize=4, color=colors,
               edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{q+1}\n{ranges[q]}\nn={ns[q]}" for q in range(n_q)], fontsize=8)
        ax.set_ylabel("Mean slope (z/s)")
        ax.set_title(name)

        for q in range(n_q):
            if ns[q] > 0 and np.isfinite(means[q]) and sems[q] > 0:
                t_stat = means[q] / sems[q]
                marker = ("***" if abs(t_stat) > 4 else "**" if abs(t_stat) > 3
                          else "*" if abs(t_stat) > 2 else "")
                if marker:
                    y_pos = (means[q] + sems[q] + 0.05 if means[q] > 0
                             else means[q] - sems[q] - 0.1)
                    ax.text(q, y_pos, marker, ha="center", fontsize=12)

    fig.suptitle(f"Mean Slope by Wait Quartile | {anchor} | Level: {level}",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / f"slope_by_quartile_{level}_{anchor}.png")
    plt.close()
    print(f"  Saved: {output_dir / f'slope_by_quartile_{level}_{anchor}.png'}")

    return summary_data


def plot_quartile_heatmap(traces, config, output_path, title):
    """Heatmap sorted by window duration with quartile boundary lines."""
    n_q    = config.N_QUARTILES
    traces = assign_quartiles(traces, n_q)

    traces_sorted = sorted(traces, key=lambda t: t["window_duration"])
    max_window    = max(t["window_duration"] for t in traces_sorted)
    time_grid     = np.arange(-min(max_window, 10), config.POST_DECISION_SEC,
                               config.BIN_SIZE_MS / 1000)

    signals, windows = interpolate_traces_to_grid(traces_sorted, time_grid)
    quartiles        = np.array([t.get("quartile", 0) for t in traces_sorted])

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = np.nanpercentile(np.abs(signals), 95)
    im   = ax.imshow(
        signals, aspect="auto",
        extent=[time_grid[0], time_grid[-1], len(signals), 0],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
    )

    for q in range(1, n_q):
        boundary = np.searchsorted(quartiles, q + 1)
        ax.axhline(boundary, color="white", linestyle="-", linewidth=2)
        ax.text(time_grid[0] - 0.5, boundary - 20, f"Q{q}", fontsize=10,
                color="white", va="center", ha="right", fontweight="bold")

    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Time from decision (s)")
    ax.set_ylabel("Trials (sorted by window duration)")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("DA (z)")
    ax.text(time_grid[-1] + 0.3, 0,           f"win={windows[-1]:.1f}s", fontsize=9, va="top")
    ax.text(time_grid[-1] + 0.3, len(signals), f"win={windows[0]:.1f}s",  fontsize=9, va="bottom")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def run_exploration(config: Config):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("=" * 70)
    print("DA Ramp Explorer + Quartile Analysis")
    print("=" * 70)

    print("\nLoading sessions...")
    sessions = load_all_sessions(config)
    print(f"  Loaded {len(sessions)} sessions")

    print("\nExtracting trial data...")
    all_trials: List[TrialData] = []
    for session in sessions:
        trials = extract_trial_data(session, config)
        all_trials.extend(trials)
        print(f"  {session['session_id']}: {len(trials)} trials")
    print(f"\nTotal trials: {len(all_trials)}")

    mice   = sorted(set(t.mouse for t in all_trials))
    groups = sorted(set(t.group for t in all_trials))
    print(f"Mice: {mice}")
    print(f"Groups: {groups}")

    summary_rows   = []
    quartile_rows  = []

    for direction in ["forward", "backward"]:
        print(f"\n{'=' * 70}")
        print(f"Direction: {direction.upper()}")
        print("=" * 70)

        # Create output directories
        trace_dir   = config.OUTPUT_DIR / f"traces_{direction}"
        scatter_dir = config.OUTPUT_DIR / f"slope_scatter_{direction}"
        heatmap_dir = config.OUTPUT_DIR / f"heatmaps_{direction}"
        for d in (trace_dir, scatter_dir, heatmap_dir):
            d.mkdir(exist_ok=True)

        get_trace_fn = get_trace_forward if direction == "forward" else get_trace_backward

        # Build trace collections
        traces_all:   Dict[str, List[Dict]] = {"all": []}
        traces_group: Dict[str, List[Dict]] = {g: [] for g in groups}
        traces_mouse: Dict[str, List[Dict]] = {m: [] for m in mice}

        for trial in all_trials:
            for anchor in config.ANCHORS:
                trace = get_trace_fn(trial, anchor, config)
                if trace is None:
                    continue
                trace["anchor"] = anchor
                traces_all["all"].append(trace)
                traces_group[trial.group].append(trace)
                traces_mouse[trial.mouse].append(trace)

        # --- Standard exploration plots ---
        print("\nPlotting average traces...")
        plot_traces_grid(traces_all,   config, direction, trace_dir, "all")
        plot_traces_grid(traces_group, config, direction, trace_dir, "group")
        plot_traces_grid(traces_mouse, config, direction, trace_dir, "mouse")

        print("\nPlotting slope correlations...")
        plot_slope_scatter(traces_all,   config, direction, scatter_dir, "all")
        plot_slope_scatter(traces_group, config, direction, scatter_dir, "group")
        plot_slope_scatter(traces_mouse, config, direction, scatter_dir, "mouse")

        print("\nPlotting heatmaps...")
        for group in groups:
            for anchor in config.ANCHORS:
                group_traces = [t for t in traces_group[group] if t.get("anchor") == anchor]
                plot_heatmap(group_traces, config, direction, heatmap_dir, group, anchor)

        # --- Summary statistics ---
        for level_name, traces_dict in [("all", traces_all), ("group", traces_group),
                                         ("mouse", traces_mouse)]:
            for name, traces in traces_dict.items():
                for anchor in config.ANCHORS:
                    anchor_traces = [t for t in traces if t.get("anchor") == anchor]
                    if len(anchor_traces) < 10:
                        continue
                    slopes  = np.array([compute_slope(t["time_rel"], t["signal"],
                                                      t["window_duration"], direction, config)[0]
                                         for t in anchor_traces])
                    windows = np.array([t["window_duration"] for t in anchor_traces])
                    r, p, n = compute_correlation(slopes, windows)
                    summary_rows.append({
                        "direction": direction, "level": level_name, "name": name,
                        "anchor": anchor, "n_trials": n,
                        "mean_slope": np.nanmean(slopes), "std_slope": np.nanstd(slopes),
                        "mean_window": np.nanmean(windows),
                        "r_slope_window": r, "p_slope_window": p,
                    })

        # --- Quartile analysis (backward only) ---
        if direction == "backward":
            print("\nRunning quartile analysis...")

            q_trace_dir  = config.OUTPUT_DIR / "quartile_traces"
            q_slope_dir  = config.OUTPUT_DIR / "quartile_slopes"
            q_heatmap_dir = config.OUTPUT_DIR / "quartile_heatmaps"
            for d in (q_trace_dir, q_slope_dir, q_heatmap_dir):
                d.mkdir(exist_ok=True)

            for anchor in config.ANCHORS:
                print(f"\n  Anchor: {anchor}")

                # Separate trace lists per anchor (needed for quartile assignment)
                q_by_group = {g: [t for t in traces_group[g] if t.get("anchor") == anchor]
                              for g in groups}
                q_by_mouse = {m: [t for t in traces_mouse[m] if t.get("anchor") == anchor]
                              for m in mice}

                # Quartile trace plots
                for group, traces in q_by_group.items():
                    if len(traces) < 50:
                        continue
                    stats = plot_quartile_traces(
                        traces, config,
                        q_trace_dir / f"{group}_{anchor}_quartiles.png",
                        f"{group.upper()} BG | {anchor} | Quartiles",
                    )
                    for s in stats:
                        s.update({"group": group, "anchor": anchor})
                        quartile_rows.append(s)

                for mouse, traces in q_by_mouse.items():
                    if len(traces) < 50:
                        continue
                    plot_quartile_traces(
                        traces, config,
                        q_trace_dir / f"{mouse}_{anchor}_quartiles.png",
                        f"{mouse} | {anchor} | Quartiles",
                    )

                # Slope-by-quartile bar plots
                quartile_rows += plot_slope_by_quartile(q_by_group, config, q_slope_dir,
                                                        "group", anchor)
                quartile_rows += plot_slope_by_quartile(q_by_mouse, config, q_slope_dir,
                                                        "mouse", anchor)

                # Quartile heatmaps
                for group, traces in q_by_group.items():
                    if len(traces) < 50:
                        continue
                    plot_quartile_heatmap(
                        traces, config,
                        q_heatmap_dir / f"{group}_{anchor}_heatmap.png",
                        f"{group.upper()} BG | {anchor} | Trials sorted by window",
                    )

    # Save summaries
    pd.DataFrame(summary_rows).to_csv(
        config.OUTPUT_DIR / "summary_statistics.csv", index=False)
    print(f"\nSaved: {config.OUTPUT_DIR / 'summary_statistics.csv'}")

    if quartile_rows:
        pd.DataFrame(quartile_rows).to_csv(
            config.OUTPUT_DIR / "quartile_summary.csv", index=False)
        print(f"Saved: {config.OUTPUT_DIR / 'quartile_summary.csv'}")

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    summary_df = pd.DataFrame(summary_rows)
    for direction in ["forward", "backward"]:
        print(f"\n{direction.upper()}:")
        group_df = summary_df[(summary_df["direction"] == direction) &
                               (summary_df["level"] == "group")]
        for _, row in group_df.iterrows():
            sig = "*" if row["p_slope_window"] < 0.05 else ""
            print(f"  {row['name']:6s} | {row['anchor']:10s} | r={row['r_slope_window']:+.3f}{sig} | "
                  f"slope={row['mean_slope']:+.3f} | n={row['n_trials']}")

    print(f"\nDone! All outputs in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    config = Config()
    run_exploration(config)

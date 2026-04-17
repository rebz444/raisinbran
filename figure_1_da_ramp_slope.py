#!/usr/bin/env python3
"""
Figure 1: DA Ramp Slope Tracks Subjective Time
===============================================

Publication-quality figure showing:
  Panel A: Average DA traces by wait duration quartile (decision-aligned)
  Panel B: Slope vs window duration scatter with regression
  Panel C: Mean slope by quartile (bar plot)

Structure:
  - Top row spans full width: quartile traces
  - Bottom row: two panels side by side (scatter + bars)

Generates single-group and combined (both groups) versions.

Usage:
    python figure_1_da_ramp_slope.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats

from config import (
    COMMITTEE_FIGURES_DIR,
    DEFAULT_SIGNAL_COL,
    PIPELINE_SESSION_LOG,
    PROCESSED_OUT,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FigureConfig:
    """Configuration for Figure 1."""

    # Paths (pulled from config.py by default)
    PROCESSED_OUT: Path = PROCESSED_OUT
    PIPELINE_SESSION_LOG: Path = PIPELINE_SESSION_LOG
    OUTPUT_DIR: Path = COMMITTEE_FIGURES_DIR
    OUTPUT_FORMATS: Tuple[str, ...] = ("png",)
    DPI: int = 300

    # Quality filter (True = GRAB tier A/B + not short session)
    QUALITY_FILTER: bool = True

    # Signal
    SIGNAL_COL: str = DEFAULT_SIGNAL_COL
    DA_CHANNEL: str = "l_str_GRAB"

    # Figure dimensions (inches)
    FIG_WIDTH: float = 7.0   # double-column ~7"
    FIG_HEIGHT: float = 8.0

    # Anchor to use for this figure
    # Options: "cue_on", "cue_off", "last_lick"
    ANCHOR: str = "cue_off"
    ANCHOR_WAIT_COL: Dict[str, str] = field(default_factory=lambda: {
        "cue_on":    "time_waited_since_cue_on",
        "cue_off":   "time_waited",
        "last_lick": "time_waited_since_last_lick",
    })

    # Timing
    MIN_WINDOW_SEC: float = 1.0          # minimum anchor→decision window
    BASELINE_WINDOW_SEC: float = 0.2    # first 200ms at anchor for baseline
    SLOPE_BUFFER_START: float = 0.3     # skip 300ms after anchor before fitting slope
    SLOPE_BUFFER_END: float = 0.2       # stop 200ms before decision
    POST_DECISION_SEC: float = 1.0      # extra signal shown past decision

    # Trace display window (decision-aligned: t=0 is decision)
    TRACE_XLIM: Tuple[float, float] = (-6.0, 1.2)
    TRACE_YLIM: Optional[Tuple[float, float]] = None   # None = auto-scale

    # Scatter x-axis: clip to this percentile to exclude extreme-wait outliers
    SCATTER_PCTILE_CAP: float = 98.0

    # Number of quartiles for trace and bar panels
    N_QUARTILES: int = 4

    # Minimum trials per quartile to include
    MIN_TRIALS_PER_QUARTILE: int = 10

    # Colors — colorblind-friendly palette
    QUARTILE_COLORS: List[str] = field(default_factory=lambda: [
        "#009E73",  # Q1 (shortest) — teal
        "#0072B2",  # Q2 — blue
        "#9467BD",  # Q3 — purple
        "#666666",  # Q4 (longest) — gray
    ])

    GROUP_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "short": "#0072B2",   # short BG — blue
        "long":  "#D55E00",   # long BG — vermillion
        "s":     "#0072B2",
        "l":     "#D55E00",
    })

    # Style
    FONT_FAMILY: str = "Arial"
    TITLE_SIZE: int = 11
    LABEL_SIZE: int = 10
    TICK_SIZE: int = 9
    LEGEND_SIZE: int = 9
    PANEL_LABEL_SIZE: int = 14
    PANEL_LABEL_WEIGHT: str = "bold"


# =============================================================================
# STYLE SETUP
# =============================================================================

def setup_publication_style(config: FigureConfig):
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family":          config.FONT_FAMILY,
        "font.size":            config.LABEL_SIZE,
        "axes.titlesize":       config.TITLE_SIZE,
        "axes.labelsize":       config.LABEL_SIZE,
        "xtick.labelsize":      config.TICK_SIZE,
        "ytick.labelsize":      config.TICK_SIZE,
        "legend.fontsize":      config.LEGEND_SIZE,
        "figure.facecolor":     "white",
        "figure.dpi":           100,
        "savefig.dpi":          config.DPI,
        "savefig.facecolor":    "white",
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.1,
        "axes.facecolor":       "white",
        "axes.edgecolor":       "black",
        "axes.linewidth":       0.8,
        "axes.grid":            False,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "xtick.major.size":     4,
        "ytick.major.size":     4,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "lines.linewidth":      1.5,
        "lines.markersize":     4,
        "legend.frameon":       False,
        "legend.borderpad":     0.3,
        "legend.handlelength":  1.5,
    })


def add_panel_label(ax, label: str, config: FigureConfig, x: float = -0.12, y: float = 1.08):
    """Add panel label (A, B, C) to an axis."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=config.PANEL_LABEL_SIZE, fontweight=config.PANEL_LABEL_WEIGHT,
            va="top", ha="left")


def add_significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return "ns"


# =============================================================================
# DATA LOADING
# =============================================================================

def _build_allowed_sessions(config: FigureConfig) -> Optional[set]:
    """Return session IDs that pass quality filter, or None if filter is off."""
    if not config.QUALITY_FILTER:
        print("  Session filter: disabled (using all sessions)")
        return None

    short_sessions: set = set()
    if config.PIPELINE_SESSION_LOG.exists():
        log = pd.read_csv(config.PIPELINE_SESSION_LOG, index_col=0)
        if "short_session" in log.columns:
            short_sessions = set(
                log.index[log["short_session"].fillna(False).astype(bool)]
            )

    allowed: set = set()
    n_total = 0
    for session_dir in sorted(config.PROCESSED_OUT.iterdir()):
        if not session_dir.is_dir():
            continue
        qc_file = session_dir / "correction_quality.csv"
        if not qc_file.exists():
            continue
        n_total += 1
        if session_dir.name in short_sessions:
            continue
        try:
            qc = pd.read_csv(qc_file)
            grab_rows = qc[qc["channel"].str.contains("GRAB", na=False)]
            if not grab_rows.empty and grab_rows.iloc[0]["quality_tier"] in ("A", "B"):
                allowed.add(session_dir.name)
        except Exception:
            continue

    print(f"  Session filter: {len(allowed)} / {n_total} sessions pass (GRAB tier A/B, not short)")
    return allowed


def load_data(config: FigureConfig) -> pd.DataFrame:
    """Load, concatenate, and filter all session data from pre-merged CSVs."""
    base = config.PROCESSED_OUT
    print(f"Scanning sessions in: {base}")
    allowed = _build_allowed_sessions(config)

    frames: List[pd.DataFrame] = []
    for session_dir in sorted(base.iterdir()):
        if not session_dir.is_dir():
            continue
        if allowed is not None and session_dir.name not in allowed:
            continue
        merged_csv = session_dir / "photometry_with_trial_data.csv"
        if not merged_csv.exists():
            continue
        try:
            df = pd.read_csv(merged_csv)
            df["session_id"] = session_dir.name
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: skipping {session_dir.name}: {e}")

    if not frames:
        raise FileNotFoundError(f"No photometry_with_trial_data.csv files found under {base}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {combined['session_id'].nunique()} sessions, {len(combined):,} rows")

    # Filter to DA (GRAB) channel
    available = combined["channel"].unique()
    if config.DA_CHANNEL in available:
        combined = combined[combined["channel"] == config.DA_CHANNEL].copy()
    else:
        grab = [c for c in available if "GRAB" in str(c)]
        if not grab:
            raise ValueError(f"No GRAB channel found. Available: {available}")
        config.DA_CHANNEL = grab[0]
        combined = combined[combined["channel"] == grab[0]].copy()
    print(f"  Channel: {config.DA_CHANNEL}")

    # Filter to non-miss trials
    n_before = len(combined)
    if "miss_trial" in combined.columns:
        combined = combined[combined["miss_trial"] == False].copy()
    else:
        combined = combined[
            combined["time_waited"].notna() & (combined["time_waited"] < 60)
        ].copy()
    print(f"  After miss filter: {n_before:,} → {len(combined):,} rows")

    # Normalize group labels  (l → long, s → short)
    _aliases = {"l": "long", "s": "short", "long": "long", "short": "short"}
    combined["group"] = combined["group"].map(lambda g: _aliases.get(str(g), str(g)))

    print(f"  Groups: {sorted(combined['group'].unique())}")
    print(f"  Mice:   {sorted(combined['mouse'].unique())}")
    return combined


# =============================================================================
# PER-TRIAL EXTRACTION
# =============================================================================

def extract_trials(df: pd.DataFrame, config: FigureConfig) -> pd.DataFrame:
    """
    Extract per-trial measures for config.ANCHOR.

    Produces decision-aligned traces (t = 0 at decision).
    Anchor is at t = -window_duration; baseline is subtracted using
    the first BASELINE_WINDOW_SEC after the anchor.

    Returns a DataFrame with one row per trial:
      group, window_duration, slope, time_rel (array), signal (array)
    """
    wait_col = config.ANCHOR_WAIT_COL[config.ANCHOR]
    records: List[Dict] = []

    for (session_id, trial_num), trial_df in df.groupby(
        ["session_id", "session_trial_num"]
    ):
        trial_df = trial_df.sort_values("trial_time")
        row = trial_df.iloc[0]

        # Decision time in trial_time coordinates (≈ time_waited_since_cue_on,
        # since trial_time = 0 ≈ cue_on)
        t_decision = row.get("time_waited_since_cue_on", np.nan)
        if not (np.isfinite(t_decision) and t_decision > 0):
            continue

        # Window duration for the selected anchor
        w = float(row.get(wait_col, np.nan))
        if not (np.isfinite(w) and w >= config.MIN_WINDOW_SEC):
            continue

        # Anchor time in trial_time coordinates
        t_anchor = t_decision - w

        trial_time = trial_df["trial_time"].to_numpy(dtype=float)
        signal     = trial_df[config.SIGNAL_COL].to_numpy(dtype=float)

        # Extract from anchor to decision + post_decision
        mask = (trial_time >= t_anchor) & (trial_time <= t_decision + config.POST_DECISION_SEC)
        if mask.sum() < 10:
            continue

        # Decision-aligned time: t = 0 at decision, anchor at t = -w
        t_trace   = trial_time[mask] - t_decision
        sig_trace = signal[mask].copy()

        # Baseline subtract: mean of first BASELINE_WINDOW_SEC at anchor
        bl_mask = (t_trace >= -w) & (t_trace <= -w + config.BASELINE_WINDOW_SEC)
        if bl_mask.sum() >= 2:
            sig_trace -= np.nanmean(sig_trace[bl_mask])

        # Slope: linear fit in [-w + buffer_start, -buffer_end]
        slope_start = -(w - config.SLOPE_BUFFER_START)
        slope_end   = -config.SLOPE_BUFFER_END
        slope = np.nan
        if slope_end > slope_start + 0.3 and len(t_trace) >= 5:
            sl_mask = (t_trace >= slope_start) & (t_trace <= slope_end)
            t_sl = t_trace[sl_mask]
            y_sl = sig_trace[sl_mask]
            if len(t_sl) >= 5:
                tm, ym = np.mean(t_sl), np.mean(y_sl)
                tt, yy = t_sl - tm, y_sl - ym
                denom = np.dot(tt, tt)
                if denom > 0:
                    slope = float(np.dot(tt, yy) / denom)

        records.append({
            "session_id":      session_id,
            "mouse":           row["mouse"],
            "group":           row["group"],
            "trial_num":       trial_num,
            "window_duration": w,
            "slope":           slope,
            "time_rel":        t_trace,
            "signal":          sig_trace,
        })

    print(f"  Extracted {len(records)} trials for anchor '{config.ANCHOR}'")
    return pd.DataFrame(records)


# =============================================================================
# QUARTILE DATA ASSEMBLY
# =============================================================================

def _build_group_data(trials_df: pd.DataFrame, config: FigureConfig) -> Dict:
    """
    Structure per-trial data into the dict format expected by plotting functions.

    Returns:
      {
        "traces":          {Q1: {time, mean, sem}, ...},
        "slopes":          {Q1: {mean, sem, n}, ...},
        "scatter":         {durations, slopes, r, p, n},
        "duration_ranges": [(min, max), ...],
      }
    """
    valid = trials_df[trials_df["slope"].notna()].copy()
    if len(valid) < config.N_QUARTILES * config.MIN_TRIALS_PER_QUARTILE:
        return {}

    # Assign quartiles by window_duration percentile
    valid["quartile"] = pd.qcut(
        valid["window_duration"], q=config.N_QUARTILES, labels=False, duplicates="drop"
    )
    n_actual = valid["quartile"].nunique()

    # Time grid for interpolation
    t0 = min(config.TRACE_XLIM[0] - 0.5, -valid["window_duration"].quantile(0.98) - 0.5)
    time_grid = np.arange(t0, config.POST_DECISION_SEC + 0.1, 0.05)

    traces: Dict       = {}
    slopes: Dict       = {}
    dur_ranges: List   = []

    for q in range(n_actual):
        q_label  = f"Q{q + 1}"
        q_trials = valid[valid["quartile"] == q]

        if len(q_trials) < config.MIN_TRIALS_PER_QUARTILE:
            continue

        # Duration range for legend
        dur_ranges.append(
            (float(q_trials["window_duration"].min()),
             float(q_trials["window_duration"].max()))
        )

        # Slope summary
        sl = q_trials["slope"].to_numpy(dtype=float)
        slopes[q_label] = {
            "mean": float(np.nanmean(sl)),
            "sem":  float(np.nanstd(sl) / np.sqrt(len(sl))),
            "n":    len(sl),
        }

        # Average decision-aligned trace
        signals = np.full((len(q_trials), len(time_grid)), np.nan)
        for i, (_, trial) in enumerate(q_trials.iterrows()):
            t = trial["time_rel"]
            s = trial["signal"]
            if len(t) > 1:
                signals[i] = np.interp(time_grid, t, s, left=np.nan, right=np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_sig = np.nanmean(signals, axis=0)
            n_valid  = np.sum(np.isfinite(signals), axis=0)
            sem_sig  = np.nanstd(signals, axis=0) / np.sqrt(np.maximum(n_valid, 1))

        # Mask low-coverage time points
        cov_min = max(config.MIN_TRIALS_PER_QUARTILE, 0.2 * len(q_trials))
        low_cov = n_valid < cov_min
        mean_sig[low_cov] = np.nan
        sem_sig[low_cov]  = np.nan

        traces[q_label] = {"time": time_grid, "mean": mean_sig, "sem": sem_sig}

    # Scatter: all trials (clipped to 98th pct for display)
    x_all = valid["window_duration"].to_numpy(dtype=float)
    y_all = valid["slope"].to_numpy(dtype=float)
    fin   = np.isfinite(x_all) & np.isfinite(y_all)
    r, p  = stats.pearsonr(x_all[fin], y_all[fin]) if fin.sum() >= 5 else (np.nan, np.nan)

    return {
        "traces":          traces,
        "slopes":          slopes,
        "scatter":         {"durations": x_all, "slopes": y_all, "r": r, "p": p, "n": int(fin.sum())},
        "duration_ranges": dur_ranges,
    }


def load_real_data(config: FigureConfig) -> Dict:
    """
    Load all sessions and build the data dict expected by the plotting functions.
    Returns {group: {traces, slopes, scatter, duration_ranges}} for each group.
    """
    df      = load_data(config)
    trials  = extract_trials(df, config)

    data: Dict = {}
    for group in ["short", "long"]:
        g_trials = trials[trials["group"] == group]
        if len(g_trials) == 0:
            print(f"  WARNING: no trials found for group '{group}'")
            continue
        print(f"  Building quartile data for {group.upper()} BG ({len(g_trials)} trials)...")
        gdata = _build_group_data(g_trials, config)
        if gdata:
            data[group] = gdata

    return data


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_quartile_traces(ax, data: Dict, group: str, config: FigureConfig):
    """
    Panel A: Average DA traces by wait duration quartile (decision-aligned).

    Q1 = shortest wait, Q4 = longest wait.
    Traces start at t = -window_duration and converge at t = 0 (decision).
    """
    traces         = data[group]["traces"]
    duration_ranges = data[group]["duration_ranges"]

    for q_idx, (q_label, trace_data) in enumerate(traces.items()):
        if q_idx >= len(config.QUARTILE_COLORS):
            break
        time  = trace_data["time"]
        mean  = trace_data["mean"]
        sem   = trace_data["sem"]
        color = config.QUARTILE_COLORS[q_idx]

        valid = np.isfinite(mean)
        ax.plot(time[valid], mean[valid], color=color, linewidth=1.8, label=q_label)
        ax.fill_between(
            time[valid], (mean - sem)[valid], (mean + sem)[valid],
            color=color, alpha=0.2, linewidth=0,
        )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Decision")
    ax.axhline(0, color="gray", linestyle="-",  linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Time from decision (s)")
    ax.set_ylabel("DA (z-score, baseline-subtracted)")

    anchor_label = {"cue_on": "Cue on", "cue_off": "Cue off", "last_lick": "Last lick"}
    group_label  = "Short BG" if group == "short" else "Long BG"
    ax.set_title(f"{group_label}  —  Quartile traces  ({anchor_label.get(config.ANCHOR, config.ANCHOR)} anchor)")

    legend_labels = []
    for q_idx, q_label in enumerate(traces.keys()):
        if q_idx >= len(duration_ranges):
            break
        lo, hi = duration_ranges[q_idx]
        suffix = " (shortest)" if q_idx == 0 else (" (longest)" if q_idx == len(traces) - 1 else "")
        legend_labels.append(f"{q_label}: {lo:.1f}–{hi:.1f} s{suffix}")

    handles = [Patch(facecolor=c, edgecolor="none", alpha=0.8)
               for c in config.QUARTILE_COLORS[:len(traces)]]
    ax.legend(handles, legend_labels, loc="upper left", fontsize=config.LEGEND_SIZE - 1)

    ax.set_xlim(config.TRACE_XLIM)
    if config.TRACE_YLIM is not None:
        ax.set_ylim(config.TRACE_YLIM)


def plot_slope_duration_scatter(ax, data: Dict, group: str, config: FigureConfig):
    """
    Panel B: Scatter of ramp slope vs window duration with regression.
    X-axis clipped to SCATTER_PCTILE_CAP to exclude extreme-wait outliers.
    """
    scatter_data = data[group]["scatter"]
    x_all = scatter_data["durations"]
    y_all = scatter_data["slopes"]

    color = config.GROUP_COLORS.get(group, "#888888")

    # Clip display to percentile cap
    x_cap = float(np.nanpercentile(x_all[np.isfinite(x_all)], config.SCATTER_PCTILE_CAP))
    keep  = np.isfinite(x_all) & np.isfinite(y_all) & (x_all <= x_cap)
    x     = x_all[keep]
    y     = y_all[keep]
    n_excl = int((~keep).sum())

    # Subsample for visual clarity
    n_plot = min(500, len(x))
    rng    = np.random.default_rng(0)
    idx    = rng.choice(len(x), n_plot, replace=False)
    ax.scatter(x[idx], y[idx], c=color, alpha=0.15, s=6, edgecolors="none", rasterized=True)

    # Regression on clipped data
    if len(x) >= 5:
        slope_reg, intercept, r, p, _ = stats.linregress(x, y)
        x_line = np.array([x.min(), x.max()])
        ax.plot(x_line, slope_reg * x_line + intercept, color="black", linewidth=2, linestyle="-")

        sig = add_significance_stars(p)
        stats_text = f"r = {r:.2f} {sig}\nn = {len(x):,}"
        if n_excl > 0:
            stats_text += f"\n({n_excl} outliers excl.)"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=config.LEGEND_SIZE,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-", alpha=0.4)
    ax.set_xlabel("Window duration (s)")
    ax.set_ylabel("Slope (z/s)")
    group_label = "Short BG" if group == "short" else "Long BG"
    ax.set_title(f"{group_label}  —  Slope vs duration")


def plot_combined_scatter(ax, data: Dict, config: FigureConfig):
    """
    Overlay scatter + regression for both groups on a single axis.
    Dots are group-colored; trend lines are group-colored and solid.
    """
    anchor_label = {"cue_on": "Cue on", "cue_off": "Cue off", "last_lick": "Last lick"}
    handles = []

    for group in ["short", "long"]:
        if group not in data:
            continue

        scatter_data = data[group]["scatter"]
        x_all = scatter_data["durations"]
        y_all = scatter_data["slopes"]
        color = config.GROUP_COLORS.get(group, "#888888")
        group_label = "Short BG" if group == "short" else "Long BG"

        x_cap = float(np.nanpercentile(x_all[np.isfinite(x_all)], config.SCATTER_PCTILE_CAP))
        keep  = np.isfinite(x_all) & np.isfinite(y_all) & (x_all <= x_cap)
        x     = x_all[keep]
        y     = y_all[keep]

        n_plot = min(500, len(x))
        rng    = np.random.default_rng(0)
        idx    = rng.choice(len(x), n_plot, replace=False)
        ax.scatter(x[idx], y[idx], c=color, alpha=0.15, s=6, edgecolors="none", rasterized=True)

        if len(x) >= 5:
            slope_reg, intercept, r, p, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 200)
            sig = add_significance_stars(p)
            line, = ax.plot(x_line, slope_reg * x_line + intercept,
                            color=color, linewidth=2, linestyle="-",
                            label=f"{group_label}  r={r:.2f}{sig}")
            handles.append(line)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-", alpha=0.4)
    ax.set_xlabel("Window duration (s)")
    ax.set_ylabel("Slope (z/s)")
    ax.set_title(f"Slope vs duration  ({anchor_label.get(config.ANCHOR, config.ANCHOR)} anchor)")
    if handles:
        ax.legend(handles=handles, fontsize=config.LEGEND_SIZE, framealpha=0.8)


def plot_quartile_bars(ax, data: Dict, group: str, config: FigureConfig):
    """
    Panel C: Mean slope ± SEM by wait duration quartile (bar plot).
    """
    slopes_data = data[group]["slopes"]
    quartiles   = list(slopes_data.keys())
    means       = [slopes_data[q]["mean"] for q in quartiles]
    sems        = [slopes_data[q]["sem"]  for q in quartiles]
    ns          = [slopes_data[q]["n"]    for q in quartiles]

    x = np.arange(len(quartiles))
    bars = ax.bar(
        x, means, yerr=sems, capsize=4,
        color=config.QUARTILE_COLORS[:len(quartiles)],
        edgecolor="black", linewidth=0.8,
        error_kw={"linewidth": 1, "capthick": 1},
    )

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")

    ax.set_xticks(x)
    ax.set_xticklabels(quartiles)
    ax.set_xlabel("Wait duration quartile")
    ax.set_ylabel("Mean slope (z/s)")
    group_label = "Short BG" if group == "short" else "Long BG"
    ax.set_title(f"{group_label}  —  Slope by quartile")

    for bar, mean, sem, _ in zip(bars, means, sems, ns):
        y_pos = mean + sem + 0.05 if mean >= 0 else mean - sem - 0.12
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"{mean:.2f}", ha="center",
            va="bottom" if mean >= 0 else "top",
            fontsize=config.TICK_SIZE,
        )


# =============================================================================
# MAIN FIGURE ASSEMBLY
# =============================================================================

def create_figure_1(data: Dict, config: FigureConfig, group: str = "short") -> plt.Figure:
    """
    Single-group Figure 1.

        ┌────────────────────────────────────────┐
        │  A. Quartile traces (full width)        │
        ├───────────────────┬────────────────────┤
        │  B. Scatter plot   │  C. Quartile bars  │
        └───────────────────┴────────────────────┘
    """
    if group not in data:
        raise ValueError(f"No data for group '{group}'")

    setup_publication_style(config)

    fig = plt.figure(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[1.2, 1],
                            hspace=0.35, wspace=0.3)

    ax_traces = fig.add_subplot(gs[0, :])
    plot_quartile_traces(ax_traces, data, group, config)
    add_panel_label(ax_traces, "A", config)

    ax_scatter = fig.add_subplot(gs[1, 0])
    plot_slope_duration_scatter(ax_scatter, data, group, config)
    add_panel_label(ax_scatter, "B", config)

    ax_bars = fig.add_subplot(gs[1, 1])
    plot_quartile_bars(ax_bars, data, group, config)
    add_panel_label(ax_bars, "C", config)

    group_label = "Short BG" if group == "short" else "Long BG"
    fig.suptitle(
        f"DA ramp slope tracks subjective time  —  {group_label}",
        fontsize=config.TITLE_SIZE + 1, fontweight="bold", y=0.99,
    )
    return fig


def create_figure_1_both_groups(data: Dict, config: FigureConfig) -> plt.Figure:
    """
    Both groups side-by-side (6 panels).

        ┌─────────────────────┬─────────────────────┐
        │  A. Short BG traces │  B. Long BG traces  │
        ├─────────────────────┼─────────────────────┤
        │  C. Short scatter   │  D. Long scatter    │
        ├─────────────────────┼─────────────────────┤
        │  E. Short bars      │  F. Long bars       │
        └─────────────────────┴─────────────────────┘
    """
    setup_publication_style(config)

    fig = plt.figure(figsize=(config.FIG_WIDTH * 1.4, config.FIG_HEIGHT * 1.3))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[1.2, 1, 1],
                            hspace=0.4, wspace=0.35)

    panel_labels = iter("ABCDEF")
    for col, group in enumerate(["short", "long"]):
        if group not in data:
            continue

        ax_tr = fig.add_subplot(gs[0, col])
        plot_quartile_traces(ax_tr, data, group, config)
        add_panel_label(ax_tr, next(panel_labels), config)

        ax_sc = fig.add_subplot(gs[1, col])
        plot_slope_duration_scatter(ax_sc, data, group, config)
        add_panel_label(ax_sc, next(panel_labels), config)

        ax_ba = fig.add_subplot(gs[2, col])
        plot_quartile_bars(ax_ba, data, group, config)
        add_panel_label(ax_ba, next(panel_labels), config)

    anchor_label = {"cue_on": "Cue on", "cue_off": "Cue off", "last_lick": "Last lick"}
    fig.suptitle(
        f"DA ramp slope tracks subjective time"
        f"  ({anchor_label.get(config.ANCHOR, config.ANCHOR)} anchor)",
        fontsize=config.TITLE_SIZE + 2, fontweight="bold", y=0.99,
    )
    return fig


def create_combined_scatter_figure(data: Dict, config: FigureConfig) -> plt.Figure:
    """Standalone figure: both groups overlaid on a single scatter."""
    setup_publication_style(config)
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_combined_scatter(ax, data, config)
    fig.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = FigureConfig()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Figure 1: DA Ramp Slope Tracks Subjective Time")
    print(f"  Anchor: {config.ANCHOR}")
    print("=" * 60)

    data = load_real_data(config)

    anchor_tag = config.ANCHOR

    # Single-group figures
    for group in ["short", "long"]:
        if group not in data:
            print(f"  Skipping {group} (no data)")
            continue
        print(f"\nCreating single-group figure: {group.upper()} BG...")
        fig = create_figure_1(data, config, group=group)
        for fmt in config.OUTPUT_FORMATS:
            out = config.OUTPUT_DIR / f"figure_1_{anchor_tag}_{group}_bg.{fmt}"
            fig.savefig(out, format=fmt, dpi=config.DPI)
            print(f"  Saved: {out}")
        plt.close(fig)

    # Combined figure
    if len(data) == 2:
        print("\nCreating combined figure (both groups)...")
        fig_combined = create_figure_1_both_groups(data, config)
        for fmt in config.OUTPUT_FORMATS:
            out = config.OUTPUT_DIR / f"figure_1_{anchor_tag}_combined.{fmt}"
            fig_combined.savefig(out, format=fmt, dpi=config.DPI)
            print(f"  Saved: {out}")
        plt.close(fig_combined)

        print("\nCreating combined scatter (groups overlaid)...")
        fig_scatter = create_combined_scatter_figure(data, config)
        for fmt in config.OUTPUT_FORMATS:
            out = config.OUTPUT_DIR / f"figure_1_{anchor_tag}_combined_scatter.{fmt}"
            fig_scatter.savefig(out, format=fmt, dpi=config.DPI)
            print(f"  Saved: {out}")
        plt.close(fig_scatter)

    print(f"\nDone. Outputs in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Committee Figures — DA Ramp Slope Tracks Subjective Time
=========================================================

Figure 1 (all trials):
  Top row    — Three anchor-aligned decision-window traces
               (short-BG group vs long-BG group), mean ± SEM
  Bottom row — Slope vs window-duration scatter
               (color-coded by BG group, regression + statistics)

Three anchors:
  cue_on   — start of background period (≈ trial_time = 0)
  cue_off  — end of background / start of wait (trial_time ≈ bg_length)
  last_lick — last lick before cue_off (can span previous trial)

Usage:
    python 5_committee_figures.py
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import COMMITTEE_FIGURES_DIR, DEFAULT_SIGNAL_COL, GROUP_COLORS, PIPELINE_SESSION_LOG, PROCESSED_OUT

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Paths
    PROCESSED_OUT: Path = PROCESSED_OUT
    OUTPUT_DIR: Path = COMMITTEE_FIGURES_DIR

    # Signal
    SIGNAL_COL: str = DEFAULT_SIGNAL_COL
    DA_CHANNEL: str = "l_str_GRAB"

    # Quality filter (True = GRAB tier A/B + not short session)
    QUALITY_FILTER: bool = True

    # Timing
    MIN_WINDOW_SEC: float = 1.0          # minimum anchor→decision window
    SLOPE_START_BUFFER: float = 0.3     # skip first 300ms after anchor
    SLOPE_END_BUFFER: float = 0.2       # stop 200ms before decision
    BASELINE_WINDOW_SEC: float = 0.2    # first 200ms at anchor for baseline subtraction
    POST_DECISION_SEC: float = 0.5      # extra signal to show past decision

    # Minimum trials to plot a group
    MIN_TRIALS: int = 10

    # Anchors
    ANCHORS: Tuple[str, ...] = ("cue_on", "cue_off", "last_lick")
    ANCHOR_LABELS: Dict[str, str] = {
        "cue_on":    "Cue on",
        "cue_off":   "Cue off",
        "last_lick": "Last lick",
    }
    # x-axis (time from anchor) shown in trace plots
    # Set to show the bulk of trials; traces naturally end earlier for short-BG group
    TRACE_XLIMS: Dict[str, Tuple[float, float]] = {
        "cue_on":    (0.0, 20.0),   # bg_length (up to ~18s) + wait
        "cue_off":   (0.0, 10.0),   # wait period only; p95 ~44s but most <10s
        "last_lick": (0.0, 20.0),   # variable; matches cue_on range
    }

    # Colors
    GROUP_COLORS: Dict[str, str] = GROUP_COLORS


# =============================================================================
# DATA LOADING
# =============================================================================

def _build_allowed_sessions(base: Path) -> set:
    """Return session IDs where GRAB is tier A/B and session is not short."""
    short_sessions: set = set()
    if PIPELINE_SESSION_LOG.exists():
        log = pd.read_csv(PIPELINE_SESSION_LOG, index_col=0)
        if "short_session" in log.columns:
            short_sessions = set(
                log.index[log["short_session"].fillna(False).astype(bool)]
            )
    allowed: set = set()
    n_total = 0
    for session_dir in sorted(base.iterdir()):
        if not session_dir.is_dir():
            continue
        qc_file = session_dir / "correction_quality.csv"
        if not qc_file.exists():
            continue
        n_total += 1
        session_id = session_dir.name
        if session_id in short_sessions:
            continue
        try:
            qc = pd.read_csv(qc_file)
            grab_rows = qc[qc["channel"].str.contains("GRAB", na=False)]
            if not grab_rows.empty and grab_rows.iloc[0]["quality_tier"] in ("A", "B"):
                allowed.add(session_id)
        except Exception:
            continue
    print(f"  Session filter: {len(allowed)} / {n_total} sessions pass (GRAB tier A/B, not short)")
    return allowed


def load_data(config: Config) -> pd.DataFrame:
    """Load, concatenate, and filter all session data from pre-merged CSVs."""
    base = config.PROCESSED_OUT
    print(f"Scanning sessions in: {base}")

    allowed = _build_allowed_sessions(base) if config.QUALITY_FILTER else None

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
            print(f"  WARNING: could not read {merged_csv}: {e}")

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
    if "group" not in combined.columns:
        raise ValueError("Missing 'group' column in merged data.")
    combined["group"] = combined["group"].map(lambda g: _aliases.get(str(g), str(g)))

    print(f"  Groups: {sorted(combined['group'].unique())}")
    print(f"  Mice:   {sorted(combined['mouse'].unique())}")
    return combined


# =============================================================================
# PER-TRIAL EXTRACTION
# =============================================================================

def extract_trial_records(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Build one record per (trial × anchor) with:
      - scalar measures: window_duration, slope
      - array data: anchor-aligned, baseline-subtracted trace (time, signal)

    Anchor timing is derived from the merged CSV:
      cue_on   ≈ trial_time = 0  (trial starts at cue on)
      cue_off  = cue_on + bg_length  ≈  t_decision - time_waited
      last_lick = t_decision - time_waited_since_last_lick
      decision  = time_waited_since_cue_on  (in trial_time units)
    """
    records: List[Dict] = []

    for (session_id, trial_num), trial_df in df.groupby(
        ["session_id", "session_trial_num"]
    ):
        trial_df = trial_df.sort_values("trial_time")
        row = trial_df.iloc[0]

        t_decision = row.get("time_waited_since_cue_on", np.nan)
        if not (np.isfinite(t_decision) and t_decision > 0):
            continue

        # Anchor times in trial_time coordinates
        t_cue_on = 0.0
        t_cue_off = t_decision - float(row.get("time_waited", np.nan))
        t_ll_dur = float(row.get("time_waited_since_last_lick", np.nan))
        t_last_lick = t_decision - t_ll_dur if np.isfinite(t_ll_dur) else np.nan

        anchor_t = {
            "cue_on":    t_cue_on,
            "cue_off":   t_cue_off,
            "last_lick": t_last_lick,
        }
        window_dur = {
            "cue_on":    float(row.get("time_waited_since_cue_on", np.nan)),
            "cue_off":   float(row.get("time_waited", np.nan)),
            "last_lick": t_ll_dur,
        }

        trial_time = trial_df["trial_time"].to_numpy(dtype=float)
        signal     = trial_df[config.SIGNAL_COL].to_numpy(dtype=float)

        for anchor in config.ANCHORS:
            ta = anchor_t[anchor]
            w  = window_dur[anchor]

            if not (np.isfinite(ta) and np.isfinite(w) and w >= config.MIN_WINDOW_SEC):
                continue

            t_end = t_decision + config.POST_DECISION_SEC
            mask  = (trial_time >= ta) & (trial_time <= t_end)
            if mask.sum() < 10:
                continue

            t_trace   = trial_time[mask] - ta   # anchor at t = 0
            sig_trace = signal[mask].copy()

            # Baseline subtract: mean of first BASELINE_WINDOW_SEC at anchor
            bl_mask = t_trace <= config.BASELINE_WINDOW_SEC
            if bl_mask.sum() >= 2:
                sig_trace -= np.nanmean(sig_trace[bl_mask])

            # Slope: linear fit from anchor+buffer to decision-buffer
            slope_start = config.SLOPE_START_BUFFER
            slope_end   = w - config.SLOPE_END_BUFFER
            slope = np.nan
            if slope_end > slope_start + 0.3:
                sl_mask = (t_trace >= slope_start) & (t_trace <= slope_end)
                t_sl = t_trace[sl_mask]
                y_sl = sig_trace[sl_mask]
                if len(t_sl) >= 5:
                    tm = np.mean(t_sl)
                    ym = np.mean(y_sl)
                    tt = t_sl - tm
                    yy = y_sl - ym
                    denom = np.dot(tt, tt)
                    if denom > 0:
                        slope = float(np.dot(tt, yy) / denom)

            records.append({
                "session_id":      session_id,
                "mouse":           row["mouse"],
                "group":           row["group"],
                "trial_num":       trial_num,
                "anchor":          anchor,
                "window_duration": w,
                "slope":           slope,
                "trace_time":      t_trace,
                "trace_signal":    sig_trace,
            })

    print(f"  Extracted {len(records)} trial-anchor records")
    return pd.DataFrame(records)


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def setup_style():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          11,
        "axes.labelsize":     12,
        "axes.titlesize":     13,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.grid":          False,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
    })


def _plot_anchor_traces(ax, trials_df: pd.DataFrame, anchor: str, config: Config):
    """
    Plot mean ± SEM anchor-aligned traces for short and long BG groups.
    x-axis: time from anchor (s); decision is at x = window_duration (varies per trial).
    """
    xlim = config.TRACE_XLIMS[anchor]
    time_grid = np.arange(0.0, xlim[1] + 0.05, 0.05)

    anchor_df = trials_df[trials_df["anchor"] == anchor]

    for group in ["short", "long"]:
        group_df = anchor_df[anchor_df["group"] == group]
        if len(group_df) < config.MIN_TRIALS:
            continue

        color = config.GROUP_COLORS.get(group, "#888888")
        n = len(group_df)

        # Interpolate each trial onto the common time grid
        signals = np.full((n, len(time_grid)), np.nan)
        for i, (_, trial) in enumerate(group_df.iterrows()):
            t = trial["trace_time"]
            s = trial["trace_signal"]
            if len(t) > 1:
                signals[i] = np.interp(time_grid, t, s, left=np.nan, right=np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_sig = np.nanmean(signals, axis=0)
            n_valid  = np.sum(np.isfinite(signals), axis=0)
            sem_sig  = np.nanstd(signals, axis=0) / np.sqrt(np.maximum(n_valid, 1))

        # Only show time points where ≥20% of trials contribute
        cov_mask = n_valid >= max(config.MIN_TRIALS, 0.2 * n)

        ax.plot(
            time_grid[cov_mask], mean_sig[cov_mask],
            color=color, linewidth=2,
            label=f"{group.capitalize()} BG (n={n})",
        )
        ax.fill_between(
            time_grid[cov_mask],
            (mean_sig - sem_sig)[cov_mask],
            (mean_sig + sem_sig)[cov_mask],
            color=color, alpha=0.2,
        )

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5, linestyle="-")
    ax.set_xlabel(f"Time from {config.ANCHOR_LABELS[anchor]} (s)")
    ax.set_ylabel("DA (z-score, baseline-subtracted)")
    ax.set_title(f"Aligned to {config.ANCHOR_LABELS[anchor]}")
    ax.set_xlim(xlim)
    ax.legend(loc="upper left", fontsize=9, frameon=False)


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _plot_slope_scatter(ax, trials_df: pd.DataFrame, anchor: str, config: Config):
    """
    Scatter: slope vs window duration, colored by BG group.
    Includes per-group regression lines and Pearson r / p annotations.
    Also shows Mann-Whitney U test p-value for group slope difference.
    Data are clipped to the 98th percentile of window duration to exclude
    extreme-wait outliers while keeping regression stats over the full dataset.
    """
    anchor_df = trials_df[
        (trials_df["anchor"] == anchor)
        & trials_df["slope"].notna()
        & trials_df["window_duration"].notna()
    ]

    # Clip to 98th percentile of window duration for display
    w_cap = anchor_df["window_duration"].quantile(0.98)
    n_total = len(anchor_df)
    anchor_df = anchor_df[anchor_df["window_duration"] <= w_cap].copy()
    n_shown = len(anchor_df)

    stat_lines: List[str] = []
    group_slopes: Dict[str, np.ndarray] = {}

    y_text = 0.97   # top of axes for stat annotations

    for group in ["short", "long"]:
        gdf = anchor_df[anchor_df["group"] == group]
        if len(gdf) < config.MIN_TRIALS:
            continue

        x = gdf["window_duration"].to_numpy(dtype=float)
        y = gdf["slope"].to_numpy(dtype=float)
        color = config.GROUP_COLORS.get(group, "#888888")
        group_slopes[group] = y

        ax.scatter(x, y, color=color, alpha=0.25, s=6,
                   label=f"{group.capitalize()} BG", rasterized=True)

        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 5:
            r, p = stats.pearsonr(x[valid], y[valid])
            m, b = np.polyfit(x[valid], y[valid], 1)
            xr = np.array([x[valid].min(), x[valid].max()])
            ax.plot(xr, m * xr + b, color=color, linewidth=2, linestyle="--", zorder=3)
            stat_lines.append(
                (group, color,
                 f"{group.capitalize()}: r={r:+.2f}, p={p:.2g} {_sig_stars(p)} (n={len(gdf)})")
            )

    # Annotate per-group stats
    for i, (_, color, txt) in enumerate(stat_lines):
        ax.text(
            0.97, y_text - i * 0.10,
            txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            color=color, fontweight="bold",
        )

    # Mann-Whitney U between groups
    if "short" in group_slopes and "long" in group_slopes:
        s_sl = group_slopes["short"]
        l_sl = group_slopes["long"]
        if len(s_sl) >= 5 and len(l_sl) >= 5:
            _, p_mw = stats.mannwhitneyu(s_sl, l_sl, alternative="two-sided")
            ax.text(
                0.97, y_text - len(stat_lines) * 0.10,
                f"Short vs long: p={p_mw:.2g} {_sig_stars(p_mw)}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="black",
            )

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(f"{config.ANCHOR_LABELS[anchor]} → Decision window (s)")
    ax.set_ylabel("DA ramp slope (z/s)")
    n_excl = n_total - n_shown
    excl_note = f"  (excl. {n_excl} outliers >98th pct)" if n_excl > 0 else ""
    ax.set_title(f"Slope vs Window | {config.ANCHOR_LABELS[anchor]}{excl_note}")
    ax.legend(loc="upper left", fontsize=9, frameon=False)


# =============================================================================
# FIGURE 1
# =============================================================================

def plot_figure1(trials_df: pd.DataFrame, config: Config):
    """
    2 × 3 figure.
    Row 0: Anchor-aligned traces (short vs long BG group)
    Row 1: Slope vs window duration scatter (short vs long BG group)
    """
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for col, anchor in enumerate(config.ANCHORS):
        _plot_anchor_traces(axes[0, col], trials_df, anchor, config)
        _plot_slope_scatter(axes[1, col], trials_df, anchor, config)

    fig.suptitle(
        "DA Ramp Slope Tracks Subjective Time  —  All Trials",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()

    for fmt in ("png",):
        out = config.OUTPUT_DIR / f"fig1_da_ramp_subjective_time.{fmt}"
        fig.savefig(out, format=fmt)
        print(f"  Saved: {out}")

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Committee Figures — DA Ramp Slope Tracks Subjective Time")
    print("=" * 65)

    df = load_data(config)

    print("\nExtracting per-trial data...")
    trials_df = extract_trial_records(df, config)

    print("\nTrials per anchor:")
    for anchor in config.ANCHORS:
        total = (trials_df["anchor"] == anchor).sum()
        n_s = ((trials_df["anchor"] == anchor) & (trials_df["group"] == "short")).sum()
        n_l = ((trials_df["anchor"] == anchor) & (trials_df["group"] == "long")).sum()
        print(f"  {anchor:12s}: {total} total  ({n_s} short, {n_l} long)")

    print("\nGenerating Figure 1...")
    plot_figure1(trials_df, config)

    print("\nDone. Output directory:", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()

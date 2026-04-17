#!/usr/bin/env python3
"""
Figure 2: Previous Reward Modulates DA Dynamics
================================================

Publication-quality figure showing trial history effects on DA:
  Panel A: Effect sizes (Cohen's d) heatmap — Group × Anchor × Measure
  Panel B/C: Average traces split by previous outcome (last_lick anchor)
  Panel D: Causal chain diagram with real effect sizes

Usage:
    python figure_2_trial_history.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm
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
    """Configuration for Figure 2."""

    # Paths
    PROCESSED_OUT: Path = PROCESSED_OUT
    PIPELINE_SESSION_LOG: Path = PIPELINE_SESSION_LOG
    OUTPUT_DIR: Path = COMMITTEE_FIGURES_DIR
    OUTPUT_FORMATS: Tuple[str, ...] = ("png",)
    DPI: int = 300

    # Quality filter
    QUALITY_FILTER: bool = True
    SIGNAL_COL: str = DEFAULT_SIGNAL_COL
    DA_CHANNEL: str = "l_str_GRAB"

    # Figure dimensions
    FIG_WIDTH: float = 11.0
    FIG_HEIGHT: float = 9.5

    # Anchors — also used as keys throughout
    ANCHORS: Tuple[str, ...] = ("cue_on", "cue_off", "last_lick")
    ANCHOR_LABELS: Dict[str, str] = field(default_factory=lambda: {
        "cue_on":    "cue_on",
        "cue_off":   "cue_off",
        "last_lick": "last_lick",
    })
    # Column in merged CSV that gives window duration per anchor
    ANCHOR_WAIT_COL: Dict[str, str] = field(default_factory=lambda: {
        "cue_on":    "time_waited_since_cue_on",
        "cue_off":   "time_waited",
        "last_lick": "time_waited_since_last_lick",
    })

    # Measures
    MEASURES: Tuple[str, ...] = ("true_baseline", "slope", "ramp_amplitude", "predec_level")
    MEASURE_LABELS: Dict[str, str] = field(default_factory=lambda: {
        "true_baseline": "Baseline",
        "slope":         "Slope",
        "ramp_amplitude":"Ramp amp.",
        "predec_level":  "Pre-decision",
    })

    # Groups — short keys used as data dict keys
    GROUPS: Tuple[str, ...] = ("s", "l")
    GROUP_LABELS: Dict[str, str] = field(default_factory=lambda: {
        "s": "Short BG",
        "l": "Long BG",
    })

    # Timing
    MIN_WINDOW_SEC: float = 1.0
    BASELINE_WINDOW_SEC: float = 0.3    # baseline window at anchor (first 300ms)
    SLOPE_START_BUFFER: float = 0.3
    SLOPE_END_BUFFER: float = 0.2
    PREDEC_START: float = -0.5          # pre-decision level window start
    PREDEC_END: float = -0.05
    POST_DECISION_SEC: float = 1.5      # signal shown past decision in traces

    # Minimum trials per condition to compute effects
    MIN_TRIALS: int = 20

    # Trace display (decision-aligned)
    PLOT_TIME_RANGE: Tuple[float, float] = (-6, 1.5)

    # Colors
    PREV_REW_COLOR: str = "#0072B2"   # Blue
    PREV_UNREW_COLOR: str = "#d62728"
    SHORT_BG_COLOR: str = "#0072B2"
    LONG_BG_COLOR: str = "#D55E00"

    # Heatmap
    EFFECT_CMAP: str = "RdBu_r"
    EFFECT_VMAX: float = 0.5

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
    plt.rcParams.update({
        "font.family":        config.FONT_FAMILY,
        "font.size":          config.LABEL_SIZE,
        "axes.titlesize":     config.TITLE_SIZE,
        "axes.labelsize":     config.LABEL_SIZE,
        "xtick.labelsize":    config.TICK_SIZE,
        "ytick.labelsize":    config.TICK_SIZE,
        "legend.fontsize":    config.LEGEND_SIZE,
        "figure.facecolor":   "white",
        "figure.dpi":         100,
        "savefig.dpi":        config.DPI,
        "savefig.facecolor":  "white",
        "savefig.bbox":       "tight",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "black",
        "axes.linewidth":     0.8,
        "axes.grid":          False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "lines.linewidth":    1.5,
        "legend.frameon":     False,
    })


def add_panel_label(ax, label: str, config: FigureConfig, x: float = -0.12, y: float = 1.08):
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
    return ""


# =============================================================================
# DATA LOADING
# =============================================================================

def _build_allowed_sessions(config: FigureConfig) -> Optional[set]:
    if not config.QUALITY_FILTER:
        print("  Session filter: disabled")
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

    print(f"  Session filter: {len(allowed)} / {n_total} sessions pass")
    return allowed


def load_data(config: FigureConfig) -> pd.DataFrame:
    """Load and filter all sessions from pre-merged CSVs."""
    print(f"Scanning sessions in: {config.PROCESSED_OUT}")
    allowed = _build_allowed_sessions(config)

    frames: List[pd.DataFrame] = []
    for session_dir in sorted(config.PROCESSED_OUT.iterdir()):
        if not session_dir.is_dir():
            continue
        if allowed is not None and session_dir.name not in allowed:
            continue
        f = session_dir / "photometry_with_trial_data.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
            df["session_id"] = session_dir.name
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: {session_dir.name}: {e}")

    if not frames:
        raise FileNotFoundError(f"No data found under {config.PROCESSED_OUT}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {combined['session_id'].nunique()} sessions, {len(combined):,} rows")

    # Filter to GRAB channel
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

    # Non-miss trials only
    n_before = len(combined)
    if "miss_trial" in combined.columns:
        combined = combined[combined["miss_trial"] == False].copy()
    else:
        combined = combined[combined["time_waited"].notna() & (combined["time_waited"] < 60)].copy()
    print(f"  After miss filter: {n_before:,} → {len(combined):,} rows")

    # Filter trials where previous trial was also valid (not miss)
    if "previous_trial_miss_trial" in combined.columns:
        combined = combined[combined["previous_trial_miss_trial"] == False].copy()
    else:
        combined = combined[combined["previous_trial_reward"].isin([0.0, 5.0])].copy()

    # Normalize group labels and map to short keys
    _long_to_short = {"l": "l", "s": "s", "long": "l", "short": "s"}
    combined["group"] = combined["group"].map(lambda g: _long_to_short.get(str(g), str(g)))

    # Binary previous reward column
    combined["prev_rewarded"] = (combined["previous_trial_reward"] > 0).astype(bool)

    print(f"  After prev-miss filter: {len(combined):,} rows")
    print(f"  Groups: {sorted(combined['group'].unique())}")
    print(f"  Mice:   {sorted(combined['mouse'].unique())}")
    return combined


# =============================================================================
# PER-TRIAL EXTRACTION
# =============================================================================

def extract_trials_for_anchor(
    df: pd.DataFrame, anchor: str, config: FigureConfig
) -> pd.DataFrame:
    """
    Extract per-trial scalar measures + decision-aligned traces for one anchor.

    Scalar measures (matching v3 script convention):
      true_baseline   — mean signal in [anchor, anchor + BASELINE_WINDOW_SEC]
      slope           — linear fit from [anchor+buffer, decision-buffer]
      predec_level    — mean signal in [decision-0.5, decision-0.05]
      ramp_amplitude  — predec_level - true_baseline

    Traces are decision-aligned (t=0 at decision), baseline-subtracted.
    """
    wait_col = config.ANCHOR_WAIT_COL[anchor]
    records: List[Dict] = []

    time_grid = np.arange(
        config.PLOT_TIME_RANGE[0], config.PLOT_TIME_RANGE[1] + 0.05, 0.05
    )

    for (session_id, trial_num), trial_df in df.groupby(
        ["session_id", "session_trial_num"]
    ):
        trial_df = trial_df.sort_values("trial_time")
        row = trial_df.iloc[0]

        # Decision time in trial_time coordinates (≈ time_waited_since_cue_on)
        t_decision = float(row.get("time_waited_since_cue_on", np.nan))
        if not (np.isfinite(t_decision) and t_decision > 0):
            continue

        w = float(row.get(wait_col, np.nan))
        if not (np.isfinite(w) and w >= config.MIN_WINDOW_SEC):
            continue

        t_anchor = t_decision - w

        trial_time = trial_df["trial_time"].to_numpy(dtype=float)
        signal     = trial_df[config.SIGNAL_COL].to_numpy(dtype=float)

        # ── BASELINE ─────────────────────────────────────────────────────────
        bl_start = t_anchor
        bl_end   = t_anchor + config.BASELINE_WINDOW_SEC
        bl_mask  = (trial_time >= bl_start) & (trial_time <= bl_end)
        if bl_mask.sum() < 3:
            continue
        true_baseline = float(np.nanmean(signal[bl_mask]))

        # ── PRE-DECISION LEVEL ────────────────────────────────────────────────
        pd_mask = (
            (trial_time >= t_decision + config.PREDEC_START)
            & (trial_time <= t_decision + config.PREDEC_END)
        )
        predec_level = float(np.nanmean(signal[pd_mask])) if pd_mask.sum() >= 3 else np.nan

        # ── SLOPE ─────────────────────────────────────────────────────────────
        sl_start = t_anchor + config.SLOPE_START_BUFFER
        sl_end   = t_decision - config.SLOPE_END_BUFFER
        slope = np.nan
        if sl_end > sl_start + 0.3:
            sl_mask = (trial_time >= sl_start) & (trial_time <= sl_end)
            t_sl = trial_time[sl_mask]
            y_sl = signal[sl_mask]
            if len(t_sl) >= 5:
                tm, ym = np.mean(t_sl), np.mean(y_sl)
                tt, yy = t_sl - tm, y_sl - ym
                denom = np.dot(tt, tt)
                if denom > 0:
                    slope = float(np.dot(tt, yy) / denom)

        ramp_amplitude = (predec_level - true_baseline
                          if np.isfinite(predec_level) else np.nan)

        # ── TRACE (decision-aligned, baseline-subtracted) ─────────────────────
        tr_mask = (
            (trial_time >= t_decision + config.PLOT_TIME_RANGE[0])
            & (trial_time <= t_decision + config.PLOT_TIME_RANGE[1])
        )
        if tr_mask.sum() < 3:
            continue

        t_trace   = trial_time[tr_mask] - t_decision   # t=0 at decision
        sig_trace = signal[tr_mask] - true_baseline     # baseline-subtracted

        # Interpolate to common grid for averaging later
        sig_interp = np.interp(time_grid, t_trace, sig_trace,
                               left=np.nan, right=np.nan)

        records.append({
            "session_id":      session_id,
            "mouse":           row["mouse"],
            "group":           row["group"],
            "prev_rewarded":   bool(row["prev_rewarded"]),
            "anchor":          anchor,
            "window_duration": w,
            "wait_time":       w,
            "true_baseline":   true_baseline,
            "slope":           slope,
            "predec_level":    predec_level,
            "ramp_amplitude":  ramp_amplitude,
            "sig_interp":      sig_interp,   # on common time_grid
        })

    print(f"  {anchor}: {len(records)} trials extracted")
    return pd.DataFrame(records)


# =============================================================================
# EFFECTS, CORRELATIONS, TRACES
# =============================================================================

def _cohens_d(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """t-test + Cohen's d (a − b). Returns (d, p)."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return 0.0, 1.0
    _, p = stats.ttest_ind(a, b)
    pooled = np.sqrt((np.var(a) + np.var(b)) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0
    return float(d), float(p)


def compute_effects(trials_df: pd.DataFrame, config: FigureConfig) -> Dict:
    """
    Compute Cohen's d (prev_rew − prev_unrew) for every group × anchor × measure.
    Returns effects[group_key][anchor][measure] = (d, p).
    """
    effects: Dict = {}
    for group_key in config.GROUPS:
        effects[group_key] = {}
        g_df = trials_df[trials_df["group"] == group_key]
        for anchor in config.ANCHORS:
            a_df = g_df[g_df["anchor"] == anchor]
            rew   = a_df[a_df["prev_rewarded"] == True]
            unrew = a_df[a_df["prev_rewarded"] == False]
            effects[group_key][anchor] = {}
            for m in config.MEASURES:
                d, p = _cohens_d(rew[m].to_numpy(float), unrew[m].to_numpy(float))
                effects[group_key][anchor][m] = (d, p)
    return effects


def compute_correlations(trials_df: pd.DataFrame, config: FigureConfig) -> Dict:
    """
    Pearson correlations for the causal chain.
    Returns correlations[group_key][anchor][pair] = (r, p).
    """
    corr_pairs = [
        ("prev_reward_to_baseline", "prev_rewarded_int", "true_baseline"),
        ("prev_reward_to_slope",    "prev_rewarded_int", "slope"),
        ("baseline_to_slope",       "true_baseline",     "slope"),
        ("slope_to_wait",           "slope",             "wait_time"),
        ("prev_reward_to_wait",     "prev_rewarded_int", "wait_time"),
    ]

    trials_df = trials_df.copy()
    trials_df["prev_rewarded_int"] = trials_df["prev_rewarded"].astype(int)

    result: Dict = {}
    for group_key in config.GROUPS:
        result[group_key] = {}
        g_df = trials_df[trials_df["group"] == group_key]
        for anchor in config.ANCHORS:
            a_df = g_df[g_df["anchor"] == anchor]
            result[group_key][anchor] = {}
            for name, col_x, col_y in corr_pairs:
                valid = a_df[[col_x, col_y]].dropna()
                if len(valid) < 20:
                    result[group_key][anchor][name] = (0.0, 1.0)
                    continue
                r, p = stats.pearsonr(
                    valid[col_x].to_numpy(float),
                    valid[col_y].to_numpy(float),
                )
                result[group_key][anchor][name] = (float(r), float(p))
    return result


def compute_traces(
    trials_df: pd.DataFrame, config: FigureConfig
) -> Dict:
    """
    Average decision-aligned traces per group × anchor × condition (prev_rew/unrew).
    Returns traces[group_key][anchor] = {time, prev_rew:{mean,sem,n}, prev_unrew:{...}}.
    """
    time_grid = np.arange(
        config.PLOT_TIME_RANGE[0], config.PLOT_TIME_RANGE[1] + 0.05, 0.05
    )
    traces: Dict = {}

    for group_key in config.GROUPS:
        traces[group_key] = {}
        g_df = trials_df[trials_df["group"] == group_key]

        for anchor in config.ANCHORS:
            a_df = g_df[g_df["anchor"] == anchor]
            traces[group_key][anchor] = {"time": time_grid}

            for cond_name, cond_bool in [("prev_rew", True), ("prev_unrew", False)]:
                c_df = a_df[a_df["prev_rewarded"] == cond_bool]
                n = len(c_df)

                if n < config.MIN_TRIALS:
                    traces[group_key][anchor][cond_name] = {
                        "mean": np.full(len(time_grid), np.nan),
                        "sem":  np.full(len(time_grid), np.nan),
                        "n":    n,
                    }
                    continue

                signals = np.vstack(c_df["sig_interp"].to_numpy())
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean_s = np.nanmean(signals, axis=0)
                    n_valid = np.sum(np.isfinite(signals), axis=0)
                    sem_s  = np.nanstd(signals, axis=0) / np.sqrt(np.maximum(n_valid, 1))

                # NaN where coverage < 20%
                low = n_valid < max(config.MIN_TRIALS, 0.2 * n)
                mean_s[low] = np.nan
                sem_s[low]  = np.nan

                traces[group_key][anchor][cond_name] = {
                    "mean": mean_s,
                    "sem":  sem_s,
                    "n":    n,
                }

    return traces


def load_real_data(config: FigureConfig) -> Dict:
    """
    Load all sessions and compute:
      effects      — Cohen's d for each group × anchor × measure
      correlations — Pearson r for causal chain pairs
      traces       — average DA traces by prev outcome
    """
    df = load_data(config)

    # Extract trials for all anchors in one pass
    all_trials = pd.concat(
        [extract_trials_for_anchor(df, anchor, config) for anchor in config.ANCHORS],
        ignore_index=True,
    )

    print("Computing effects...")
    effects = compute_effects(all_trials, config)

    print("Computing correlations...")
    correlations = compute_correlations(all_trials, config)

    print("Computing traces...")
    traces = compute_traces(all_trials, config)

    # Print summary
    print("\nEffect sizes (Cohen's d, prev_rew − prev_unrew):")
    for group_key in config.GROUPS:
        print(f"  {config.GROUP_LABELS[group_key]}:")
        for anchor in config.ANCHORS:
            vals = effects[group_key][anchor]
            d_bl  = vals["true_baseline"][0]
            p_bl  = vals["true_baseline"][1]
            d_sl  = vals["slope"][0]
            p_sl  = vals["slope"][1]
            print(
                f"    {anchor:12s}  baseline d={d_bl:+.3f}{add_significance_stars(p_bl)}"
                f"   slope d={d_sl:+.3f}{add_significance_stars(p_sl)}"
            )

    return {"effects": effects, "correlations": correlations, "traces": traces}


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_effect_heatmap(ax, data: Dict, config: FigureConfig):
    """
    Panel A: Effect sizes heatmap.
    Rows = measures, Columns = anchor × group (6 cols).
    """
    effects = data["effects"]

    n_rows = len(config.MEASURES)
    n_cols = len(config.ANCHORS) * len(config.GROUPS)
    matrix    = np.zeros((n_rows, n_cols))
    sig_matrix = np.empty((n_rows, n_cols), dtype=object)

    col_labels = []
    for anchor in config.ANCHORS:
        for group in config.GROUPS:
            col_labels.append(
                f"{config.GROUP_LABELS[group]}\n{config.ANCHOR_LABELS[anchor]}"
            )

    col_idx = 0
    for anchor in config.ANCHORS:
        for group in config.GROUPS:
            for row_idx, measure in enumerate(config.MEASURES):
                d, p = effects[group][anchor][measure]
                matrix[row_idx, col_idx]    = d
                sig_matrix[row_idx, col_idx] = add_significance_stars(p)
            col_idx += 1

    norm = TwoSlopeNorm(vmin=-config.EFFECT_VMAX, vcenter=0, vmax=config.EFFECT_VMAX)
    im   = ax.imshow(matrix, cmap=config.EFFECT_CMAP, norm=norm, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            sig = sig_matrix[i, j]
            color = "white" if abs(val) > 0.25 else "black"
            ax.text(j, i, f"{val:+.2f}{sig}", ha="center", va="center",
                    fontsize=config.TICK_SIZE - 1, color=color)

    # Highlight last_lick × Short BG column (index 4: ll×s, ll×l → col 4 = ll×s)
    key_col = 4
    rect = Rectangle((key_col - 0.5, -0.5), 1, n_rows,
                      linewidth=2, edgecolor="black", facecolor="none")
    ax.add_patch(rect)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=config.TICK_SIZE - 1)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([config.MEASURE_LABELS[m] for m in config.MEASURES])
    ax.set_xlabel("Group × Anchor")
    ax.set_title("Effect of previous reward (Cohen's d)")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cohen's d\n(prev_rew − prev_unrew)")

    for i in range(1, len(config.ANCHORS)):
        ax.axvline(i * 2 - 0.5, color="white", linewidth=2)


def plot_traces_comparison(ax, data: Dict, group: str, anchor: str, config: FigureConfig):
    """
    Panel B/C: Average traces split by previous outcome for one group/anchor.

    Significance bar: pointwise independent t-test reconstructed from mean/SEM/n
    at each time bin, shown as a black strip at the bottom of the axes.
    Effect size annotation: slope Cohen's d + p from precomputed effects.
    """
    from matplotlib.transforms import blended_transform_factory

    trace_data = data["traces"][group][anchor]
    time = trace_data["time"]

    stored: Dict = {}
    for outcome, label_suffix in [("prev_rew", "rewarded"), ("prev_unrew", "unrewarded")]:
        mean  = trace_data[outcome]["mean"]
        sem   = trace_data[outcome]["sem"]
        n     = trace_data[outcome]["n"]
        color = config.PREV_REW_COLOR if outcome == "prev_rew" else config.PREV_UNREW_COLOR
        stored[outcome] = {"mean": mean, "sem": sem, "n": n}

        valid = np.isfinite(mean)
        ax.plot(time[valid], mean[valid], color=color, linewidth=1.8,
                label=f"Prev {label_suffix} (n={n:,})")
        ax.fill_between(time[valid],
                        (mean - sem)[valid], (mean + sem)[valid],
                        color=color, alpha=0.2, linewidth=0)

    ax.axvline(0,  color="gray", linestyle="--", linewidth=1,   alpha=0.7)
    ax.axhline(0,  color="gray", linestyle="-",  linewidth=0.5, alpha=0.3)

    # ── Pointwise significance bar ─────────────────────────────────────────
    # Reconstruct t = (mean_r - mean_u) / sqrt(sem_r² + sem_u²)
    m_r, s_r, n_r = stored["prev_rew"]["mean"],  stored["prev_rew"]["sem"],  stored["prev_rew"]["n"]
    m_u, s_u, n_u = stored["prev_unrew"]["mean"], stored["prev_unrew"]["sem"], stored["prev_unrew"]["n"]

    both_valid = np.isfinite(m_r) & np.isfinite(m_u) & (s_r > 0) & (s_u > 0)
    denom  = np.where(both_valid, np.sqrt(s_r ** 2 + s_u ** 2), np.nan)
    t_stat = np.where(both_valid, (m_r - m_u) / denom, np.nan)
    df     = n_r + n_u - 2
    p_vals = np.where(both_valid, 2 * stats.t.sf(np.abs(t_stat), df), 1.0)

    sig05  = both_valid & (p_vals < 0.05)
    sig001 = both_valid & (p_vals < 0.001)

    # Use blended transform: data coords for x, axes (0–1) for y
    # so the bar sits at the bottom regardless of ylim
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    if sig05.any():
        ax.fill_between(time, 0, 0.04, where=sig05,
                        transform=trans, color="gray", alpha=0.5,
                        linewidth=0, zorder=5)
    if sig001.any():
        ax.fill_between(time, 0, 0.04, where=sig001,
                        transform=trans, color="black", alpha=0.8,
                        linewidth=0, zorder=6)

    # ── Effect size annotation (slope) ────────────────────────────────────
    if "effects" in data and group in data["effects"] and anchor in data["effects"][group]:
        d, p = data["effects"][group][anchor]["slope"]
        sig  = add_significance_stars(p)
        ax.text(0.97, 0.97,
                f"slope: d = {d:+.2f}{sig}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=config.TICK_SIZE,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="gray", alpha=0.85))

    ax.set_xlabel("Time from decision (s)")
    ax.set_ylabel("DA (z-score, baseline-subtracted)")
    ax.set_title(f"{config.GROUP_LABELS[group]}  —  {config.ANCHOR_LABELS[anchor]} anchor")
    ax.legend(loc="lower left", fontsize=config.LEGEND_SIZE - 1)
    ax.set_xlim(config.PLOT_TIME_RANGE)


def plot_causal_chain(ax, data: Dict, config: FigureConfig):
    """
    Panel D: Causal chain diagram with real effect sizes from SHORT BG × last_lick.

    REWARD(n−1) → BASELINE(n) → SLOPE(n) → WAIT(n)
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Pull real values
    corr_s  = data["correlations"]["s"]["last_lick"]
    eff_s   = data["effects"]["s"]["last_lick"]

    def _fmt_r(key):
        r, p = corr_s[key]
        return f"r = {r:+.2f}{add_significance_stars(p)}"

    def _fmt_d(measure):
        d, p = eff_s[measure]
        return f"d = {d:+.2f}{add_significance_stars(p)}"

    boxes = {
        "reward":   (0.3,  2, 2, 0.8),
        "baseline": (3.0,  2, 2, 0.8),
        "slope":    (5.7,  2, 2, 0.8),
        "wait":     (8.4,  2, 2, 0.8),
    }
    box_labels = {
        "reward":   "Previous\nreward",
        "baseline": "DA\nbaseline",
        "slope":    "DA ramp\nslope",
        "wait":     "Wait\ntime",
    }
    box_style = dict(
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor="white", edgecolor="black", linewidth=1.5,
    )

    for name, (x, y, w, h) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, **box_style))
        ax.text(x + w / 2, y + h / 2, box_labels[name],
                ha="center", va="center",
                fontsize=config.LABEL_SIZE, fontweight="500")

    # Sequential arrows with real annotation strings
    arrow_defs = [
        ("reward",   "baseline", _fmt_d("true_baseline")),
        ("baseline", "slope",    _fmt_r("baseline_to_slope") + "\n(ceiling)"),
        ("slope",    "wait",     _fmt_r("slope_to_wait")),
    ]
    label_bg = dict(boxstyle="round,pad=0.15", facecolor="lightyellow",
                    edgecolor="gray", alpha=0.9)

    for start, end, label in arrow_defs:
        x1, y1, w1, h1 = boxes[start]
        x2, y2, w2, h2 = boxes[end]
        ax.add_patch(FancyArrowPatch(
            (x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2),
            arrowstyle="->", mutation_scale=15,
            color="black", linewidth=1.5,
        ))
        mid_x = (x1 + w1 + x2) / 2
        ax.text(mid_x, y1 + h1 / 2 + 0.35, label,
                ha="center", va="bottom",
                fontsize=config.TICK_SIZE,
                bbox=label_bg)

    # Direct reward → wait (curved, dashed)
    x1, y1, w1, h1 = boxes["reward"]
    x2, y2, w2, h2 = boxes["wait"]
    ax.add_patch(FancyArrowPatch(
        (x1 + w1 / 2, y1), (x2 + w2 / 2, y2),
        arrowstyle="->", mutation_scale=12,
        color="gray", linewidth=1, linestyle="--",
        connectionstyle="arc3,rad=-0.3",
    ))
    r_str = _fmt_r("prev_reward_to_wait")
    ax.text(4.5, 0.8, f"{r_str}\n(direct)",
            ha="center", va="top", fontsize=config.TICK_SIZE, color="gray",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="gray", alpha=0.8))

    ax.set_title(
        "SHORT BG causal chain (last_lick anchor): DA mediates trial history",
        fontsize=config.TITLE_SIZE, fontweight="bold", pad=10,
    )
    ax.text(5, 3.65,
            "Complete mediation: reward → baseline ↑ → slope ↓ → wait ↑",
            ha="center", va="bottom",
            fontsize=config.LABEL_SIZE - 1, style="italic", color="darkgreen")


def plot_group_comparison_bars(ax, data: Dict, anchor: str, measure: str, config: FigureConfig):
    """Small bar chart: group comparison for one measure at one anchor."""
    effects = data["effects"]
    groups  = list(config.GROUPS)
    x       = np.arange(len(groups))

    vals   = [effects[g][anchor][measure][0] for g in groups]
    ps     = [effects[g][anchor][measure][1] for g in groups]
    colors = [config.SHORT_BG_COLOR, config.LONG_BG_COLOR]

    bars = ax.bar(x, vals, 0.6, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)

    for bar, p in zip(bars, ps):
        stars = add_significance_stars(p)
        y_pos = (bar.get_height() + 0.02
                 if bar.get_height() >= 0
                 else bar.get_height() - 0.05)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, stars,
                ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                fontsize=config.TICK_SIZE + 1, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([config.GROUP_LABELS[g] for g in groups])
    ax.set_ylabel("Cohen's d")
    ax.set_title(f"{config.MEASURE_LABELS[measure]} at {anchor}")


# =============================================================================
# MAIN FIGURE ASSEMBLY
# =============================================================================

def create_figure_2a_heatmap(data: Dict, config: FigureConfig) -> plt.Figure:
    """Panel A only: effect sizes heatmap as a standalone figure."""
    setup_publication_style(config)
    fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, 3.5))
    plot_effect_heatmap(ax, data, config)
    add_panel_label(ax, "A", config, x=-0.08)
    plt.tight_layout()
    return fig


def create_figure_2(data: Dict, config: FigureConfig) -> plt.Figure:
    """
    Main Figure 2.

        ┌──────────────────────────────────────────────┐
        │  A. Effect sizes heatmap (Group × Anchor)    │
        ├──────────────────────┬───────────────────────┤
        │  B. SHORT BG traces  │  C. LONG BG traces    │
        │     (last_lick)      │      (last_lick)      │
        ├──────────────────────┴───────────────────────┤
        │  D. Causal chain diagram (SHORT BG)          │
        └──────────────────────────────────────────────┘
    """
    setup_publication_style(config)

    fig = plt.figure(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[1.3, 1, 1],
                            hspace=0.4, wspace=0.3)

    ax_heatmap = fig.add_subplot(gs[0, :])
    plot_effect_heatmap(ax_heatmap, data, config)
    add_panel_label(ax_heatmap, "A", config, x=-0.08)

    ax_tr_short = fig.add_subplot(gs[1, 0])
    plot_traces_comparison(ax_tr_short, data, "s", "last_lick", config)
    add_panel_label(ax_tr_short, "B", config)

    ax_tr_long = fig.add_subplot(gs[1, 1])
    plot_traces_comparison(ax_tr_long, data, "l", "last_lick", config)
    add_panel_label(ax_tr_long, "C", config)

    ax_causal = fig.add_subplot(gs[2, :])
    plot_causal_chain(ax_causal, data, config)
    add_panel_label(ax_causal, "D", config, x=-0.08, y=1.02)

    return fig


def create_figure_2_extended(data: Dict, config: FigureConfig) -> plt.Figure:
    """
    Extended Figure 2: traces at all anchors, both groups (2 × 3 = 6 panels).
    """
    setup_publication_style(config)

    fig = plt.figure(figsize=(config.FIG_WIDTH * 1.3, config.FIG_HEIGHT * 0.8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    panel_labels = iter("ABCDEF")
    for row, group in enumerate(config.GROUPS):
        for col, anchor in enumerate(config.ANCHORS):
            ax = fig.add_subplot(gs[row, col])
            plot_traces_comparison(ax, data, group, anchor, config)
            add_panel_label(ax, next(panel_labels), config)

            # Highlight the key panel (Short BG × last_lick)
            if group == "s" and anchor == "last_lick":
                for spine in ax.spines.values():
                    spine.set_color("red")
                    spine.set_linewidth(2)

    fig.suptitle(
        "Figure 2 (Extended): DA traces by previous outcome — All anchors",
        fontsize=config.TITLE_SIZE + 1, fontweight="bold", y=0.98,
    )
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = FigureConfig()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Figure 2: Previous Reward Modulates DA Dynamics")
    print("=" * 60)

    data = load_real_data(config)

    print("\nCreating Figure 2a (heatmap only)...")
    fig_heatmap = create_figure_2a_heatmap(data, config)
    for fmt in config.OUTPUT_FORMATS:
        out = config.OUTPUT_DIR / f"figure_2a_heatmap.{fmt}"
        fig_heatmap.savefig(out, format=fmt, dpi=config.DPI)
        print(f"  Saved: {out}")
    plt.close(fig_heatmap)

    print("\nCreating Figure 2 (main)...")
    fig = create_figure_2(data, config)
    for fmt in config.OUTPUT_FORMATS:
        out = config.OUTPUT_DIR / f"figure_2_trial_history.{fmt}"
        fig.savefig(out, format=fmt, dpi=config.DPI)
        print(f"  Saved: {out}")
    plt.close(fig)

    print("\nCreating Figure 2 (extended — all anchors)...")
    fig_ext = create_figure_2_extended(data, config)
    for fmt in config.OUTPUT_FORMATS:
        out = config.OUTPUT_DIR / f"figure_2_extended.{fmt}"
        fig_ext.savefig(out, format=fmt, dpi=config.DPI)
        print(f"  Saved: {out}")
    plt.close(fig_ext)

    print(f"\nDone. Outputs in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

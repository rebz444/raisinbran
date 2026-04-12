#!/usr/bin/env python3
"""
Committee Meeting Figures: DA Ramp and Timing Analysis
=======================================================

Generates minimum viable figures for April 14, 2026 committee meeting.

Based on framework from:
- Kim & Uchida 2020 (Cell): DA ramps = TD RPE, not value
- Jakob & Gershman 2022: Bidirectional timing updates
- Hamilos et al. 2021 (eLife): Single-trial ramp-behavior correlation (R² up to 0.82)

Figures to generate:
1. GRAB-DA Ramp Dynamics (trial-averaged, decision-aligned)
2. Ramp Slope Predicts Wait Time (scatter + mixed-effects regression)
3. Trial History Chain (outcome → slope → wait)
4. Per-Mouse Consistency (individual animal panels)

Usage:
    python 5_committee_figures.py

Assumes you have already run 4_photometry_wait_analysis.py in BATCH mode
to generate trial_metrics.csv for each session.
"""

from pathlib import Path
from typing import Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from utils import get_session_groups

# Try importing statsmodels for mixed-effects; fall back to simple regression
try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not available; using simple regression instead of mixed-effects")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration for the analysis."""
    
    # Paths - adjust these to your setup
    PROCESSED_OUT: Path = Path("/Volumes/T7 Shield/photometry/processed_output")
    OUTPUT_DIR: Path = Path("/Volumes/T7 Shield/photometry/committee_figures")
    # Pre-computed by 6_decision_aligned_traces.py
    DECISION_ALIGNED_CSV: Path = OUTPUT_DIR / "decision_aligned_averages.csv"
    
    # Analysis parameters (from Jakob & Gershman / Hamilos methods)
    PRE_DECISION_BUFFER = 0.2  # seconds before decision to exclude
    T_MIN_AFTER_ALIGN = 0.3    # seconds after alignment to start ramp fit
    MIN_WAIT_FOR_RAMP = 0.8    # minimum wait time to include in ramp analysis
    
    # Figure aesthetics
    COLORS = {
        "long": "#2E86AB",      # Blue for long BG
        "short": "#A23B72",     # Magenta for short BG
        "da_signal": "#1B9E77", # Teal for DA
        "iso": "#999999",       # Gray for isosbestic
        "null": "#CCCCCC",      # Light gray for null distributions
    }
    
    MOUSE_COLORS = {
        "RZ074": "#1f77b4",
        "RZ075": "#ff7f0e",
        "RZ081": "#2ca02c",
        "RZ082": "#d62728",
        "RZ083": "#9467bd",
        "RZ085": "#8c564b",
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_session_metrics(config: Config) -> pd.DataFrame:
    """
    Load and concatenate trial_metrics.csv from all sessions.
    
    Returns DataFrame with columns:
        - mouse, session_id, group (long/short)
        - All trial-level metrics from the per-session analysis
    """
    all_data = []
    group_map = get_session_groups()

    # Find all wait_analysis directories
    for session_dir in config.PROCESSED_OUT.iterdir():
        if not session_dir.is_dir():
            continue

        metrics_file = session_dir / "wait_analysis" / "trial_metrics.csv"
        if not metrics_file.exists():
            continue

        # Parse session info from directory name (format: MOUSE_YYYYMMDD_HHMMSS)
        session_id = session_dir.name
        parts = session_id.split("_")
        if len(parts) < 2:
            continue
        mouse = parts[0]

        if mouse not in group_map:
            print(f"Skipping {session_id}: mouse {mouse} not in sessions_dff_behav_merged")
            continue

        # Load metrics
        try:
            df = pd.read_csv(metrics_file)
            all_data.append(df.assign(mouse=mouse, session_id=session_id, group=group_map[mouse]))
        except Exception as e:
            print(f"Error loading {metrics_file}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No session data found! Check paths and run 4_photometry_wait_analysis.py first.")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined)} trials from {len(all_data)} sessions")
    print(f"Mice: {combined['mouse'].unique()}")
    print(f"Groups: {combined.groupby('group').size().to_dict()}")
    
    return combined


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def filter_good_trials(df: pd.DataFrame, min_wait: float = 0.5) -> pd.DataFrame:
    """
    Filter to good trials for ramp analysis.

    Excludes:
    - Miss trials (no decision) — uses miss_trial column written by file 4
    - BG restart trials (bg_repeats > 1) — uses bg_restart column written by file 4
    - Impulsive trials — uses impulsive column written by file 4
      (time_waited < impulsive_sec, default 0.25 s)
    - Trials where wait_from_cue_off < min_wait (additional ramp-specific threshold)
    """
    mask = (
        (df["has_decision"] == True) &
        (df.get("miss_trial", False) == False) &
        (df.get("bg_restart", False) == False)
    )
    # Use the impulsive flag computed by file 4 (time_waited < impulsive_sec)
    # rather than re-deriving from wait_from_cue_off with a different threshold.
    if "impulsive" in df.columns:
        mask = mask & (~df["impulsive"].astype(bool))
    # Additional minimum wait threshold for ramp slope validity
    if min_wait > 0:
        mask = mask & (df["wait_from_cue_off"] >= min_wait)
    return df[mask].copy()


def get_alignment_column(group: str) -> Tuple[str, str]:
    """
    Return (slope_col, wait_col) based on group alignment.
    
    Long BG → align to cue_on
    Short BG → align to cue_off
    """
    if group == "long":
        return "slope_cue_on", "wait_from_cue_on"
    else:
        return "slope_cue_off", "wait_from_cue_off"


def compute_slope_wait_correlation(df: pd.DataFrame, group: str) -> dict:
    """Compute correlation between ramp slope and wait time."""
    slope_col, wait_col = get_alignment_column(group)
    
    valid = df[np.isfinite(df[slope_col]) & np.isfinite(df[wait_col])].copy()
    if len(valid) < 10:
        return {"r": np.nan, "p": np.nan, "n": len(valid)}
    
    r, p = stats.pearsonr(valid[slope_col], valid[wait_col])
    return {"r": r, "p": p, "n": len(valid)}


def run_mixed_effects_regression(df: pd.DataFrame) -> dict:
    """
    Run mixed-effects regression: wait_time ~ slope + (1|mouse/session)
    
    Falls back to simple regression if statsmodels unavailable.
    """
    # Prepare data with proper alignment per group
    rows = []
    for _, row in df.iterrows():
        slope_col, wait_col = get_alignment_column(row["group"])
        if np.isfinite(row.get(slope_col, np.nan)) and np.isfinite(row.get(wait_col, np.nan)):
            rows.append({
                "slope": row[slope_col],
                "wait": row[wait_col],
                "mouse": row["mouse"],
                "session_id": row["session_id"],
                "group": row["group"],
            })
    
    reg_df = pd.DataFrame(rows)
    if len(reg_df) < 20:
        return {"beta": np.nan, "se": np.nan, "p": np.nan, "n": len(reg_df)}
    
    if HAS_STATSMODELS:
        try:
            # Mixed-effects model: wait ~ slope with random intercept for mouse
            model = smf.mixedlm("wait ~ slope", reg_df, groups=reg_df["mouse"])
            result = model.fit(disp=False)
            return {
                "beta": result.params["slope"],
                "se": result.bse["slope"],
                "p": result.pvalues["slope"],
                "n": len(reg_df),
                "model": result,
            }
        except Exception as e:
            print(f"Mixed-effects failed: {e}; falling back to simple regression")
    
    # Fallback: simple OLS
    slope, intercept, r, p, se = stats.linregress(reg_df["slope"], reg_df["wait"])
    return {"beta": slope, "se": se, "p": p, "n": len(reg_df), "intercept": intercept}


def compute_trial_history_chain(df: pd.DataFrame) -> dict:
    """
    Compute the three-link chain from Jakob & Gershman:
    
    Link 1: reward_outcome(n) → slope(n+1)
    Link 2: slope(n) → wait(n)
    Link 3: Full mediation test
    
    Returns dict with correlation/regression results for each link.
    """
    results = {}
    
    # Need to compute lagged variables within each session
    chain_rows = []
    for (mouse, session), sdf in df.groupby(["mouse", "session_id"]):
        sdf = sdf.sort_values("session_trial_num").copy()
        group = sdf["group"].iloc[0]
        slope_col, wait_col = get_alignment_column(group)
        
        # Add lagged columns (previous_trial_reward is precomputed in trial_metrics.csv)
        sdf["prev_rewarded"] = sdf["previous_trial_reward"] > 0
        sdf["prev_wait"] = sdf[wait_col].shift(1)
        sdf["prev_slope"] = sdf[slope_col].shift(1)
        sdf["next_slope"] = sdf[slope_col].shift(-1)
        
        # Current trial values
        sdf["slope"] = sdf[slope_col]
        sdf["wait"] = sdf[wait_col]
        
        chain_rows.append(sdf)
    
    chain_df = pd.concat(chain_rows, ignore_index=True)
    
    # Link 1: Previous reward → Current slope
    valid_1 = chain_df.dropna(subset=["prev_rewarded", "slope"])
    if len(valid_1) >= 20:
        # Group by prev_rewarded and compare slopes
        rewarded_slopes = valid_1[valid_1["prev_rewarded"] == True]["slope"]
        unrewarded_slopes = valid_1[valid_1["prev_rewarded"] == False]["slope"]
        
        if len(rewarded_slopes) >= 5 and len(unrewarded_slopes) >= 5:
            t_stat, p_val = stats.ttest_ind(rewarded_slopes, unrewarded_slopes)
            effect_size = rewarded_slopes.mean() - unrewarded_slopes.mean()
            results["link1"] = {
                "rewarded_mean": rewarded_slopes.mean(),
                "unrewarded_mean": unrewarded_slopes.mean(),
                "effect": effect_size,
                "t": t_stat,
                "p": p_val,
                "n_rewarded": len(rewarded_slopes),
                "n_unrewarded": len(unrewarded_slopes),
            }
    
    # Link 2: Current slope → Current wait
    valid_2 = chain_df.dropna(subset=["slope", "wait"])
    if len(valid_2) >= 20:
        r, p = stats.pearsonr(valid_2["slope"], valid_2["wait"])
        results["link2"] = {"r": r, "p": p, "n": len(valid_2)}
    
    # Link 3: Previous reward → Current wait (total effect)
    valid_3 = chain_df.dropna(subset=["prev_rewarded", "wait"])
    if len(valid_3) >= 20:
        rewarded_waits = valid_3[valid_3["prev_rewarded"] == True]["wait"]
        unrewarded_waits = valid_3[valid_3["prev_rewarded"] == False]["wait"]
        
        if len(rewarded_waits) >= 5 and len(unrewarded_waits) >= 5:
            t_stat, p_val = stats.ttest_ind(rewarded_waits, unrewarded_waits)
            effect_size = rewarded_waits.mean() - unrewarded_waits.mean()
            results["link3_total"] = {
                "rewarded_mean": rewarded_waits.mean(),
                "unrewarded_mean": unrewarded_waits.mean(),
                "effect": effect_size,
                "t": t_stat,
                "p": p_val,
            }
    
    return results, chain_df


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def setup_figure_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def figure_1_ramp_dynamics(df: pd.DataFrame, config: Config, output_dir: Path):
    """
    Figure 3.1: GRAB-DA Ramp Dynamics

    Loads pre-computed decision-aligned averages from 6_decision_aligned_traces.py.
    Run that script first to generate decision_aligned_averages.csv.

    Panel A: Grand average across all mice
    Panel B: Long BG vs Short BG group comparison
    Panel C: Per-mouse traces
    """
    if not config.DECISION_ALIGNED_CSV.exists():
        print(
            f"  Skipping Figure 3.1: {config.DECISION_ALIGNED_CSV} not found.\n"
            "  Run 6_decision_aligned_traces.py first to generate it."
        )
        return

    avg = pd.read_csv(config.DECISION_ALIGNED_CSV)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    def _declines(ax):
        ax.axvline(0, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
        ax.set_xlabel("Time from decision (s)")
        ax.set_ylabel("DA (z-scored dF/F)")

    # Panel A: Grand average across all mice (mean of per-mouse means)
    ax = axes[0]
    grand = avg.groupby("time").agg(
        mean=("mean", "mean"),
        sem=("sem", lambda x: np.sqrt(np.sum(x**2)) / len(x)),
    ).reset_index()
    ax.plot(grand["time"], grand["mean"], color=config.COLORS["da_signal"], linewidth=2)
    ax.fill_between(grand["time"],
                    grand["mean"] - grand["sem"],
                    grand["mean"] + grand["sem"],
                    color=config.COLORS["da_signal"], alpha=0.3)
    _declines(ax)
    n_mice = avg["mouse"].nunique()
    n_trials = int(avg.groupby("time")["n"].mean().mean())
    ax.set_title(f"A. All mice (n={n_mice}, ~{n_trials} trials/bin)")

    # Panel B: Group comparison
    ax = axes[1]
    for group in ("long", "short"):
        gdf = avg[avg["group"] == group]
        if len(gdf) == 0:
            continue
        grp = gdf.groupby("time").agg(
            mean=("mean", "mean"),
            sem=("sem", lambda x: np.sqrt(np.sum(x**2)) / len(x)),
        ).reset_index()
        color = config.COLORS.get(group, "gray")
        ax.plot(grp["time"], grp["mean"], color=color, linewidth=2,
                label=f"{group.upper()} BG")
        ax.fill_between(grp["time"],
                        grp["mean"] - grp["sem"],
                        grp["mean"] + grp["sem"],
                        color=color, alpha=0.2)
    _declines(ax)
    ax.set_title("B. Long BG vs Short BG")
    ax.legend()

    # Panel C: Per-mouse traces
    ax = axes[2]
    for mouse in sorted(avg["mouse"].unique()):
        mdf = avg[avg["mouse"] == mouse].sort_values("time")
        color = config.MOUSE_COLORS.get(mouse, "gray")
        group = mdf["group"].iloc[0]
        ax.plot(mdf["time"], mdf["mean"], color=color, linewidth=1.5,
                alpha=0.85, label=f"{mouse} ({group[0].upper()})")
    _declines(ax)
    ax.set_title("C. Per-mouse traces")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_1_ramp_dynamics.png")
    plt.close()

    print("Saved: fig3_1_ramp_dynamics.png")


def figure_2_slope_vs_wait(df: pd.DataFrame, config: Config, output_dir: Path):
    """
    Figure 3.2: Ramp Slope Predicts Wait Time
    
    Panel A: Scatter plot of ramp slope vs time_waited (all trials, colored by mouse)
    Panel B: Mixed-effects regression line with confidence band
    Panel C: Per-mouse slopes (showing consistency)
    """
    # Filter to good trials
    good_df = filter_good_trials(df, min_wait=config.MIN_WAIT_FOR_RAMP)
    
    # Prepare data with proper alignment
    plot_data = []
    for _, row in good_df.iterrows():
        slope_col, wait_col = get_alignment_column(row["group"])
        if np.isfinite(row.get(slope_col, np.nan)) and np.isfinite(row.get(wait_col, np.nan)):
            plot_data.append({
                "slope": row[slope_col],
                "wait": row[wait_col],
                "mouse": row["mouse"],
                "group": row["group"],
                "session_id": row["session_id"],
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    if len(plot_df) < 20:
        print("Not enough data for Figure 3.2")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Panel A: Scatter plot colored by mouse
    ax = axes[0]
    for mouse in plot_df["mouse"].unique():
        mdf = plot_df[plot_df["mouse"] == mouse]
        ax.scatter(mdf["slope"], mdf["wait"], 
                   c=config.MOUSE_COLORS.get(mouse, "gray"),
                   alpha=0.4, s=15, label=mouse)
    
    # Overall regression line
    slope_reg, intercept, r_val, p_val, _ = stats.linregress(plot_df["slope"], plot_df["wait"])
    x_line = np.array([plot_df["slope"].min(), plot_df["slope"].max()])
    ax.plot(x_line, intercept + slope_reg * x_line, "k-", linewidth=2, label=f"r={r_val:.3f}")
    
    ax.set_xlabel("DA Ramp Slope (z/s)")
    ax.set_ylabel("Wait Time (s)")
    ax.set_title(f"A. All trials (n={len(plot_df)})")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    
    # Panel B: Mixed-effects result
    ax = axes[1]
    me_result = run_mixed_effects_regression(good_df)
    
    # Plot with confidence band
    ax.scatter(plot_df["slope"], plot_df["wait"], c="gray", alpha=0.2, s=10)
    
    if not np.isnan(me_result["beta"]):
        x_range = np.linspace(plot_df["slope"].min(), plot_df["slope"].max(), 100)
        y_pred = me_result.get("intercept", plot_df["wait"].mean()) + me_result["beta"] * x_range
        ax.plot(x_range, y_pred, "k-", linewidth=2)
        
        # Add text with statistics
        stats_text = f"β = {me_result['beta']:.3f}\np = {me_result['p']:.2e}\nn = {me_result['n']}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    ax.set_xlabel("DA Ramp Slope (z/s)")
    ax.set_ylabel("Wait Time (s)")
    ax.set_title("B. Mixed-effects regression")
    
    # Panel C: Per-mouse correlations
    ax = axes[2]
    mouse_stats = []
    for mouse in sorted(plot_df["mouse"].unique()):
        mdf = plot_df[plot_df["mouse"] == mouse]
        if len(mdf) >= 10:
            r, p = stats.pearsonr(mdf["slope"], mdf["wait"])
            mouse_stats.append({"mouse": mouse, "r": r, "p": p, "n": len(mdf)})
    
    mouse_stats_df = pd.DataFrame(mouse_stats)
    if len(mouse_stats_df) > 0:
        colors = [config.MOUSE_COLORS.get(m, "gray") for m in mouse_stats_df["mouse"]]
        bars = ax.bar(range(len(mouse_stats_df)), mouse_stats_df["r"], color=colors)
        ax.set_xticks(range(len(mouse_stats_df)))
        ax.set_xticklabels(mouse_stats_df["mouse"], rotation=45, ha="right")
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax.set_ylabel("Correlation (r)")
        ax.set_title("C. Per-mouse slope-wait correlation")
        
        # Add significance markers
        for i, row in mouse_stats_df.iterrows():
            marker = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
            if marker:
                ax.text(i, row["r"] + 0.02 * np.sign(row["r"]), marker, 
                        ha="center", va="bottom" if row["r"] > 0 else "top", fontsize=10)
    
    plt.tight_layout()
    fig.savefig(output_dir / "fig3_2_slope_vs_wait.png")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Figure 3.2: Slope vs Wait Time")
    print("="*60)
    print(f"Total trials: {len(plot_df)}")
    print(f"Overall correlation: r = {r_val:.3f}, p = {p_val:.2e}")
    if not np.isnan(me_result["beta"]):
        print(f"Mixed-effects: β = {me_result['beta']:.3f}, p = {me_result['p']:.2e}")
    print("\nPer-mouse correlations:")
    for _, row in mouse_stats_df.iterrows():
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
        print(f"  {row['mouse']}: r = {row['r']:.3f}, p = {row['p']:.3e}, n = {row['n']} {sig}")


def figure_3_trial_history(df: pd.DataFrame, config: Config, output_dir: Path):
    """
    Figure 3.3: Trial History Chain (Jakob & Gershman replication)
    
    Panel A: Prior reward → Current slope
    Panel B: Current slope → Current wait time
    Panel C: Mediation diagram with effect sizes
    """
    good_df = filter_good_trials(df, min_wait=config.MIN_WAIT_FOR_RAMP)
    chain_results, chain_df = compute_trial_history_chain(good_df)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Prior reward → Current slope
    ax = axes[0]
    if "link1" in chain_results:
        link1 = chain_results["link1"]

        # Bar plot
        means = [link1["unrewarded_mean"], link1["rewarded_mean"]]
        labels = ["Unrewarded", "Rewarded"]
        colors = ["#999999", config.COLORS["da_signal"]]

        bars = ax.bar(range(2), means, color=colors, width=0.6)
        ax.set_xticks(range(2))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Current Trial Slope (z/s)")
        ax.set_title("A. Previous Outcome -> Current Slope")

        # Add significance
        sig_marker = "***" if link1["p"] < 0.001 else "**" if link1["p"] < 0.01 else "*" if link1["p"] < 0.05 else "n.s."
        max_y = max(means) * 1.1
        ax.plot([0, 1], [max_y, max_y], "k-", linewidth=1)
        ax.text(0.5, max_y * 1.02, sig_marker, ha="center", fontsize=12)

        # Stats text
        stats_text = f"Δ = {link1['effect']:.3f}\nt = {link1['t']:.2f}\np = {link1['p']:.3e}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment="top", horizontalalignment="right", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")

    # Panel B: Current slope → Current wait
    ax = axes[1]
    if "link2" in chain_results:
        link2 = chain_results["link2"]

        plot_df = chain_df.dropna(subset=["slope", "wait"])

        # Scatter with regression line
        ax.scatter(plot_df["slope"], plot_df["wait"], c="gray", alpha=0.3, s=10)
        slope_reg, intercept, _, _, _ = stats.linregress(plot_df["slope"], plot_df["wait"])
        x_line = np.array([plot_df["slope"].min(), plot_df["slope"].max()])
        ax.plot(x_line, intercept + slope_reg * x_line, "k-", linewidth=2)
        
        ax.set_xlabel("Current Slope (z/s)")
        ax.set_ylabel("Current Wait Time (s)")
        ax.set_title("B. Current Slope -> Current Wait")
        
        # Stats text
        sig_marker = "***" if link2["p"] < 0.001 else "**" if link2["p"] < 0.01 else "*" if link2["p"] < 0.05 else ""
        stats_text = f"r = {link2['r']:.3f} {sig_marker}\np = {link2['p']:.2e}\nn = {link2['n']}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
    
    # Panel C: Mediation diagram
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("C. Three-Link Chain (Mediation)")
    
    # Draw boxes
    box_style = dict(boxstyle="round,pad=0.3", facecolor="lightblue", edgecolor="black")
    ax.text(1, 7, "Reward\nOutcome(n)", ha="center", va="center", fontsize=10, bbox=box_style)
    ax.text(5, 7, "DA Ramp\nSlope(n+1)", ha="center", va="center", fontsize=10, bbox=box_style)
    ax.text(9, 7, "Wait\nTime(n+1)", ha="center", va="center", fontsize=10, bbox=box_style)
    
    # Draw arrows with effect sizes
    if "link1" in chain_results:
        effect1 = chain_results["link1"]["effect"]
        p1 = chain_results["link1"]["p"]
        sig1 = "***" if p1 < 0.001 else "**" if p1 < 0.01 else "*" if p1 < 0.05 else ""
        ax.annotate("", xy=(3.5, 7), xytext=(2.2, 7),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2))
        ax.text(2.85, 7.5, f"Δ={effect1:.2f}{sig1}", ha="center", fontsize=8)
    
    if "link2" in chain_results:
        r2 = chain_results["link2"]["r"]
        p2 = chain_results["link2"]["p"]
        sig2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else ""
        ax.annotate("", xy=(7.5, 7), xytext=(6.5, 7),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2))
        ax.text(7, 7.5, f"r={r2:.2f}{sig2}", ha="center", fontsize=8)
    
    if "link3_total" in chain_results:
        effect3 = chain_results["link3_total"]["effect"]
        p3 = chain_results["link3_total"]["p"]
        sig3 = "***" if p3 < 0.001 else "**" if p3 < 0.01 else "*" if p3 < 0.05 else ""
        ax.annotate("", xy=(8, 5.5), xytext=(2, 5.5),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))
        ax.text(5, 5, f"Total: Δ={effect3:.2f}s {sig3}", ha="center", fontsize=8, color="gray")
    
    plt.tight_layout()
    fig.savefig(output_dir / "fig3_3_trial_history.png")
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("Figure 3.3: Trial History Chain")
    print("="*60)
    if "link1" in chain_results:
        l1 = chain_results["link1"]
        print(f"Link 1 (Outcome → Slope): effect = {l1['effect']:.3f}, t = {l1['t']:.2f}, p = {l1['p']:.3e}")
    if "link2" in chain_results:
        l2 = chain_results["link2"]
        print(f"Link 2 (Slope → Wait): r = {l2['r']:.3f}, p = {l2['p']:.3e}")
    if "link3_total" in chain_results:
        l3 = chain_results["link3_total"]
        print(f"Link 3 (Total: Outcome → Wait): effect = {l3['effect']:.3f}s, t = {l3['t']:.2f}, p = {l3['p']:.3e}")


def figure_4_per_mouse(df: pd.DataFrame, config: Config, output_dir: Path):
    """
    Figure 3.4: Per-Mouse Consistency
    
    One panel per mouse showing slope vs wait scatter with regression line.
    Critical for n=6 to show each animal replicates the effect.
    """
    good_df = filter_good_trials(df, min_wait=config.MIN_WAIT_FOR_RAMP)
    mice = sorted(good_df["mouse"].unique())
    
    n_mice = len(mice)
    n_cols = min(3, n_mice)
    n_rows = int(np.ceil(n_mice / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    if n_mice == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, mouse in enumerate(mice):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        mdf = good_df[good_df["mouse"] == mouse].copy()
        group = mdf["group"].iloc[0] if len(mdf) > 0 else "long"
        slope_col, wait_col = get_alignment_column(group)
        
        # Filter to valid data
        valid = mdf[np.isfinite(mdf[slope_col]) & np.isfinite(mdf[wait_col])].copy()
        
        if len(valid) < 10:
            ax.text(0.5, 0.5, f"{mouse}\nInsufficient data\nn={len(valid)}", 
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{mouse} ({group.upper()} BG)")
            continue
        
        # Scatter plot
        color = config.MOUSE_COLORS.get(mouse, "gray")
        ax.scatter(valid[slope_col], valid[wait_col], c=color, alpha=0.5, s=20)
        
        # Regression line
        slope_reg, intercept, r_val, p_val, _ = stats.linregress(valid[slope_col], valid[wait_col])
        x_line = np.array([valid[slope_col].min(), valid[slope_col].max()])
        ax.plot(x_line, intercept + slope_reg * x_line, "k-", linewidth=2)
        
        # Labels and title
        ax.set_xlabel("DA Ramp Slope (z/s)")
        ax.set_ylabel("Wait Time (s)")
        
        sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        n_sessions = mdf["session_id"].nunique()
        ax.set_title(f"{mouse} ({group.upper()} BG)\nr={r_val:.3f}{sig_marker}, n={len(valid)} trials, {n_sessions} sessions")
    
    # Hide empty subplots
    for i in range(n_mice, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    fig.savefig(output_dir / "fig3_4_per_mouse.png")
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("Figure 3.4: Per-Mouse Consistency")
    print("="*60)
    
    consistent_count = 0
    for mouse in mice:
        mdf = good_df[good_df["mouse"] == mouse].copy()
        group = mdf["group"].iloc[0] if len(mdf) > 0 else "long"
        slope_col, wait_col = get_alignment_column(group)
        valid = mdf[np.isfinite(mdf[slope_col]) & np.isfinite(mdf[wait_col])]
        
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid[slope_col], valid[wait_col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            direction = "✓" if r > 0 else "✗"
            consistent_count += 1 if r > 0 and p < 0.05 else 0
            print(f"  {mouse} ({group}): r = {r:.3f} {sig} {direction}, n = {len(valid)} trials")
        else:
            print(f"  {mouse}: insufficient data (n={len(valid)})")
    
    print(f"\nConsistent direction: {consistent_count}/{len(mice)} mice")


def generate_summary_table(df: pd.DataFrame, config: Config, output_dir: Path):
    """Generate a summary statistics table for the committee."""
    good_df = filter_good_trials(df, min_wait=config.MIN_WAIT_FOR_RAMP)
    
    rows = []
    for mouse in sorted(df["mouse"].unique()):
        mdf = good_df[good_df["mouse"] == mouse]
        group = mdf["group"].iloc[0] if len(mdf) > 0 else "unknown"
        slope_col, wait_col = get_alignment_column(group)
        
        valid = mdf[np.isfinite(mdf[slope_col]) & np.isfinite(mdf[wait_col])]
        
        row = {
            "mouse": mouse,
            "group": group,
            "n_sessions": mdf["session_id"].nunique(),
            "n_trials": len(mdf),
            "n_valid": len(valid),
            "mean_wait": valid[wait_col].mean() if len(valid) > 0 else np.nan,
            "mean_slope": valid[slope_col].mean() if len(valid) > 0 else np.nan,
        }
        
        if len(valid) >= 10:
            r, p = stats.pearsonr(valid[slope_col], valid[wait_col])
            row["slope_wait_r"] = r
            row["slope_wait_p"] = p
        else:
            row["slope_wait_r"] = np.nan
            row["slope_wait_p"] = np.nan
        
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)
    
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    return summary_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all committee figures."""
    config = Config()
    
    # Setup
    setup_figure_style()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Committee Figure Generation")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    try:
        df = load_all_session_metrics(config)
    except ValueError as e:
        print(f"ERROR: {e}")
        print("\nMake sure you have run 4_photometry_wait_analysis.py in BATCH mode first.")
        return
    
    # Generate figures
    print("\nGenerating figures...")
    
    print("\n--- Figure 3.1: Ramp Dynamics ---")
    figure_1_ramp_dynamics(df, config, config.OUTPUT_DIR)
    
    print("\n--- Figure 3.2: Slope vs Wait ---")
    figure_2_slope_vs_wait(df, config, config.OUTPUT_DIR)
    
    print("\n--- Figure 3.3: Trial History ---")
    figure_3_trial_history(df, config, config.OUTPUT_DIR)
    
    print("\n--- Figure 3.4: Per-Mouse ---")
    figure_4_per_mouse(df, config, config.OUTPUT_DIR)
    
    # Summary table
    print("\n--- Summary Table ---")
    generate_summary_table(df, config, config.OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("DONE!")
    print(f"Figures saved to: {config.OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()

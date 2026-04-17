#!/usr/bin/env python3
"""
DA Trial History Analysis (v2)
==============================

Analyzes how previous trial outcome affects current trial DA dynamics.

Key measures (properly separated):
1. TRUE BASELINE: First 300ms after anchor — tests tonic carry-over from previous trial
2. PRE-DECISION PEAK: Last 500ms before decision — where the ramp ends up
3. RAMP SLOPE: Linear fit from anchor to decision — rate of DA change

Key questions:
1. Does true baseline differ after rewarded vs. unrewarded trials? (Hamid & Berke prediction)
2. Does ramp slope differ after rewarded vs. unrewarded trials?
3. Does pre-decision peak differ? (Could reflect baseline + slope effects)
4. Does slope predict wait time on the same trial?

Input: Merged photometry + trial data CSV with columns:
    - reward, previous_trial_reward (0.0 or 5.0)
    - time_waited_since_cue_on, previous_trial_time_waited_since_cue_on
    - time_waited_since_last_lick
    - dff_zscored (or other signal column)
    - trial_time, decision_time
    - session_id, session_trial_num, mouse, group

Usage:
    # Run over all sessions (auto-discovers photometry_with_trial_data.csv under PROCESSED_OUT):
    python 9_da_trial_history_v2.py

    # Explicit sessions directory:
    python 9_da_trial_history_v2.py --sessions-dir /path/to/fp_processed

    # Single pre-merged CSV (original behaviour):
    python 9_da_trial_history_v2.py --input photometry_with_trial_data.csv
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import GROUP_COLORS, PIPELINE_SESSION_LOG, PROCESSED_OUT, TRIAL_HISTORY_V2_DIR

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Quality filter (True = GRAB tier A/B only + no short sessions; False = all sessions)
    QUALITY_FILTER = True

    # Signal column
    SIGNAL_COL = "dff_zscored"
    
    # Which channel to analyze
    DA_CHANNEL = "l_str_GRAB"
    
    # Timing parameters
    PRE_ANCHOR_SEC = 0.5      # Time before anchor to include
    POST_DECISION_SEC = 1.5   # Time after decision to include
    MIN_WINDOW_SEC = 0.5      # Minimum anchor→decision window
    
    # Measure windows (all relative to decision at t=0)
    # True baseline: measured at the anchor point (start of wait)
    BASELINE_WINDOW_SEC = 0.3  # First 300ms after anchor
    
    # Pre-decision: measured just before decision
    PREDEC_START = -0.5        # 500ms before decision
    PREDEC_END = -0.05         # Stop 50ms before (avoid decision transient)
    
    # Slope window: exclude edges
    SLOPE_START_BUFFER = 0.3   # Skip first 300ms after anchor
    SLOPE_END_BUFFER = 0.2     # Stop 200ms before decision
    
    # Plotting
    PLOT_TIME_RANGE = (-8, 1.5)
    
    # Minimum trials
    MIN_TRIALS = 20
    
    # Output
    OUTPUT_DIR = TRIAL_HISTORY_V2_DIR
    
    # Colors
    COLORS = {
        "prev_rew": "#2ca02c",
        "prev_unrew": "#d62728",
        **GROUP_COLORS,
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def _process_dataframe(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Filter and annotate a merged photometry DataFrame."""
    # Filter to DA channel
    available_channels = df['channel'].unique()
    if config.DA_CHANNEL in available_channels:
        df = df[df['channel'] == config.DA_CHANNEL].copy()
        print(f"  Using channel: {config.DA_CHANNEL}")
    else:
        grab_channels = [c for c in available_channels if 'GRAB' in str(c)]
        if grab_channels:
            config.DA_CHANNEL = grab_channels[0]
            df = df[df['channel'] == config.DA_CHANNEL].copy()
            print(f"  Using channel: {config.DA_CHANNEL}")
        else:
            print(f"  Available channels: {available_channels}")
            raise ValueError("No GRAB channel found")

    # Create binary reward columns
    df['rewarded'] = (df['reward'] > 0).astype(int)
    df['prev_rewarded'] = (df['previous_trial_reward'] > 0).astype(int)

    # Time relative to decision (decision at t=0)
    df['time_from_decision'] = df['trial_time'] - df['time_waited_since_cue_on']

    # Window duration (anchor to decision)
    df['window_duration'] = df['time_waited_since_last_lick']

    print(f"  Total rows: {len(df):,}")
    print(f"  Sessions: {df['session_id'].nunique()}")
    print(f"  Mice: {sorted(df['mouse'].unique())}")
    print(f"  Groups: {sorted(df['group'].unique())}")

    return df


def load_data(filepath: Path, config: Config) -> pd.DataFrame:
    """Load merged photometry data from a single CSV file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return _process_dataframe(df, config)


def _build_allowed_sessions(base: Path) -> Optional[set]:
    """Return session IDs where the GRAB channel is tier A/B and session is not short."""
    short_sessions: set = set()
    if PIPELINE_SESSION_LOG.exists():
        log = pd.read_csv(PIPELINE_SESSION_LOG, index_col=0)
        if "short_session" in log.columns:
            short_sessions = set(log.index[log["short_session"].fillna(False).astype(bool)])

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
    print(f"  Session filter: {len(allowed)} of {n_total} sessions pass (GRAB tier A/B, not short)")
    return allowed


def load_all_sessions_data(config: Config, sessions_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Discover and concatenate photometry_with_trial_data.csv from every session
    directory under sessions_dir (defaults to PROCESSED_OUT from config.py).
    If config.QUALITY_FILTER is True, restricts to sessions where the GRAB
    channel is tier A or B and the session is not short.
    """
    base = sessions_dir or PROCESSED_OUT
    print(f"Scanning for sessions in: {base}")

    if config.QUALITY_FILTER:
        allowed_sessions = _build_allowed_sessions(base)
    else:
        allowed_sessions = None
        print("  Session filter: disabled (using all sessions)")

    frames = []
    for session_dir in sorted(base.iterdir()):
        if not session_dir.is_dir():
            continue
        if allowed_sessions is not None and session_dir.name not in allowed_sessions:
            continue
        merged_csv = session_dir / "photometry_with_trial_data.csv"
        if not merged_csv.exists():
            continue
        try:
            df = pd.read_csv(merged_csv)
            df["session_id"] = session_dir.name        # ensure session_id is set
            frames.append(df)
            print(f"  Loaded {session_dir.name}  ({len(df):,} rows)")
        except Exception as e:
            print(f"  WARNING: could not read {merged_csv}: {e}")

    if not frames:
        raise FileNotFoundError(f"No photometry_with_trial_data.csv files found under {base}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows after concatenation: {len(combined):,}")
    print(f"  Sessions loaded: {combined['session_id'].nunique()}")
    return combined


# =============================================================================
# TRIAL EXTRACTION
# =============================================================================

def extract_trials(df: pd.DataFrame, config: Config) -> List[Dict]:
    """Extract per-trial measures."""
    trials = []
    
    grouped = df.groupby(['session_id', 'session_trial_num'])
    
    for (session_id, trial_num), trial_df in grouped:
        trial_df = trial_df.sort_values('trial_time')
        row = trial_df.iloc[0]
        
        # Skip first trial (no previous trial data)
        if pd.isna(row['previous_trial_reward']):
            continue
        
        # Get window duration
        window_duration = row['window_duration']
        if pd.isna(window_duration) or window_duration < config.MIN_WINDOW_SEC:
            continue
        
        # Extract time and signal
        time_rel = trial_df['time_from_decision'].values
        signal = trial_df[config.SIGNAL_COL].values
        
        # Anchor time (relative to decision)
        anchor_time = -window_duration
        
        # =================================================================
        # MEASURE 1: TRUE BASELINE (at anchor, before any ramping)
        # =================================================================
        baseline_start = anchor_time
        baseline_end = anchor_time + config.BASELINE_WINDOW_SEC
        baseline_mask = (time_rel >= baseline_start) & (time_rel <= baseline_end)
        
        if baseline_mask.sum() < 3:
            continue
        
        true_baseline = np.nanmean(signal[baseline_mask])
        
        # =================================================================
        # MEASURE 2: PRE-DECISION LEVEL (where the ramp ends up)
        # =================================================================
        predec_mask = (time_rel >= config.PREDEC_START) & (time_rel <= config.PREDEC_END)
        predec_level = np.nanmean(signal[predec_mask]) if predec_mask.sum() >= 3 else np.nan
        
        # =================================================================
        # MEASURE 3: RAMP SLOPE (rate of change during wait)
        # =================================================================
        slope_start = anchor_time + config.SLOPE_START_BUFFER
        slope_end = -config.SLOPE_END_BUFFER
        
        slope, r2 = np.nan, np.nan
        if slope_end > slope_start + 0.3:
            slope_mask = (time_rel >= slope_start) & (time_rel <= slope_end)
            t_slope = time_rel[slope_mask]
            y_slope = signal[slope_mask]
            
            if len(t_slope) >= 5:
                # Linear regression
                t_mean = np.mean(t_slope)
                y_mean = np.mean(y_slope)
                tt = t_slope - t_mean
                yy = y_slope - y_mean
                denom = np.sum(tt ** 2)
                
                if denom > 0:
                    slope = np.sum(tt * yy) / denom
                    yhat = slope * (t_slope - t_mean) + y_mean
                    ss_res = np.sum((y_slope - yhat) ** 2)
                    ss_tot = np.sum(yy ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        
        # =================================================================
        # MEASURE 4: BASELINE-SUBTRACTED PRE-DECISION (ramp amplitude)
        # =================================================================
        ramp_amplitude = predec_level - true_baseline if np.isfinite(predec_level) else np.nan
        
        # =================================================================
        # Store trace for plotting (baseline-subtracted)
        # =================================================================
        plot_mask = (time_rel >= config.PLOT_TIME_RANGE[0]) & (time_rel <= config.PLOT_TIME_RANGE[1])
        
        trials.append({
            # Identifiers
            'session_id': session_id,
            'trial_num': trial_num,
            'mouse': row['mouse'],
            'group': row['group'],
            
            # Timing
            'window_duration': window_duration,
            'wait_time': row['time_waited_since_cue_on'],
            'prev_wait_time': row['previous_trial_time_waited_since_cue_on'],
            
            # Outcome
            'rewarded': bool(row['rewarded']),
            'prev_rewarded': bool(row['prev_rewarded']),
            
            # DA measures (RAW - not baseline subtracted)
            'true_baseline': true_baseline,
            'predec_level': predec_level,
            'slope': slope,
            'slope_r2': r2,
            'ramp_amplitude': ramp_amplitude,
            
            # For plotting (baseline-subtracted trace)
            'time_rel': time_rel[plot_mask],
            'signal_raw': signal[plot_mask],
            'signal_baselined': signal[plot_mask] - true_baseline,
        })
    
    print(f"  Extracted {len(trials)} valid trials")
    return trials


# =============================================================================
# STATISTICS
# =============================================================================

def compare_by_prev_outcome(
    trials: List[Dict],
    measure: str,
    config: Config,
) -> Optional[Dict]:
    """Compare a measure between prev-rewarded and prev-unrewarded trials."""
    prev_rew = [t[measure] for t in trials if t['prev_rewarded'] and np.isfinite(t[measure])]
    prev_unrew = [t[measure] for t in trials if not t['prev_rewarded'] and np.isfinite(t[measure])]
    
    if len(prev_rew) < config.MIN_TRIALS or len(prev_unrew) < config.MIN_TRIALS:
        return None
    
    t_stat, p_val = stats.ttest_ind(prev_rew, prev_unrew)
    
    # Cohen's d
    pooled_std = np.sqrt((np.var(prev_rew) + np.var(prev_unrew)) / 2)
    cohens_d = (np.mean(prev_rew) - np.mean(prev_unrew)) / pooled_std if pooled_std > 0 else 0
    
    return {
        'measure': measure,
        'n_prev_rew': len(prev_rew),
        'n_prev_unrew': len(prev_unrew),
        'mean_prev_rew': np.mean(prev_rew),
        'mean_prev_unrew': np.mean(prev_unrew),
        'std_prev_rew': np.std(prev_rew),
        'std_prev_unrew': np.std(prev_unrew),
        't_stat': t_stat,
        'p_val': p_val,
        'cohens_d': cohens_d,
    }


def correlation_analysis(trials: List[Dict]) -> pd.DataFrame:
    """Compute key correlations."""
    # Build dataframe
    df = pd.DataFrame([{
        'prev_rewarded': int(t['prev_rewarded']),
        'true_baseline': t['true_baseline'],
        'predec_level': t['predec_level'],
        'slope': t['slope'],
        'ramp_amplitude': t['ramp_amplitude'],
        'wait_time': t['wait_time'],
        'prev_wait_time': t['prev_wait_time'],
        'window_duration': t['window_duration'],
        'mouse': t['mouse'],
        'group': t['group'],
    } for t in trials])
    
    # Key correlations to test
    pairs = [
        ('prev_rewarded', 'true_baseline', 'Prev reward → True baseline (Hamid & Berke test)'),
        ('prev_rewarded', 'slope', 'Prev reward → Slope'),
        ('prev_rewarded', 'ramp_amplitude', 'Prev reward → Ramp amplitude'),
        ('prev_rewarded', 'predec_level', 'Prev reward → Pre-decision level'),
        ('true_baseline', 'slope', 'True baseline → Slope'),
        ('slope', 'wait_time', 'Slope → Wait time (clock speed test)'),
        ('true_baseline', 'wait_time', 'True baseline → Wait time'),
        ('prev_rewarded', 'wait_time', 'Prev reward → Wait time'),
        ('prev_wait_time', 'slope', 'Prev wait time → Slope'),
    ]
    
    results = []
    for x_col, y_col, description in pairs:
        valid = df[[x_col, y_col]].dropna()
        if len(valid) < 20:
            continue
        
        r, p = stats.pearsonr(valid[x_col], valid[y_col])
        results.append({
            'x': x_col,
            'y': y_col,
            'description': description,
            'r': r,
            'p': p,
            'n': len(valid),
        })
    
    return pd.DataFrame(results)


# =============================================================================
# PLOTTING
# =============================================================================

def setup_style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_traces(
    trials: List[Dict],
    config: Config,
    output_path: Path,
    title: str,
    baseline_subtract: bool = True,
):
    """Plot average traces split by previous outcome."""
    prev_rew = [t for t in trials if t['prev_rewarded']]
    prev_unrew = [t for t in trials if not t['prev_rewarded']]
    
    if len(prev_rew) < config.MIN_TRIALS or len(prev_unrew) < config.MIN_TRIALS:
        print(f"  Skipping {title}: insufficient trials")
        return
    
    # Time grid
    time_grid = np.arange(config.PLOT_TIME_RANGE[0], config.PLOT_TIME_RANGE[1], 0.05)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    signal_key = 'signal_baselined' if baseline_subtract else 'signal_raw'
    
    for subset, label, color in [
        (prev_rew, f"Prev rewarded (n={len(prev_rew)})", config.COLORS["prev_rew"]),
        (prev_unrew, f"Prev unrewarded (n={len(prev_unrew)})", config.COLORS["prev_unrew"]),
    ]:
        # Interpolate
        signals = np.full((len(subset), len(time_grid)), np.nan)
        for i, t in enumerate(subset):
            signals[i, :] = np.interp(time_grid, t['time_rel'], t[signal_key], left=np.nan, right=np.nan)
        
        mean = np.nanmean(signals, axis=0)
        sem = np.nanstd(signals, axis=0) / np.sqrt(np.sum(~np.isnan(signals), axis=0))
        
        ax.plot(time_grid, mean, color=color, linewidth=2, label=label)
        ax.fill_between(time_grid, mean - sem, mean + sem, color=color, alpha=0.2)
    
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, label='Decision')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ylabel = "DA (z, baseline-subtracted)" if baseline_subtract else "DA (z, raw)"
    ax.set_xlabel("Time from decision (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.set_xlim(config.PLOT_TIME_RANGE)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()


def plot_measure_comparison(
    trials: List[Dict],
    measures: List[str],
    measure_labels: List[str],
    config: Config,
    output_path: Path,
    suptitle: str,
):
    """Plot violin comparisons for multiple measures."""
    n_measures = len(measures)
    fig, axes = plt.subplots(1, n_measures, figsize=(4 * n_measures, 5))
    if n_measures == 1:
        axes = [axes]
    
    for ax, measure, label in zip(axes, measures, measure_labels):
        prev_rew = [t[measure] for t in trials if t['prev_rewarded'] and np.isfinite(t[measure])]
        prev_unrew = [t[measure] for t in trials if not t['prev_rewarded'] and np.isfinite(t[measure])]
        
        if len(prev_rew) < 10 or len(prev_unrew) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue
        
        parts = ax.violinplot([prev_rew, prev_unrew], positions=[0, 1], showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor([config.COLORS["prev_rew"], config.COLORS["prev_unrew"]][i])
            pc.set_alpha(0.6)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Prev\nrewarded', 'Prev\nunrewarded'])
        ax.set_title(label)
        
        # Stats
        t_stat, p_val = stats.ttest_ind(prev_rew, prev_unrew)
        pooled_std = np.sqrt((np.var(prev_rew) + np.var(prev_unrew)) / 2)
        d = (np.mean(prev_rew) - np.mean(prev_unrew)) / pooled_std if pooled_std > 0 else 0
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.text(0.5, 0.95, f"d={d:+.2f}, p={p_val:.3f} {sig}",
                transform=ax.transAxes, ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()


def plot_slope_behavior(
    trials: List[Dict],
    config: Config,
    output_path: Path,
    title: str,
):
    """Plot slope vs wait time relationship."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Slope vs current wait time (the key prediction)
    ax = axes[0]
    for subset, color, label in [
        ([t for t in trials if t['prev_rewarded']], config.COLORS["prev_rew"], "Prev rew"),
        ([t for t in trials if not t['prev_rewarded']], config.COLORS["prev_unrew"], "Prev unrew"),
    ]:
        x = [t['slope'] for t in subset if np.isfinite(t['slope']) and np.isfinite(t['wait_time'])]
        y = [t['wait_time'] for t in subset if np.isfinite(t['slope']) and np.isfinite(t['wait_time'])]
        
        if len(x) > 10:
            ax.scatter(x, y, color=color, alpha=0.3, s=15, label=label)
            # Regression line
            m, b = np.polyfit(x, y, 1)
            x_line = np.array([min(x), max(x)])
            ax.plot(x_line, m * x_line + b, color=color, linewidth=2, linestyle='--')
            
            r, p = stats.pearsonr(x, y)
            ax.text(0.95, 0.95 if color == config.COLORS["prev_rew"] else 0.85,
                    f"{label}: r={r:.2f}, p={p:.3f}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=9, color=color)
    
    ax.set_xlabel("DA slope (z/s)")
    ax.set_ylabel("Wait time (s)")
    ax.set_title("Slope predicts wait time")
    ax.legend(loc='upper left')
    
    # Panel 2: True baseline vs wait time
    ax = axes[1]
    for subset, color, label in [
        ([t for t in trials if t['prev_rewarded']], config.COLORS["prev_rew"], "Prev rew"),
        ([t for t in trials if not t['prev_rewarded']], config.COLORS["prev_unrew"], "Prev unrew"),
    ]:
        x = [t['true_baseline'] for t in subset if np.isfinite(t['true_baseline']) and np.isfinite(t['wait_time'])]
        y = [t['wait_time'] for t in subset if np.isfinite(t['true_baseline']) and np.isfinite(t['wait_time'])]
        
        if len(x) > 10:
            ax.scatter(x, y, color=color, alpha=0.3, s=15, label=label)
            m, b = np.polyfit(x, y, 1)
            x_line = np.array([min(x), max(x)])
            ax.plot(x_line, m * x_line + b, color=color, linewidth=2, linestyle='--')
            
            r, p = stats.pearsonr(x, y)
            ax.text(0.95, 0.95 if color == config.COLORS["prev_rew"] else 0.85,
                    f"{label}: r={r:.2f}, p={p:.3f}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=9, color=color)
    
    ax.set_xlabel("True baseline DA (z)")
    ax.set_ylabel("Wait time (s)")
    ax.set_title("Baseline vs wait time")
    ax.legend(loc='upper left')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def run_analysis(
    config: Config,
    input_path: Optional[Path] = None,
    sessions_dir: Optional[Path] = None,
):
    """Main analysis."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("=" * 70)
    print("DA Trial History Analysis (v2)")
    print("=" * 70)
    print("\nMeasures:")
    print("  - TRUE BASELINE: First 300ms after anchor (tests Hamid & Berke)")
    print("  - PRE-DECISION LEVEL: Last 500ms before decision")
    print("  - RAMP SLOPE: Linear fit, anchor → decision")
    print("  - RAMP AMPLITUDE: Pre-decision minus baseline")
    print()

    # Load data — either a single pre-merged CSV or auto-discover all sessions
    if input_path is not None:
        df = load_data(input_path, config)
    else:
        raw = load_all_sessions_data(config, sessions_dir)
        df = _process_dataframe(raw, config)
    
    # Extract trials
    print("\nExtracting trials...")
    trials = extract_trials(df, config)
    
    n_prev_rew = sum(1 for t in trials if t['prev_rewarded'])
    n_prev_unrew = sum(1 for t in trials if not t['prev_rewarded'])
    print(f"  After prev reward: {n_prev_rew}")
    print(f"  After prev unreward: {n_prev_unrew}")
    
    if len(trials) < config.MIN_TRIALS * 2:
        print("ERROR: Insufficient trials")
        return
    
    groups = sorted(set(t['group'] for t in trials))
    mice = sorted(set(t['mouse'] for t in trials))
    
    # =========================================================================
    # 1. MAIN COMPARISON: All measures
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. Effect of Previous Outcome on DA Measures")
    print("=" * 70)
    
    measures = ['true_baseline', 'slope', 'ramp_amplitude', 'predec_level']
    measure_labels = ['True baseline\n(at anchor)', 'Slope\n(z/s)', 
                      'Ramp amplitude\n(predec - baseline)', 'Pre-decision\nlevel']
    
    results = []
    
    print("\n  ALL TRIALS:")
    for measure, label in zip(measures, measure_labels):
        stats_result = compare_by_prev_outcome(trials, measure, config)
        if stats_result:
            stats_result['level'] = 'all'
            results.append(stats_result)
            sig = "***" if stats_result['p_val'] < 0.001 else "**" if stats_result['p_val'] < 0.01 else "*" if stats_result['p_val'] < 0.05 else "ns"
            print(f"    {measure:20s}: d={stats_result['cohens_d']:+.3f}, p={stats_result['p_val']:.4f} {sig}")
    
    # Plot
    plot_measure_comparison(
        trials, measures, measure_labels, config,
        config.OUTPUT_DIR / "all_measures_comparison.png",
        "All Trials: DA Measures by Previous Outcome"
    )
    
    # By group
    for group in groups:
        group_trials = [t for t in trials if t['group'] == group]
        if len(group_trials) >= config.MIN_TRIALS * 2:
            print(f"\n  {group.upper()} GROUP:")
            for measure, label in zip(measures, measure_labels):
                stats_result = compare_by_prev_outcome(group_trials, measure, config)
                if stats_result:
                    stats_result['level'] = group
                    results.append(stats_result)
                    sig = "***" if stats_result['p_val'] < 0.001 else "**" if stats_result['p_val'] < 0.01 else "*" if stats_result['p_val'] < 0.05 else "ns"
                    print(f"    {measure:20s}: d={stats_result['cohens_d']:+.3f}, p={stats_result['p_val']:.4f} {sig}")
            
            plot_measure_comparison(
                group_trials, measures, measure_labels, config,
                config.OUTPUT_DIR / f"{group}_measures_comparison.png",
                f"{group.upper()} BG: DA Measures by Previous Outcome"
            )
    
    # By mouse
    for mouse in mice:
        mouse_trials = [t for t in trials if t['mouse'] == mouse]
        if len(mouse_trials) >= config.MIN_TRIALS * 2:
            print(f"\n  {mouse}:")
            for measure, label in zip(measures, measure_labels):
                stats_result = compare_by_prev_outcome(mouse_trials, measure, config)
                if stats_result:
                    stats_result['level'] = mouse
                    results.append(stats_result)
                    sig = "***" if stats_result['p_val'] < 0.001 else "**" if stats_result['p_val'] < 0.01 else "*" if stats_result['p_val'] < 0.05 else "ns"
                    print(f"    {measure:20s}: d={stats_result['cohens_d']:+.3f}, p={stats_result['p_val']:.4f} {sig}")
            
            plot_measure_comparison(
                mouse_trials, measures, measure_labels, config,
                config.OUTPUT_DIR / f"{mouse}_measures_comparison.png",
                f"{mouse}: DA Measures by Previous Outcome"
            )
    
    # =========================================================================
    # 2. TRACES
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. Trace Plots")
    print("=" * 70)
    
    trace_dir = config.OUTPUT_DIR / "traces"
    trace_dir.mkdir(exist_ok=True)
    
    # Baseline-subtracted (shows ramp dynamics)
    plot_traces(trials, config, trace_dir / "all_traces_baselined.png",
                "All Trials (baseline-subtracted)", baseline_subtract=True)
    
    # Raw (shows absolute levels)
    plot_traces(trials, config, trace_dir / "all_traces_raw.png",
                "All Trials (raw signal)", baseline_subtract=False)
    
    for group in groups:
        group_trials = [t for t in trials if t['group'] == group]
        if len(group_trials) >= config.MIN_TRIALS * 2:
            plot_traces(group_trials, config, trace_dir / f"{group}_traces_baselined.png",
                        f"{group.upper()} BG (baseline-subtracted)", baseline_subtract=True)
            plot_traces(group_trials, config, trace_dir / f"{group}_traces_raw.png",
                        f"{group.upper()} BG (raw signal)", baseline_subtract=False)
    
    for mouse in mice:
        mouse_trials = [t for t in trials if t['mouse'] == mouse]
        if len(mouse_trials) >= config.MIN_TRIALS * 2:
            plot_traces(mouse_trials, config, trace_dir / f"{mouse}_traces_baselined.png",
                        f"{mouse} (baseline-subtracted)", baseline_subtract=True)
    
    print("  Saved trace plots")
    
    # =========================================================================
    # 3. SLOPE → BEHAVIOR
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. DA → Behavior Relationships")
    print("=" * 70)
    
    behavior_dir = config.OUTPUT_DIR / "behavior"
    behavior_dir.mkdir(exist_ok=True)
    
    plot_slope_behavior(trials, config, behavior_dir / "all_slope_behavior.png", "All Trials")
    
    for group in groups:
        group_trials = [t for t in trials if t['group'] == group]
        if len(group_trials) >= config.MIN_TRIALS * 2:
            plot_slope_behavior(group_trials, config, 
                                behavior_dir / f"{group}_slope_behavior.png", f"{group.upper()} BG")
    
    print("  Saved behavior plots")
    
    # =========================================================================
    # 4. CORRELATION MATRIX
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. Correlation Analysis")
    print("=" * 70)
    
    corr_df = correlation_analysis(trials)
    corr_df.to_csv(config.OUTPUT_DIR / "correlations.csv", index=False)
    
    print("\n  Key correlations:")
    for _, row in corr_df.iterrows():
        sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
        print(f"    {row['description']:45s}: r={row['r']:+.3f}, p={row['p']:.4f} {sig}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(config.OUTPUT_DIR / "measure_comparisons.csv", index=False)
    
    # Save trial-level data for further analysis
    trial_df = pd.DataFrame([{
        'session_id': t['session_id'],
        'trial_num': t['trial_num'],
        'mouse': t['mouse'],
        'group': t['group'],
        'prev_rewarded': t['prev_rewarded'],
        'rewarded': t['rewarded'],
        'true_baseline': t['true_baseline'],
        'predec_level': t['predec_level'],
        'slope': t['slope'],
        'ramp_amplitude': t['ramp_amplitude'],
        'wait_time': t['wait_time'],
        'prev_wait_time': t['prev_wait_time'],
        'window_duration': t['window_duration'],
    } for t in trials])
    trial_df.to_csv(config.OUTPUT_DIR / "trial_measures.csv", index=False)
    
    print(f"\n{'=' * 70}")
    print(f"DONE. All outputs saved to: {config.OUTPUT_DIR}")
    print("=" * 70)


# =============================================================================
# ENTRY POINT — edit settings here, then run
# =============================================================================

if __name__ == "__main__":
    # ── Settings ──────────────────────────────────────────────────────────────
    # Set INPUT_CSV to a Path to use a single pre-merged file, or leave as None
    # to auto-discover all sessions under SESSIONS_DIR.
    INPUT_CSV: Optional[Path] = None

    # Override the sessions root (None = use PROCESSED_OUT from config.py)
    SESSIONS_DIR: Optional[Path] = None

    # Override output directory (None = use Config.OUTPUT_DIR default)
    OUTPUT_DIR: Optional[Path] = None
    # ──────────────────────────────────────────────────────────────────────────

    config = Config()
    if OUTPUT_DIR is not None:
        config.OUTPUT_DIR = OUTPUT_DIR

    run_analysis(config, input_path=INPUT_CSV, sessions_dir=SESSIONS_DIR)

#!/usr/bin/env python3
"""
DA Trial History Analysis (v3) — Multi-Anchor
==============================================

Analyzes how previous trial outcome affects current trial DA dynamics,
testing all three behavioral anchors separately.

TRIAL STRUCTURE:
    cue_on ──── BACKGROUND ────► cue_off ──── WAIT ────► decision ──► CONSUMPTION
                (visual on)                   (silence)     (first lick)
                
    Licks can occur during background. The decision IS the first lick during
    the wait period — there are no licks between cue_off and decision.
    
    The "last_lick" is the last lick before cue_off. If the mouse didn't lick
    during the background period, last_lick could be from the PREVIOUS trial
    (during consumption). So time_waited_since_last_lick can span trials!

THREE ANCHORS (each tested separately):
    1. cue_on → decision:     External reference, includes background period
    2. cue_off → decision:    Wait period only (what we call "time_waited")
    3. last_lick → decision:  From last self-generated action (may span trials)

COLUMNS USED:
    - time_waited:                 cue_off → decision (wait period)
    - time_waited_since_cue_on:    cue_on → decision (includes background)
    - time_waited_since_last_lick: last_lick → decision (can span trials)

For each anchor, we compute:
    - TRUE BASELINE: First 300ms after anchor
    - SLOPE: Linear fit from anchor+300ms to decision-200ms
    - RAMP AMPLITUDE: Pre-decision minus baseline
    - WAIT TIME: Time from anchor to decision (matched to anchor!)

FILTERS:
    - Exclude miss trials on CURRENT trial (no decision)
    - Exclude trials where PREVIOUS trial was a miss (no reward outcome)
    - For each anchor: exclude if that window < 1.0s (need time for measures)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import GROUP_COLORS, PIPELINE_SESSION_LOG, PROCESSED_OUT, TRIAL_HISTORY_V3_DIR

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Quality filter (True = GRAB tier A/B only + no short sessions; False = all sessions)
    QUALITY_FILTER = False

    # Signal column
    SIGNAL_COL = "dff_zscored"

    # Which channel to analyze
    DA_CHANNEL = "l_str_GRAB"

    # Anchors to test
    ANCHORS = ['cue_on', 'cue_off', 'last_lick']

    # Column names for each anchor's wait time
    # Note: time_waited = cue_off → decision (the wait period)
    #       time_waited_since_cue_on = cue_on → decision (includes background)
    #       time_waited_since_last_lick = last_lick → decision (could span trials!)
    WAIT_TIME_COLS = {
        'cue_on': 'time_waited_since_cue_on',
        'cue_off': 'time_waited',  # This is cue_off → decision
        'last_lick': 'time_waited_since_last_lick',
    }

    # Timing parameters
    MIN_WINDOW_SEC = 1.0      # Minimum anchor→decision window (need time for baseline)
    BASELINE_WINDOW_SEC = 0.3  # First 300ms after anchor
    SLOPE_START_BUFFER = 0.3   # Skip first 300ms after anchor (same as baseline)
    SLOPE_END_BUFFER = 0.2     # Stop 200ms before decision
    PREDEC_START = -0.5        # 500ms before decision
    PREDEC_END = -0.05         # Stop 50ms before decision

    # Plotting
    PLOT_TIME_RANGE = (-8, 1.5)

    # Minimum trials per condition
    MIN_TRIALS = 20

    # Output
    OUTPUT_DIR = TRIAL_HISTORY_V3_DIR

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
            raise ValueError(f"No GRAB channel found. Available: {available_channels}")

    # Create binary reward columns
    df['rewarded'] = (df['reward'] > 0).astype(int)
    df['prev_rewarded'] = (df['previous_trial_reward'] > 0).astype(int)

    # Filter out miss trials (current trial must have decision)
    n_before = len(df)
    if 'miss_trial' in df.columns:
        df = df[df['miss_trial'] == False].copy()
        print(f"  Filtered current miss trials: {n_before:,} → {len(df):,} rows")
    elif 'time_waited' in df.columns:
        df = df[df['time_waited'].notna() & (df['time_waited'] < 60)].copy()
        print(f"  Filtered by valid time_waited: {n_before:,} → {len(df):,} rows")

    # Filter out trials where PREVIOUS trial was a miss
    n_before = len(df)
    if 'previous_trial_miss' in df.columns:
        df = df[df['previous_trial_miss'] == False].copy()
        print(f"  Filtered prev miss trials: {n_before:,} → {len(df):,} rows")
    else:
        valid_prev = df['previous_trial_reward'].isin([0.0, 5.0])
        df = df[valid_prev].copy()
        print(f"  Filtered invalid prev reward: {n_before:,} → {len(df):,} rows")

    print(f"  Total rows after filtering: {len(df):,}")
    print(f"  Sessions: {df['session_id'].nunique()}")
    print(f"  Mice: {sorted(df['mouse'].unique())}")
    print(f"  Groups: {sorted(df['group'].unique())}")

    return df


def load_data(filepath: Path, config: Config) -> pd.DataFrame:
    """Load merged photometry data from a single CSV file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return _process_dataframe(df, config)


def _build_allowed_sessions(base: Path) -> set:
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
            df["session_id"] = session_dir.name
            frames.append(df)
            print(f"  Loaded {session_dir.name}  ({len(df):,} rows)")
        except Exception as e:
            print(f"  WARNING: could not read {merged_csv}: {e}")

    if not frames:
        raise FileNotFoundError(f"No photometry_with_trial_data.csv files found under {base}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows after concatenation: {len(combined):,}")
    print(f"  Sessions loaded: {combined['session_id'].nunique()}")
    return _process_dataframe(combined, config)


# =============================================================================
# TRIAL EXTRACTION (per anchor)
# =============================================================================

def extract_trials_for_anchor(
    df: pd.DataFrame, 
    anchor: str,
    config: Config
) -> List[Dict]:
    """Extract per-trial measures for a specific anchor."""
    
    # Get the wait time column for this anchor
    wait_col = config.WAIT_TIME_COLS.get(anchor)
    if wait_col not in df.columns:
        print(f"  WARNING: Column {wait_col} not found for anchor {anchor}")
        return []
    
    trials = []
    grouped = df.groupby(['session_id', 'session_trial_num'])
    
    for (session_id, trial_num), trial_df in grouped:
        trial_df = trial_df.sort_values('trial_time')
        row = trial_df.iloc[0]
        
        # Skip first trial (no previous trial data)
        if pd.isna(row['previous_trial_reward']):
            continue
        
        # Get window duration for this anchor
        window_duration = row[wait_col]
        if pd.isna(window_duration) or window_duration < config.MIN_WINDOW_SEC:
            continue
        
        # Extract time and signal
        # trial_time is time within trial, decision_time is when decision happened
        # We need time relative to decision
        if 'decision_time' in trial_df.columns:
            time_rel = trial_df['trial_time'].values - row['decision_time']
        else:
            # Fallback: use time_waited_since_cue_on as decision time proxy
            time_rel = trial_df['trial_time'].values - row['time_waited_since_cue_on']
        
        signal = trial_df[config.SIGNAL_COL].values
        
        # Anchor time (relative to decision at t=0)
        anchor_time = -window_duration
        
        # =================================================================
        # MEASURE 1: TRUE BASELINE (first 300ms after anchor)
        # =================================================================
        baseline_start = anchor_time
        baseline_end = anchor_time + config.BASELINE_WINDOW_SEC
        baseline_mask = (time_rel >= baseline_start) & (time_rel <= baseline_end)
        
        if baseline_mask.sum() < 3:
            continue
        
        true_baseline = np.nanmean(signal[baseline_mask])
        
        # =================================================================
        # MEASURE 2: PRE-DECISION LEVEL (last 500ms before decision)
        # =================================================================
        predec_mask = (time_rel >= config.PREDEC_START) & (time_rel <= config.PREDEC_END)
        predec_level = np.nanmean(signal[predec_mask]) if predec_mask.sum() >= 3 else np.nan
        
        # =================================================================
        # MEASURE 3: SLOPE (rate of change from anchor to decision)
        # =================================================================
        slope_start = anchor_time + config.SLOPE_START_BUFFER
        slope_end = -config.SLOPE_END_BUFFER
        
        slope, r2 = np.nan, np.nan
        if slope_end > slope_start + 0.3:  # Need at least 300ms for slope
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
        # MEASURE 4: RAMP AMPLITUDE (pre-decision minus baseline)
        # =================================================================
        ramp_amplitude = predec_level - true_baseline if np.isfinite(predec_level) else np.nan
        
        # =================================================================
        # Store trial data
        # =================================================================
        plot_mask = (time_rel >= config.PLOT_TIME_RANGE[0]) & (time_rel <= config.PLOT_TIME_RANGE[1])

        if plot_mask.sum() < 3:
            continue

        trials.append({
            # Identifiers
            'session_id': session_id,
            'trial_num': trial_num,
            'mouse': row['mouse'],
            'group': row['group'],
            'anchor': anchor,
            
            # Timing (matched to anchor!)
            'window_duration': window_duration,
            'wait_time': window_duration,  # Now matched to anchor
            'prev_wait_time': row.get(f'previous_trial_{wait_col}', np.nan),
            
            # Outcome
            'rewarded': bool(row['rewarded']),
            'prev_rewarded': bool(row['prev_rewarded']),
            
            # DA measures
            'true_baseline': true_baseline,
            'predec_level': predec_level,
            'slope': slope,
            'slope_r2': r2,
            'ramp_amplitude': ramp_amplitude,
            
            # For plotting
            'time_rel': time_rel[plot_mask],
            'signal_raw': signal[plot_mask],
            'signal_baselined': signal[plot_mask] - true_baseline,
        })
    
    print(f"  Anchor {anchor}: extracted {len(trials)} valid trials")
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


def correlation_analysis(trials: List[Dict], anchor: str) -> pd.DataFrame:
    """Compute key correlations for a specific anchor."""
    df = pd.DataFrame([{
        'prev_rewarded': int(t['prev_rewarded']),
        'true_baseline': t['true_baseline'],
        'predec_level': t['predec_level'],
        'slope': t['slope'],
        'ramp_amplitude': t['ramp_amplitude'],
        'wait_time': t['wait_time'],
        'window_duration': t['window_duration'],
        'mouse': t['mouse'],
        'group': t['group'],
    } for t in trials])
    
    # Key correlations
    pairs = [
        ('prev_rewarded', 'true_baseline', f'Prev reward → Baseline ({anchor})'),
        ('prev_rewarded', 'slope', f'Prev reward → Slope ({anchor})'),
        ('prev_rewarded', 'ramp_amplitude', f'Prev reward → Ramp amplitude ({anchor})'),
        ('prev_rewarded', 'predec_level', f'Prev reward → Pre-decision ({anchor})'),
        ('true_baseline', 'slope', f'Baseline → Slope ({anchor})'),
        ('slope', 'wait_time', f'Slope → Wait time ({anchor})'),
        ('true_baseline', 'wait_time', f'Baseline → Wait time ({anchor})'),
        ('prev_rewarded', 'wait_time', f'Prev reward → Wait time ({anchor})'),
    ]
    
    results = []
    for x_col, y_col, description in pairs:
        valid = df[[x_col, y_col]].dropna()
        if len(valid) < 20:
            continue
        
        r, p = stats.pearsonr(valid[x_col], valid[y_col])
        results.append({
            'anchor': anchor,
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
        return
    
    time_grid = np.arange(config.PLOT_TIME_RANGE[0], config.PLOT_TIME_RANGE[1], 0.05)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    signal_key = 'signal_baselined' if baseline_subtract else 'signal_raw'
    
    for subset, label, color in [
        (prev_rew, f"Prev rewarded (n={len(prev_rew)})", config.COLORS["prev_rew"]),
        (prev_unrew, f"Prev unrewarded (n={len(prev_unrew)})", config.COLORS["prev_unrew"]),
    ]:
        signals = np.full((len(subset), len(time_grid)), np.nan)
        for i, t in enumerate(subset):
            if len(t['time_rel']) == 0:
                continue
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
    config: Config,
    output_path: Path,
    title: str,
):
    """Plot violin comparisons for all four measures."""
    measures = ['true_baseline', 'slope', 'ramp_amplitude', 'predec_level']
    labels = ['Baseline\n(at anchor)', 'Slope\n(z/s)', 'Ramp amplitude\n(Δ from baseline)', 'Pre-decision\nlevel']
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    
    for ax, measure, label in zip(axes, measures, labels):
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
        
        t_stat, p_val = stats.ttest_ind(prev_rew, prev_unrew)
        pooled_std = np.sqrt((np.var(prev_rew) + np.var(prev_unrew)) / 2)
        d = (np.mean(prev_rew) - np.mean(prev_unrew)) / pooled_std if pooled_std > 0 else 0
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.text(0.5, 0.95, f"d={d:+.2f}, p={p_val:.3f} {sig}",
                transform=ax.transAxes, ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()


def plot_slope_vs_wait(
    trials: List[Dict],
    config: Config,
    output_path: Path,
    title: str,
):
    """Plot slope vs wait time relationship."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for subset, color, label in [
        ([t for t in trials if t['prev_rewarded']], config.COLORS["prev_rew"], "Prev rew"),
        ([t for t in trials if not t['prev_rewarded']], config.COLORS["prev_unrew"], "Prev unrew"),
    ]:
        x = [t['slope'] for t in subset if np.isfinite(t['slope']) and np.isfinite(t['wait_time'])]
        y = [t['wait_time'] for t in subset if np.isfinite(t['slope']) and np.isfinite(t['wait_time'])]
        
        if len(x) > 10:
            ax.scatter(x, y, color=color, alpha=0.3, s=15, label=label)
            m, b = np.polyfit(x, y, 1)
            x_line = np.array([min(x), max(x)])
            ax.plot(x_line, m * x_line + b, color=color, linewidth=2, linestyle='--')
            
            r, p = stats.pearsonr(x, y)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.text(0.95, 0.95 if color == config.COLORS["prev_rew"] else 0.88,
                    f"{label}: r={r:.2f} {sig}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=10, color=color)
    
    ax.set_xlabel("Slope (z/s)")
    ax.set_ylabel("Wait time (s)")
    ax.set_title(title)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis(config: Config):
    """Main analysis — test all three anchors."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("=" * 70)
    print("DA Trial History Analysis (v3) — Multi-Anchor")
    print("=" * 70)

    # Load data — auto-discover all sessions under PROCESSED_OUT
    df = load_all_sessions_data(config)
    
    all_results = []
    all_correlations = []
    
    # Process each anchor
    for anchor in config.ANCHORS:
        print(f"\n{'=' * 70}")
        print(f"ANCHOR: {anchor}")
        print("=" * 70)
        
        # Create output directory for this anchor
        anchor_dir = config.OUTPUT_DIR / anchor
        anchor_dir.mkdir(exist_ok=True)
        
        # Extract trials
        trials = extract_trials_for_anchor(df, anchor, config)
        
        if len(trials) < config.MIN_TRIALS * 2:
            print(f"  Insufficient trials for {anchor}, skipping")
            continue
        
        n_prev_rew = sum(1 for t in trials if t['prev_rewarded'])
        n_prev_unrew = sum(1 for t in trials if not t['prev_rewarded'])
        print(f"  After prev reward: {n_prev_rew}")
        print(f"  After prev unreward: {n_prev_unrew}")
        
        groups = sorted(set(t['group'] for t in trials))
        
        # =====================================================================
        # 1. MEASURE COMPARISONS
        # =====================================================================
        print(f"\n  --- Effect of Previous Outcome ({anchor}) ---")
        
        measures = ['true_baseline', 'slope', 'ramp_amplitude', 'predec_level']
        
        # All trials
        print("\n  ALL TRIALS:")
        for measure in measures:
            result = compare_by_prev_outcome(trials, measure, config)
            if result:
                result['anchor'] = anchor
                result['level'] = 'all'
                all_results.append(result)
                sig = "***" if result['p_val'] < 0.001 else "**" if result['p_val'] < 0.01 else "*" if result['p_val'] < 0.05 else "ns"
                print(f"    {measure:18s}: d={result['cohens_d']:+.3f} {sig}")
        
        # Plot
        plot_measure_comparison(trials, config, anchor_dir / "all_measures.png",
                                f"All Trials | Anchor: {anchor}")
        
        # By group
        for group in groups:
            group_trials = [t for t in trials if t['group'] == group]
            if len(group_trials) >= config.MIN_TRIALS * 2:
                print(f"\n  {group.upper()} GROUP:")
                for measure in measures:
                    result = compare_by_prev_outcome(group_trials, measure, config)
                    if result:
                        result['anchor'] = anchor
                        result['level'] = group
                        all_results.append(result)
                        sig = "***" if result['p_val'] < 0.001 else "**" if result['p_val'] < 0.01 else "*" if result['p_val'] < 0.05 else "ns"
                        print(f"    {measure:18s}: d={result['cohens_d']:+.3f} {sig}")
                
                plot_measure_comparison(group_trials, config, anchor_dir / f"{group}_measures.png",
                                        f"{group.upper()} BG | Anchor: {anchor}")
        
        # =====================================================================
        # 2. TRACES
        # =====================================================================
        plot_traces(trials, config, anchor_dir / "traces_baselined.png",
                    f"All Trials | Anchor: {anchor}", baseline_subtract=True)
        
        for group in groups:
            group_trials = [t for t in trials if t['group'] == group]
            if len(group_trials) >= config.MIN_TRIALS * 2:
                plot_traces(group_trials, config, anchor_dir / f"{group}_traces.png",
                            f"{group.upper()} BG | Anchor: {anchor}", baseline_subtract=True)
        
        # =====================================================================
        # 3. SLOPE VS WAIT TIME
        # =====================================================================
        plot_slope_vs_wait(trials, config, anchor_dir / "slope_vs_wait.png",
                           f"Slope → Wait Time | Anchor: {anchor}")
        
        for group in groups:
            group_trials = [t for t in trials if t['group'] == group]
            if len(group_trials) >= config.MIN_TRIALS * 2:
                plot_slope_vs_wait(group_trials, config, anchor_dir / f"{group}_slope_vs_wait.png",
                                   f"{group.upper()} BG | Slope → Wait | Anchor: {anchor}")
        
        # =====================================================================
        # 4. CORRELATIONS
        # =====================================================================
        corr_df = correlation_analysis(trials, anchor)
        all_correlations.append(corr_df)
        
        print(f"\n  --- Correlations ({anchor}) ---")
        for _, row in corr_df.iterrows():
            sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
            print(f"    {row['description']:45s}: r={row['r']:+.3f} {sig}")
    
    # =========================================================================
    # SAVE ALL RESULTS
    # =========================================================================
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(config.OUTPUT_DIR / "measure_comparisons.csv", index=False)
        print(f"\nSaved measure comparisons to {config.OUTPUT_DIR / 'measure_comparisons.csv'}")
    
    if all_correlations:
        corr_df = pd.concat(all_correlations, ignore_index=True)
        corr_df.to_csv(config.OUTPUT_DIR / "correlations.csv", index=False)
        print(f"Saved correlations to {config.OUTPUT_DIR / 'correlations.csv'}")
    
    # =========================================================================
    # SUMMARY COMPARISON ACROSS ANCHORS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Effect sizes (Cohen's d) by anchor")
    print("=" * 70)
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        for level in ['all', 's', 'l']:
            level_df = summary_df[summary_df['level'] == level]
            if len(level_df) == 0:
                continue
            print(f"\n  {level.upper()}:")
            for anchor in config.ANCHORS:
                anchor_df = level_df[level_df['anchor'] == anchor]
                if len(anchor_df) == 0:
                    continue
                baseline_d = anchor_df[anchor_df['measure'] == 'true_baseline']['cohens_d'].values
                slope_d = anchor_df[anchor_df['measure'] == 'slope']['cohens_d'].values
                baseline_str = f"{baseline_d[0]:+.2f}" if len(baseline_d) > 0 else "N/A"
                slope_str = f"{slope_d[0]:+.2f}" if len(slope_d) > 0 else "N/A"
                print(f"    {anchor:12s}: baseline d={baseline_str}, slope d={slope_str}")
    
    print(f"\n{'=' * 70}")
    print(f"DONE. All outputs saved to: {config.OUTPUT_DIR}")
    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    config = Config()
    run_analysis(config)

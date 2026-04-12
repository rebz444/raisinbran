"""
Quality Tier System for Photometry Signal Processing (v3)

Tier A: Excellent - clean signal appropriate for the indicator type
Tier B: Good - minor issues, still usable for most analyses  
Tier C: Review - significant artifacts or failed correction

Key improvements in v3:
1. Indicator-aware skewness: GCaMP positive skew is expected, not penalized
2. running_mean_range captures U-shapes that endpoint drift misses
3. Self-bleach channels CAN be Tier A if the correction worked well
"""

import numpy as np


# =============================================================================
# INDICATOR TYPES
# =============================================================================

def get_indicator_type(channel_name):
    """
    Determine indicator type from channel name.
    
    Returns
    -------
    str : 'dopamine', 'calcium', or 'unknown'
    """
    ch = channel_name.lower()
    if 'grab' in ch:
        return 'dopamine'  # GRAB-DA, GRAB-ACh, etc. - relatively symmetric
    elif 'gcamp' in ch or 'rcamp' in ch:
        return 'calcium'   # Calcium indicators - positive skew expected
    else:
        return 'unknown'


# =============================================================================
# TIER THRESHOLDS
# =============================================================================

# Tier A (excellent)
TIER_A = {
    'drift_max': 0.40,           # Low endpoint drift
    'running_range_max': 3.0,    # Based on actual data: good sessions ~2-3
    'r2_min': 0.85,              # Good baseline fit
    'r_iso_dff_max': 0.08,       # Good iso correction
    'outlier_max': 0.003,        # Allow some outliers (GCaMP transients)
    'noise_std_min': 0.0015,     # Very low noise suggests over-smoothing
    # Skewness thresholds depend on indicator type (see below)
}

# Tier B (good) - green channels
TIER_B_GREEN = {
    'drift_max': 0.80,
    'running_range_max': 5.0,    # Based on actual data distribution
    'r2_min': 0.70,
    'r_iso_dff_max': 0.20,
    'outlier_max': 0.005,
    'noise_std_min': 0.0008,     # Flag if almost no high-freq content
}

# Tier B (good) - red/self-bleach channels (more lenient on drift)
TIER_B_RED = {
    'drift_max': 1.20,           # Reduced from 1.5 - the bad sessions had drift ~1.1
    'running_range_max': 5.0,
    'r2_min': 0.70,
    'r_iso_dff_max': 0.20,
    'outlier_max': 0.005,
    'noise_std_min': 0.0008,
}

# Skewness thresholds by indicator type
SKEW_THRESHOLDS = {
    'dopamine': {
        'tier_a_max': 0.50,      # Dopamine should be fairly symmetric
        'tier_b_max': 2.00,      # Flag if very asymmetric
    },
    'calcium': {
        'tier_a_max': 2.50,      # Calcium transients cause positive skew - that's OK
        'tier_b_max': 5.00,      # Only flag extreme values
        'negative_max': -1.50,   # Negative skew IS a problem for calcium
    },
    'unknown': {
        'tier_a_max': 1.00,
        'tier_b_max': 2.50,
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_running_mean_range(dff, fs, window_sec=30):
    """
    Compute the range of a smoothed dF/F signal, normalized by std.
    Catches U-shapes and slow nonlinear structure.
    """
    if len(dff) < 10:
        return 0.0
    
    window_samples = int(window_sec * fs)
    window_samples = max(3, min(window_samples, len(dff) // 3))
    
    kernel = np.ones(window_samples) / window_samples
    smoothed = np.convolve(dff, kernel, mode='same')
    
    edge = window_samples // 2
    if edge > 0 and len(smoothed) > 2 * edge:
        smoothed = smoothed[edge:-edge]
    
    smooth_range = np.max(smoothed) - np.min(smoothed)
    dff_std = np.std(dff)
    
    return smooth_range / dff_std if dff_std > 0 else 0.0


# =============================================================================
# MAIN TIER COMPUTATION
# =============================================================================

def compute_quality_tier(metrics, note, channel_name, fs=20.0, dff=None):
    """
    Assign quality tier (A/B/C) based on correction metrics.
    
    Parameters
    ----------
    metrics : dict
        Output from compute_correction_metrics()
    note : str
        Processing note string
    channel_name : str
        Channel name (e.g., 'l_str_GRAB', 'r_str_GCaMP')
    fs : float
        Sampling frequency
    dff : array, optional
        Raw dF/F for running_mean_range calculation
    
    Returns
    -------
    tier : str
        'A', 'B', or 'C'
    reasons : list of str
        Reasons for tier assignment
    """
    reasons = []
    
    # Determine channel characteristics
    is_self_bleach = "self_bleach_560" in note
    is_green = "green" in note
    indicator = get_indicator_type(channel_name)
    
    # Get appropriate thresholds
    tier_b = TIER_B_RED if is_self_bleach else TIER_B_GREEN
    skew_thresh = SKEW_THRESHOLDS.get(indicator, SKEW_THRESHOLDS['unknown'])
    
    # Extract metrics with safe defaults
    drift = abs(metrics.get("drift_delta_z", 0) or 0)
    skew = metrics.get("skewness", 0) or 0
    r2 = metrics.get("r2_baseline", 1.0) or 1.0
    outliers = metrics.get("outlier_frac", 0) or 0
    r_iso_dff = abs(metrics.get("r_iso_dff", 0) or 0)
    noise_std = metrics.get("noise_std", 0.01) or 0.01  # Default to high if missing
    
    # Compute or retrieve running_mean_range
    if dff is not None:
        running_range = compute_running_mean_range(dff, fs)
    elif "running_mean_range" in metrics:
        running_range = metrics["running_mean_range"]
    else:
        # Estimate when dff not available
        running_range = drift * 1.2 + abs(skew) * 0.2
    
    # =========================================================================
    # CHECK FOR TIER C (critical failures)
    # =========================================================================
    
    # Poor baseline fit
    if r2 < tier_b['r2_min']:
        reasons.append(f"R²={r2:.2f}")
    
    # Processing fallbacks
    if metrics.get("flag_iso_unreliable"):
        reasons.append("iso_unreliable")
    if metrics.get("flag_fit_failed"):
        reasons.append("fit_failed")
    
    # Excessive drift (key indicator of U/V shapes for self-bleach)
    if drift > tier_b['drift_max']:
        reasons.append(f"drift={drift:.2f}")
    
    # Excessive slow structure (U-shapes)
    if running_range > tier_b['running_range_max']:
        reasons.append(f"slow_structure={running_range:.2f}")
    
    # Very low noise suggests over-smoothing / baseline captured real signal
    if noise_std < tier_b.get('noise_std_min', 0):
        reasons.append(f"low_noise={noise_std:.4f}")
    
    # Skewness check (indicator-aware)
    if indicator == 'calcium':
        # For calcium: flag extreme positive OR any significant negative
        if skew > skew_thresh['tier_b_max']:
            reasons.append(f"skew={skew:.2f}")
        elif skew < skew_thresh.get('negative_max', -2.0):
            reasons.append(f"neg_skew={skew:.2f}")
    else:
        # For dopamine/unknown: flag any extreme asymmetry
        if abs(skew) > skew_thresh['tier_b_max']:
            reasons.append(f"skew={skew:.2f}")
    
    # Poor iso correction (green only)
    if is_green and r_iso_dff > tier_b['r_iso_dff_max']:
        reasons.append(f"r_iso_dff={r_iso_dff:.2f}")
    
    # Excessive outliers
    if outliers > tier_b['outlier_max']:
        reasons.append(f"outliers={outliers:.3f}")
    
    # If any critical issues, return Tier C
    if reasons:
        return "C", reasons
    
    # =========================================================================
    # CHECK FOR TIER A (excellent)
    # =========================================================================
    
    tier_b_reasons = []
    
    if drift > TIER_A['drift_max']:
        tier_b_reasons.append(f"drift={drift:.2f}")
    
    if running_range > TIER_A['running_range_max']:
        tier_b_reasons.append(f"slow_structure={running_range:.2f}")
    
    if r2 < TIER_A['r2_min']:
        tier_b_reasons.append(f"R²={r2:.2f}")
    
    if is_green and r_iso_dff > TIER_A['r_iso_dff_max']:
        tier_b_reasons.append(f"r_iso_dff={r_iso_dff:.2f}")
    
    if outliers > TIER_A['outlier_max']:
        tier_b_reasons.append(f"outliers={outliers:.4f}")
    
    # Skewness for Tier A (indicator-aware)
    if indicator == 'calcium':
        # Calcium: allow positive skew, penalize negative
        if skew > skew_thresh['tier_a_max']:
            tier_b_reasons.append(f"skew={skew:.2f}")
        elif skew < -0.50:  # Some negative skew is concerning
            tier_b_reasons.append(f"neg_skew={skew:.2f}")
    else:
        # Dopamine: should be fairly symmetric
        if abs(skew) > skew_thresh['tier_a_max']:
            tier_b_reasons.append(f"skew={skew:.2f}")
    
    if tier_b_reasons:
        return "B", tier_b_reasons
    
    return "A", []


# =============================================================================
# CONVENIENCE FUNCTION FOR DATAFRAME ROWS
# =============================================================================

def compute_tier_from_row(row, fs=20.0):
    """Apply tier computation to a DataFrame row."""
    metrics = row.to_dict()
    note = row.get('note', '')
    channel = row.get('channel', '')
    
    # Estimate running_mean_range if not present
    if 'running_mean_range' not in metrics:
        drift = abs(metrics.get('drift_delta_z', 0) or 0)
        skew = abs(metrics.get('skewness', 0) or 0)
        metrics['running_mean_range'] = drift * 1.2 + skew * 0.2
    
    return compute_quality_tier(metrics, note, channel, fs=fs)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("QUALITY TIER SYSTEM v3 - INDICATOR-AWARE")
    print("="*70)
    
    test_cases = [
        # Good GRAB - should be Tier A
        {
            "name": "RZ074 08-08 GRAB (good dopamine)",
            "channel": "l_str_GRAB",
            "note": "green_iso415_sig470",
            "metrics": {"r2_baseline": 0.92, "drift_delta_z": -0.34, "skewness": 0.13,
                       "r_iso_dff": 0.01, "outlier_frac": 0.0003, "running_mean_range": 0.4},
        },
        # Good GCaMP with positive skew - should be Tier A (skew is expected!)
        {
            "name": "RZ082 02-26 GCaMP (good calcium, skew=1.94)",
            "channel": "r_str_GCaMP", 
            "note": "green_iso415_sig470",
            "metrics": {"r2_baseline": 0.93, "drift_delta_z": -0.20, "skewness": 1.94,
                       "r_iso_dff": 0.02, "outlier_frac": 0.0028, "running_mean_range": 0.5},
        },
        # Good rCaMP self-bleach - should be Tier A
        {
            "name": "RZ081 02-18 rCaMP (good self-bleach)",
            "channel": "l_str_rCaMP",
            "note": "red_sig560_only|self_bleach_560",
            "metrics": {"r2_baseline": 0.99, "drift_delta_z": -0.02, "skewness": 0.04,
                       "outlier_frac": 0.0, "running_mean_range": 0.3},
        },
        # Bad rCaMP with U-shape - should be Tier C
        {
            "name": "RZ074 08-14 rCaMP (bad U-shape)",
            "channel": "l_str_rCaMP",
            "note": "red_sig560_only|self_bleach_560", 
            "metrics": {"r2_baseline": 0.96, "drift_delta_z": 0.51, "skewness": 0.64,
                       "outlier_frac": 0.0, "running_mean_range": 2.5},
        },
    ]
    
    for tc in test_cases:
        tier, reasons = compute_quality_tier(
            tc["metrics"], tc["note"], tc["channel"]
        )
        print(f"\n{tc['name']}:")
        print(f"  indicator: {get_indicator_type(tc['channel'])}")
        print(f"  → Tier {tier}: {reasons if reasons else 'excellent'}")

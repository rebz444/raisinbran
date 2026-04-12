"""
Step 1: Fiber Photometry Signal Processing
===========================================
Matches FP and behavior CSV files by mouse/timestamp, corrects for
photobleaching using isosbestic (415 nm) or self-bleach models, computes
dF/F and z-scored traces, runs QC metrics, and saves per-session outputs.

Outputs per session (saved to fp_processed/{session_id}/):
    photometry_long.csv         — long-form dF/F trace (all channels)
    {channel}.csv               — per-channel trace
    channel_summary.csv         — sampling rate, n_samples, duration
    correction_quality.csv      — QC metrics and quality tier per channel
    qc/{channel}_qc.png         — 4-panel QC diagnostic plot

Run before:
    2_fp_qc_summary.py
    3_fp_behavior_align.py
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, skew
from datetime import datetime
from pathlib import Path

from config import (
    FP_DIR,
    PROCESSED_OUT,
    QC_GATE_ENABLED,
    QC_STOP_ON_FAIL,
    QC_MIN_SAMPLES_PER_STATE,
    get_channel_color,
)
from quality_tiers import compute_quality_tier
from utils import get_channels_for_session, load_photometry_log, update_session_log

# --- Configuration ---
REGENERATE = False  # Set to True to reprocess already-processed sessions
POST_LOWPASS_HZ = None  # Set to e.g. 10.0 to apply a post-hoc lowpass filter on dF/F; None = no filter

# --- 1. Signal Processing Logic ---

class SignalProcessor:
    """Encapsulates signal processing logic (filtering, fitting)."""
    
    @staticmethod
    def butter_lowpass(x, fs, cutoff_hz, order=2):
        """Zero-phase lowpass filter."""
        if cutoff_hz is None:
            return x
        nyq = 0.5 * fs
        if cutoff_hz >= nyq:
            return x  # Filter useless if cutoff > nyquist
        b, a = butter(order, cutoff_hz / nyq, btype="low")
        return filtfilt(b, a, x)

    @staticmethod
    def exp2_func(t, a, b, c, d):
        """Bi-exponential function for curve fitting."""
        return a * np.exp(b * t) + c * np.exp(d * t)

    @staticmethod
    def robust_line_fit(x, y, n_iter=20, c=4.685):
        """
        Robust linear fit y = slope * x + intercept using Tukey bisquare IRLS.
        Used to align iso to signal baseline.
        """
        # Add intercept column
        X = np.column_stack([x, np.ones_like(x)])
        
        # Initial OLS
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        for _ in range(n_iter):
            resid = y - (X @ beta)
            mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
            u = resid / (c * mad)
            w = (1 - u**2)**2
            w[np.abs(u) >= 1] = 0
            
            # Optimized weighted least squares:
            # X is (N, 2), w is (N,)
            # Broadcasting: X * w[:, None] multiplies each row of X by weight w
            # Weighted least squares uses sqrt(weights)
            sw = np.sqrt(w)
            WX = X * sw[:, np.newaxis]
            wy = y * sw

            beta = np.linalg.lstsq(WX, wy, rcond=None)[0]

        return beta[0], beta[1]

    @staticmethod
    def compute_fs(ts):
        """Estimate sampling frequency from timestamps."""
        ts = np.asarray(ts)
        dt = np.diff(ts)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) == 0:
            return 1.0 # Fallback
        return 1.0 / np.mean(dt)

    @classmethod
    def fit_bleaching_from_iso(cls, ts_iso, iso, method="exp2", smooth_cutoff_hz=0.01):
        """
        Derives a bleaching model from the isosbestic channel (415nm).
        """
        iso = np.asarray(iso, dtype=float)
        # Use index as time proxy for fitting stability
        t = np.arange(len(iso), dtype=float) 

        if method == "exp2":
            # Initial guess: two decaying components
            p0 = (iso[0], -1e-4, iso[0] * 0.5, -1e-5)
            bounds = (
                (-np.inf, -1.0, -np.inf, -1.0),  # Decay rates must be negative
                ( np.inf,  0.0,  np.inf,  0.0),
            )
            try:
                popt, _ = curve_fit(cls.exp2_func, t, iso, p0=p0, bounds=bounds, maxfev=20000)
                bleach = cls.exp2_func(t, *popt)
            except Exception:
                # Fallback to smoothing
                fs_iso = cls.compute_fs(ts_iso)
                bleach = cls.butter_lowpass(iso, fs_iso, cutoff_hz=smooth_cutoff_hz)
        
        elif method == "lowpass":
            fs_iso = cls.compute_fs(ts_iso)
            bleach = cls.butter_lowpass(iso, fs_iso, cutoff_hz=smooth_cutoff_hz)
        else:
            raise ValueError("method must be 'exp2' or 'lowpass'")

        return bleach

    @classmethod
    def correct_signal(cls, sig, bleach_iso_model, robust=True, eps=1e-12):
        """
        Scales the iso-derived bleach model to the signal channel and normalizes.
        Returns:
            scaled_bleach: The fitted baseline for the signal
            normalized: sig / scaled_bleach
            dff: normalized - 1
        """
        x = np.asarray(bleach_iso_model, dtype=float)
        y = np.asarray(sig, dtype=float)

        if robust:
            slope, intercept = cls.robust_line_fit(x, y)
        else:
            A = np.column_stack([x, np.ones_like(x)])
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

        scaled_bleach = slope * x + intercept
        scaled_bleach = np.maximum(scaled_bleach, eps) # Prevent division by zero/negative

        normalized = y / scaled_bleach
        dff = normalized - 1.0

        return scaled_bleach, normalized, dff


# --- 2. Correction Quality Metrics ---

def compute_edge_mismatch(sig, baseline):
    """
    Detect if the baseline fit is poor at session start but good at the end.
    Flags sessions that are candidates for cropping the first N seconds.
    """
    n = len(sig)
    seg = n // 10

    if seg < 10:
        return {"r2_start": float("nan"), "r2_end": float("nan"), "flag_start_mismatch": False}

    residual = sig - baseline

    ss_res_start = np.sum(residual[:seg] ** 2)
    ss_tot_start = np.sum((sig[:seg] - np.mean(sig[:seg])) ** 2)
    r2_start = 1 - ss_res_start / ss_tot_start if ss_tot_start > 1e-12 else 0.0

    ss_res_end = np.sum(residual[-seg:] ** 2)
    ss_tot_end = np.sum((sig[-seg:] - np.mean(sig[-seg:])) ** 2)
    r2_end = 1 - ss_res_end / ss_tot_end if ss_tot_end > 1e-12 else 0.0

    flag = (r2_start < 0.70) and (r2_end > 0.85) and (r2_end - r2_start > 0.20)

    return {
        "r2_start": float(r2_start),
        "r2_end": float(r2_end),
        "flag_start_mismatch": bool(flag),
    }


# Flag thresholds (tunable)
QC_R2_BASELINE_MIN    = 0.70   # R² of baseline fit below this = poor tracking
QC_DFF_SLOPE_MAX      = 5e-4   # |dff slope| > this (ΔF/F per second) = residual drift
QC_DRIFT_DELTA_Z_MAX  = 0.50   # |first-vs-last chunk delta / std| > this = drift
QC_ISO_DFF_CORR_MAX   = 0.15   # |r(iso, dff)| > this = incomplete iso correction
QC_OUTLIER_FRAC_MAX   = 0.005  # fraction of |dff_zscored| > 5 above this = artifact
QC_SKEWNESS_MAX       = 2.0    # |skew(dff)| above this = asymmetric distribution


def compute_correction_metrics(sig, iso, scaled_bleach, dff, dff_zscored, ts_sig, fs, note, channel_name=""):
    """
    Compute quantitative quality metrics for a single channel's bleaching correction.

    Returns a flat dict of scalar metrics and boolean flags. All inputs are numpy arrays
    (iso may be empty for red/self-bleach channels).

    Metrics
    -------
    r2_baseline      : R² of scaled_bleach vs raw sig. Fraction of slow-trend variance
                       captured by the baseline model.
    dff_slope        : Linear slope of dff vs time (ΔF/F per second). Near-zero = good.
    drift_delta_z    : (median(dff[last 10%]) - median(dff[first 10%])) / std(dff).
                       Catches nonlinear residual drift.
    r_iso_sig        : Pearson r(iso, sig) before correction (green only; NaN for red).
    r_iso_dff        : Pearson r(iso, dff) after correction (green only; NaN for red).
                       Near-zero = iso correction fully removed shared variance.
    noise_std        : std of dff - lowpass(dff, 2 Hz). Estimates high-freq noise floor.
    outlier_frac     : Fraction of samples where |dff_zscored| > 5.
    skewness         : scipy.stats.skew(dff). ~0 for well-behaved distributions.
    flag_*           : Boolean flags derived from the thresholds above.
    """
    metrics = {}

    # --- A. Baseline tracking quality ---
    sig_var = np.var(sig - np.mean(sig))
    if sig_var > 0:
        resid_var = np.var(sig - scaled_bleach)
        metrics["r2_baseline"] = float(1.0 - resid_var / sig_var)
    else:
        metrics["r2_baseline"] = float("nan")

    metrics.update(compute_edge_mismatch(sig, scaled_bleach))

    # --- B. Residual temporal drift ---
    t = np.asarray(ts_sig, dtype=float)
    t_norm = t - t[0]  # seconds from start
    if len(t_norm) > 1:
        slope, _ = np.polyfit(t_norm, dff, 1)
        metrics["dff_slope"] = float(slope)
    else:
        metrics["dff_slope"] = float("nan")

    chunk = max(1, len(dff) // 10)
    dff_std = np.std(dff)
    if dff_std > 0:
        delta = np.median(dff[-chunk:]) - np.median(dff[:chunk])
        metrics["drift_delta_z"] = float(delta / dff_std)
    else:
        metrics["drift_delta_z"] = float("nan")

    # --- C. Iso–dff residual correlation (green channels only) ---
    iso = np.asarray(iso)
    if len(iso) == len(sig) and len(iso) > 2:
        try:
            metrics["r_iso_sig"] = float(pearsonr(iso, sig)[0])
        except Exception:
            metrics["r_iso_sig"] = float("nan")
        try:
            metrics["r_iso_dff"] = float(pearsonr(iso, dff)[0])
        except Exception:
            metrics["r_iso_dff"] = float("nan")
    else:
        metrics["r_iso_sig"] = float("nan")
        metrics["r_iso_dff"] = float("nan")

    # --- D. Noise and distribution quality ---
    try:
        proc = SignalProcessor()
        dff_lp = proc.butter_lowpass(dff, fs, cutoff_hz=2.0)
        metrics["noise_std"] = float(np.std(dff - dff_lp))
    except Exception:
        metrics["noise_std"] = float("nan")

    metrics["outlier_frac"] = float(np.mean(np.abs(dff_zscored) > 5))
    metrics["skewness"] = float(skew(dff)) if len(dff) > 2 else float("nan")

    # --- Fallback/method flags from note string ---
    metrics["flag_iso_unreliable"]  = "iso_unreliable_fallback" in note
    metrics["flag_fit_failed"]      = "fit_failed_fallback" in note
    metrics["flag_self_bleach_560"] = "self_bleach_560" in note

    # --- Threshold flags ---
    r2 = metrics["r2_baseline"]
    metrics["flag_poor_baseline"]   = (not np.isnan(r2)) and (r2 < QC_R2_BASELINE_MIN)
    slope = metrics["dff_slope"]
    metrics["flag_residual_drift"]  = (not np.isnan(slope)) and (abs(slope) > QC_DFF_SLOPE_MAX)
    ddz = metrics["drift_delta_z"]
    metrics["flag_drift_delta"]     = (not np.isnan(ddz)) and (abs(ddz) > QC_DRIFT_DELTA_Z_MAX)
    r_id = metrics["r_iso_dff"]
    metrics["flag_iso_dff_corr"]    = (not np.isnan(r_id)) and (abs(r_id) > QC_ISO_DFF_CORR_MAX)
    metrics["flag_outliers"]        = metrics["outlier_frac"] > QC_OUTLIER_FRAC_MAX
    sk = metrics["skewness"]
    metrics["flag_skewed"]          = (not np.isnan(sk)) and (abs(sk) > QC_SKEWNESS_MAX)

    # Summary: any flag raised (flag_self_bleach_560 is expected for all red channels,
    # not a problem, so it is excluded from the any_flag rollup)
    flag_cols = [v for k, v in metrics.items()
                 if k.startswith("flag_") and k != "flag_self_bleach_560"]
    metrics["any_flag"] = any(flag_cols)

    # --- E. Running-mean range (slow modulation amplitude) ---
    window_samples = int(30 * fs)
    window_samples = max(3, min(window_samples, len(dff) // 3))
    kernel = np.ones(window_samples) / window_samples
    dff_smooth = np.convolve(dff, kernel, mode='same')
    edge = window_samples // 2
    if edge > 0 and len(dff_smooth) > 2 * edge:
        dff_smooth = dff_smooth[edge:-edge]
    metrics["running_mean_range"] = float(
        (np.max(dff_smooth) - np.min(dff_smooth)) / (np.std(dff) + 1e-12)
    )

    # --- Quality tier ---
    metrics["quality_tier"], tier_reasons = compute_quality_tier(metrics, note, channel_name, fs=fs, dff=dff)
    metrics["tier_reasons"] = "|".join(tier_reasons) if tier_reasons else ""

    return metrics


# --- 3. Data Handling & Utilities ---


def get_session_info(filename):
    """
    Parses common filename patterns for mouse, date, and time.
    Expected format examples:
      Type_Mouse_YYYY-MM-DD_HH_MM_SS.csv
      Type_Mouse_YYYY-MM-DDTHH_MM_SS.csv
    """
    # Regex handles separator T or _, and allows flexible ID chars
    pattern = r'([A-Za-z]+)_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})[T_](\d{2}_\d{2}_\d{2})'
    match = re.search(pattern, str(filename))
    
    if not match: 
        return None
    
    # Standardize time format
    dt_str = f"{match.group(3)} {match.group(4)}"
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H_%M_%S")
    
    return {
        'type': match.group(1),     # e.g., FP or Behav
        'mouse': match.group(2),    # e.g., RZ075
        'datetime': dt,
        'filename': filename
    }

def match_sessions(data_dir):
    """
    Pairs each FP_*.csv with its Behav_*.csv by mouse ID and nearest timestamp (within 300s).
    Searches recursively so no flattening step is needed.
    Stores absolute paths so process_session can load files directly.
    """
    data_dir = Path(data_dir)
    all_files = [f for f in data_dir.rglob("*.csv") if not f.name.startswith(".")]

    parsed = []
    for f in all_files:
        info = get_session_info(f.name)
        if info:
            info['path'] = f  # store full path
            parsed.append(info)

    fp_files    = [x for x in parsed if x['type'] == 'FP']
    behav_files = [x for x in parsed if x['type'] == 'Behav']

    sessions = []

    for fp in fp_files:
        candidates = [b for b in behav_files if b['mouse'] == fp['mouse']]

        best_match = None
        min_diff = float('inf')

        for b in candidates:
            diff = abs((fp['datetime'] - b['datetime']).total_seconds())
            if diff < 300 and diff < min_diff:
                min_diff = diff
                best_match = b

        if best_match:
            sessions.append({
                'mouse':      fp['mouse'],
                'date':       fp['datetime'].strftime('%Y-%m-%d'),
                'time':       fp['datetime'].strftime('%H:%M:%S'),
                'FP_file':    str(fp['path']),
                'Behav_file': str(best_match['path']),
                'session_id': f"{fp['mouse']}_{fp['datetime'].strftime('%Y%m%d_%H%M%S')}"
            })
        else:
            unmatched_id = f"{fp['mouse']}_{fp['datetime'].strftime('%Y%m%d_%H%M%S')}"
            update_session_log(unmatched_id, {
                "session_id":      unmatched_id,
                "mouse":           fp["mouse"],
                "date":            fp["datetime"].strftime("%Y-%m-%d"),
                "FP_file":         str(fp["path"]),
                "Behav_file":      float("nan"),
                "fp_behav_matched": False,
            })
            print(f"Unmatched FP session (no Behav within 300s): {fp['path'].name}")

    df = pd.DataFrame(sessions)
    if not df.empty:
        df.sort_values(['date', 'time'], inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df

def get_channel_config(channel_name):
    """
    Returns configuration (iso_state, sig_state) based on channel name.

    Standard Configuration (per FP3002 guide):
    - Green channels (G#): sig=470nm (State 2), iso=415nm (State 1)
    - Red channels (R#):   sig=560nm (State 4), NO isosbestic by default (single-wavelength)
    """
    if channel_name.startswith("G"):
        return 1, 2, "green_iso415_sig470"
    elif channel_name.startswith("R"):
        # Treat red as single-wavelength unless you truly acquired a red isosbestic channel.
        return None, 4, "red_sig560_only"
    else:
        raise ValueError(f"Unknown channel prefix: {channel_name}")

def extract_channel_data(fp_df, channel_col, led_state):
    """
    Extracts timestamps and signal for a specific LED state and channel column.
    """
    subset = fp_df[fp_df["LedState"] == led_state]
    if subset.empty:
        return np.array([]), np.array([])
        
    ts = subset["SystemTimestamp"].to_numpy(dtype=float)
    sig = subset[channel_col].to_numpy(dtype=float)
    return ts, sig


def align_iso_to_sig(ts_iso, iso, ts_sig, max_gap=None):
    """Align iso samples onto signal timestamps using nearest timestamp matching.

    Returns aligned (ts_sig_aligned, iso_aligned, sig_aligned) with any unmatched points dropped.
    """
    ts_iso = np.asarray(ts_iso, dtype=float)
    iso = np.asarray(iso, dtype=float)
    ts_sig = np.asarray(ts_sig, dtype=float)

    if len(ts_iso) == 0 or len(ts_sig) == 0:
        return ts_sig, np.array([]), np.array([])

    # Compute a reasonable tolerance from sampling intervals if not provided
    def _median_dt(ts):
        d = np.diff(ts)
        d = d[np.isfinite(d) & (d > 0)]
        return np.median(d) if len(d) else np.inf

    tol = max_gap
    if tol is None:
        tol = 0.5 * min(_median_dt(ts_iso), _median_dt(ts_sig))
        if not np.isfinite(tol) or tol <= 0:
            tol = 1e-3  # 1 ms fallback

    df_sig = pd.DataFrame({"SystemTimestamp": ts_sig})
    df_iso = pd.DataFrame({"SystemTimestamp": ts_iso, "iso": iso})
    df_sig.sort_values("SystemTimestamp", inplace=True)
    df_iso.sort_values("SystemTimestamp", inplace=True)

    merged = pd.merge_asof(
        df_sig,
        df_iso,
        on="SystemTimestamp",
        direction="nearest",
        tolerance=tol,
    )

    keep = merged["iso"].notna().to_numpy()
    ts_out = merged.loc[keep, "SystemTimestamp"].to_numpy(dtype=float)
    iso_out = merged.loc[keep, "iso"].to_numpy(dtype=float)
    idx = np.where(keep)[0]
    return ts_out, iso_out, idx

def is_iso_reliable(iso, sig, min_std=1e-5, min_ratio=0.10):
    """
    Heuristic check: is the isosbestic signal "alive"?
    Returns False if iso is flat or noise is too small relative to signal.
    """
    if len(iso) < 2: return False
    iso_std = np.std(iso)
    sig_std = np.std(sig) + 1e-12
    return (iso_std >= min_std) and (iso_std >= min_ratio * sig_std)


def make_longform_channel_df(session_info, res):
    """Return long-form dataframe for a single channel result."""
    ts = np.asarray(res["ts_sig"], dtype=float)
    df = pd.DataFrame({
        "session_id": session_info["session_id"],
        "mouse": session_info["mouse"],
        "date": session_info["date"],
        "time": session_info["time"],
        "channel": res["channel"],
        "t_sec": ts,  # Time in seconds since behavior start (t_start = 0)
        "sig": np.asarray(res["sig"], dtype=float),
        "baseline": np.asarray(res["scaled_bleach"], dtype=float),
        "dff": np.asarray(res["dff"], dtype=float),
        "dff_filtered": np.asarray(res["dff_filtered"], dtype=float),
        "dff_zscored": np.asarray(res["dff_zscored"], dtype=float),
    })
    iso = res.get("iso", None)
    if iso is not None and len(iso) == len(df):
        df["iso"] = np.asarray(iso, dtype=float)
    return df


# --- 4. Core Processing ---

def process_roi(fp_df, channel_name, post_lowpass_hz=None, min_len=50,
               bleach_method="exp2", iso_smooth_cutoff=0.01):
    try:
        iso_state, sig_state, note = get_channel_config(channel_name)
    except ValueError as e:
        return {"channel": channel_name, "error": str(e)}

    ts_sig, sig = extract_channel_data(fp_df, channel_name, sig_state)
    if len(sig) == 0:
        return {"channel": channel_name, "error": "Missing signal data for LED state"}

    # Check raw timestamps before any sorting (catches glitches in all channel types).
    # Small backward steps (>= -1s) are treated as jitter/duplicates and dropped.
    # Large jumps (< -1s) indicate a real clock reset and hard-fail the channel.
    if len(ts_sig) > 1 and np.any(np.diff(ts_sig) <= 0):
        diffs = np.diff(ts_sig)
        bad = np.where(diffs <= 0)[0]
        worst = diffs[bad].min()
        print(f"  [{channel_name}] {len(bad)} non-monotonic step(s) at indices {bad[:5]}, "
              f"worst jump: {worst:.4f}s")
        if worst < -1.0:
            return {"channel": channel_name, "error": f"Non-monotonic ts_sig (clock reset, worst jump {worst:.4f}s)"}
        # Drop the out-of-order samples and continue
        mask = np.ones(len(ts_sig), dtype=bool)
        mask[bad + 1] = False
        ts_sig, sig = ts_sig[mask], sig[mask]
        print(f"  [{channel_name}] Dropped {(~mask).sum()} sample(s) as jitter/duplicate, continuing.")

    # For green channels, align iso onto signal timestamps (do NOT truncate by length).
    ts_iso, iso = np.array([]), np.array([])
    if iso_state is not None:
        ts_iso_raw, iso_raw = extract_channel_data(fp_df, channel_name, iso_state)
        if len(iso_raw) == 0:
            return {"channel": channel_name, "error": "Missing iso data for LED state"}

        ts_aligned, iso_aligned, idx = align_iso_to_sig(ts_iso_raw, iso_raw, ts_sig)
        sig = sig[idx]
        ts_sig = ts_aligned
        ts_iso, iso = ts_sig, iso_aligned  # iso is now aligned to signal timestamps

    if len(sig) < min_len:
        return {"channel": channel_name, "error": f"Too few samples ({len(sig)})"}

    use_iso = (iso_state is not None)

    # Red is treated as single-wavelength (560-only): always self-bleach model.
    if channel_name.startswith("R"):
        use_iso = False
        note += "|self_bleach_560"
    elif use_iso and (not is_iso_reliable(iso, sig)):
        use_iso = False
        note += "|iso_unreliable_fallback"

    processor = SignalProcessor()

    if use_iso:
        try:
            bleach_iso = processor.fit_bleaching_from_iso(
                ts_iso, iso, method=bleach_method, smooth_cutoff_hz=iso_smooth_cutoff
            )
            scaled_bleach, _, dff = processor.correct_signal(sig, bleach_iso)
        except Exception as e:
            print(f"  [{channel_name}] Iso fit failed ({e}), falling back to self-bleach.")
            use_iso = False
            note += "|fit_failed_fallback"

    if not use_iso:
        # Self-bleach modeling directly on the signal (e.g., red 560-only or iso fallback)
        try:
            scaled_bleach = processor.fit_bleaching_from_iso(
                ts_sig, sig, method=bleach_method, smooth_cutoff_hz=iso_smooth_cutoff
            )
        except Exception:
            fs_sig = processor.compute_fs(ts_sig)
            scaled_bleach = processor.butter_lowpass(sig, fs_sig, cutoff_hz=iso_smooth_cutoff)

        scaled_bleach = np.maximum(scaled_bleach, 1e-12)

        dff = sig / scaled_bleach - 1.0

    # Baseline sanity (after either path)
    if (not np.all(np.isfinite(scaled_bleach))) or (np.median(scaled_bleach) <= 0):
        return {"channel": channel_name, "error": "Invalid scaled_bleach (NaN/inf or non-positive)"}

    ratio = np.median(scaled_bleach) / (np.median(sig) + 1e-12)
    if ratio < 0.2 or ratio > 5.0:
        return {"channel": channel_name, "error": f"Baseline/signal ratio suspicious ({ratio:.3g})"}

    dff_filtered = dff
    if post_lowpass_hz is not None:
        fs_sig = processor.compute_fs(ts_sig)
        dff_filtered = processor.butter_lowpass(dff, fs_sig, cutoff_hz=post_lowpass_hz)

    # Compute z-score within this channel
    dff_mean = np.mean(dff_filtered)
    dff_std = np.std(dff_filtered)
    if dff_std > 0:
        dff_zscored = (dff_filtered - dff_mean) / dff_std
    else:
        dff_zscored = np.zeros_like(dff_filtered)

    fs_out = processor.compute_fs(ts_sig)
    qc_metrics = compute_correction_metrics(
        sig, iso, scaled_bleach, dff, dff_zscored, ts_sig, fs_out, note, channel_name
    )

    return {
        "channel": channel_name,
        "note": note,
        "ts_sig": ts_sig,
        "iso": iso,
        "sig": sig,
        "scaled_bleach": scaled_bleach,
        "dff": dff,
        "dff_filtered": dff_filtered,
        "dff_zscored": dff_zscored,
        "fs": fs_out,
        "qc_metrics": qc_metrics,
        "channel_type": "green" if channel_name.startswith("G") else "red",
    }

def process_session(session_info, output_dir, regenerate=False):
    """
    Loads data for a session and processes all ROI channels.
    If regenerate=False, skips processing if output already exists.
    """
    output_dir = Path(output_dir)

    session_outdir = output_dir / session_info['session_id']

    if not regenerate and (session_outdir / "photometry_long.csv").exists():
        print(f"  Skipping {session_info['session_id']} (already processed; use regenerate=True to rerun)")
        return

    session_outdir.mkdir(parents=True, exist_ok=True)

    fp_path    = Path(session_info['FP_file'])
    behav_path = Path(session_info['Behav_file'])

    fp = pd.read_csv(fp_path)
    fp = fp[fp["LedState"] != 7].copy()  # 7 is often 'unknown' or 'inter-trial'

    # Temporal Alignment with Behavior (Crop Start/End)
    try:
        trigger = pd.read_csv(behav_path)
        if "SystemTimestamp" in trigger.columns:
            sys_ts = trigger["SystemTimestamp"]
        else:
            sys_ts = trigger.iloc[:, 3]

        t_start, t_end = sys_ts.iloc[0], sys_ts.iloc[-1]
        if fp["SystemTimestamp"].min() > t_end or fp["SystemTimestamp"].max() < t_start:
            print("Warning: behavior timestamps do not overlap FP timestamps; skipping crop.")
            fp["SystemTimestamp"] -= fp["SystemTimestamp"].min()
        else:
            fp = fp[fp["SystemTimestamp"].between(t_start, t_end)]
            fp["SystemTimestamp"] -= t_start
    except Exception as e:
        print(f"Warning: Could not crop to behavior ({e}). Zero-offsetting full FP file.")
        fp["SystemTimestamp"] -= fp["SystemTimestamp"].min()

    # Session duration (in minutes) from the cropped/zero-offset FP trace
    duration_min = float(fp["SystemTimestamp"].max()) / 60.0

    # Identify Channels (R0, R1, G2, etc.)
    channels = [c for c in fp.columns if re.match(r'^[RG]\d+$', c)]
    
    session_results = []
    
    for ch in channels:
        res = process_roi(fp, ch, post_lowpass_hz=POST_LOWPASS_HZ)
        if "error" in res:
            print(f"  Skipping {ch}: {res['error']}")
        else:
            session_results.append(res)
    # --- QC Gate: raw deinterleaved scan (recommended) ---
    qc_dir = session_outdir / "qc"
    tag = f"{session_info['mouse']}_{session_info['date']}_{session_info['time'].replace(':','_')}"
    if QC_GATE_ENABLED:
        ok, issues = qc_gate_raw_deinterleaved(fp, channels, qc_dir, tag)
        if (not ok) and QC_STOP_ON_FAIL:
            print("Raw QC gate failed; aborting session. See raw QC issues file.")
            return

    # --- Rename hardware channel indices to biological labels ---
    # Capture hardware channel names before relabeling for the session log
    channels_found_raw = [r["channel"] for r in session_results]

    try:
        photometry_log = load_photometry_log()
        label_map = get_channels_for_session(session_info["mouse"], session_info["date"], photometry_log)
    except Exception as e:
        print(f"  Warning: could not load photometry log for label mapping ({e}). Using hardware channel names.")
        label_map = {}

    for res in session_results:
        res["channel_raw"] = res["channel"]  # preserve hardware name
        res["channel"] = label_map.get(res["channel"], res["channel"])
        # Recompute tier now that we have the bio label (indicator type depends on it)
        tier, reasons = compute_quality_tier(
            res["qc_metrics"], res["note"], res["channel"], fs=res["fs"], dff=res["dff"]
        )
        res["qc_metrics"]["quality_tier"] = tier
        res["qc_metrics"]["tier_reasons"] = "|".join(reasons) if reasons else ""

    # Drop channels that have no bio label (still named G2, R0, etc.) — unmapped hardware
    # channels have no meaningful signal, so there's nothing to analyse or QC.
    session_results = [r for r in session_results if not re.match(r'^[RG]\d+$', r["channel"])]

    # --- Save outputs (long-form + per-channel) ---
    if session_results:
        # Long-form combined
        long_dfs = [make_longform_channel_df(session_info, r) for r in session_results]
        long_df = pd.concat(long_dfs, ignore_index=True)
        long_df.to_csv(session_outdir / "photometry_long.csv", index=False)

        # Per-channel files + summary
        summary_rows = []
        qc_rows = []
        for r, df_ch in zip(session_results, long_dfs):
            ch = r["channel"]
            df_ch.to_csv(session_outdir / f"{ch}.csv", index=False)
            plot_channel_qc(r, qc_dir, session_info["session_id"])
            summary_rows.append({
                "channel": ch,
                "note": r.get("note", ""),
                "fs_hz": r.get("fs", np.nan),
                "n_samples": len(r.get("sig", [])),
                "duration_min": duration_min,
            })
            qc = r.get("qc_metrics", {})
            qc_rows.append({
                "session_id": session_info["session_id"],
                "mouse": session_info["mouse"],
                "date": session_info["date"],
                "channel": ch,
                "note": r.get("note", ""),
                "duration_min": duration_min,
                **qc,
            })

        pd.DataFrame(summary_rows).to_csv(session_outdir / "channel_summary.csv", index=False)

        qc_df = pd.DataFrame(qc_rows)
        qc_df.to_csv(session_outdir / "correction_quality.csv", index=False)

        print(f"Saved session outputs to: {session_outdir}")

    # --- Write to comprehensive pipeline session log ---
    bio_labels_str = ",".join(sorted([r["channel"] for r in session_results]))
    channels_found_str = ",".join(sorted(channels_found_raw))
    update_session_log(session_info["session_id"], {
        "session_id":             session_info["session_id"],
        "mouse":                  session_info["mouse"],
        "date":                   session_info["date"],
        "FP_file":                session_info["FP_file"],
        "Behav_file":             session_info.get("Behav_file", float("nan")),
        "fp_behav_matched":       True,
        "duration_min":           duration_min,
        "channels_found":         channels_found_str,
        "photometry_log_matched": bool(label_map),
        "bio_labels":             bio_labels_str,
    })


def qc_gate_raw_deinterleaved(fp_df, channels, outdir, tag):
    """Run basic sanity checks on raw deinterleaved traces (sample counts, timestamp monotonicity).

    Returns (ok: bool, issues: list[str]).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    issues = []
    for ch in channels:
        try:
            iso_state, sig_state, note = get_channel_config(ch)
        except Exception as e:
            issues.append(f"{ch}: config error {e}")
            continue

        # Extract raw streams
        ts_sig, sig = extract_channel_data(fp_df, ch, sig_state)
        if len(sig) < QC_MIN_SAMPLES_PER_STATE:
            issues.append(f"{ch}: too few signal samples for state {sig_state} ({len(sig)})")

        if len(ts_sig) > 2 and np.any(np.diff(ts_sig) <= 0):
            issues.append(f"{ch}: non-monotonic signal timestamps")

        ts_iso, iso = np.array([]), np.array([])
        if iso_state is not None:
            ts_iso, iso = extract_channel_data(fp_df, ch, iso_state)
            if len(iso) < QC_MIN_SAMPLES_PER_STATE:
                issues.append(f"{ch}: too few iso samples for state {iso_state} ({len(iso)})")
            if len(ts_iso) > 2 and np.any(np.diff(ts_iso) <= 0):
                issues.append(f"{ch}: non-monotonic iso timestamps")


    ok = len(issues) == 0
    # Save issues log
    if issues:
        (outdir / f"{tag}_raw_qc_issues.txt").write_text("\n".join(issues))

    return ok, issues


def plot_channel_qc(res, outdir, session_id, max_points=150000):
    """Save a 4-panel QC plot for one channel alongside its CSV in the session output dir."""
    ch  = res["channel"]
    qc  = res.get("qc_metrics", {})
    tier        = qc.get("quality_tier", "?")
    tier_reasons = qc.get("tier_reasons", "") or ""
    tier_color  = {"A": "#27ae60", "B": "#e67e22", "C": "#e74c3c"}.get(tier, "#888888")

    n    = len(res["sig"])
    step = max(1, n // max_points)
    t       = res["ts_sig"][::step] - res["ts_sig"][0]
    sig     = res["sig"][::step]
    base    = res["scaled_bleach"][::step]
    dff_z   = res["dff_zscored"][::step]
    iso_raw = res.get("iso", np.array([]))
    iso     = iso_raw[::step] if len(iso_raw) == n else np.array([])

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), constrained_layout=True)

    tier_label = f"Tier {tier}" + (f"  |  {tier_reasons}" if tier_reasons else "")
    fig.suptitle(
        f"{session_id}  |  {ch}  |  {tier_label}",
        fontsize=12, fontweight="bold", color=tier_color,
        bbox=dict(facecolor="white", edgecolor=tier_color, linewidth=2, boxstyle="round,pad=0.4"),
    )

    if len(iso) == len(sig) and len(iso) > 0:
        axes[0].plot(t, iso, color="violet")
        axes[0].set_title(f"{ch} - Raw Isosbestic (415 nm)")
    else:
        axes[0].plot(t, sig, color="black", alpha=0.7)
        axes[0].set_title(f"{ch} - Raw Signal (no iso)")

    axes[1].plot(t, sig,  label="Raw Signal",      color=get_channel_color(ch), alpha=0.8)
    axes[1].plot(t, base, label="Scaled Baseline", color="black", linestyle="--")
    axes[1].set_title(f"{ch} - Signal vs Baseline")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, dff_z, color="steelblue", lw=1)
    axes[2].set_title(f"{ch} - dF/F Z-scored")
    axes[2].set_ylabel("Z-score")
    axes[2].axhline(0, color="gray", linestyle="--", alpha=0.5)
    if qc:
        metrics_str = (
            f"R²={qc.get('r2_baseline', float('nan')):.2f}   "
            f"drift_z={qc.get('drift_delta_z', float('nan')):.2f}   "
            f"r_iso_dff={qc.get('r_iso_dff', float('nan')):.3f}   "
            f"skew={qc.get('skewness', float('nan')):.2f}   "
            f"outlier_frac={qc.get('outlier_frac', float('nan')):.4f}"
        )
        axes[2].text(
            0.01, 0.97, metrics_str,
            transform=axes[2].transAxes, fontsize=7.5, va="top", ha="left",
            bbox=dict(facecolor="lightyellow", alpha=0.85, boxstyle="round,pad=0.3"),
        )

    axes[3].hist(res["dff_zscored"], bins=100, color="gray", alpha=0.7)
    axes[3].set_title(f"{ch} - dF/F Z-scored Distribution")
    axes[3].set_xlabel("Z-score")
    axes[3].set_ylabel("Count")

    fig.savefig(Path(outdir) / f"{ch}_qc.png", dpi=150)
    plt.close(fig)

# --- 5. Main Execution ---

def main():
    print("Starting FP Processing...")

    PROCESSED_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Match FP sessions with their Behav files
    sessions_df = match_sessions(FP_DIR)

    if sessions_df.empty:
        print("No matched FP+Behav sessions found. Exiting.")
        return

    print(f"Found {len(sessions_df)} sessions to process.")

    # 2. Process sessions
    for idx, session in sessions_df.iterrows():
        print(f"Processing {idx+1}/{len(sessions_df)}: {session['session_id']}")
        process_session(session, PROCESSED_OUT, regenerate=REGENERATE)

    print("All done.")

    # 3. Aggregate correction quality across all sessions
    aggregate_correction_quality(PROCESSED_OUT)


def aggregate_correction_quality(output_dir):
    """
    Collect per-session correction_quality.csv files into a single summary table.
    Writes all_sessions_correction_quality.csv to output_dir root and prints flagged channels.
    """
    output_dir = Path(output_dir)
    qc_files = sorted(output_dir.glob("*/correction_quality.csv"))

    if not qc_files:
        print("No correction_quality.csv files found; skipping aggregation.")
        return

    all_dfs = [pd.read_csv(f) for f in qc_files]
    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = output_dir / "all_sessions_correction_quality.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nCorrection quality summary: {len(combined)} channels across {len(qc_files)} sessions → {out_path}")


if __name__ == "__main__":
    main()


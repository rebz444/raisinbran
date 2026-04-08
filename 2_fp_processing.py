
import shutil
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from datetime import datetime
from pathlib import Path

from config import (
    PHOTO_ROOT,
    FP_DIR,
    FP_DIR_FLATTENED,
    PROCESSED_OUT,
    QC_GATE_ENABLED,
    QC_STOP_ON_FAIL,
    QC_MIN_SAMPLES_PER_STATE,
)

# --- Configuration ---
PHOTOMETRY_DIR = PHOTO_ROOT
OUT_DIR = PROCESSED_OUT

REGENERATE = False  # Set to True to reprocess already-processed sessions

# Ensure directories exist
FP_DIR_FLATTENED.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
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


# --- 2. Data Handling & Utilities ---

def flatten_directory(source_dir, target_dir):
    """
    Copies all .csv files from source_dir (recursive) to target_dir (flat).
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    copied = 0
    for file_path in source_dir.rglob("*.csv"):
        target_path = target_dir / file_path.name
        if not target_path.exists():
            shutil.copy2(file_path, target_path)
            copied += 1
            
    if copied > 0:
        print(f"Flattening complete. Copied {copied} new files.")

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
    Matches FP and Behavior files based on Mouse ID and Timestamp (within 60s).
    """
    data_dir = Path(data_dir)
    files = [f.name for f in data_dir.glob("*.csv")]
    
    parsed = []
    for f in files:
        info = get_session_info(f)
        if info:
            parsed.append(info)
            
    fp_files = [x for x in parsed if 'FP' in x['type']]
    behav_files = [x for x in parsed if 'Behav' in x['type']]
    
    sessions = []
    
    for fp in fp_files:
        # Find behavior files for same mouse
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
                'mouse': fp['mouse'],
                'date': fp['datetime'].strftime('%Y-%m-%d'),
                'time': fp['datetime'].strftime('%H:%M:%S'),
                'FP_file': fp['filename'],
                'Behav_file': best_match['filename'],
                'session_id': f"{fp['mouse']}_{fp['datetime'].strftime('%Y%m%d_%H%M%S')}"
            })
        else:
            print(f"Unmatched FP session: {fp['filename']}")

    df = pd.DataFrame(sessions)
    if not df.empty:
        df.sort_values(['date', 'time'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        # Save matched list for reference
        df.to_csv(PHOTOMETRY_DIR / "matched_sessions.csv", index=False)
        
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


# --- 3. Core Processing ---

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
            scaled_bleach, normalized, dff = processor.correct_signal(sig, bleach_iso)
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

        normalized = sig / scaled_bleach
        dff = normalized - 1.0

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

    return {
        "channel": channel_name,
        "note": note,
        "ts_iso": ts_iso,
        "ts_sig": ts_sig,
        "iso": iso,
        "sig": sig,
        "scaled_bleach": scaled_bleach,
        "normalized": normalized,
        "dff": dff,
        "dff_filtered": dff_filtered,
        "dff_zscored": dff_zscored,
        "fs": processor.compute_fs(ts_sig)
    }

def process_session(session_info, input_dir, output_dir, regenerate=False):
    """
    Loads data for a session and processes all ROI channels.
    If regenerate=False, skips processing if output already exists.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    session_outdir = output_dir / session_info['session_id']

    if not regenerate and (session_outdir / "photometry_long.csv").exists():
        print(f"  Skipping {session_info['session_id']} (already processed; use regenerate=True to rerun)")
        return

    session_outdir.mkdir(parents=True, exist_ok=True)

    fp_path = input_dir / session_info['FP_file']
    behav_path = input_dir / session_info['Behav_file']
    
    fp = pd.read_csv(fp_path)     
    fp = fp[fp["LedState"] != 7].copy() # 7 is often 'unknown' or 'inter-trial'

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
        else:
            fp = fp[fp["SystemTimestamp"].between(t_start, t_end)]
            fp["SystemTimestamp"] -= t_start
    except Exception as e:
        print(f"Warning: Could not crop to behavior ({e}). Using full FP file.")

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

    # --- Save outputs (long-form + per-channel) ---
    if session_results:
        # Long-form combined
        long_dfs = [make_longform_channel_df(session_info, r) for r in session_results]
        long_df = pd.concat(long_dfs, ignore_index=True)
        long_df.to_csv(session_outdir / "photometry_long.csv", index=False)

        # Per-channel files + summary
        summary_rows = []
        for r in session_results:
            ch = r["channel"]
            df_ch = make_longform_channel_df(session_info, r)
            df_ch.to_csv(session_outdir / f"{ch}.csv", index=False)
            summary_rows.append({
                "channel": ch,
                "note": r.get("note", ""),
                "fs_hz": r.get("fs", np.nan),
                "n_samples": len(r.get("sig", [])),
            })

        pd.DataFrame(summary_rows).to_csv(session_outdir / "channel_summary.csv", index=False)
        print(f"Saved session outputs to: {session_outdir}")

        # Standard QC plots (baseline/dff, etc.)
        plot_qc_session(session_results, qc_dir, tag)


# --- 4. Plotting ---

def qc_gate_raw_deinterleaved(fp_df, channels, outdir, tag):
    """Plot raw deinterleaved traces and run basic sanity checks.

    Returns (ok: bool, issues: list[str]).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    issues = []
    # Quick checks + plots
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

        # Plot (downsample)
        max_points = 150000
        def _ds(ts, x):
            if len(x) == 0:
                return np.array([]), np.array([])
            step = max(1, len(x) // max_points)
            t = ts[::step] - ts[0]
            return t, x[::step]

        t_sig, sig_ds = _ds(ts_sig, sig)
        t_iso, iso_ds = _ds(ts_iso, iso)

        fig, ax = plt.subplots(figsize=(10, 4))
        if len(sig_ds):
            ax.plot(t_sig, sig_ds, label=f"sig state {sig_state}", alpha=0.8)
        if len(iso_ds):
            ax.plot(t_iso, iso_ds, label=f"iso state {iso_state}", alpha=0.8)
        ax.set_title(f"{tag} {ch} - Raw deinterleaved")
        ax.set_xlabel("time (s, relative)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{tag}_{ch}_raw_deinterleaved.png", dpi=150)
        plt.close(fig)

    ok = len(issues) == 0
    # Save issues log
    if issues:
        (outdir / f"{tag}_raw_qc_issues.txt").write_text("\n".join(issues))

    return ok, issues

def plot_qc_session(roi_results, outdir, tag):
    """Generates QC plots for all processed ROIs in a session."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. Individual ROI Plots
    for res in roi_results:
        plot_single_roi(res, outdir, tag)
        
    # 2. Multi-Green Summary (if applicable)
    greens = [r for r in roi_results if r['channel'].startswith('G')]
    if len(greens) > 1:
        plot_multi_green(greens, outdir, tag)

def plot_single_roi(data, outdir, tag, max_points=150000):
    ch = data["channel"]
    
    # Downsample for big plots
    n = len(data["sig"])
    step = max(1, n // max_points)
    
    t = (data["ts_sig"][::step] - data["ts_sig"][0])
    iso = data.get("iso", np.array([]))[::step]
    sig = data["sig"][::step]
    base = data["scaled_bleach"][::step]
    dff_zscored = data["dff_zscored"][::step]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)

    # Raw deinterleaved reference trace (iso for green; signal for red-only)
    if iso is not None and len(iso) == len(sig) and len(iso) > 0:
        axes[0].plot(t, iso, color='violet')
        axes[0].set_title(f"{tag} {ch} - Raw Isosbestic (415nm)")
    else:
        axes[0].plot(t, sig, color='black', alpha=0.7)
        axes[0].set_title(f"{tag} {ch} - Raw Signal (no iso)")
    
    # Raw Signal + Fit
    axes[1].plot(t, sig, label="Raw Signal", color='green' if 'G' in ch else 'red', alpha=0.8)
    axes[1].plot(t, base, label="Scaled Iso/Baseline", color='black', linestyle='--')
    axes[1].set_title(f"{tag} {ch} - Signal vs Baseline")
    axes[1].legend()
    
    # dF/F Z-scored
    axes[2].plot(t, dff_zscored, color='blue', lw=1)
    axes[2].set_title(f"{tag} {ch} - dF/F Z-scored")
    axes[2].set_ylabel("Z-score")
    axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Histogram of z-scored data
    axes[3].hist(data["dff_zscored"], bins=100, color='gray', alpha=0.7)
    axes[3].set_title(f"{tag} {ch} - dF/F Z-scored Distribution")
    
    fig.savefig(outdir / f"{tag}_{ch}_qc.png", dpi=150)
    plt.close(fig)

def plot_multi_green(greens, outdir, tag):
    """Stacked traces for green channels to check correlation."""
    # Align lengths
    n = min(len(g["dff"]) for g in greens)
    dffs = np.vstack([g["dff"][:n] for g in greens])
    labels = [g["channel"] for g in greens]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (trace, label) in enumerate(zip(dffs, labels)):
        ax.plot(trace + i*0.5, label=label)
        
    ax.set_title(f"{tag} - Green Channels Stacked")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_multi_green.png", dpi=150)
    plt.close(fig)


# --- 5. Main Execution ---

def main():
    print("Starting FP Processing...")
    
    # 1. Flatten Data Structure
    flatten_directory(FP_DIR, FP_DIR_FLATTENED)
    
    # 2. Match Sessions
    sessions_df = match_sessions(FP_DIR_FLATTENED)
    
    if sessions_df.empty:
        print("No matched sessions found. Exiting.")
        return
        
    print(f"Found {len(sessions_df)} matched sessions.")
    
    # 3. Process Loops
    for idx, session in sessions_df.iterrows():
        print(f"Processing {idx+1}/{len(sessions_df)}: {session['session_id']}")
        process_session(session, FP_DIR_FLATTENED, OUT_DIR, regenerate=REGENERATE)
        
    print("All done.")

if __name__ == "__main__":
    main()
    
    
    
    

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.signal import butter, filtfilt


# -----------------------------
# Data Cleanup & Filtering
# -----------------------------

def check_monotonic(timestamps):
    """
    Checks if timestamps are monotonically increasing.
    """
    diffs = np.diff(timestamps)
    if np.any(diffs <= 0):
        print(f"Warning: {np.sum(diffs <= 0)} non-monotonic timestamps found.")
        return False
    return True


def detect_saturation(signal, low_thresh=0.001, high_thresh=3.3):
    """
    Detects saturated samples based on thresholds.
    Returns a boolean mask (True where saturated).
    """
    mask = (signal <= low_thresh) | (signal >= high_thresh)
    count = np.sum(mask)
    if count > 0:
        print(f"Warning: {count} samples saturated (<= {low_thresh} or >= {high_thresh}).")
    return mask


def lowpass_filter(data, fs, cutoff=10, order=2):
    """
    Applies a zero-phase Butterworth low-pass filter.
    """
    nyquist = 0.5 * fs
    if cutoff >= nyquist:
        print(f"Warning: Cutoff {cutoff}Hz >= Nyquist {nyquist}Hz. Filtering skipped.")
        return data
        
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def find_latest_files(mouse_id, idN=1, year_pattern="2025", base_path=None):
    """
    Replicates the MATLAB logic:
        behavFiles = dir('*Behav*mouseID*.csv');
        [~,index] = sortrows({behavFiles.date}.');
        behavFiles = behavFiles(index(end:-1:1));
        behavFileName = behavFiles(idN).name;

        fpFiles = dir('*FP*mouseID*2025*.csv');
        (same sorting and picking idN)

    Returns:
        behav_path, fp_path (Path objects)
    """
    if base_path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_path)

    # Find behavioral files
    behav_candidates = list(base_path.glob(f"*Behav*{mouse_id}*.csv"))
    if not behav_candidates:
        raise FileNotFoundError(f"No behavior files found for {mouse_id} in {base_path}")

    # Sort by modified time (most recent first)
    behav_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # MATLAB is 1-based, Python is 0-based -> idN-1
    behav_path = behav_candidates[idN - 1]

    # Find FP files (with year constraint)
    fp_candidates = list(base_path.glob(f"*FP*{mouse_id}*{year_pattern}*.csv"))
    if not fp_candidates:
        raise FileNotFoundError(f"No FP files found for {mouse_id} and year {year_pattern} in {base_path}")

    fp_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    fp_path = fp_candidates[idN - 1]

    print("Selected behavior file:", behav_path.name)
    print("Selected FP file      :", fp_path.name)
    return behav_path, fp_path


def read_behavior(behav_path):
    """
    MATLAB:

    opts.NumVariables = 5
    opts.VariableNames = ["Input1","VarName2","True","VarName4","VarName5"]
    behavData = readtable(...);
    tb = behavData.VarName4;

    Here we'll just load with pandas and use the 4th column as tb,
    which matches VarName4 in the MATLAB import.
    """
    df = pd.read_csv(behav_path)
    if df.shape[1] < 4:
        raise ValueError(f"Behavior file {behav_path} has fewer than 4 columns")

    # Equivalent of behavData.VarName4
    tb = df.iloc[:, 3].to_numpy()
    return df, tb


def read_fp(fp_path):
    """
    MATLAB FP import:

    VariableNames = ["FrameCounter","SystemTimestamp","LedState",
                     "ComputerTimestamp","R0","R1","G2","G3"]

    We replicate those names here and then access the same columns.
    """
    df = pd.read_csv(fp_path, skiprows=1, header=None)  # DataLines = [2, Inf] in MATLAB
    df.columns = [
        "FrameCounter",
        "SystemTimestamp",
        "LedState",
        "ComputerTimestamp",
        "R0",
        "R1",
        "G2",
        "G3",
    ]
    return df


def assign_fp_channels(fp_df):
    """
    Direct translation of:

    isoSig(:,1)   = fpData.G2(2:3:end);
    isoSig(:,2)   = fpData.G3(2:3:end);
    isoSig(:,3)   = fpData.R0(2:3:end);
    isoSig(:,4)   = fpData.R1(2:3:end);
    greenSig(:,1) = fpData.G2(3:3:end);
    greenSig(:,2) = fpData.G3(3:3:end);
    redSig(:,1)   = fpData.R0(4:3:end);
    redSig(:,2)   = fpData.R1(4:3:end);

    MATLAB indexing is 1-based, Python 0-based so:
        2:3:end -> start index 1 -> [1::3]
        3:3:end -> start index 2 -> [2::3]
        4:3:end -> start index 3 -> [3::3]
    """
    G2 = fp_df["G2"].to_numpy()
    G3 = fp_df["G3"].to_numpy()
    R0 = fp_df["R0"].to_numpy()
    R1 = fp_df["R1"].to_numpy()

    isoSig = np.column_stack([
        G2[1::3],
        G3[1::3],
        R0[1::3],
        R1[1::3],
    ])

    greenSig = np.column_stack([
        G2[2::3],
        G3[2::3],
    ])

    redSig = np.column_stack([
        R0[3::3],
        R1[3::3],
    ])

    return isoSig, greenSig, redSig


def align_time(fp_df, isoSig, greenSig, redSig, tb):
    """
    MATLAB:

    tf = fpData.SystemTimestamp(2:3:end);
    tb = behavData.VarName4;

    [minValue, minX] = min(abs(tf - tb(1)));
    [maxValue, maxX] = min(abs(tf - tb(end)));

    Then slice minX:maxX for tf and all signals,
    and define tf0 = tf - min(tf).

    We'll do the same.
    """
    SystemTimestamp = fp_df["SystemTimestamp"].to_numpy()
    tf = SystemTimestamp[1::3]  # 2:3:end in MATLAB

    # Find closest indices
    minX = np.argmin(np.abs(tf - tb[0]))
    maxX = np.argmin(np.abs(tf - tb[-1]))

    if maxX < minX:
        minX, maxX = maxX, minX

    tf = tf[minX:maxX]
    tf0 = tf - tf[0]  # relative time starting at 0

    isoSig = isoSig[minX:maxX, :]
    greenSig = greenSig[minX:maxX, :]
    redSig = redSig[minX:maxX, :]

    return tf0, isoSig, greenSig, redSig


def get_dff(iso_sig, signal, plot_flag=0, method='isosbestic_division'):
    """
    Calculates dF/F using isosbestic correction.
    
    Args:
        iso_sig (array): Isosbestic control signal
        signal (array): Calcium-dependent signal
        plot_flag (int): If 1, generate plot (not implemented in this minimal version)
        method (str): 'isosbestic_subtraction' (original) or 'isosbestic_division' (field standard)
    
    Returns:
        dff (array): Calculated dF/F
    """
    iso_sig = np.asarray(iso_sig)
    signal = np.asarray(signal)

    # 1) Fit Isosbestic to Signal: signal_hat = a * iso + b
    # Uses robust regression logic if available, otherwise OLS
    A = np.vstack([iso_sig, np.ones_like(iso_sig)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, signal, rcond=None)
    a, b = coeffs
    signal_hat = a * iso_sig + b

    # 2) Calculate dF/F
    if method == 'isosbestic_subtraction':
        # (Signal - Fit) / F0_constant
        corrected = signal - signal_hat
        F0 = np.percentile(corrected, 10)
        if F0 == 0: F0 = 1.0 # Protect div zero
        dff = (corrected - F0) / F0
        
    elif method == 'isosbestic_division':
        # (Signal - Fit) / Fit
        # This accounts for baseline drift in the denominator too (photobleaching)
        # Avoid divide by zero
        denom = signal_hat.copy()
        denom[denom == 0] = np.mean(denom) 
        dff = (signal - signal_hat) / denom
        
    else:
        raise ValueError(f"Unknown dFF method: {method}")

    return dff


def compute_all_dff(isoSig, greenSig, redSig):
    """
    MATLAB:

    [dFF(:,1)] = getdFF(isoSig(:,1), greenSig(:,1),0); % right branch
    [dFF(:,2)] = getdFF(isoSig(:,2), greenSig(:,2),0); % left branch
    [dFF(:,3)] = getdFF(isoSig(:,3), redSig(:,1),0);   % right branch
    [dFF(:,4)] = getdFF(isoSig(:,4), redSig(:,2),0);   % left branch
    """
    dFF1 = get_dff(isoSig[:, 0], greenSig[:, 0], 0)
    dFF2 = get_dff(isoSig[:, 1], greenSig[:, 1], 0)
    dFF3 = get_dff(isoSig[:, 2], redSig[:, 0], 0)
    dFF4 = get_dff(isoSig[:, 3], redSig[:, 1], 0)

    dFF = np.column_stack([dFF1, dFF2, dFF3, dFF4])
    return dFF


def plot_dff(tf0, dFF, behav_file_name, chan_names, brain_regions=None, offset=0.2):
    """
    Replicates:

    nChannels = size(dFF,2);
    offset = 0.2;

    for ch = 1:nChannels
        plot(tf0, dFF(:,ch) + ch*offset, ...)
    yticks((1:nChannels)*offset);
    yticklabels(brainRegions);

    If brainRegions is None, we’ll label y-axis with chan_names instead.
    """
    tf0 = np.asarray(tf0)
    dFF = np.asarray(dFF)
    nChannels = dFF.shape[1]

    if brain_regions is None:
        brain_regions = chan_names

    if len(brain_regions) != nChannels:
        raise ValueError("brain_regions / chan_names length mismatch with dFF channels.")

    fig, ax = plt.subplots(figsize=(10, 6))
    # use default color cycle
    for ch in range(nChannels):
        ax.plot(tf0, dFF[:, ch] + (ch + 1) * offset, linewidth=1.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F (a.u.)")
    # behavFileName(7:end-4) in MATLAB = drop first 6 chars and last 4 (.csv)
    stem = behav_file_name
    if stem.endswith(".csv"):
        stem = stem[:-4]
    if len(stem) > 6:
        title_str = stem[6:]
    else:
        title_str = stem
    ax.set_title(title_str)

    yticks = [(i + 1) * offset for i in range(nChannels)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(brain_regions)

    ax.grid(True)
    plt.tight_layout()
    return fig, ax


def save_outputs(tf0, dFF, chan_names, fp_path, behav_path, out_dir=None):
    """
    MATLAB:

    T = array2table([tf0', dFF], 'VariableNames', [{'time'}, chanNames]);
    writetable(T, ['dFF_', fpFileName]);
    save([behavFileName(7:end-4), '.mat']);

    We’ll do:

        CSV name: dFF_<fpFileName>
        MAT name: <behavFileName(7:end-4)>.mat
    """
    if out_dir is None:
        out_dir = behav_path.parent
    out_dir = Path(out_dir)

    # CSV
    data = np.column_stack([tf0, dFF])
    columns = ["time"] + list(chan_names)
    df_out = pd.DataFrame(data, columns=columns)
    csv_name = out_dir / f"dFF_{fp_path.name}"
    df_out.to_csv(csv_name, index=False)

    # MAT
    behav_name = behav_path.name
    if behav_name.endswith(".csv"):
        behav_stem = behav_name[:-4]
    else:
        behav_stem = behav_name

    if len(behav_stem) > 6:
        mat_stem = behav_stem[6:]
    else:
        mat_stem = behav_stem

    mat_name = out_dir / f"{mat_stem}.mat"
    savemat(mat_name, {
        "tf0": tf0,
        "dFF": dFF,
        "chanNames": np.array(chan_names, dtype=object),
    })

    print("Saved dF/F CSV to:", csv_name)
    print("Saved MAT to      :", mat_name)


def preprocess_session(
    mouse_id="RZ075",
    idN=1,
    Fs=20.0,
    chan_names=("g1", "g2", "r1", "r2"),
    brain_regions=None,
    year_pattern="2025",
    base_path=None,
):
    """
    High-level wrapper: replicates your MATLAB script end-to-end.
    """
    # 1) Find latest files
    behav_path, fp_path = find_latest_files(
        mouse_id=mouse_id,
        idN=idN,
        year_pattern=year_pattern,
        base_path=base_path,
    )

    # 2) Read behavior
    behav_df, tb = read_behavior(behav_path)

    # 3) Read FP
    fp_df = read_fp(fp_path)

    # 4) Assign channels
    isoSig, greenSig, redSig = assign_fp_channels(fp_df)

    # --- Data Cleanup & Filtering ---
    # Check monotonicity of timestamps (using SystemTimestamp from raw df)
    check_monotonic(fp_df["SystemTimestamp"].to_numpy())

    # Detect saturation (example check on raw G2 channel)
    detect_saturation(fp_df["G2"].to_numpy())

    # Filter signals (using 10Hz cutoff by default)
    # Filter each column of the de-interleaved signals
    for sig_array in [isoSig, greenSig, redSig]:
        for col in range(sig_array.shape[1]):
            sig_array[:, col] = lowpass_filter(sig_array[:, col], fs=Fs, cutoff=10)
    # --------------------------------

    # 5) Align FP & behavior in time (get tf0, cut signals)
    tf0, isoSig, greenSig, redSig = align_time(fp_df, isoSig, greenSig, redSig, tb)

    # 6) Compute dFF for all 4 channels
    dFF = compute_all_dff(isoSig, greenSig, redSig)

    # 7) Plot
    fig, ax = plot_dff(
        tf0,
        dFF,
        behav_file_name=behav_path.name,
        chan_names=chan_names,
        brain_regions=brain_regions if brain_regions is not None else chan_names,
        offset=0.2,
    )

    # Save figure like MATLAB: [behavFileName(7:end-4), '_DFF_','.png']
    behav_name = behav_path.name
    if behav_name.endswith(".csv"):
        behav_stem = behav_name[:-4]
    else:
        behav_stem = behav_name

    if len(behav_stem) > 6:
        png_stem = behav_stem[6:]
    else:
        png_stem = behav_stem

    png_name = behav_path.parent / f"{png_stem}_DFF_.png"
    fig.savefig(png_name, dpi=300)
    plt.close(fig)
    print("Saved figure to    :", png_name)

    # 8) Save CSV + MAT
    save_outputs(tf0, dFF, chan_names, fp_path, behav_path, out_dir=behav_path.parent)


if __name__ == "__main__":
    # Example usage: run from the folder with your CSVs
    # and adjust mouse_id, year_pattern, etc. if needed.
    preprocess_session(
        mouse_id="RZ083",
        idN=1,
        Fs=20.0,
        chan_names=("g1", "g2", "r1", "r2"),
        brain_regions=("gV1", "gSTR", "rV1", "rSTR"),
        year_pattern="2026",
        base_path=Path.cwd(),
    )
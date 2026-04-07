#!/usr/bin/env python3
"""
Batch process all sessions in merged log:
1) Load photometry_long.csv for each session
2) Load trials/events for each session
3) Merge photometry to trials (trial_time)
4) Save merged per-session signal file
5) Plot each trial and save PNGs
"""
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt

# ============================
# CONFIGURATION
# ============================
PHOTO_ROOT = Path("/Users/rebekahzhang/data/photometry")
PROCESSED_OUT = PHOTO_ROOT / "processed_output"
BEHAV_DIR = PHOTO_ROOT / "behav_analyzed"
MATCHED_SESSIONS_CSV = PHOTO_ROOT / "matched_sessions.csv"
BEHAV_LOG_CSV = BEHAV_DIR / "sessions_photometry_exp2.csv"
OUT_PLOTS_DIR = PHOTO_ROOT / "trial_plots"

PHOTOMETRY_LOG_URL = (
    "https://docs.google.com/spreadsheets/d/1B8KCnku1vQInKOFBuoLeuDND6CEUA4hT6qQ5zuhqvyU/"
    "export?format=csv&gid=1751471403"
)

SIGNAL_COL = "dff_zscored"  # or "dff_filtered"

# Channel pairs (red, green) per fiber number, keyed by total fiber count
_FIBER_CHANNEL_MAP = {
    2: {1: ("R0", "G2"), 2: ("R1", "G3")},
    4: {1: ("R0", "G4"), 2: ("R1", "G5"), 3: ("R2", "G6"), 4: ("R3", "G7")},
}

DEFAULT_CHANNEL_COLORS = {
    "R0": "#e74c3c",
    "R1": "#c0392b",
    "R2": "#ff6b35",
    "R3": "#ff006e",
    "G2": "#27ae60",
    "G3": "#4cc9f0",
    "G4": "#2ecc71",
    "G5": "#1abc9c",
    "G6": "#16a085",
    "G7": "#0e6655",
}

# ============================
# HELPERS
# ============================

def get_channels_for_session(mouse: str, date: str, photometry_log: pd.DataFrame) -> Dict[str, str]:
    """
    Look up the photometry log for a session and return a dict of {channel: label}.
    Label format: {channel}_{side}_{area}_{sensor}  e.g. "R0_l_v1_rCaMP"
    - str/striatum fibers: include both red and green channels
    - v1 fibers: include red channel only
    Returns empty dict if the session is not found in the log.
    """
    row = photometry_log[
        (photometry_log["mouse"] == mouse) & (photometry_log["date"] == date)
    ]
    if row.empty:
        return {}

    row = row.iloc[0]

    # Count active fibers (non-null area)
    n_fibers = sum(
        1 for i in range(1, 5)
        if pd.notna(row.get(f"fiber_{i}_area")) and str(row.get(f"fiber_{i}_area")).strip() != ""
    )

    if n_fibers not in _FIBER_CHANNEL_MAP:
        return {}

    channel_labels = {}
    for fiber_num, (r_ch, g_ch) in _FIBER_CHANNEL_MAP[n_fibers].items():
        area = str(row.get(f"fiber_{fiber_num}_area", "")).strip().lower()
        side = str(row.get(f"fiber_{fiber_num}_side", "")).strip().lower()
        sensor = str(row.get(f"fiber_{fiber_num}_sensor", "")).strip()
        if area in ("str", "striatum"):
            channel_labels[r_ch] = f"{r_ch}_{side}_{area}_{sensor}"
            channel_labels[g_ch] = f"{g_ch}_{side}_{area}_{sensor}"
        elif area == "v1":
            channel_labels[r_ch] = f"{r_ch}_{side}_{area}_{sensor}"

    return channel_labels


def assign_trials_to_photometry(phot: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Align photometry to trials and compute per-sample trial time."""
    trials = trials.copy()
    session_first_start = float(trials.iloc[0]["start_time"])
    trials["start_time"] = trials["start_time"].astype(float) - session_first_start
    trials["end_time"] = trials["end_time"].astype(float) - session_first_start

    phot_sorted = phot.sort_values("t_sec").reset_index(drop=True)
    trials_sorted = trials.sort_values("start_time").reset_index(drop=True)

    merged = pd.merge_asof(
        phot_sorted,
        trials_sorted,
        left_on="t_sec",
        right_on="start_time",
        direction="backward",
    )

    merged = merged[merged["t_sec"] <= merged["end_time"]].reset_index(drop=True)
    merged["trial_time"] = merged["t_sec"] - merged["start_time"]
    merged["decision_time"] = merged["bg_length"] + merged["time_waited"]
    return merged


def plot_trial(
    trial_sig: pd.DataFrame,
    trial_events: pd.DataFrame,
    signal_col: str,
    channel_labels: Dict[str, str],
    title: str,
    save_path: Path,
) -> None:
    channel_colors = DEFAULT_CHANNEL_COLORS

    fig = plt.figure(figsize=(14, 6))

    for ch, label in channel_labels.items():
        ch_df = trial_sig[trial_sig["channel"] == ch]
        if ch_df.empty:
            continue
        plt.plot(
            ch_df["trial_time"],
            ch_df[signal_col],
            label=label,
            linewidth=1.4,
            alpha=0.85,
            color=channel_colors.get(ch, None),
        )

    # Event markers
    trial_data = trial_sig.iloc[0]
    plt.axvline(0, color="green", linestyle="--", linewidth=2, alpha=0.7, label="BG start")
    plt.axvline(trial_data["bg_length"], color="orange", linestyle="--", linewidth=2, alpha=0.7, label="WAIT start")
    if not trial_data["miss_trial"]:
        if trial_data["reward"] > 0:
            plt.axvline(trial_data["decision_time"], color="blue", linestyle="--", linewidth=2, alpha=0.7, label="Decision")
        else:
            plt.axvline(trial_data["decision_time"], color="red", linestyle="--", linewidth=2, alpha=0.7, label="Decision")

    if "key" in trial_events.columns:
        lick_events = trial_events[trial_events["key"] == "lick"]
        if len(lick_events) > 0:
            lick_times = lick_events["trial_time"].values
            for i, t in enumerate(lick_times):
                plt.axvline(t, color="grey", linestyle="-", linewidth=1, alpha=0.25, label="Licks" if i == 0 else "")

    plt.xlabel("Time (s)")
    plt.ylabel(signal_col)
    plt.title(title)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_merged_log() -> pd.DataFrame:
    data_log = pd.read_csv(MATCHED_SESSIONS_CSV)
    behav_log = pd.read_csv(BEHAV_LOG_CSV, index_col=0)
    merged_log = pd.merge(data_log, behav_log, on=["mouse", "date"], how="inner")
    merged_log.to_csv(PHOTO_ROOT / "sessions_dff_behav_merged.csv")
    return merged_log


def process(regenerate: bool = False) -> None:
    """Merge photometry with trial data and save photometry_with_trial_data.csv per session.
    If regenerate=False, skips sessions where the output already exists.
    """
    merged_log = _load_merged_log()

    for _, session_info in merged_log.iterrows():
        session_id = str(session_info["session_id"])
        behav_subdir = str(session_info["dir"])

        merged_out = PROCESSED_OUT / session_id / "photometry_with_trial_data.csv"
        if not regenerate and merged_out.exists():
            print(f"  Skipping {session_id} (already processed; use regenerate=True to rerun)")
            continue

        phot_path = PROCESSED_OUT / session_id / "photometry_long.csv"
        trials_path = BEHAV_DIR / behav_subdir / f"trials_analyzed_{behav_subdir}.csv"

        if not phot_path.exists() or not trials_path.exists():
            print(f"Skipping {session_id}: missing photometry or trials")
            continue

        phot = pd.read_csv(phot_path)
        trials = pd.read_csv(trials_path, index_col=0)

        merged = assign_trials_to_photometry(phot, trials)
        merged.to_csv(merged_out, index=False)
        print(f"Saved merged data: {session_id}")

    print("Processing done.")


def plot() -> None:
    """Plot per-trial photometry traces using saved photometry_with_trial_data.csv."""
    OUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    merged_log = _load_merged_log()

    photometry_log = pd.read_csv(PHOTOMETRY_LOG_URL)
    photometry_log["date"] = pd.to_datetime(photometry_log["date"]).dt.strftime("%Y-%m-%d")

    for _, session_info in merged_log.iterrows():
        session_id = str(session_info["session_id"])
        behav_subdir = str(session_info["dir"])

        merged_path = PROCESSED_OUT / session_id / "photometry_with_trial_data.csv"
        events_path = BEHAV_DIR / behav_subdir / f"events_processed_{behav_subdir}.csv"

        if not merged_path.exists() or not events_path.exists():
            print(f"Skipping {session_id}: run process() first or missing events")
            continue

        merged = pd.read_csv(merged_path)
        events = pd.read_csv(events_path, index_col=0)

        signal_col = SIGNAL_COL if SIGNAL_COL in merged.columns else "dff_filtered"

        channel_labels = get_channels_for_session(session_info["mouse"], session_info["date"], photometry_log)
        if not channel_labels:
            channel_labels = {ch: ch for ch in sorted(merged["channel"].unique().tolist())}
            print(f"  [{session_id}] No log entry found; plotting all available channels: {list(channel_labels)}")

        trial_nums = sorted(merged["session_trial_num"].dropna().unique())
        session_plot_dir = OUT_PLOTS_DIR / f"{session_info['mouse']}_{session_info['date']}"

        for trial_num in trial_nums:
            trial_sig = merged[merged["session_trial_num"] == trial_num]
            trial_events = events[events["session_trial_num"] == trial_num]
            if trial_sig.empty or trial_events.empty:
                continue

            save_path = session_plot_dir / f"trial_{int(trial_num):03d}.png"
            title = f"{session_info['mouse']} {session_info['date']} - Trial {int(trial_num)}"
            plot_trial(
                trial_sig=trial_sig,
                trial_events=trial_events,
                signal_col=signal_col,
                channel_labels=channel_labels,
                title=title,
                save_path=save_path,
            )

        print(f"Done plotting session {session_id}")

    print("Plotting done.")


if __name__ == "__main__":
    # process(regenerate=False)
    plot()

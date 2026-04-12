#!/usr/bin/env python3
"""
Step 4: Per-trial photometry check plots
=========================================
Generates per-trial PNG plots using photometry_with_trial_data.csv produced
by 3_fp_behavior_align.py.

Each plot shows the dF/F trace with event markers:
  - Background start (t=0, green)
  - Wait period start (orange)
  - Decision (blue = rewarded, red = unrewarded)
  - Individual licks (grey)

Run after 3_fp_behavior_align.py.
"""
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt

from config import (
    BEHAV_DIR,
    get_channel_color,
    DEFAULT_SIGNAL_COL,
    OUT_PLOTS_DIR,
    PROCESSED_OUT,
)
from utils import get_channels_for_session, load_merged_log, load_photometry_log

SIGNAL_COL = DEFAULT_SIGNAL_COL


def plot_trial(
    trial_sig: pd.DataFrame,
    trial_events: pd.DataFrame,
    signal_col: str,
    channel_labels: Dict[str, str],
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    for _, label in channel_labels.items():
        ch_df = trial_sig[trial_sig["channel"] == label]
        if ch_df.empty:
            continue
        ax.plot(
            ch_df["trial_time"],
            ch_df[signal_col],
            label=label,
            linewidth=1.4,
            alpha=0.85,
            color=get_channel_color(label),
        )

    # Event markers
    trial_data = trial_sig.iloc[0]
    ax.axvline(0, color="green", linestyle="--", linewidth=2, alpha=0.7, label="BG start")
    ax.axvline(trial_data["bg_length"], color="orange", linestyle="--", linewidth=2, alpha=0.7, label="WAIT start")
    if not trial_data["miss_trial"]:
        if trial_data["reward"] > 0:
            ax.axvline(trial_data["decision_time"], color="blue", linestyle="--", linewidth=2, alpha=0.7, label="Decision")
        else:
            ax.axvline(trial_data["decision_time"], color="red", linestyle="--", linewidth=2, alpha=0.7, label="Decision")

    if "key" in trial_events.columns:
        lick_events = trial_events[trial_events["key"] == "lick"]
        if len(lick_events) > 0:
            lick_times = lick_events["trial_time"].values
            for i, t in enumerate(lick_times):
                ax.axvline(t, color="grey", linestyle="-", linewidth=1, alpha=0.25, label="Licks" if i == 0 else "")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(signal_col)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot(regenerate: bool = False) -> None:
    """Plot per-trial photometry traces using saved photometry_with_trial_data.csv.
    If regenerate=False, skips trials where the plot already exists.
    """
    OUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    merged_log = load_merged_log()

    photometry_log = load_photometry_log()

    for _, session_info in merged_log.iterrows():
        session_id = str(session_info["session_id"])
        behav_subdir = str(session_info["behav_dir"])

        merged_path = PROCESSED_OUT / session_id / "photometry_with_trial_data.csv"
        events_path = BEHAV_DIR / behav_subdir / f"events_processed_{behav_subdir}.csv"

        if not merged_path.exists() or not events_path.exists():
            print(f"Skipping {session_id}: run process() first or missing events")
            continue

        merged = pd.read_csv(merged_path, low_memory=False)
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
            if not regenerate and save_path.exists():
                continue
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
    plot(regenerate=False)

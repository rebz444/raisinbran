"""
Step 3: Behavior Data Sourcing + FP–Trial Alignment
=====================================================
1. Sources behavioral event data directories (from BEHAVIOR_DATA_DIR → BEHAV_DIR)
2. For each processed FP session in pipeline_session_log.csv, finds the matching
   behavioral session directory and aligns photometry to trial events.
3. Updates the pipeline session log with match status and trial counts.

Outputs per session (saved to fp_processed/{session_id}/):
    photometry_with_trial_data.csv  — photometry_long.csv merged with trial events

Run after:
    1_fp_processing.py
    2_fp_qc_summary.py  (optional but recommended)
"""

import os
import shutil
from pathlib import Path

import pandas as pd

from config import (
    BEHAVIOR_DATA_DIR,
    BEHAV_DIR,
    PIPELINE_SESSION_LOG,
    PROCESSED_OUT,
    PHOTOMETRY_LOG_URL,
)
from utils import assign_trials_to_photometry, load_merged_log, update_session_log

EXP = "exp2"


# ---------------------------------------------------------------------------
# Step 3a: Source behavioral event directories
# ---------------------------------------------------------------------------

def source_behavior_dirs() -> pd.DataFrame:
    """
    Find behavioral sessions that have photometry data and copy their
    directories into BEHAV_DIR.

    Matches on (mouse, date) between the photometry log (Google Sheets)
    and the training sessions CSV in BEHAVIOR_DATA_DIR.

    Writes behav_dir (and group/bg_length if present) into pipeline_session_log.csv
    for each matched session so downstream steps can read it from one place.

    Returns the matched sessions DataFrame.
    """
    photometry_log = pd.read_csv(PHOTOMETRY_LOG_URL)
    data_folder = Path(BEHAVIOR_DATA_DIR) / EXP
    sessions_training = pd.read_csv(
        data_folder / f"sessions_training_{EXP}.csv",
        index_col=0,
    )

    # Normalize dates
    photometry_log["date"] = pd.to_datetime(photometry_log["date"]).dt.strftime("%Y-%m-%d")
    sessions_training["date"] = pd.to_datetime(sessions_training["date"]).dt.strftime("%Y-%m-%d")

    # Deduplicate photometry log before merging
    dupes = photometry_log[["date", "mouse"]].duplicated().sum()
    if dupes:
        print(f"Note: dropping {dupes} duplicate row(s) from photometry log before merge.")
        photometry_log = photometry_log.drop_duplicates(subset=["date", "mouse"])

    # Find sessions with both training and photometry data
    sessions_photometry = pd.merge(
        sessions_training,
        photometry_log[["date", "mouse"]],
        on=["date", "mouse"],
        how="inner",
    )

    os.makedirs(BEHAV_DIR, exist_ok=True)

    # Write behav_dir (and any group/bg_length) into the pipeline session log
    if PIPELINE_SESSION_LOG.exists():
        pipe_log = pd.read_csv(PIPELINE_SESSION_LOG, index_col=0)
        if {"mouse", "date"}.issubset(pipe_log.columns):
            extra_cols = [c for c in ("group", "bg_length") if c in sessions_photometry.columns]
            for _, row in sessions_photometry.iterrows():
                matched_ids = pipe_log[
                    (pipe_log["mouse"] == row["mouse"]) &
                    (pipe_log["date"] == row["date"])
                ].index
                for sid in matched_ids:
                    updates = {"behav_dir": row["dir"]}
                    for col in extra_cols:
                        updates[col] = row[col]
                    update_session_log(sid, updates)

    # Report photometry sessions with no matching behavior data
    unmatched = photometry_log[["date", "mouse"]].merge(
        sessions_training[["date", "mouse"]],
        on=["date", "mouse"],
        how="left",
        indicator=True,
    )
    no_behavior = unmatched[unmatched["_merge"] == "left_only"][["date", "mouse"]]
    if not no_behavior.empty:
        print(f"\n{len(no_behavior)} photometry session(s) with no matching behavior data:")
        for _, row in no_behavior.iterrows():
            print(f"  {row['mouse']}  {row['date']}")

        # Log behav_session_found=False for any already-processed FP sessions
        if PIPELINE_SESSION_LOG.exists():
            pipe_log = pd.read_csv(PIPELINE_SESSION_LOG, index_col=0)
            if {"mouse", "date"}.issubset(pipe_log.columns):
                for _, row in no_behavior.iterrows():
                    matched_ids = pipe_log[
                        (pipe_log["mouse"] == row["mouse"]) &
                        (pipe_log["date"] == row["date"])
                    ].index
                    for sid in matched_ids:
                        update_session_log(sid, {"behav_session_found": False})
                        print(f"    logged behav_session_found=False for {sid}")
    else:
        print("All photometry sessions have matching behavior data.")

    # Copy directories
    copied_count = 0
    for src_dir in sessions_photometry["dir"]:
        src_path = data_folder / src_dir
        dest_path = BEHAV_DIR / Path(src_dir).name

        if not src_path.exists():
            print(f"Warning: Source directory does not exist: {src_path}")
            continue

        if dest_path.exists():
            print(f"Skipped: {Path(src_dir).name} (already exists)")
            continue

        shutil.copytree(str(src_path), str(dest_path))
        copied_count += 1
        print(f"Copied: {Path(src_dir).name}")

    print(f"\nCompleted: Copied {copied_count} of {len(sessions_photometry)} directories.")

    # Dataset overview
    mice = sessions_photometry["mouse"].unique()
    print(f"\n{'='*50}")
    print(f"Dataset Summary")
    print(f"{'='*50}")
    print(f"  Mice: {len(mice)}")
    print(f"  Total sessions: {len(sessions_photometry)}")
    print()
    for mouse in sorted(mice):
        mouse_sessions = sessions_photometry[sessions_photometry["mouse"] == mouse]
        print(f"  {mouse}: {len(mouse_sessions)} session(s)")
        for _, row in mouse_sessions.iterrows():
            session_dir = data_folder / row["dir"]
            trial_files = list(session_dir.glob("trials_analyzed_*.csv"))
            if trial_files:
                try:
                    n_trials = len(pd.read_csv(trial_files[0]))
                except Exception:
                    n_trials = "?"
            else:
                n_trials = "?"
            print(f"    {row['date']}  —  {n_trials} trials")
    print(f"{'='*50}")

    return sessions_photometry


# ---------------------------------------------------------------------------
# Step 3b: Align FP photometry to behavioral trial events
# ---------------------------------------------------------------------------

def align_session(session_info: pd.Series, regenerate: bool = False) -> None:
    """
    For one FP session, merge photometry_long.csv with trial events.

    Reads behav_dir from the session row (written by source_behavior_dirs()).
    Saves photometry_with_trial_data.csv to the session's processed output dir.
    Updates pipeline_session_log.csv with behav_session_found and n_trials.
    """
    session_id = str(session_info["session_id"])
    merged_out = PROCESSED_OUT / session_id / "photometry_with_trial_data.csv"

    if not regenerate and merged_out.exists():
        print(f"  Skipping {session_id} (already aligned; use regenerate=True to rerun)")
        return

    phot_path = PROCESSED_OUT / session_id / "photometry_long.csv"
    if not phot_path.exists():
        print(f"  Skipping {session_id}: photometry_long.csv not found — run 1_fp_processing.py first")
        return

    behav_subdir = session_info.get("behav_dir")
    if pd.isna(behav_subdir) or not behav_subdir:
        print(f"  {session_id}: behav_dir not set — run source_behavior_dirs() first")
        update_session_log(session_id, {"behav_session_found": False})
        return

    behav_subdir = str(behav_subdir)
    trials_path = BEHAV_DIR / behav_subdir / f"trials_analyzed_{behav_subdir}.csv"

    if not trials_path.exists():
        print(f"  {session_id}: trials file not found at {trials_path}")
        update_session_log(session_id, {"behav_session_found": False})
        return

    phot = pd.read_csv(phot_path, low_memory=False)
    trials = pd.read_csv(trials_path, index_col=0)

    merged = assign_trials_to_photometry(phot, trials)
    merged.to_csv(merged_out, index=False)

    n_trials = len(trials)
    print(f"  {session_id}: aligned {n_trials} trials → {merged_out}")

    update_session_log(session_id, {
        "behav_session_found": True,
        "behav_dir":           behav_subdir,
        "n_trials":            n_trials,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(regenerate: bool = False) -> None:
    # Step 3a: source behavioral directories and write behav_dir to pipeline log
    source_behavior_dirs()

    # Step 3b: align each processed FP session to its behavioral trial data
    print("\nAligning FP sessions to behavioral events...")
    sessions = load_merged_log()
    if sessions.empty:
        print("No sessions found with behav_dir set. Run 1_fp_processing.py first.")
        return

    for _, session_info in sessions.iterrows():
        align_session(session_info, regenerate=regenerate)

    print("\nAlignment done.")


if __name__ == "__main__":
    main(regenerate=False)

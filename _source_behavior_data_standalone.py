"""Copy behavior data directories for sessions that have photometry data."""

import os
import shutil
from pathlib import Path

import pandas as pd

from config import BEHAVIOR_DATA_DIR, BEHAV_DIR, PHOTOMETRY_LOG_URL

DATA_DIR = BEHAVIOR_DATA_DIR
EXP = "exp2"
PHOTOMETRY_FOLDER = BEHAV_DIR


def main():
    """Main processing function."""
    # Load data
    photometry_log = pd.read_csv(PHOTOMETRY_LOG_URL)
    data_folder = os.path.join(DATA_DIR, EXP)
    sessions_training = pd.read_csv(
        os.path.join(data_folder, f'sessions_training_{EXP}.csv'),
        index_col=0
    )

    # Normalize date formats to YYYY-MM-DD before merging
    photometry_log['date'] = pd.to_datetime(photometry_log['date']).dt.strftime('%Y-%m-%d')
    sessions_training['date'] = pd.to_datetime(sessions_training['date']).dt.strftime('%Y-%m-%d')

    # Deduplicate photometry log before merging to avoid row multiplication
    dupes = photometry_log[['date', 'mouse']].duplicated().sum()
    if dupes:
        print(f"Note: dropping {dupes} duplicate row(s) from photometry log before merge.")
        photometry_log = photometry_log.drop_duplicates(subset=['date', 'mouse'])

    # Find sessions that have both training and photometry data
    sessions_photometry = pd.merge(
        sessions_training,
        photometry_log[['date', 'mouse']],
        on=['date', 'mouse'],
        how='inner'
    )
    os.makedirs(PHOTOMETRY_FOLDER, exist_ok=True)
    sessions_photometry.to_csv(os.path.join(PHOTOMETRY_FOLDER, f'sessions_photometry_{EXP}.csv'))

    # Report photometry sessions with no matching behavior data
    unmatched = photometry_log[['date', 'mouse']].merge(
        sessions_training[['date', 'mouse']],
        on=['date', 'mouse'],
        how='left',
        indicator=True
    )
    no_behavior = unmatched[unmatched['_merge'] == 'left_only'][['date', 'mouse']]
    if not no_behavior.empty:
        print(f"\n{len(no_behavior)} photometry session(s) with no matching behavior data:")
        for _, row in no_behavior.iterrows():
            print(f"  {row['mouse']}  {row['date']}")
    else:
        print("All photometry sessions have matching behavior data.")

    # Copy directories
    copied_count = 0
    for src_dir in sessions_photometry['dir']:
        src_path = os.path.join(data_folder, src_dir)
        dest_path = os.path.join(PHOTOMETRY_FOLDER, os.path.basename(src_dir))

        if not os.path.exists(src_path):
            print(f"Warning: Source directory does not exist: {src_path}")
            continue

        if os.path.exists(dest_path):
            print(f"Skipped: {os.path.basename(src_dir)} (already exists)")
            continue

        shutil.copytree(src_path, dest_path)
        copied_count += 1
        print(f"Copied: {os.path.basename(src_dir)}")

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
            session_dir = Path(data_folder) / row["dir"]
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


if __name__ == '__main__':
    main()
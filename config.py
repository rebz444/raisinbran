"""Centralized configuration for the photometry analysis pipeline."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — all data lives on the external drive
# ---------------------------------------------------------------------------
PHOTO_ROOT = Path("/Volumes/T7 Shield/photometry")
PROCESSED_OUT = PHOTO_ROOT / "processed_output"
BEHAV_DIR = PHOTO_ROOT / "behav_analyzed"
MATCHED_SESSIONS_CSV = PHOTO_ROOT / "matched_sessions.csv"
BEHAV_LOG_CSV = BEHAV_DIR / "sessions_photometry_exp2.csv"
MERGED_SESSIONS_CSV = PHOTO_ROOT / "sessions_dff_behav_merged.csv"
OUT_PLOTS_DIR = PHOTO_ROOT / "trial_plots"

# Raw FP data
FP_DIR = PHOTO_ROOT / "fp"
FP_DIR_FLATTENED = PHOTO_ROOT / "fp_flattened"

# Behavior data (still on internal drive — not moved)
BEHAVIOR_DATA_DIR = Path("/Users/rebekahzhang/data/behavior_data")

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
PHOTOMETRY_LOG_URL = (
    "https://docs.google.com/spreadsheets/d/1B8KCnku1vQInKOFBuoLeuDND6CEUA4hT6qQ5zuhqvyU/"
    "export?format=csv&gid=1751471403"
)

# ---------------------------------------------------------------------------
# Channel configuration
# ---------------------------------------------------------------------------
FIBER_CHANNEL_MAP = {
    2: {1: ("R0", "G2"), 2: ("R1", "G3")},
    4: {1: ("R0", "G4"), 2: ("R1", "G5"), 3: ("R2", "G6"), 4: ("R3", "G7")},
}

CHANNEL_COLORS = {
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

# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
DEFAULT_SIGNAL_COL = "dff_zscored"
NON_IMPULSIVE_CUTOFF = 0.5  # seconds; trials below this are considered impulsive

# QC
QC_GATE_ENABLED = True
QC_STOP_ON_FAIL = False
QC_MIN_SAMPLES_PER_STATE = 50

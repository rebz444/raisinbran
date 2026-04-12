"""Centralized configuration for the photometry analysis pipeline."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — all data lives on the external drive
# ---------------------------------------------------------------------------
PHOTO_ROOT = Path("/Volumes/T7 Shield/photometry")
PROCESSED_OUT = PHOTO_ROOT / "fp_processed"
BEHAV_DIR = PHOTO_ROOT / "behav_analyzed"
OUT_PLOTS_DIR = PHOTO_ROOT / "trial_plots"
QC_OUT_DIR = PHOTO_ROOT / "quality_control"

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

# Per-label colors: each sensor × side × area combination gets a unique, visually
# distinct color. Sensor families use different hue families; left/right within a
# family differ enough to tell apart clearly.
_CHANNEL_COLORS: dict[str, str] = {
    # GRAB (dopamine) — green family
    "l_str_GRAB":  "#2ecc71",   # bright green
    "r_str_GRAB":  "#1a7a43",   # dark forest green
    # GCaMP (calcium) — blue family
    "l_str_GCaMP": "#3498db",   # cornflower blue
    "r_str_GCaMP": "#1a5276",   # deep navy
    # rCaMP in striatum (V1 axon terminals) — orange/red family
    "l_str_rCaMP": "#e67e22",   # amber orange
    "r_str_rCaMP": "#c0392b",   # crimson
    # rCaMP in V1 (cell bodies) — purple family
    "l_v1_rCaMP":  "#9b59b6",   # medium purple
    "r_v1_rCaMP":  "#6c3483",   # deep purple
}


def get_channel_color(label: str) -> str:
    """Return a plot color for a biological channel label (e.g. 'l_str_GRAB').
    Falls back to grey for any label not in the lookup table."""
    return _CHANNEL_COLORS.get(label, "#888888")

# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
DEFAULT_SIGNAL_COL = "dff_zscored"
NON_IMPULSIVE_CUTOFF = 0.5  # seconds; trials below this are considered impulsive

# QC
QC_GATE_ENABLED = True
QC_STOP_ON_FAIL = False
QC_MIN_SAMPLES_PER_STATE = 50

# ---------------------------------------------------------------------------
# Pipeline session log
# ---------------------------------------------------------------------------
SHORT_SESSION_THRESHOLD_MIN = 40
PIPELINE_SESSION_LOG = PHOTO_ROOT / "pipeline_session_log.csv"

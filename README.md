# raisinbran — Fiber Photometry Analysis Pipeline

Photometry signal processing and dopamine dynamics analysis for Shuler Lab behavioral experiments.

---

## Setup

### 1. Clone the repo
```bash
git clone <repo-url>
cd raisinbran
```

### 2. Create the conda environment
```bash
conda env create -f environment.yml
conda activate raisinbran
```

### 3. Configure paths

Open `config.py` and edit the two variables at the top of the **CONFIGURE THESE FOR YOUR MACHINE** section:

```python
# Where your T7 Shield (or equivalent) is mounted:
PHOTO_ROOT = Path("/Volumes/T7 Shield/photometry")

# Where local behavior data lives:
BEHAVIOR_DATA_DIR = Path("/Users/yourname/data/behavior_data")
```

All other paths (processed output, QC, figures, etc.) are derived from `PHOTO_ROOT` automatically.

---

## Pipeline order

Run scripts in sequence — each step produces outputs consumed by the next.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_fp_processing.py` | Match FP + behavior files, correct photobleaching, compute dF/F & z-scores, run QC |
| 2 | `2_fp_qc_summary.py` | Visualize QC metrics and assign quality tiers (A/B/C) |
| 3 | `3_fp_behavior_align.py` | Source behavior data, align photometry with trial events |
| 4 | `4_fp_behavior_check_plots.py` | Per-trial dF/F trace plots with behavioral event markers |

After step 4, the following analysis scripts can be run independently:

- `5_committee_figures.py` — DA ramp slope figure (three anchors)
- `7_da_ramp_explorer.py` — Systematic DA dynamics (forward/backward, quartiles, heatmaps)
- `9_da_trial_history_v2.py` — Previous trial outcome effects on DA (v2)
- `9_da_trial_history_v3.py` — Previous trial outcome effects, multi-anchor (v3)
- `figure_1_da_ramp_slope.py` — Publication Figure 1 (quartile traces, slope scatter)
- `figure_2_trial_history.py` — Publication Figure 2 (effect sizes, trace splits)

---

## Data layout (on external drive)

```
PHOTO_ROOT/
├── fp/                        # Raw FP data
├── fp_flattened/              # Flattened raw data
├── fp_processed/              # Per-session processed output (step 1)
├── behav_analyzed/            # Aligned behavior data (step 3)
├── trial_plots/               # Per-trial plots (step 4)
├── quality_control/           # QC visualizations (step 2)
├── committee_figures/         # Committee/publication figures
├── da_ramp_analysis/          # DA ramp explorer outputs
├── trial_history_results_v2/  # Trial history v2 outputs
├── trial_history_results_v3/  # Trial history v3 outputs
└── pipeline_session_log.csv   # Central session tracking file
```

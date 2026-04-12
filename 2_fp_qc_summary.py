"""
FP Processing Quality Control Summary
======================================
Reads the correction_quality.csv and per-channel CSVs written by 1_fp_processing.py
and produces all QC visualisations saved to PHOTO_ROOT/quality_control/.

Run this independently — no reprocessing needed.

Outputs
-------
quality_control/
  tier_A/  tier_B/  tier_C/
      {session_id}_{channel}_qc.png   — per-channel 4-panel plot (copied from session dir)
  tier_by_sensor.png                  — stacked bar: tier counts per sensor group
  tier_by_sensor_and_side.png         — same, left vs right split
  metric_<name>.png                   — violin plots per QC metric
  tier_over_time.png                  — tier scatter over session dates
"""

import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from config import PROCESSED_OUT, QC_OUT_DIR, SHORT_SESSION_THRESHOLD_MIN
from utils import update_session_log


TIER_COLORS = {"A": "#27ae60", "B": "#e67e22", "C": "#e74c3c"}


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_combined_qc(processed_dir=PROCESSED_OUT):
    """Concatenate all per-session correction_quality.csv files."""
    qc_files = sorted(Path(processed_dir).glob("*/correction_quality.csv"))
    if not qc_files:
        raise FileNotFoundError(f"No correction_quality.csv files found under {processed_dir}")
    dfs = [pd.read_csv(f) for f in qc_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined["sensor_group"] = combined["channel"].str.replace(r'^[lr]_', '', regex=True)

    # Older CSVs lack duration_min; derive from channel_summary.csv (n_samples / fs_hz / 60)
    if "duration_min" not in combined.columns:
        dur_rows = []
        for qc_f in qc_files:
            summary_csv = qc_f.parent / "channel_summary.csv"
            if summary_csv.exists():
                sdf = pd.read_csv(summary_csv)
                if {"n_samples", "fs_hz"}.issubset(sdf.columns):
                    dur = (sdf["n_samples"] / sdf["fs_hz"] / 60).median()
                    dur_rows.append({"session_id": qc_f.parent.name, "duration_min": dur})
        if dur_rows:
            combined = combined.merge(pd.DataFrame(dur_rows), on="session_id", how="left")
            print(f"  Backfilled duration_min from channel_summary.csv for {len(dur_rows)} sessions.")

    if "duration_min" in combined.columns:
        combined["short_session"] = combined["duration_min"] < SHORT_SESSION_THRESHOLD_MIN
    print(f"Loaded {len(combined)} channel-sessions from {len(qc_files)} sessions.")
    return combined


def collect_plots_into_tiers(processed_dir=PROCESSED_OUT, qc_out_dir=QC_OUT_DIR):
    """
    Copy per-channel QC plots (saved alongside their CSVs by 2_fp_processing.py)
    into qc_out_dir/tier_A|B|C/ for easy manual review.

    Source:  {processed_dir}/{session_id}/{channel}_qc.png
    Dest:    {qc_out_dir}/tier_{tier}/{session_id}_{channel}_qc.png
    """
    processed_dir = Path(processed_dir)
    qc_out_dir    = Path(qc_out_dir)

    session_dirs = sorted(d for d in processed_dir.iterdir() if d.is_dir())
    if not session_dirs:
        print(f"No session directories found under {processed_dir}")
        return

    n_copied = 0
    for session_dir in session_dirs:
        qc_csv = session_dir / "correction_quality.csv"
        if not qc_csv.exists():
            continue

        qc_df = pd.read_csv(qc_csv)
        session_id = session_dir.name

        for _, row in qc_df.iterrows():
            ch           = row["channel"]
            tier         = str(row.get("quality_tier", "unknown"))
            sensor_group = re.sub(r'^[lr]_', '', ch)   # strip l_/r_ → str_GRAB, v1_rCaMP, …
            src          = session_dir / "qc" / f"{ch}_qc.png"
            if not src.exists():
                continue
            dest_dir = qc_out_dir / f"tier_{tier}" / sensor_group
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_dir / f"{session_id}_{ch}_qc.png")
            n_copied += 1

    print(f"  Copied {n_copied} QC plots into {qc_out_dir}/tier_*/.")


# ---------------------------------------------------------------------------
# Cross-session summary plots
# ---------------------------------------------------------------------------

def plot_tier_by_sensor(combined, outdir):
    """Stacked bar: Tier A/B/C counts per sensor-location group (l/r pooled)."""
    if "quality_tier" not in combined.columns:
        print("No quality_tier column found; skipping.")
        return

    counts = (
        combined.groupby(["sensor_group", "quality_tier"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["A", "B", "C"], fill_value=0)
    )
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

    n_groups = len(counts)
    fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.4 + 2), 5), constrained_layout=True)

    bottoms = np.zeros(n_groups)
    x = np.arange(n_groups)
    for tier in ["A", "B", "C"]:
        vals = counts[tier].values
        ax.bar(x, vals, bottom=bottoms, color=TIER_COLORS[tier],
               label=f"Tier {tier}", width=0.6, edgecolor="white", linewidth=0.5)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax.text(xi, b + v / 2, str(v), ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Channel-sessions")
    ax.set_title("Quality Tier Distribution by Sensor Location", fontsize=13, fontweight="bold")
    ax.legend(title="Tier", loc="upper right")
    ax.set_ylim(0, bottoms.max() * 1.15)
    _save(fig, outdir, "tier_by_sensor.png")


def plot_tier_by_sensor_and_side(combined, outdir):
    """Grouped bar: tier counts per sensor group, left vs right side separated."""
    if "quality_tier" not in combined.columns:
        return

    df = combined.copy()
    df["side"] = df["channel"].str.extract(r'^([lr])_')[0].fillna("?")

    counts = (
        df.groupby(["sensor_group", "side", "quality_tier"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["A", "B", "C"], fill_value=0)
    )

    sensor_groups = counts.index.get_level_values("sensor_group").unique()
    sides = sorted(counts.index.get_level_values("side").unique())
    n_groups = len(sensor_groups)
    n_sides = len(sides)
    bar_w = 0.35
    offsets = np.linspace(-(n_sides - 1) * bar_w / 2, (n_sides - 1) * bar_w / 2, n_sides)

    fig, ax = plt.subplots(figsize=(max(7, n_groups * 1.8 + 2), 5), constrained_layout=True)
    x = np.arange(n_groups)

    for side_i, side in enumerate(sides):
        bottoms = np.zeros(n_groups)
        for tier in ["A", "B", "C"]:
            vals = np.array([
                counts.loc[(sg, side), tier] if (sg, side) in counts.index else 0
                for sg in sensor_groups
            ])
            ax.bar(x + offsets[side_i], vals, bottom=bottoms, width=bar_w,
                   color=TIER_COLORS[tier], edgecolor="white", linewidth=0.5,
                   alpha=0.9 if side == "l" else 0.6)
            bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(sensor_groups, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Channel-sessions")
    ax.set_title("Quality Tier Distribution by Sensor × Side", fontsize=13, fontweight="bold")

    legend_els = [mpatches.Patch(facecolor=TIER_COLORS[t], label=f"Tier {t}") for t in ["A", "B", "C"]]
    legend_els += [mpatches.Patch(facecolor="gray", alpha=0.9, label="left"),
                   mpatches.Patch(facecolor="gray", alpha=0.6, label="right")]
    ax.legend(handles=legend_els, loc="upper right", fontsize=8)
    _save(fig, outdir, "tier_by_sensor_and_side.png")


def plot_metric_distributions(combined, outdir):
    """Violin plots of key QC metrics split by tier and sensor group."""
    metrics = {
        "r2_baseline":   "R² (baseline tracking)",
        "drift_delta_z": "Drift Δz (first vs last 10%)",
        "r_iso_dff":     "r(iso, dF/F) — green only",
        "skewness":      "Skewness of dF/F",
        "outlier_frac":  "Outlier fraction (|z|>5)",
    }

    for col, label in metrics.items():
        if col not in combined.columns:
            continue
        df = combined.dropna(subset=[col, "quality_tier", "sensor_group"]).copy()
        if df.empty:
            continue

        sensor_groups = sorted(df["sensor_group"].unique())
        n = len(sensor_groups)
        fig, axes = plt.subplots(1, n, figsize=(max(6, n * 2.5), 4),
                                 sharey=True, constrained_layout=True)
        if n == 1:
            axes = [axes]
        fig.suptitle(label, fontsize=12, fontweight="bold")

        for ax, sg in zip(axes, sensor_groups):
            sub = df[df["sensor_group"] == sg]
            plot_data = [
                (sub.loc[sub["quality_tier"] == t, col].values, TIER_COLORS[t], t)
                for t in ["A", "B", "C"]
                if (sub["quality_tier"] == t).any()
            ]
            if not plot_data:
                ax.set_visible(False)
                continue

            vp = ax.violinplot([d for d, _, _ in plot_data], showmedians=True, showextrema=False)
            for body, (_, color, _) in zip(vp["bodies"], plot_data):
                body.set_facecolor(color)
                body.set_alpha(0.7)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(1.5)
            ax.set_xticks(range(1, len(plot_data) + 1))
            ax.set_xticklabels([t for _, _, t in plot_data])
            ax.set_title(sg, fontsize=10)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.4, lw=0.8)

        axes[0].set_ylabel(col)
        _save(fig, outdir, f"metric_{col}.png")


def plot_tier_over_time(combined, outdir):
    """Tier scatter over session dates, coloured by sensor group."""
    if "quality_tier" not in combined.columns or "date" not in combined.columns:
        return

    df = combined.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["tier_num"] = df["quality_tier"].map({"A": 2, "B": 1, "C": 0})

    sensor_groups = sorted(df["sensor_group"].unique())
    color_map = dict(zip(sensor_groups, plt.cm.tab10(np.linspace(0, 1, len(sensor_groups)))))

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for sg in sensor_groups:
        sub = df[df["sensor_group"] == sg].sort_values("date")
        jitter = np.random.default_rng(abs(hash(sg)) % 2**31).uniform(-0.1, 0.1, len(sub))
        ax.scatter(sub["date"], sub["tier_num"] + jitter,
                   color=color_map[sg], label=sg, alpha=0.7, s=25, zorder=3)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["C", "B", "A"], fontsize=11)
    ax.set_ylabel("Quality Tier")
    ax.set_xlabel("Session date")
    ax.set_title("Quality Tier Over Time by Sensor Group", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="x", alpha=0.3)

    # Mark short sessions with a black X
    if "short_session" in df.columns:
        short_df = df[df["short_session"]]
        if not short_df.empty:
            ax.scatter(short_df["date"], short_df["tier_num"],
                       marker="x", color="black", s=60, linewidths=1.5,
                       zorder=5, label="short session")
            ax.legend(loc="lower right", fontsize=8, ncol=2)

    _save(fig, outdir, "tier_over_time.png")


def plot_short_session_summary(combined, outdir):
    """Horizontal bar chart of sessions flagged as short (duration < threshold)."""
    if "short_session" not in combined.columns or "duration_min" not in combined.columns:
        print("No short_session data available; skipping short session summary.")
        return

    session_cols = ["session_id", "mouse", "date", "duration_min", "short_session"]
    available = [c for c in session_cols if c in combined.columns]
    short = (
        combined[combined["short_session"]][available]
        .drop_duplicates("session_id")
        .sort_values("duration_min")
    )

    if short.empty:
        print("No short sessions flagged.")
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(short) * 0.4 + 1)), constrained_layout=True)
    y = range(len(short))
    ax.barh(list(y), short["duration_min"].values, color="#e74c3c", alpha=0.75)
    ax.axvline(SHORT_SESSION_THRESHOLD_MIN, color="black", linestyle="--",
               linewidth=1.2, label=f"{SHORT_SESSION_THRESHOLD_MIN} min threshold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(
        [f"{r['mouse']}  {r['date']}" for _, r in short.iterrows()],
        fontsize=9,
    )
    ax.set_xlabel("Recording duration (min)")
    ax.set_title(f"Short Sessions Flagged ({len(short)} sessions)", fontweight="bold")
    ax.legend(fontsize=9)
    _save(fig, outdir, "short_sessions_flagged.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, outdir, filename):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading QC data...")
    combined = load_combined_qc()

    print("\nCollecting per-channel QC plots into tier folders...")
    collect_plots_into_tiers()

    print(f"\nGenerating cross-session summary plots → {QC_OUT_DIR}")
    plot_tier_by_sensor(combined, QC_OUT_DIR)
    plot_tier_by_sensor_and_side(combined, QC_OUT_DIR)
    plot_metric_distributions(combined, QC_OUT_DIR)
    plot_tier_over_time(combined, QC_OUT_DIR)
    plot_short_session_summary(combined, QC_OUT_DIR)

    print("\nUpdating pipeline session log with QC results...")
    tier_map = {"A": 0, "B": 1, "C": 2}
    for session_id, g in combined.groupby("session_id"):
        worst_idx = int(g["quality_tier"].map(tier_map).max()) if "quality_tier" in g.columns else 0
        worst_tier = ["A", "B", "C"][worst_idx]
        tier_summary = ", ".join(
            f"{row['channel']}:{row['quality_tier']}"
            for _, row in g.iterrows()
            if "quality_tier" in g.columns
        )
        reasons = "|".join(filter(None, g["tier_reasons"].dropna().tolist())) if "tier_reasons" in g.columns else ""
        short = bool(g["short_session"].any()) if "short_session" in g.columns else False
        update_session_log(session_id, {
            "short_session":   short,
            "worst_tier":      worst_tier,
            "tier_summary":    tier_summary,
            "qc_flag_reasons": reasons,
        })

    print("\nDone.")


if __name__ == "__main__":
    main()

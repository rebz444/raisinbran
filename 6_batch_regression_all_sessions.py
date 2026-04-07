#!/usr/bin/env python3
"""
Batch regression across all sessions.
For each session:
- Load photometry_with_trial_data.csv
- Compute per-trial linear slope for G4 (trial_time 0 -> decision_time)
- Save per-trial plots and per-session slope CSV
- Merge with behavior trials to apply filters (good_trial, time_waited)
- Run regressions: decision_time ~ slope (all / non-impulsive / good / good+non-impulsive)
- Save per-session summary and a global summary CSV
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ============================
# CONFIGURATION
# ============================
PHOTO_ROOT = Path("/Users/rebekahzhang/data/photometry")
PROCESSED_OUT = PHOTO_ROOT / "processed_output"
BEHAV_DIR = PHOTO_ROOT / "behav_analyzed"
MATCHED_SESSIONS_CSV = PHOTO_ROOT / "matched_sessions.csv"
BEHAV_LOG_CSV = BEHAV_DIR / "sessions_photometry_exp2.csv"
MERGED_SESSIONS_CSV = PHOTO_ROOT / "sessions_dff_behav_merged.csv"
OUT_PLOTS_DIR = PHOTO_ROOT / "trial_plots"

SIGNAL_COL = "dff_zscored"  # or "dff_filtered"
CHANNEL = "G4"
NON_IMPULSIVE_CUTOFF = 0.5

PLOT_COLOR = "#27ae60"


# ============================
# HELPERS
# ============================

def compute_trial_slopes(sig: pd.DataFrame) -> pd.DataFrame:
    sig = sig.copy()
    sig["decision_time"] = sig["bg_length"] + sig["time_waited"]

    sig = sig.loc[sig["miss_trial"] == False].copy()
    sig = sig.loc[sig["channel"] == CHANNEL].copy()

    results: List[Dict[str, object]] = []
    for trial_num, trial_sig in sig.groupby("session_trial_num"):
        trial_data = trial_sig.iloc[0]
        decision_time = trial_data["decision_time"]

        sig_window = trial_sig[(trial_sig["trial_time"] >= 0) & (trial_sig["trial_time"] <= decision_time)]
        if len(sig_window) < 2:
            continue

        slope, intercept, r_value, p_value, std_err = linregress(
            sig_window["trial_time"],
            sig_window[SIGNAL_COL],
        )

        results.append(
            {
                "trial_num": int(trial_num),
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "p_value": p_value,
                "std_err": std_err,
                "bg_length": trial_data.get("bg_length", np.nan),
                "time_waited": trial_data.get("time_waited", np.nan),
                "decision_time": decision_time,
                "reward": trial_data.get("reward", np.nan),
                "n_points": len(sig_window),
            }
        )
    return pd.DataFrame(results)


def plot_trial_fit(
    trial_sig: pd.DataFrame,
    decision_time: float,
    slope: float,
    intercept: float,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        trial_sig["trial_time"],
        trial_sig[SIGNAL_COL],
        label=CHANNEL,
        linewidth=1.4,
        alpha=0.85,
        color=PLOT_COLOR,
    )

    fit_x = np.array([0, decision_time])
    fit_y = slope * fit_x + intercept
    ax.plot(fit_x, fit_y, "--", linewidth=2, alpha=0.7, color=PLOT_COLOR, label="Fit")

    ax.axvline(0, color="green", linestyle="--", linewidth=2, alpha=0.7, label="Cue On")
    bg_length = float(trial_sig.iloc[0].get("bg_length", np.nan))
    if np.isfinite(bg_length):
        ax.axvline(bg_length, color="orange", linestyle="--", linewidth=2, alpha=0.7, label="BG End")
    ax.axvline(decision_time, color="blue", linestyle="--", linewidth=2, alpha=0.7, label="Decision")

    ax.set_xlabel("Trial Time (s)")
    ax.set_ylabel(SIGNAL_COL)
    ax.set_title(f"Trial {int(trial_sig.iloc[0]['session_trial_num'])}: {CHANNEL} Fit")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_regression(df: pd.DataFrame) -> Dict[str, float]:
    slope_coef, intercept, r_value, p_value, std_err = linregress(df["slope"], df["decision_time"])
    return {
        "n": int(len(df)),
        "coef": slope_coef,
        "intercept": intercept,
        "r_squared": r_value**2,
        "r_value": r_value,
        "p_value": p_value,
        "std_err": std_err,
    }


# ============================
# MAIN
# ============================

def main() -> None:
    merged_log = pd.read_csv(PHOTO_ROOT / "sessions_dff_behav_merged.csv", index_col=0)
    session_summaries: List[Dict[str, object]] = []

    for _, session_info in merged_log.iterrows():
        session_id = str(session_info["session_id"])
        print(f"processing {session_id}")
        behav_subdir = str(session_info["dir"])

        sig_path = PROCESSED_OUT / session_id / "photometry_with_trial_data.csv"
        trials_path = BEHAV_DIR / behav_subdir / f"trials_analyzed_{behav_subdir}.csv"

        sig = pd.read_csv(sig_path, dtype={"previous_trial_miss_trial": "object"})
        trials = pd.read_csv(trials_path, index_col=0)

        results_df = compute_trial_slopes(sig)

        # Save per-session slope results
        results_path = PROCESSED_OUT / session_id / f"linear_fit_results_{CHANNEL}.csv"
        results_df.to_csv(results_path, index=False)

        # Plot each trial
        plot_dir = OUT_PLOTS_DIR / f"{session_info['mouse']}_{session_info['date']}_linear_fits"
        sig["decision_time"] = sig["bg_length"] + sig["time_waited"]
        sig = sig.loc[(sig["miss_trial"] == False) & (sig["channel"] == CHANNEL)].copy()

        for _, row in results_df.iterrows():
            tr = row["trial_num"]
            trial_sig = sig[sig["session_trial_num"] == tr]
            if trial_sig.empty:
                continue
            plot_trial_fit(
                trial_sig=trial_sig,
                decision_time=float(row["decision_time"]),
                slope=float(row["slope"]),
                intercept=float(row["intercept"]),
                save_path=plot_dir / f"trial_{int(tr):03d}_linear_fit.png",
            )

        # Merge with trials to bring in good_trial and time_waited
        trials_with_slope = trials.merge(
            results_df,
            left_on="session_trial_num",
            right_on="trial_num",
            how="left",
        )
        # decision_time already computed in results_df from photometry data
        trials_with_slope = trials_with_slope.dropna(subset=["slope", "decision_time"])

        # Filters
        all_df = trials_with_slope
        non_impulsive_df = trials_with_slope[trials_with_slope["time_waited"] > NON_IMPULSIVE_CUTOFF]
        good_df = trials_with_slope[trials_with_slope["good_trial"] == True]
        good_non_impulsive_df = trials_with_slope[
            (trials_with_slope["good_trial"] == True)
            & (trials_with_slope["time_waited"] > NON_IMPULSIVE_CUTOFF)
        ]

        # Regression summaries
        summary = {
            "session_id": session_id,
            "mouse": session_info.get("mouse", ""),
            "date": session_info.get("date", ""),
        }

        for label, df in [
            ("all", all_df),
            ("non_impulsive", non_impulsive_df),
            ("good", good_df),
            ("good_non_impulsive", good_non_impulsive_df),
        ]:
            stats = run_regression(df)
            summary.update({
                f"{label}_n": stats["n"],
                f"{label}_coef": stats["coef"],
                f"{label}_intercept": stats["intercept"],
                f"{label}_r2": stats["r_squared"],
                f"{label}_r": stats["r_value"],
                f"{label}_p": stats["p_value"],
                f"{label}_stderr": stats["std_err"],
            })

        # Save per-session summary
        summary_path = PROCESSED_OUT / session_id / f"decision_time_regression_summary_{CHANNEL}.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        session_summaries.append(summary)

        print(f"Processed {session_id}: {len(results_df)} trials")

    # Save global summary
    if session_summaries:
        summary_df = pd.DataFrame(session_summaries)
        global_summary_path = PHOTO_ROOT / f"decision_time_regression_summary_{CHANNEL}_all_sessions.csv"
        summary_df.to_csv(global_summary_path, index=False)
        print(f"Saved summary to: {global_summary_path}")
    else:
        print("No sessions processed.")


if __name__ == "__main__":
    main()

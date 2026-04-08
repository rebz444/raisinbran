#!/usr/bin/env python3
"""
Per-trial dopamine (GRAB) photometry analysis for time investment (waiting) behavior.

Designed for your current processed outputs:
- trials_analyzed_<dir>.csv  (contains time_waited, bg_repeats, miss_trial, prev-trial columns)
- events_processed_<dir>.csv (contains per-event trial_time and state transitions)
- G4.csv                     (processed photometry for the GRAB channel; includes t_sec, dff_zscored, iso)

Core goals:
1) Relate *pre-decision* dopamine activity to how long the animal waited on the same trial
   (no cheating: we only use photometry samples strictly before the decision lick).
2) Provide three complementary analyses:
   A) Ramping slope during waiting
   B) Mean DA in multiple pre-decision windows
   C) Time-resolved correlation DA(t) vs wait_time
3) Provide negative controls:
   - Isosbestic trace
   - Trial-label shuffles (permutation and circular shift)
   - Time-reversal control for ramp slopes

Outputs:
- trial_metrics.csv
- figures/*.png

Usage:
Set BATCH and file paths in the IDE RUN CONFIG block at the bottom of this file, then run.

Notes on timebases:
- This script assumes photometry t_sec is aligned to "session seconds since first trial start".
  We verify this by comparing photometry t_sec range with (trials.start_time - trials.start_time[0]).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_OUT


def _find_file(patterns: List[str]) -> Optional[Path]:
    """Search the current directory for the first file matching any of the given glob patterns."""
    for pattern in patterns:
        matches = sorted(Path(".").glob(pattern))
        if matches:
            return matches[0]
    return None


# ----------------------------
# Utilities
# ----------------------------

def robust_linear_slope(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Simple OLS slope fit y = a*t + b.
    Returns (slope a, intercept b, r2).
    We keep this minimal on purpose. (You can swap to statsmodels RLM later.)
    """
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if len(t) < 5:
        return np.nan, np.nan, np.nan
    t0 = np.mean(t)
    y0 = np.mean(y)
    tt = t - t0
    yy = y - y0
    denom = np.sum(tt**2)
    if denom == 0:
        return np.nan, np.nan, np.nan
    a = np.sum(tt * yy) / denom
    b = y0 - a * t0
    yhat = a * t + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(a), float(b), float(r2)


# ----------------------------
# Event extraction (matches your notebook logic)
# ----------------------------

@dataclass
class TrialTimes:
    cue_on: float
    cue_off: float
    decision: float  # first lick in wait / consumption start
    has_decision: bool


def extract_trial_times(events: pd.DataFrame) -> Dict[int, TrialTimes]:
    """
    Build trial -> key event times (in *trial_time* seconds).
    Uses state transitions:
      cue_on  = first 'in_background'
      cue_off = first 'in_wait'
      decision = first 'in_consumption' (if any; missed otherwise)
    """
    out: Dict[int, TrialTimes] = {}
    for tr, df in events.groupby("session_trial_num"):
        df = df.sort_values("trial_time")
        cue_on = df.loc[df["state"] == "in_background", "trial_time"]
        cue_off = df.loc[df["state"] == "in_wait", "trial_time"]
        cue_on_t = cue_on.iloc[0] if len(cue_on) else np.nan
        cue_off_t = cue_off.iloc[0] if len(cue_off) else np.nan

        has_decision = (df["state"] == "in_consumption").any()
        decision_t = np.nan
        if has_decision:
            decision_t = df.loc[df["state"] == "in_consumption", "trial_time"].iloc[0]

        out[int(tr)] = TrialTimes(cue_on=cue_on_t, cue_off=cue_off_t, decision=decision_t, has_decision=bool(has_decision))
    return out


# ----------------------------
# Photometry -> trial assignment
# ----------------------------

def assign_photometry_to_trials(phot: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each photometry sample to a trial using merge_asof on session time.

    Assumptions:
    - trials.start_time is an epoch-ish timestamp
    - phot.t_sec is seconds since photometry start AND (in these processed outputs)
      phot.t_sec aligns to (trials.start_time - first_trial_start_time).

    Steps:
    - Compute trial_start_sec = trials.start_time - first_trial_start
    - merge_asof(phot.t_sec, trial_start_sec) backward
    - keep samples with t_sec <= trial_end_sec
    - compute trial_time = t_sec - trial_start_sec
    """
    required_trials = {"start_time", "end_time", "session_trial_num"}
    required_phot = {"t_sec"}
    if not required_trials.issubset(trials.columns):
        raise ValueError(f"Trials missing required columns: {sorted(required_trials - set(trials.columns))}")
    if not required_phot.issubset(phot.columns):
        raise ValueError(f"Photometry missing required columns: {sorted(required_phot - set(phot.columns))}")

    trials = trials.sort_values("start_time").copy()
    t0 = float(trials.iloc[0]["start_time"])
    trials["trial_start_sec"] = trials["start_time"].astype(float) - t0
    trials["trial_end_sec"] = trials["end_time"].astype(float) - t0

    phot = phot.sort_values("t_sec").copy()
    phot["t_sec"] = phot["t_sec"].astype(float)

    # sanity: check alignment roughly matches
    phot_range = float(phot["t_sec"].max()) - float(phot["t_sec"].min())
    trial_range = float(trials["trial_end_sec"].max()) - float(trials["trial_start_sec"].min())
    if abs(phot_range - trial_range) > 10.0:
        print("[WARN] photometry and trials time ranges differ by >10s; check alignment.")

    merged = pd.merge_asof(
        phot,
        trials[["session_trial_num", "trial_start_sec", "trial_end_sec"]],
        left_on="t_sec",
        right_on="trial_start_sec",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.dropna(subset=["session_trial_num"])
    merged["session_trial_num"] = merged["session_trial_num"].astype(int)
    merged = merged[merged["t_sec"] <= merged["trial_end_sec"]].copy()
    merged["trial_time"] = merged["t_sec"] - merged["trial_start_sec"]
    return merged


# ----------------------------
# Metrics
# ----------------------------

def compute_ramp_slope_per_trial(
    phot_trial: pd.DataFrame,
    tt: TrialTimes,
    align: str,
    signal_col: str,
    pre_decision_buffer: float,
    t_min_after_align: float,
) -> Tuple[float, float, float]:
    """
    Compute slope in a window during waiting:
      start = align_time + t_min_after_align
      end   = decision_time - pre_decision_buffer
    align in {"cue_on", "cue_off"}; we require has_decision.
    """
    if not tt.has_decision or not np.isfinite(tt.decision):
        return np.nan, np.nan, np.nan

    align_t = getattr(tt, align)
    if not np.isfinite(align_t):
        return np.nan, np.nan, np.nan

    start = align_t + t_min_after_align
    end = tt.decision - pre_decision_buffer
    if not np.isfinite(end) or end <= start + 0.2:
        return np.nan, np.nan, np.nan

    seg = phot_trial[(phot_trial["trial_time"] >= start) & (phot_trial["trial_time"] <= end)]
    if seg.empty:
        return np.nan, np.nan, np.nan
    t = seg["trial_time"].to_numpy()
    y = seg[signal_col].to_numpy()
    return robust_linear_slope(t, y)


def compute_mean_in_windows(
    phot_trial: pd.DataFrame,
    tt: TrialTimes,
    signal_col: str,
    windows: List[Tuple[float, float]],
    relative_to: str = "decision",
) -> Dict[str, float]:
    """
    Compute mean DA in multiple windows.
    windows are (t_start, t_end) in seconds relative to `relative_to`.
    Example: [(-1.0, -0.5), (-0.5, -0.2)] relative_to="decision"
    """
    out: Dict[str, float] = {}
    if relative_to not in {"decision", "cue_on", "cue_off"}:
        raise ValueError("relative_to must be one of decision/cue_on/cue_off")

    ref_t = getattr(tt, relative_to) if relative_to != "decision" else tt.decision
    if not np.isfinite(ref_t):
        for w in windows:
            out[f"mean_{relative_to}_{w[0]:.3f}_{w[1]:.3f}"] = np.nan
        return out

    for (a, b) in windows:
        start = ref_t + a
        end = ref_t + b
        seg = phot_trial[(phot_trial["trial_time"] >= start) & (phot_trial["trial_time"] <= end)]
        key = f"mean_{relative_to}_{a:.3f}_{b:.3f}"
        out[key] = float(np.nanmean(seg[signal_col].to_numpy())) if len(seg) else np.nan
    return out


def time_resolved_correlation(
    phot: pd.DataFrame,
    trial_times: Dict[int, TrialTimes],
    trials: pd.DataFrame,
    align: str,
    signal_col: str,
    t_grid: np.ndarray,
    min_fraction_waiting: float = 0.8,
    pre_decision_buffer: float = 0.2,
) -> pd.DataFrame:
    """
    Compute corr(DA(t), wait_time) across trials at each time in t_grid,
    only using trials that are still waiting at that time.

    We restrict to trials with decisions (not missed).
    """
    assert align in {"cue_on", "cue_off"}

    # map trial -> wait_time label depending on align (cue_on vs cue_off)
    if align == "cue_on":
        wait_label = "wait_from_cue_on"
    else:
        wait_label = "wait_from_cue_off"

    # precompute decision times etc
    good_trials = []
    for tr, tt in trial_times.items():
        if not tt.has_decision or not np.isfinite(tt.decision):
            continue
        if not np.isfinite(getattr(tt, align)):
            continue
        good_trials.append(tr)
    good_trials = np.array(sorted(good_trials), dtype=int)
    if len(good_trials) < 10:
        raise ValueError("Not enough decision trials for time-resolved correlation.")

    # build per-trial wait time
    tt_df = pd.DataFrame({
        "session_trial_num": good_trials,
        "cue_on_t": [trial_times[int(tr)].cue_on for tr in good_trials],
        "cue_off_t": [trial_times[int(tr)].cue_off for tr in good_trials],
        "decision_t": [trial_times[int(tr)].decision for tr in good_trials],
    })
    tt_df["wait_from_cue_on"] = tt_df["decision_t"] - tt_df["cue_on_t"]
    tt_df["wait_from_cue_off"] = tt_df["decision_t"] - tt_df["cue_off_t"]

    # join to ensure same trial set as trials table (for flags, etc.)
    tt_df = tt_df.merge(trials[["session_trial_num"]], on="session_trial_num", how="inner")
    if tt_df.empty:
        raise ValueError("No overlapping trials between trial_times and trials table.")
    wait = tt_df.set_index("session_trial_num")[wait_label]

    # pre-index phot by trial
    phot_g = phot[phot["session_trial_num"].isin(tt_df["session_trial_num"])].copy()

    rows = []
    n_trials_total = len(tt_df)
    for t in t_grid:
        # trials still waiting at this relative time:
        # condition: t <= (decision - buffer) - align_time
        # i.e. decision_t - buffer >= align_t + t
        still_waiting = []
        da_vals = []
        wt_vals = []
        for tr in tt_df["session_trial_num"].to_numpy(dtype=int):
            tt = trial_times[int(tr)]
            align_t = getattr(tt, align)
            if not np.isfinite(align_t) or not np.isfinite(tt.decision):
                continue
            if tt.decision - pre_decision_buffer < align_t + t:
                continue
            # sample DA at that trial_time (align_t + t)
            target = align_t + t
            seg = phot_g[(phot_g["session_trial_num"] == tr)]
            if seg.empty:
                continue
            # nearest neighbor sample
            idx = np.argmin(np.abs(seg["trial_time"].to_numpy() - target))
            da = float(seg.iloc[idx][signal_col])
            if not np.isfinite(da):
                continue
            da_vals.append(da)
            wt_vals.append(float(wait.loc[tr]))
            still_waiting.append(tr)

        frac = len(still_waiting) / n_trials_total if n_trials_total else 0
        if frac < min_fraction_waiting or len(da_vals) < 10:
            rows.append({"t": t, "r": np.nan, "n": len(da_vals), "frac_waiting": frac})
            continue

        da_vals = np.asarray(da_vals)
        wt_vals = np.asarray(wt_vals)
        # Pearson r
        r = np.corrcoef(da_vals, wt_vals)[0, 1] if np.nanstd(da_vals) > 0 and np.nanstd(wt_vals) > 0 else np.nan
        rows.append({"t": t, "r": float(r), "n": len(da_vals), "frac_waiting": float(frac)})

    return pd.DataFrame(rows)


# ----------------------------
# Negative controls
# ----------------------------

def permutation_null(effect_fn, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Compute null distribution by permuting labels inside effect_fn."""
    vals = []
    for _ in range(n_perm):
        vals.append(effect_fn(shuffle="perm"))
    return np.asarray(vals, dtype=float)


def circular_shift_null(effect_fn, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Null distribution by circularly shifting labels (preserves slow drifts)."""
    vals = []
    for _ in range(n_perm):
        vals.append(effect_fn(shuffle="cshift"))
    return np.asarray(vals, dtype=float)


# ----------------------------
# Plot helpers
# ----------------------------

def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------

@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for photometry wait analysis."""
    trials_file: Optional[Path] = None
    events_file: Optional[Path] = None
    phot_file: Optional[Path] = None
    outdir: Path = Path("wait_analysis_out")
    signal_col: str = "dff_zscored"
    iso_col: str = "iso"
    impulsive_sec: float = 0.25
    pre_decision_buffer: float = 0.2
    t_min_after_align: float = 0.3
    n_perm: int = 1000
    seed: int = 0


def run_session(cfg: "AnalysisConfig", session_label: str = "") -> dict:
    """
    Run the full wait-time analysis for one session.
    Returns a summary dict of key per-session statistics.
    """
    label = f"[{session_label}] " if session_label else ""

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"session": session_label}

    # Load data
    trials = pd.read_csv(cfg.trials_file)
    events = pd.read_csv(cfg.events_file)
    phot = pd.read_csv(cfg.phot_file)

    # Validate required columns early
    for col in ["session_trial_num", "start_time", "end_time"]:
        if col not in trials.columns:
            raise ValueError(f"Trials file missing column: {col}")
    for col in ["session_trial_num", "trial_time", "state"]:
        if col not in events.columns:
            raise ValueError(f"Events file missing column: {col}")

    # Basic trial flags
    trials = trials.copy()
    trials["bg_restart"] = trials.get("bg_repeats", 0).fillna(0).astype(float) > 1
    trials["miss_trial"] = trials.get("miss_trial", False).astype(bool)
    trials["impulsive"] = trials.get("time_waited", np.nan).astype(float) < float(cfg.impulsive_sec)

    # Extract event times per trial
    trial_times = extract_trial_times(events)

    # Assign photometry to trials
    phot_t = assign_photometry_to_trials(phot, trials)

    # Restrict to the GRAB channel in the file (should already be)
    # Keep only signal columns we need
    for col in [cfg.signal_col, cfg.iso_col]:
        if col not in phot_t.columns:
            raise ValueError(f"Photometry file missing column: {col}")

    # Build trial metric table
    rows = []
    windows = [(-2.0, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, -0.2)]

    for tr in sorted(trials["session_trial_num"].astype(int).unique()):
        tr_row = trials.loc[trials["session_trial_num"].astype(int) == tr].iloc[0].to_dict()
        tt = trial_times.get(int(tr), None)
        if tt is None:
            continue

        phot_trial = phot_t[phot_t["session_trial_num"] == int(tr)]

        # Compute wait labels
        tr_row["cue_on_t"] = tt.cue_on
        tr_row["cue_off_t"] = tt.cue_off
        tr_row["decision_t"] = tt.decision
        tr_row["has_decision"] = tt.has_decision
        tr_row["wait_from_cue_on"] = (tt.decision - tt.cue_on) if tt.has_decision else np.nan
        tr_row["wait_from_cue_off"] = (tt.decision - tt.cue_off) if tt.has_decision else np.nan

        # Slopes (cue on and cue off), signal and iso
        for align in ["cue_on", "cue_off"]:
            a, b, r2 = compute_ramp_slope_per_trial(
                phot_trial, tt, align=align, signal_col=cfg.signal_col,
                pre_decision_buffer=cfg.pre_decision_buffer,
                t_min_after_align=cfg.t_min_after_align,
            )
            tr_row[f"slope_{align}"] = a
            tr_row[f"slope_{align}_r2"] = r2

            a_iso, _, r2_iso = compute_ramp_slope_per_trial(
                phot_trial, tt, align=align, signal_col=cfg.iso_col,
                pre_decision_buffer=cfg.pre_decision_buffer,
                t_min_after_align=cfg.t_min_after_align,
            )
            tr_row[f"slope_{align}_iso"] = a_iso
            tr_row[f"slope_{align}_iso_r2"] = r2_iso

            # time-reversal slope control (signal only)
            if tt.has_decision and np.isfinite(tt.decision) and np.isfinite(getattr(tt, align)):
                align_t = getattr(tt, align)
                start = align_t + cfg.t_min_after_align
                end = tt.decision - cfg.pre_decision_buffer
                seg = phot_trial[(phot_trial["trial_time"] >= start) & (phot_trial["trial_time"] <= end)]
                if len(seg) >= 5:
                    t = seg["trial_time"].to_numpy()
                    y = seg[cfg.signal_col].to_numpy()
                    # reverse time within segment
                    t_rev = (t.max() - t) + t.min()
                    a_rev, _, _ = robust_linear_slope(t_rev, y)
                else:
                    a_rev = np.nan
            else:
                a_rev = np.nan
            tr_row[f"slope_{align}_time_reversed"] = a_rev

        # Mean DA in windows before decision (signal and iso)
        means = compute_mean_in_windows(phot_trial, tt, signal_col=cfg.signal_col, windows=windows, relative_to="decision")
        means_iso = compute_mean_in_windows(phot_trial, tt, signal_col=cfg.iso_col, windows=windows, relative_to="decision")
        tr_row.update(means)
        tr_row.update({k + "_iso": v for k, v in means_iso.items()})

        rows.append(tr_row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(cfg.outdir / "trial_metrics.csv", index=False)

    # ----------------------------
    # Core correlations (single-session)
    # ----------------------------
    rng = np.random.default_rng(cfg.seed)

    def effect_slope_vs_wait(align: str, use_iso: bool = False, shuffle: Optional[str] = None) -> float:
        """
        Effect summary: Pearson r between slope and wait time (decision trials only).
        shuffle can be:
          - None: real
          - "perm": permute wait labels
          - "cshift": circular shift wait labels
        """
        slope_col = f"slope_{align}_iso" if use_iso else f"slope_{align}"
        wait_col = "wait_from_cue_on" if align == "cue_on" else "wait_from_cue_off"

        df = metrics.copy()
        df = df[df["has_decision"] == True]
        df = df[np.isfinite(df[slope_col]) & np.isfinite(df[wait_col])]
        if len(df) < 10:
            return np.nan

        y = df[wait_col].to_numpy(dtype=float)
        x = df[slope_col].to_numpy(dtype=float)

        if shuffle == "perm":
            y = rng.permutation(y)
        elif shuffle == "cshift":
            k = int(rng.integers(0, len(y)))
            y = np.roll(y, k)

        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    # Figure: slope vs wait scatter
    figdir = cfg.outdir / "figures"
    figdir.mkdir(exist_ok=True, parents=True)

    for align in ["cue_off", "cue_on"]:
        wait_col = "wait_from_cue_off" if align == "cue_off" else "wait_from_cue_on"
        slope_col = f"slope_{align}"
        df = metrics[(metrics["has_decision"] == True) & np.isfinite(metrics[wait_col]) & np.isfinite(metrics[slope_col])].copy()

        plt.figure(figsize=(6, 5))
        plt.scatter(df[wait_col], df[slope_col], alpha=0.7)
        r = np.corrcoef(df[wait_col], df[slope_col])[0, 1] if len(df) >= 2 else np.nan
        plt.xlabel(wait_col)
        plt.ylabel(slope_col)
        plt.title(f"Slope vs wait ({align})  r={r:.3f}")
        savefig(figdir / f"scatter_slope_vs_wait_{align}.png")

        # Negative controls: iso, shuffle, time-reversal
        r_real = effect_slope_vs_wait(align, use_iso=False, shuffle=None)
        r_iso = effect_slope_vs_wait(align, use_iso=True, shuffle=None)
        r_rev = np.corrcoef(
            df[wait_col].to_numpy(float),
            df[f"slope_{align}_time_reversed"].to_numpy(float)
        )[0, 1] if len(df) >= 2 and np.nanstd(df[f"slope_{align}_time_reversed"].to_numpy(float)) > 0 else np.nan

        # Null distributions
        null_perm = np.asarray([effect_slope_vs_wait(align, use_iso=False, shuffle="perm") for _ in range(cfg.n_perm)])
        null_csh = np.asarray([effect_slope_vs_wait(align, use_iso=False, shuffle="cshift") for _ in range(cfg.n_perm)])

        plt.figure(figsize=(7, 4))
        plt.hist(null_perm[np.isfinite(null_perm)], bins=40, alpha=0.6, label="perm null")
        plt.hist(null_csh[np.isfinite(null_csh)], bins=40, alpha=0.6, label="cshift null")
        plt.axvline(r_real, linewidth=2, label=f"real r={r_real:.3f}")
        plt.axvline(r_iso, linewidth=2, linestyle="--", label=f"iso r={r_iso:.3f}")
        if np.isfinite(r_rev):
            plt.axvline(r_rev, linewidth=2, linestyle=":", label=f"time-rev r={r_rev:.3f}")
        plt.xlabel("correlation r")
        plt.ylabel("count")
        plt.title(f"Null controls for slope vs wait ({align})")
        plt.legend()
        savefig(figdir / f"null_controls_slope_vs_wait_{align}.png")

    # ----------------------------
    # Option B: pre-decision window means (heatmap-ish summary)
    # ----------------------------
    for align in ["cue_off", "cue_on"]:
        wait_col = "wait_from_cue_off" if align == "cue_off" else "wait_from_cue_on"
        df = metrics[(metrics["has_decision"] == True) & np.isfinite(metrics[wait_col])].copy()
        if df.empty:
            continue

        # window columns
        win_cols = [c for c in df.columns if c.startswith("mean_decision_") and (not c.endswith("_iso"))]
        win_cols = sorted(win_cols)

        rs = []
        for c in win_cols:
            d = df[np.isfinite(df[c])].copy()
            if len(d) < 10:
                rs.append(np.nan)
                continue
            rs.append(float(np.corrcoef(d[c].to_numpy(float), d[wait_col].to_numpy(float))[0, 1]))

        plt.figure(figsize=(10, 3))
        plt.plot(range(len(win_cols)), rs, marker="o")
        plt.xticks(range(len(win_cols)), [c.replace("mean_decision_", "") for c in win_cols], rotation=45, ha="right")
        plt.ylabel("corr r")
        plt.title(f"Pre-decision window mean DA vs wait ({align})")
        savefig(figdir / f"pred_lick_window_corr_{align}.png")

    # ----------------------------
    # Option C: time-resolved correlation
    # ----------------------------
    t_grid = np.arange(0.0, 6.0, 0.1)  # seconds after align
    for align in ["cue_off", "cue_on"]:
        try:
            trcorr = time_resolved_correlation(
                phot=phot_t,
                trial_times=trial_times,
                trials=trials,
                align=align,
                signal_col=cfg.signal_col,
                t_grid=t_grid,
                min_fraction_waiting=0.8,
                pre_decision_buffer=cfg.pre_decision_buffer,
            )
        except Exception as e:
            print(f"[WARN] time-resolved correlation failed for {align}: {e}")
            continue

        trcorr.to_csv(cfg.outdir / f"time_resolved_corr_{align}.csv", index=False)

        plt.figure(figsize=(8, 4))
        plt.plot(trcorr["t"], trcorr["r"])
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.xlabel(f"time since {align} (s)")
        plt.ylabel("corr r (DA vs wait)")
        plt.title(f"Time-resolved correlation ({align})")
        savefig(figdir / f"time_resolved_corr_{align}.png")

    print(f"{label}Done. Wrote:\n- {cfg.outdir / 'trial_metrics.csv'}\n- {figdir}/*.png")
    return summary



if __name__ == "__main__":
    # ----------------------------------------------------------------
    # IDE RUN CONFIG — edit these instead of using the terminal
    # ----------------------------------------------------------------
    BATCH = True          # True = all sessions, False = single session below

    # Single-session paths (only used when BATCH = False)
    TRIALS_FILE = None    # e.g. Path("/Volumes/T7 Shield/photometry/behav_analyzed/2026-01-13_11-32-01_RZ083/trials_analyzed_2026-01-13_11-32-01_RZ083.csv")
    EVENTS_FILE = None    # e.g. Path("/Volumes/T7 Shield/photometry/behav_analyzed/2026-01-13_11-32-01_RZ083/events_processed_2026-01-13_11-32-01_RZ083.csv")
    PHOT_FILE   = None    # e.g. Path("/Volumes/T7 Shield/photometry/processed_output/RZ083_20260113_113201/G4.csv")
    OUTDIR      = None    # None = default (wait_analysis_out for single, PROCESSED_OUT/<id>/wait_analysis for batch)

    # Analysis settings
    SIGNAL_COL          = "dff_zscored"
    ISO_COL             = "iso"
    IMPULSIVE_SEC       = 0.25
    PRE_DECISION_BUFFER = 0.2
    T_MIN_AFTER_ALIGN   = 0.3
    N_PERM              = 1000
    SEED                = 0
    # ----------------------------------------------------------------

    if BATCH:
        from config import BEHAV_DIR, PHOTO_ROOT, PROCESSED_OUT
        from utils import get_grab_channel, load_merged_log, load_photometry_log

        merged_log = load_merged_log()
        photometry_log = load_photometry_log()
        all_summaries = []
        base_outdir = Path(OUTDIR) if OUTDIR else None

        for _, session_info in merged_log.iterrows():
            session_id = str(session_info["session_id"])
            behav_subdir = str(session_info["dir"])
            mouse = str(session_info["mouse"])
            date = str(session_info["date"])

            grab_channel = get_grab_channel(mouse, date, photometry_log)
            if grab_channel is None:
                print(f"[{session_id}] Skipping: no striatum GRAB channel in photometry log")
                continue

            trials_file = BEHAV_DIR / behav_subdir / f"trials_analyzed_{behav_subdir}.csv"
            events_file = BEHAV_DIR / behav_subdir / f"events_processed_{behav_subdir}.csv"
            phot_file = PROCESSED_OUT / session_id / f"{grab_channel}.csv"

            if not trials_file.exists() or not events_file.exists() or not phot_file.exists():
                print(f"[{session_id}] Skipping: missing input files")
                continue

            outdir = (base_outdir / session_id) if base_outdir else (PROCESSED_OUT / session_id / "wait_analysis")
            cfg = AnalysisConfig(
                trials_file=trials_file, events_file=events_file, phot_file=phot_file,
                outdir=outdir, signal_col=SIGNAL_COL, iso_col=ISO_COL,
                impulsive_sec=IMPULSIVE_SEC, pre_decision_buffer=PRE_DECISION_BUFFER,
                t_min_after_align=T_MIN_AFTER_ALIGN, n_perm=N_PERM, seed=SEED,
            )
            try:
                summary = run_session(cfg, session_label=session_id)
                summary.update({"mouse": mouse, "date": date, "grab_channel": grab_channel})
                all_summaries.append(summary)
            except Exception as e:
                print(f"[{session_id}] ERROR: {e}")

        if all_summaries:
            out_csv = (base_outdir or PHOTO_ROOT) / "wait_analysis_summary_all_sessions.csv"
            pd.DataFrame(all_summaries).to_csv(out_csv, index=False)
            print(f"\nBatch done. Global summary: {out_csv}")
        else:
            print("No sessions processed.")

    else:
        # Single-session mode
        trials_file = Path(TRIALS_FILE) if TRIALS_FILE else _find_file(["trials_analyzed_*.csv", "*trials*analyzed*.csv"])
        events_file = Path(EVENTS_FILE) if EVENTS_FILE else _find_file(["events_processed_*.csv", "*events*processed*.csv"])
        phot_file = Path(PHOT_FILE) if PHOT_FILE else _find_file(["G4.csv", "*G4*.csv", "*phot*processed*.csv", "photometry_long.csv"])
        outdir = Path(OUTDIR) if OUTDIR else Path("wait_analysis_out")
        cfg = AnalysisConfig(
            trials_file=trials_file, events_file=events_file, phot_file=phot_file,
            outdir=outdir, signal_col=SIGNAL_COL, iso_col=ISO_COL,
            impulsive_sec=IMPULSIVE_SEC, pre_decision_buffer=PRE_DECISION_BUFFER,
            t_min_after_align=T_MIN_AFTER_ALIGN, n_perm=N_PERM, seed=SEED,
        )
        run_session(cfg)

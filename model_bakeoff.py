# model_bakeoff.py — Phase 2 model bake-off (project checklist item 8).
#
# The tournament pipeline (mm_model) currently picks between RandomForest and
# BaggingClassifier on cross-val *accuracy*. But the betting work (item 7) showed
# the model is overconfident on chalk — its *probabilities* are off even when its
# picks are fine. Accuracy can't see that; Brier score, log-loss and calibration
# can. So this bakes a broader roster of classifiers off against each other on
# **probabilistic** quality, and tests whether post-hoc **calibration** (isotonic)
# fixes the overconfidence.
#
# All models are sklearn-only (no new deploy dependency): Logistic Regression (a
# naturally calibrated baseline), the incumbent RandomForest + Bagging,
# HistGradientBoosting, an MLP (the "neural net" from the notes), and an SVM.
#
# Method: walk-forward, same as precompute.py. For each test year we train on the
# prior years (the `all_prior` window) and predict the *actual* tournament games
# of that year — one out-of-sample probability per game that the higher seed wins
# (build_model_dataset gives both the feature diffs and the HIGH_SEED_WINS label,
# so no bracket cascade is involved). Metrics are pooled over all test years.
#
# Output (committed, consumed by app.py; the app can also call run_bakeoff live):
#   data/model_bakeoff_summary.csv       one row per model variant (raw + calib)
#   data/model_bakeoff_reliability.csv   reliability-curve bins per model variant
#   data/model_bakeoff_meta.csv          window, test-year span, generated time
#
# Usage:  python model_bakeoff.py
#
# NOTE: heavier than betting_strategies.py (it trains models), so it's a
# precompute step, not computed live on every app load — the app reads the CSVs.

import os
import sys

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

import mm_model as mm

DATA_DIR = "data"
WINDOW = "all_prior"        # most-data window — the strategy-lab headline window
N_BINS = 10                # reliability-curve resolution


# ──────────────────────────────────────────────────────────────
# MODEL ROSTER  (sklearn-only; each is scaled + variance-filtered)
# ──────────────────────────────────────────────────────────────

def _pipe(clf):
    """Standard preprocessing in front of a classifier (matches mm_model.PIPE:
    scale, then drop zero-variance columns). Harmless for trees, needed for the
    linear / SVM / MLP models."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("selector", VarianceThreshold()),
        ("classifier", clf),
    ])


def model_roster():
    """The base estimators to compare, keyed by display name. Fixed, sensible
    configs (not grid-searched) so the comparison is about model *family* and
    calibration, not hyperparameter tuning."""
    return {
        "Logistic Regression": _pipe(LogisticRegression(max_iter=2000)),
        "Random Forest": _pipe(RandomForestClassifier(
            n_estimators=200, random_state=0, n_jobs=-1)),
        "Bagging": _pipe(BaggingClassifier(n_estimators=20, random_state=0)),
        "HistGradientBoosting": _pipe(HistGradientBoostingClassifier(
            random_state=0)),
        "MLP (neural net)": _pipe(MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=2000, random_state=0)),
        "SVM (RBF)": _pipe(SVC(probability=True, random_state=0)),
    }


# ──────────────────────────────────────────────────────────────
# WALK-FORWARD OUT-OF-SAMPLE PREDICTIONS
# ──────────────────────────────────────────────────────────────

def _build_frames(df, features):
    """Precompute, per tournament year, the labeled feature-diff frame of that
    year's actual games. Returns (frames_by_year, tournament_years)."""
    all_rows, years = mm.build_all_matchups(df)
    rows_by_year = {}
    for r in all_rows:
        rows_by_year.setdefault(r["YEAR"], []).append(r)
    frames = {y: mm.build_model_dataset(rows_by_year.get(y, []), df, features)
              for y in years}
    return frames, years


def walk_forward_predictions(df=None, features=None, window=WINDOW,
                             roster=None, calibrate=True):
    """Walk-forward out-of-sample predictions for every model in the roster.

    For each test year (2009+), train on the window's prior years and predict the
    higher-seed-wins probability for that year's actual games. Optionally also
    fit an isotonic-calibrated version of each model. Returns a long DataFrame:
    columns [model, calibrated, year, p, y]."""
    if df is None:
        df = mm.load_data()
    if features is None:
        features = mm.get_features(df)
    roster = roster or model_roster()

    frames, years = _build_frames(df, features)
    window_size = mm.TRAINING_WINDOWS[window]
    feature_diffs = [f"{c}_DIFF" for c in features]
    test_years = [y for y in years if y >= 2009]

    recs = []
    for test_year in test_years:
        train_years = mm.get_train_years_for_window(years, test_year, window_size)
        train = pd.concat([frames[y] for y in train_years if not frames[y].empty],
                          ignore_index=True) if train_years else pd.DataFrame()
        test = frames.get(test_year, pd.DataFrame())
        if len(train) < 10 or test.empty:
            continue

        Xtr, ytr = train[feature_diffs].values, train["HIGH_SEED_WINS"].values
        Xte, yte = test[feature_diffs].values, test["HIGH_SEED_WINS"].values

        for name, est in roster.items():
            variants = [("raw", _fresh(est))]
            if calibrate:
                variants.append(("isotonic", CalibratedClassifierCV(
                    _fresh(est), method="isotonic", cv=3)))
            for tag, model in variants:
                try:
                    model.fit(Xtr, ytr)
                    p = _proba_high(model, Xte)
                except Exception as exc:                       # noqa: BLE001
                    print(f"  [skip] {name}/{tag} {test_year}: {exc}", flush=True)
                    continue
                for pi, yi in zip(p, yte):
                    recs.append({"model": name, "calibrated": tag,
                                 "year": int(test_year),
                                 "p": float(pi), "y": int(yi)})
        print(f"  {test_year}: trained {len(roster)} models "
              f"on {len(train)} games, scored {len(yte)}", flush=True)

    return pd.DataFrame(recs)


def _fresh(est):
    """A clean, unfitted clone so each (year, variant) trains independently."""
    from sklearn.base import clone
    return clone(est)


def _proba_high(model, X):
    """P(high seed wins) column from a fitted classifier's predict_proba."""
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else (proba.shape[1] - 1)
    return proba[:, idx]


# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────

def reliability_bins(p, y, n_bins=N_BINS):
    """Bin predictions into [0,1] deciles; return per-bin mean predicted prob,
    observed frequency, and count. The gap between the two is miscalibration."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin": b, "bin_mid": (edges[b] + edges[b + 1]) / 2,
                     "mean_pred": float(p[m].mean()),
                     "obs_freq": float(y[m].mean()), "count": int(m.sum())})
    return pd.DataFrame(rows)


def expected_calibration_error(p, y, n_bins=N_BINS):
    """ECE: count-weighted average gap between confidence and accuracy across
    bins. 0 = perfectly calibrated."""
    b = reliability_bins(p, y, n_bins)
    if b.empty:
        return float("nan")
    w = b["count"] / b["count"].sum()
    return float((w * (b["mean_pred"] - b["obs_freq"]).abs()).sum())


def summarize(preds):
    """One metrics row per (model, calibrated) variant, pooled over all years."""
    rows = []
    for (name, tag), g in preds.groupby(["model", "calibrated"]):
        p, y = g["p"].values, g["y"].values
        rows.append({
            "model": name, "calibrated": tag, "games": len(g),
            "accuracy": round(accuracy_score(y, p >= 0.5), 4),
            "brier": round(brier_score_loss(y, p), 4),
            "log_loss": round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 4),
            "ece": round(expected_calibration_error(p, y), 4),
        })
    out = pd.DataFrame(rows).sort_values(["brier", "log_loss"]).reset_index(drop=True)
    return out


def run_bakeoff(df=None, features=None, window=WINDOW, calibrate=True):
    """Full bake-off: returns (summary_df, reliability_df)."""
    preds = walk_forward_predictions(df, features, window, calibrate=calibrate)
    summary = summarize(preds)
    rel_rows = []
    for (name, tag), g in preds.groupby(["model", "calibrated"]):
        b = reliability_bins(g["p"].values, g["y"].values)
        b.insert(0, "calibrated", tag)
        b.insert(0, "model", name)
        rel_rows.append(b)
    reliability = pd.concat(rel_rows, ignore_index=True) if rel_rows else pd.DataFrame()
    return summary, reliability


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    df = mm.load_data()
    features = mm.get_features(df)
    print(f"Model bake-off — window '{WINDOW}', walk-forward from 2009.\n"
          f"Roster: {list(model_roster())}\n")
    summary, reliability = run_bakeoff(df, features)

    summary.to_csv(os.path.join(DATA_DIR, "model_bakeoff_summary.csv"), index=False)
    reliability.to_csv(os.path.join(DATA_DIR, "model_bakeoff_reliability.csv"),
                       index=False)
    pd.DataFrame([{
        "window": WINDOW,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "n_variants": summary[["model", "calibrated"]].drop_duplicates().shape[0],
    }]).to_csv(os.path.join(DATA_DIR, "model_bakeoff_meta.csv"), index=False)

    print("\nRanked by Brier score (lower = better calibrated probabilities):\n")
    print(f"{'model':22} {'cal':9} {'acc':>6} {'brier':>7} {'logloss':>8} {'ECE':>6}")
    for r in summary.itertuples(index=False):
        print(f"{r.model:22} {r.calibrated:9} {r.accuracy:>6.3f} {r.brier:>7.4f} "
              f"{r.log_loss:>8.4f} {r.ece:>6.4f}")
    print(f"\nWrote model_bakeoff_summary.csv ({len(summary)} rows) and "
          f"model_bakeoff_reliability.csv ({len(reliability)} rows).")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:          # noqa: BLE001
        pass
    main()

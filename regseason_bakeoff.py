# regseason_bakeoff.py — does training on the full ~103k-game log fix the
# tournament model's overfitting? (The "biggest modelling lever" follow-up.)
#
# The bake-off (model_bakeoff.py) showed the tournament classifiers are
# overconfident — they train on only ~1k tournament games. The general game model
# (game_model.py) already trains on ~90k regular-season games, but in a different
# feature space, so the comparison there isn't apples-to-apples.
#
# This isolates the ONE variable that matters — training-set SIZE — by holding the
# feature space (game_model's ratings.csv features + HOME_COURT) and the model
# family (HistGradientBoosting) fixed, and only swapping the training data:
#
#   * "Tournament-only"  — trained walk-forward on prior NCAA tournament games
#     only (~1k), the small-data regime.
#   * "Full game log"    — trained walk-forward on ALL prior games (conf, nonconf,
#     conf tourney, postseason; up to ~90k), the big-data regime.
#
# Both are evaluated on the SAME test set the bake-off uses: each season's actual
# NCAA tournament matchups (oriented high-seed-as-home, neutral court), scoring
# P(high seed wins) against HIGH_SEED_WINS. Isotonic-calibrated variants of each
# are included to ask whether more data makes post-hoc calibration unnecessary.
#
# No leakage: for test year Y, training uses only games/tournaments from years < Y.
#
# Output (committed, consumed by app.py):
#   data/regseason_compare_summary.csv       one row per (training set, calibration)
#   data/regseason_compare_reliability.csv   reliability-curve bins per variant
#
# Usage:  python regseason_bakeoff.py

import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

import mm_model as mm
import game_model as gm
import model_bakeoff as mb          # reuse the metric helpers

DATA_DIR = "data"


def _matchup_frame(matchups, ratings_year, features, cols):
    """Feature-diff rows (game_model space, neutral, high-seed-as-home) for a set
    of tournament matchups, plus the HIGH_SEED_WINS label. Skips matchups whose
    teams lack a ratings row that season."""
    idx = ratings_year.set_index("TEAM")
    rows = []
    for m in matchups:
        th, tl = m["TEAM_HIGH"], m["TEAM_LOW"]
        if th not in idx.index or tl not in idx.index:
            continue
        h, a = idx.loc[th], idx.loc[tl]
        row = {f"{c}_DIFF": float(h[c]) - float(a[c]) for c in features}
        row[gm.HOME_COURT] = 0.0
        row["y"] = int(m["HIGH_SEED_WINS"])
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return np.empty((0, len(cols))), np.empty((0,))
    return df[cols].values, df["y"].values


def walk_forward():
    """Return a long DataFrame [model, calibrated, year, p, y] of out-of-sample
    tournament-game predictions for each training regime."""
    kp = mm.load_data()
    ratings = gm.load_ratings()
    games = gm.load_games()
    features = gm.get_features(ratings)
    cols = gm.feature_columns(features)

    all_rows, tour_years = mm.build_all_matchups(kp)
    by_year = {}
    for r in all_rows:
        by_year.setdefault(r["YEAR"], []).append(r)
    test_years = [y for y in tour_years if y >= 2009]

    recs = []
    for ty in test_years:
        ry = ratings[ratings["YEAR"] == ty]
        Xte, yte = _matchup_frame(by_year.get(ty, []), ry, features, cols)
        if len(yte) == 0:
            continue
        prior_years = [y for y in tour_years if y < ty]

        # --- small-data regime: prior tournament matchups only ---
        tour_frames = []
        for py in prior_years:
            rpy = ratings[ratings["YEAR"] == py]
            Xt, yt = _matchup_frame(by_year.get(py, []), rpy, features, cols)
            if len(yt):
                tour_frames.append((Xt, yt))
        if not tour_frames:
            continue
        X_tour = np.vstack([x for x, _ in tour_frames])
        y_tour = np.concatenate([y for _, y in tour_frames])

        # --- big-data regime: ALL prior games ---
        big = gm.build_game_dataset(games[games["YEAR"] < ty], ratings, features)
        X_big, y_big = big[cols].values, big[gm.TARGET].values

        regimes = [("Tournament-only (~1k)", X_tour, y_tour),
                   ("Full game log (~90k)", X_big, y_big)]
        for name, Xtr, ytr in regimes:
            for tag, est in (("raw", gm.build_pipeline()),
                             ("isotonic", CalibratedClassifierCV(
                                 gm.build_pipeline(), method="isotonic", cv=3))):
                model = clone(est)
                model.fit(Xtr, ytr)
                p = mb._proba_high(model, Xte)
                for pi, yi in zip(p, yte):
                    recs.append({"model": name, "calibrated": tag,
                                 "year": int(ty), "p": float(pi), "y": int(yi)})
        print(f"  {ty}: tourney-train={len(y_tour)}, full-train={len(y_big):,}, "
              f"test={len(yte)}", flush=True)

    return pd.DataFrame(recs)


def main():
    print("Regular-season training bake-off — walk-forward from 2009.\n"
          "Same feature space + model (HistGradientBoosting); only the training "
          "set size changes.\n")
    preds = walk_forward()
    summary = mb.summarize(preds)

    rel_rows = []
    for (name, tag), g in preds.groupby(["model", "calibrated"]):
        b = mb.reliability_bins(g["p"].values, g["y"].values)
        b.insert(0, "calibrated", tag)
        b.insert(0, "model", name)
        rel_rows.append(b)
    reliability = pd.concat(rel_rows, ignore_index=True)

    summary.to_csv(os.path.join(DATA_DIR, "regseason_compare_summary.csv"),
                   index=False)
    reliability.to_csv(os.path.join(DATA_DIR, "regseason_compare_reliability.csv"),
                       index=False)

    print("\nPooled over every 2009–2025 NCAA tournament game "
          "(lower Brier/log-loss/ECE = better):\n")
    print(f"{'training set':24} {'cal':9} {'acc':>6} {'brier':>7} {'logloss':>8} {'ECE':>6}")
    for r in summary.itertuples(index=False):
        print(f"{r.model:24} {r.calibrated:9} {r.accuracy:>6.3f} {r.brier:>7.4f} "
              f"{r.log_loss:>8.4f} {r.ece:>6.4f}")
    print(f"\nWrote regseason_compare_summary.csv ({len(summary)} rows) and "
          f"regseason_compare_reliability.csv ({len(reliability)} rows).")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:          # noqa: BLE001
        pass
    main()

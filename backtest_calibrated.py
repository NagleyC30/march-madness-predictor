# backtest_calibrated.py — Phase 2, sub-step b: does the betting edge survive
# calibration?
#
# The strategy lab (item 7) found flat +EV betting turns a profit on the longer
# windows, but suspected it was largely an artifact of the model being
# OVERCONFIDENT on favorites: if the model says 90% when the truth is 84%, every
# such game looks +EV against the market even when it isn't. The bake-off
# (model_bakeoff.py) then measured that overconfidence directly.
#
# This script settles the SAME walk-forward bets as backtest_odds.py, at the SAME
# real closing lines, but replaces the model's raw probabilities with ISOTONIC-
# CALIBRATED ones (fit on the training seasons only, no leakage). It reuses
# backtest_odds.simulate_real_betting unchanged — only the model passed in
# differs — and writes a calibrated twin of bet_games.csv:
#
#   data/bet_games_calibrated.csv   same columns as bet_games.csv, calibrated model_p
#
# The strategy lab runs on this file identically, so the app can put raw vs
# calibrated ROI side by side. If the +EV profit shrinks toward zero once the
# probabilities are honest, the "edge" was mostly overconfidence.
#
# Usage:  python backtest_calibrated.py

import os
import sys

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

import mm_model as mm
import fetch_odds as fo
import backtest_odds as bo

DATA_DIR = "data"
CAL_METHOD = "isotonic"
CAL_CV = 3


def train_calibrated_model(df_train, features, method=CAL_METHOD, cv=CAL_CV):
    """Train the same best model backtest_odds uses, then wrap it in isotonic
    calibration refit (internally cross-validated) on the training seasons.

    Returns (calibrated_model, base_name, calibrated) where ``calibrated`` is
    False when calibration couldn't be fit (too few samples) and we fell back to
    the raw model so the game is still bettable."""
    model, name, _ = mm.train_best_model(df_train, features)
    feature_diffs = [f"{c}_DIFF" for c in features]
    X = df_train[feature_diffs].values
    y = df_train["HIGH_SEED_WINS"].values
    try:
        cal = CalibratedClassifierCV(clone(model), method=method, cv=cv)
        cal.fit(X, y)
        return cal, name, True
    except Exception as exc:                       # noqa: BLE001
        print(f"    [calib fallback] {name}: {exc}", flush=True)
        return model, name, False


def main():
    df = mm.load_data()
    features = mm.get_features(df)
    all_rows, tour_years = mm.build_all_matchups(df)

    lookup = bo.load_odds_lookup()
    odds_years = sorted(set(fo.AVAILABLE_YEARS) & set(tour_years))
    print(f"Calibrated real-odds backtest over {odds_years}")
    print(f"Windows: {list(mm.TRAINING_WINDOWS.keys())} "
          f"({CAL_METHOD}, cv={CAL_CV})\n")

    game_rows = []
    n_fallback = 0
    total = len(mm.TRAINING_WINDOWS) * len(odds_years)
    done = 0
    for window_name, window_size in mm.TRAINING_WINDOWS.items():
        for test_year in odds_years:
            done += 1
            train_years = mm.get_train_years_for_window(
                tour_years, test_year, window_size)
            if not train_years:
                continue
            train_rows = [r for r in all_rows if r["YEAR"] in train_years]
            df_train = mm.build_model_dataset(train_rows, df, features)
            if len(df_train) < 10:
                continue

            cal_model, name, calibrated = train_calibrated_model(df_train, features)
            n_fallback += (not calibrated)

            bo.model_df = df[df["YEAR"] == test_year]
            actual_games = [r for r in all_rows if r["YEAR"] == test_year]
            sim = bo.simulate_real_betting(
                actual_games, test_year, cal_model, features, lookup)

            for gr in sim["games"]:
                game_rows.append({"window": window_name, "year": test_year, **gr})
            print(f"[{done}/{total}] {window_name:13} {test_year} | {name:17} "
                  f"| {'calibrated' if calibrated else 'RAW (fallback)':14} "
                  f"| {len(sim['games'])} settleable", flush=True)

    out = pd.DataFrame(game_rows)
    out.to_csv(os.path.join(DATA_DIR, "bet_games_calibrated.csv"), index=False)
    print(f"\nWrote bet_games_calibrated.csv ({len(out)} game-predictions, "
          f"{n_fallback} window/year cells fell back to raw).")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:          # noqa: BLE001
        pass
    main()

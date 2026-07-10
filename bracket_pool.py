# bracket_pool.py — Phase 2, sub-step c: which model fills the best bracket?
#
# Accuracy and Brier score treat every game equally, but a bracket POOL doesn't:
# calling the champion is worth 320 points, a first-round game only 10. A model
# can be middling on raw accuracy yet win the pool by nailing the deep rounds
# (or tank it by busting a Final Four pick). This scores each model's full
# walk-forward bracket under the standard 10/20/40/80/160/320-by-round pool.
#
# Method: for each model and each test year (2009+), train on the prior seasons
# (all_prior window), simulate a COMPLETE bracket (mm.simulate_full_bracket), and
# score it with mm.score_predictions_cascade — the same cascade convention the
# app's walk-forward accuracy already uses (a game scores only if the matchup you
# predicted actually occurred). Points are summed across every tournament.
#
# A "Chalk (always higher seed)" baseline is included so the ML models have a
# meaningful benchmark: does any of them actually beat just picking the favorite?
#
# Roster is reused from model_bakeoff.py (sklearn-only, no new dependency).
#
# Output (committed, consumed by app.py):
#   data/bracket_pool_summary.csv   one row per model: total + per-round points
#   data/bracket_pool_by_year.csv   per (model, year) total, for the chart
#
# Usage:  python bracket_pool.py

import os
import sys

import numpy as np
import pandas as pd

import mm_model as mm
import model_bakeoff as mb

DATA_DIR = "data"
WINDOW = "all_prior"

# Classic ESPN-style by-round pool points.
POOL_POINTS = {"R64": 10, "R32": 20, "S16": 40, "E8": 80,
               "F4": 160, "Championship": 320}


class ChalkModel:
    """Baseline 'model' that always favors the better seed. Because
    predict_game_proba orders the higher seed as the 'high' class, returning a
    high P(high-seed-wins) makes every pick the favorite — a pure chalk bracket."""
    classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return np.tile([0.1, 0.9], (len(X), 1))


def pool_points(pred_rows, actual_df, year):
    """Per-round pool points for one simulated bracket, via the app's cascade
    scorer. Returns {round: points} plus 'total'."""
    scored = mm.score_predictions_cascade(pred_rows, actual_df, year)
    pts = {r: scored.get(r, {}).get("correct", 0) * POOL_POINTS[r]
           for r in POOL_POINTS}
    pts["total"] = sum(pts.values())
    return pts


def run_pool(df=None, features=None, window=WINDOW, roster=None):
    """Walk-forward bracket-pool score for every model. Returns
    (summary_df, by_year_df)."""
    if df is None:
        df = mm.load_data()
    if features is None:
        features = mm.get_features(df)
    if roster is None:
        roster = {"Chalk (always higher seed)": ChalkModel(), **mb.model_roster()}

    all_rows, years = mm.build_all_matchups(df)
    actual_df = pd.DataFrame(all_rows)
    window_size = mm.TRAINING_WINDOWS[window]
    feature_diffs = [f"{c}_DIFF" for c in features]
    test_years = [y for y in years if y >= 2009]

    by_year = []
    for name, est in roster.items():
        for test_year in test_years:
            train_years = mm.get_train_years_for_window(years, test_year, window_size)
            train_rows = [r for r in all_rows if r["YEAR"] in train_years]
            df_train = mm.build_model_dataset(train_rows, df, features)
            if len(df_train) < 10:
                continue
            model = mb._fresh(est) if not isinstance(est, ChalkModel) else ChalkModel()
            model.fit(df_train[feature_diffs].values, df_train["HIGH_SEED_WINS"].values)

            df_test = df[df["YEAR"] == test_year]
            preds = mm.simulate_full_bracket(df_test, model, test_year, features)
            pts = pool_points(preds, actual_df, test_year)
            by_year.append({"model": name, "year": int(test_year), **pts})
        print(f"  scored {name}", flush=True)

    by_year_df = pd.DataFrame(by_year)
    n_years = by_year_df["year"].nunique()
    round_cols = list(POOL_POINTS)
    summary = (by_year_df.groupby("model")[round_cols + ["total"]].sum()
               .reset_index())
    summary["avg_per_year"] = (summary["total"] / n_years).round(1)
    # Perfect bracket each year = 1920 pts; report share of the ceiling.
    summary["pct_of_max"] = (summary["total"] / (n_years * 1920) * 100).round(1)
    summary = summary.sort_values("total", ascending=False).reset_index(drop=True)
    return summary, by_year_df


def main():
    df = mm.load_data()
    features = mm.get_features(df)
    print(f"Bracket-pool bake-off — window '{WINDOW}', walk-forward from 2009.\n")
    summary, by_year = run_pool(df, features)

    summary.to_csv(os.path.join(DATA_DIR, "bracket_pool_summary.csv"), index=False)
    by_year.to_csv(os.path.join(DATA_DIR, "bracket_pool_by_year.csv"), index=False)

    print("\nRanked by total pool points "
          "(10/20/40/80/160/320 by round; perfect = 1920/yr):\n")
    print(f"{'model':28} {'total':>6} {'avg/yr':>7} {'%max':>6}  "
          f"{'R64':>4} {'R32':>4} {'S16':>4} {'E8':>4} {'F4':>5} {'Chip':>5}")
    for r in summary.itertuples(index=False):
        print(f"{r.model:28} {r.total:>6.0f} {r.avg_per_year:>7.1f} {r.pct_of_max:>5.1f}%  "
              f"{r.R64:>4.0f} {r.R32:>4.0f} {r.S16:>4.0f} {r.E8:>4.0f} "
              f"{r.F4:>5.0f} {r.Championship:>5.0f}")
    print(f"\nWrote bracket_pool_summary.csv ({len(summary)} rows) and "
          f"bracket_pool_by_year.csv ({len(by_year)} rows).")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:          # noqa: BLE001
        pass
    main()

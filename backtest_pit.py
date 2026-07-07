# backtest_pit.py — Point-in-time vs. season-aggregate reality check.
#
# Runs the walk-forward backtest twice on the SAME games: once with point-in-time
# ratings (as-of game date) and once with season-aggregate ratings (end of
# season). The accuracy/Brier gap is exactly how much the season-aggregate
# leakage flatters the numbers.
#
# Needs data/games_pit.csv (python fetch_data.py --pit 2011 2026) and
# data/ratings.csv. Writes small aggregate CSVs the dashboard reads:
#   data/game_backtest_pit_summary.csv   per (season, game_type), both methods
#   data/game_backtest_pit_meta.csv      headline metrics + the gap
#
# Usage:  python backtest_pit.py

import os

import numpy as np
import pandas as pd

import game_model as gm

DATA_DIR = "data"
PIT_FILE = os.path.join(DATA_DIR, "games_pit.csv")
MIN_TRAIN_SEASONS = 3
EPS = 1e-15
KEY = ["YEAR", "DATE", "HOME", "AWAY"]
# Older seasons only have time-machine snapshots from mid-season on, so their
# point-in-time coverage is partial and biased toward late games. Only compare
# on seasons where snapshots cover most of the schedule.
COVERAGE_MIN = 0.85


def _aligned_datasets():
    """Return (pit, agg) frames covering the identical set of games, row-aligned.

    pit comes straight from games_pit.csv; agg is the season-aggregate feature
    matrix built for those same games from ratings.csv.
    """
    pit = pd.read_csv(PIT_FILE)
    ratings = gm.load_ratings()
    features = gm.get_features(ratings)

    games_like = pit[["YEAR", "DATE", "HOME", "AWAY", "NEUTRAL", gm.TARGET]].copy()
    agg = gm.build_game_dataset(games_like, ratings, features, keep_meta=True)

    pit = pit.set_index(KEY)
    agg = agg.set_index(KEY)
    common = pit.index.intersection(agg.index).sort_values()
    pit = pit.loc[common].reset_index()
    agg = agg.loc[common].reset_index()

    # Keep only well-covered seasons (same row-mask applied to both, so the two
    # frames stay aligned).
    totals = gm.load_games().groupby("YEAR").size()
    cov = pit.groupby("YEAR").size() / totals
    usable = sorted(y for y in cov.index if cov.get(y, 0) >= COVERAGE_MIN)
    print("  season coverage: " + ", ".join(
        f"{y}:{cov[y]:.0%}" for y in sorted(cov.index)), flush=True)
    print(f"  usable seasons (>={COVERAGE_MIN:.0%}): {usable}", flush=True)

    mask = pit["YEAR"].isin(usable).values
    pit = pit[mask].reset_index(drop=True)
    agg = agg[mask].reset_index(drop=True)
    return pit, agg, features


def _walk_forward(pit, agg, features):
    """Train both methods on prior seasons, predict each season; return per-game
    predictions with P_PIT and P_AGG side by side."""
    cols = gm.feature_columns(features)
    seasons = sorted(pit["YEAR"].unique())
    parts = []
    for i, year in enumerate(seasons):
        if i < MIN_TRAIN_SEASONS:
            continue
        tr = pit["YEAR"] < year
        te = (pit["YEAR"] == year).values

        m_pit, _ = gm.train_model(pit[tr], features, do_cv=False)
        m_agg, _ = gm.train_model(agg[tr], features, do_cv=False)
        ip = list(m_pit.classes_).index(1)
        ia = list(m_agg.classes_).index(1)
        p_pit = m_pit.predict_proba(pit.loc[te, cols].values)[:, ip]
        p_agg = m_agg.predict_proba(agg.loc[te, cols].values)[:, ia]

        part = pit.loc[te, ["YEAR", "GAME_TYPE", "NEUTRAL", gm.TARGET]].copy()
        part["P_PIT"] = p_pit
        part["P_AGG"] = p_agg
        parts.append(part)
        print(f"[{year}] scored {int(te.sum()):,} games (trained on {int(tr.sum()):,})",
              flush=True)
    return pd.concat(parts, ignore_index=True)


def _metrics(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    acc = float(((p >= 0.5).astype(int) == y).mean())
    brier = float(np.mean((p - y) ** 2))
    ll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    return acc, brier, ll


def _summary(preds):
    y = preds[gm.TARGET].values
    rows = preds.copy()
    for m in ("PIT", "AGG"):
        p = np.clip(rows[f"P_{m}"].values, EPS, 1 - EPS)
        rows[f"correct_{m.lower()}"] = ((p >= 0.5).astype(int) == y).astype(int)
        rows[f"sum_sq_{m.lower()}"] = (p - y) ** 2
        rows[f"sum_ll_{m.lower()}"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    g = rows.groupby(["YEAR", "GAME_TYPE"])
    return pd.DataFrame({
        "n": g.size(),
        "correct_pit": g["correct_pit"].sum(),
        "correct_agg": g["correct_agg"].sum(),
        "sum_sq_pit": g["sum_sq_pit"].sum(),
        "sum_sq_agg": g["sum_sq_agg"].sum(),
        "sum_ll_pit": g["sum_ll_pit"].sum(),
        "sum_ll_agg": g["sum_ll_agg"].sum(),
    }).reset_index()


def build():
    print("Aligning point-in-time and season-aggregate game sets…", flush=True)
    pit, agg, features = _aligned_datasets()
    print(f"  {len(pit):,} common games", flush=True)

    preds = _walk_forward(pit, agg, features)
    y = preds[gm.TARGET].values
    acc_p, brier_p, ll_p = _metrics(preds["P_PIT"].values, y)
    acc_a, brier_a, ll_a = _metrics(preds["P_AGG"].values, y)

    meta = pd.DataFrame([{
        "n_games": len(preds),
        "season_min": int(preds["YEAR"].min()),
        "season_max": int(preds["YEAR"].max()),
        "n_seasons": preds["YEAR"].nunique(),
        "acc_pit": acc_p, "acc_agg": acc_a,
        "brier_pit": brier_p, "brier_agg": brier_a,
        "logloss_pit": ll_p, "logloss_agg": ll_a,
        "base_home_acc": float(y.mean()),
    }])

    _summary(preds).to_csv(
        os.path.join(DATA_DIR, "game_backtest_pit_summary.csv"), index=False)
    meta.to_csv(os.path.join(DATA_DIR, "game_backtest_pit_meta.csv"), index=False)

    m = meta.iloc[0]
    print(f"\nCompared on {int(m['n_games']):,} games, "
          f"{int(m['season_min'])}–{int(m['season_max'])}")
    print(f"  accuracy  point-in-time {acc_p:.4f}  vs  season-aggregate {acc_a:.4f}  "
          f"(gap {acc_a - acc_p:+.4f})")
    print(f"  Brier     point-in-time {brier_p:.4f}  vs  season-aggregate {brier_a:.4f}")
    print(f"  log loss  point-in-time {ll_p:.4f}  vs  season-aggregate {ll_a:.4f}")


if __name__ == "__main__":
    build()

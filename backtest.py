# backtest.py — Walk-forward backtest + calibration for the general game model.
#
# For each season, trains ONLY on prior seasons and predicts that season's
# games (honest out-of-sample). Writes small aggregate CSVs to ./data that the
# "Backtest & Calibration" Streamlit page reads — no per-game file is shipped.
#
#   data/game_backtest_summary.csv  — per (season, game_type): n, correct, and
#                                      summed squared error / log loss (additive,
#                                      so the app can re-aggregate any slice).
#   data/game_calibration.csv       — per group (all / game type / venue) and
#                                      probability bin: n, summed pred, summed
#                                      actual → reliability diagram.
#   data/game_backtest_meta.csv     — one row of headline metrics + baselines.
#
# Caveat (documented in the UI): ratings are season-aggregate, so a game's
# features already reflect that game — in-season accuracy is mildly optimistic.
# Walk-forward training removes the separate train-on-the-future leakage.
#
# Usage:  python backtest.py            (needs data/games.csv + data/ratings.csv)

import os

import numpy as np
import pandas as pd

import game_model as gm

DATA_DIR = "data"
MIN_TRAIN_SEASONS = 3          # need a few prior seasons before we trust a fit
N_BINS = 20                    # calibration resolution (0.05-wide bins)
EPS = 1e-15                    # log-loss clipping


def run_walk_forward():
    """Return a per-game frame of out-of-sample predictions across all seasons
    that have at least MIN_TRAIN_SEASONS of prior data to train on."""
    ratings = gm.load_ratings()
    games = gm.load_games()
    features = gm.get_features(ratings)
    ds = gm.build_game_dataset(games, ratings, features, keep_meta=True)
    cols = gm.feature_columns(features)

    seasons = sorted(ds["YEAR"].unique())
    parts = []
    for i, year in enumerate(seasons):
        if i < MIN_TRAIN_SEASONS:
            continue
        train = ds[ds["YEAR"] < year]
        test = ds[ds["YEAR"] == year]
        model, _ = gm.train_model(train, features, do_cv=False)
        idx1 = list(model.classes_).index(1)
        p_home = model.predict_proba(test[cols].values)[:, idx1]

        part = test[["YEAR", "GAME_TYPE", "NEUTRAL", gm.TARGET]].copy()
        part["P_HOME"] = p_home
        parts.append(part)
        print(f"[{year}] trained on {len(train):,} prior games, "
              f"scored {len(test):,}", flush=True)

    preds = pd.concat(parts, ignore_index=True)
    preds["CORRECT"] = ((preds["P_HOME"] >= 0.5).astype(int)
                        == preds[gm.TARGET]).astype(int)
    return preds


def summarize(preds):
    """Per (season, game_type) rollup with additive error sums."""
    y = preds[gm.TARGET].values
    p = preds["P_HOME"].values
    preds = preds.assign(
        SQ_ERR=(p - y) ** 2,
        LOG_LOSS=-(y * np.log(np.clip(p, EPS, 1))
                   + (1 - y) * np.log(np.clip(1 - p, EPS, 1))),
    )
    g = preds.groupby(["YEAR", "GAME_TYPE"])
    return pd.DataFrame({
        "n": g.size(),
        "correct": g["CORRECT"].sum(),
        "sum_sq": g["SQ_ERR"].sum(),
        "sum_ll": g["LOG_LOSS"].sum(),
    }).reset_index()


def calibrate(preds):
    """Reliability bins per group: all games, each game type, home, neutral."""
    edges = np.linspace(0, 1, N_BINS + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    groups = {"all": preds,
              "home": preds[~preds["NEUTRAL"]],
              "neutral": preds[preds["NEUTRAL"]]}
    for gt in sorted(preds["GAME_TYPE"].unique()):
        groups[gt] = preds[preds["GAME_TYPE"] == gt]

    rows = []
    for name, sub in groups.items():
        b = np.clip(np.digitize(sub["P_HOME"].values, edges) - 1, 0, N_BINS - 1)
        for k in range(N_BINS):
            mask = b == k
            n = int(mask.sum())
            if n == 0:
                continue
            rows.append({
                "group": name, "bin_mid": mids[k], "n": n,
                "sum_pred": float(sub["P_HOME"].values[mask].sum()),
                "sum_actual": float(sub[gm.TARGET].values[mask].sum()),
            })
    return pd.DataFrame(rows)


def build():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Running walk-forward backtest…", flush=True)
    preds = run_walk_forward()

    summary = summarize(preds)
    calib = calibrate(preds)

    n = len(preds)
    y = preds[gm.TARGET].values
    p = preds["P_HOME"].values
    meta = pd.DataFrame([{
        "n_games": n,
        "season_min": int(preds["YEAR"].min()),
        "season_max": int(preds["YEAR"].max()),
        "n_seasons": preds["YEAR"].nunique(),
        "accuracy": float(preds["CORRECT"].mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(np.mean(-(y * np.log(np.clip(p, EPS, 1))
                                    + (1 - y) * np.log(np.clip(1 - p, EPS, 1))))),
        "base_home_acc": float(y.mean()),           # always-pick-home baseline
    }])

    summary.to_csv(os.path.join(DATA_DIR, "game_backtest_summary.csv"), index=False)
    calib.to_csv(os.path.join(DATA_DIR, "game_calibration.csv"), index=False)
    meta.to_csv(os.path.join(DATA_DIR, "game_backtest_meta.csv"), index=False)

    m = meta.iloc[0]
    print(f"\nEvaluated {n:,} games, {int(m['season_min'])}–{int(m['season_max'])}")
    print(f"  accuracy {m['accuracy']:.4f}  (always-home {m['base_home_acc']:.4f})")
    print(f"  Brier {m['brier']:.4f}   log loss {m['log_loss']:.4f}")


if __name__ == "__main__":
    build()

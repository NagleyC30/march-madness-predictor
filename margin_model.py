# margin_model.py — Tier-2 betting expansion: a MARGIN model that bets the SPREAD.
#
# Every other model in this project is a classifier (win/lose). The betting work
# then showed the tournament *moneyline* and the chalk-flip *spread* are both
# efficient — no edge. This asks a genuinely different question: instead of "who
# wins," predict the **point margin** and bet it against the **real closing
# spread** on every game (not just the heavy-chalk flip). The honest, full
# version of "beat the spread."
#
# Model: the same feature space as game_model.py (HOME-minus-AWAY rating diffs +
# a HOME_COURT flag) but a **regression** target — the actual point margin
# (HOME_SCORE - AWAY_SCORE) from data/games.csv. Trained walk-forward: to bet a
# tournament year it trains only on games from strictly-earlier seasons.
#
# TWO RATINGS REGIMES, exactly like the Backtest & Calibration page's reality
# check — because a spread is a razor-thin target and leakage flatters it hugely:
#   * point-in-time (pit)   ratings as they stood the morning of each game
#                           (data/games_pit.csv). The HONEST number.
#   * season-aggregate (agg) end-of-season ratings (data/ratings.csv), which
#                           already encode the tournament result being predicted —
#                           so its ATS "edge" is inflated, kept only as a foil.
#
# Betting: for each real NCAA-tournament game (mm_model's bracket matchups) with a
# real closing spread + final score (data/odds.csv), predict the neutral-court
# margin, compare it to the number, and bet the side the model thinks covers.
# Settle at the real result and the real spread price when odds.csv carries one
# (else standard -110). A **points-edge** sweep (only bet when the model disagrees
# with the number by >= K points) is the spread analogue of the moneyline lab's
# +EV edge threshold.
#
# Output (committed, consumed by app.py):
#   data/margin_model_ats.csv    per (method, test_year-pooled, edge_pts) ATS P&L
#   data/margin_model_meta.csv   per method: coverage, ATS cover %, margin MAE
#   data/margin_bets.csv         one row per settled game (method-tagged)
#
# Usage:  python margin_model.py

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import game_model as gm
import mm_model as mm
import fetch_odds as fo
import backtest_odds as bo

DATA_DIR = "data"
GAMES_FILE = os.path.join(DATA_DIR, "games.csv")
PIT_FILE = os.path.join(DATA_DIR, "games_pit.csv")
KEY = ["YEAR", "DATE", "HOME", "AWAY"]

STAKE = 100.0
SPREAD_ODDS = -110                        # fallback juice; real price used when present
EDGE_PTS_SWEEP = (0.0, 1.0, 2.0, 3.0, 5.0)   # min points the model must beat the number by
MARGIN = "MARGIN"
MIN_TRAIN_GAMES = 2000


# ──────────────────────────────────────────────────────────────
# MODEL  (shared regressor head)
# ──────────────────────────────────────────────────────────────

def fit_regressor(X, y, random_state=0):
    """StandardScaler + HistGradientBoostingRegressor. Returns (model, in-sample
    MAE) — the same light stack game_model uses, just a regressor."""
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("reg", HistGradientBoostingRegressor(random_state=random_state))])
    pipe.fit(X, y)
    return pipe, float(np.mean(np.abs(pipe.predict(X) - y)))


# ──────────────────────────────────────────────────────────────
# SEASON-AGGREGATE regime — features from ratings.csv (leakage-inflated)
# ──────────────────────────────────────────────────────────────

def build_agg_dataset(games, ratings, features):
    """Every game as HOME-minus-AWAY season-aggregate rating diffs + HOME_COURT,
    with the real point MARGIN as the regression target."""
    g = games.dropna(subset=["HOME_SCORE", "AWAY_SCORE"]).copy()
    g[MARGIN] = g["HOME_SCORE"].astype(float) - g["AWAY_SCORE"].astype(float)
    feat = ratings[["YEAR", "TEAM"] + features]
    m = g.merge(feat.rename(columns={"TEAM": "HOME"}), on=["YEAR", "HOME"], how="inner")
    m = m.merge(feat.rename(columns={"TEAM": "AWAY"}), on=["YEAR", "AWAY"],
                how="inner", suffixes=("_H", "_A"))
    out = pd.DataFrame({"YEAR": m["YEAR"].values})
    for c in features:
        out[f"{c}_DIFF"] = (m[f"{c}_H"] - m[f"{c}_A"]).values
    out[gm.HOME_COURT] = np.where(m["NEUTRAL"].values, 0.0, 1.0)
    out[MARGIN] = m[MARGIN].values
    return out


def agg_ratings_index(ratings, features):
    """(year, team_key) -> season-aggregate feature vector, keyed by the same
    normalizer fetch_odds uses so ratings, the bracket and the odds all align."""
    r = ratings.copy()
    r["__key"] = r["TEAM"].map(fo.team_key)
    r = r.drop_duplicates(subset=["YEAR", "__key"], keep="first")
    keys = list(zip(r["YEAR"].astype(int), r["__key"]))
    mat = r[features].to_numpy(dtype=float)          # avoids itertuples name-mangling
    return {k: mat[i] for i, k in enumerate(keys)}


# ──────────────────────────────────────────────────────────────
# POINT-IN-TIME regime — pre-differenced features from games_pit.csv (honest)
# ──────────────────────────────────────────────────────────────

def load_pit_with_margin():
    """games_pit.csv (pre-differenced point-in-time features + HOME_WIN) with the
    real MARGIN attached from games.csv. Returns (frame, diff_cols)."""
    pit = pd.read_csv(PIT_FILE)
    g = pd.read_csv(GAMES_FILE)[KEY + ["HOME_SCORE", "AWAY_SCORE"]]
    pit = pit.merge(g, on=KEY, how="left").dropna(subset=["HOME_SCORE", "AWAY_SCORE"])
    pit[MARGIN] = pit["HOME_SCORE"].astype(float) - pit["AWAY_SCORE"].astype(float)
    diff_cols = [c for c in pit.columns if c.endswith("_DIFF")]
    return pit, diff_cols


def pit_matchup_index(pit, diff_cols):
    """(year, frozenset{home_key, away_key}) -> (home-minus-away diff vector,
    home_key) for postseason games, so a bracket matchup can pull its own
    point-in-time features (oriented later to high-minus-low)."""
    post = pit[pit["GAME_TYPE"] == "postseason"]
    years = post["YEAR"].astype(int).to_numpy()
    hk = post["HOME"].map(fo.team_key).to_numpy()
    ak = post["AWAY"].map(fo.team_key).to_numpy()
    mat = post[diff_cols].to_numpy(dtype=float)
    idx = {}
    for i in range(len(post)):
        idx[(int(years[i]), frozenset((hk[i], ak[i])))] = (mat[i], hk[i])
    return idx


# ──────────────────────────────────────────────────────────────
# ATS SETTLEMENT  (regime-agnostic: give it a predict_fn)
# ──────────────────────────────────────────────────────────────

def _spread_price(game, key):
    p = game.get("spread_price", {}).get(key)
    return int(p) if p is not None else SPREAD_ODDS


def settle_games(matchups, year, predict_fn, lookup):
    """Grade every real tournament game whose pair has a rating pair and a real
    spread+score. predict_fn(kh, kl) -> predicted HIGH-minus-LOW margin (or None).
    Actual margin + push are read from the real odds/score lookup, so both regimes
    settle identically — only the prediction differs."""
    recs = []
    for g in matchups:
        kh, kl = fo.team_key(g["TEAM_HIGH"]), fo.team_key(g["TEAM_LOW"])
        game = lookup.get((year, frozenset((kh, kl))))
        if game is None:
            continue
        spread, score = game.get("spread", {}), game.get("score", {})
        if not ({kh, kl} <= spread.keys() and {kh, kl} <= score.keys()):
            continue
        pred = predict_fn(kh, kl)
        if pred is None:
            continue
        sp_h = float(spread[kh])                     # high seed's number (– if favored)
        actual = float(score[kh] - score[kl])        # real high-minus-low margin
        bet_high = (pred + sp_h) > 0                  # model thinks high covers?
        edge_pts = abs(pred + sp_h)
        pick_key = kh if bet_high else kl
        pick_spread = sp_h if bet_high else float(spread[kl])
        pick_actual = actual if bet_high else -actual
        cover = pick_actual + pick_spread            # >0 cover, ==0 push, <0 loss
        recs.append({
            "year": year, "round": g["ROUND"],
            "pick": g["TEAM_HIGH"] if bet_high else g["TEAM_LOW"],
            "pick_is_high": bet_high, "pred_margin": round(pred, 2),
            "spread_high": sp_h, "actual_margin": actual,
            "edge_pts": round(edge_pts, 2),
            "covered": cover > 0, "push": cover == 0,
            "price": _spread_price(game, pick_key),
        })
    return recs


def sweep_ats(bets, thresholds=EDGE_PTS_SWEEP, stake=STAKE):
    """Roll per-game ATS records up into (edge_pts -> P&L), pooled over years."""
    rows = []
    for k in thresholds:
        sub = bets[bets["edge_pts"] >= k]
        placed = won = lost = push = 0
        net = 0.0
        for r in sub.itertuples(index=False):
            if r.push:
                push += 1
                continue
            placed += 1
            if r.covered:
                net += mm.american_odds_payout(r.price, stake); won += 1
            else:
                net -= stake; lost += 1
        wagered = placed * stake
        rows.append({
            "edge_pts": k, "placed": placed, "won": won, "lost": lost, "push": push,
            "net_pnl": round(net, 2), "total_wagered": round(wagered, 2),
            "roi_pct": round(net / wagered * 100, 1) if wagered else 0.0,
            "cover_pct": round(won / placed * 100, 1) if placed else 0.0,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# WALK-FORWARD DRIVERS
# ──────────────────────────────────────────────────────────────

def run_agg(all_rows, odds_years, lookup, ratings, games, features):
    """Season-aggregate regime over every odds year."""
    ridx = agg_ratings_index(ratings, features)
    cols = gm.feature_columns(features)
    bets = []
    for year in odds_years:
        ds = build_agg_dataset(games[games["YEAR"] < year], ratings, features)
        if len(ds) < MIN_TRAIN_GAMES:
            continue
        model, _ = fit_regressor(ds[cols].values, ds[MARGIN].values)

        def _pred(kh, kl, _m=model):
            hv, av = ridx.get((year, kh)), ridx.get((year, kl))
            if hv is None or av is None:
                return None
            x = np.concatenate([hv - av, [0.0]]).reshape(1, -1)   # neutral court
            return float(_m.predict(x)[0])

        recs = settle_games([r for r in all_rows if r["YEAR"] == year],
                            year, _pred, lookup)
        for r in recs:
            r["method"] = "agg"
        bets.extend(recs)
        print(f"  agg {year}: {len(recs):>2} bets", flush=True)
    return bets


def run_pit(all_rows, odds_years, lookup):
    """Point-in-time regime — the honest one — over the years games_pit covers."""
    pit, diff_cols = load_pit_with_margin()
    cols = diff_cols + [gm.HOME_COURT]
    pidx = pit_matchup_index(pit, diff_cols)
    bets = []
    for year in odds_years:
        train = pit[pit["YEAR"] < year]
        if len(train) < MIN_TRAIN_GAMES:
            continue
        model, _ = fit_regressor(train[cols].values, train[MARGIN].values)

        def _pred(kh, kl, _m=model):
            entry = pidx.get((year, frozenset((kh, kl))))
            if entry is None:
                return None
            diff, home_key = entry
            oriented = diff if home_key == kh else -diff      # -> high-minus-low
            x = np.concatenate([oriented, [0.0]]).reshape(1, -1)
            return float(_m.predict(x)[0])

        recs = settle_games([r for r in all_rows if r["YEAR"] == year],
                            year, _pred, lookup)
        for r in recs:
            r["method"] = "pit"
        bets.extend(recs)
        print(f"  pit {year}: {len(recs):>2} bets", flush=True)
    return bets


def _method_meta(bets, method):
    b = bets[bets["method"] == method]
    non_push = b[~b["push"]]
    cover = float(non_push["covered"].sum() / max(len(non_push), 1) * 100)
    mae = float(np.mean(np.abs(b["pred_margin"] - b["actual_margin"]))) if len(b) else np.nan
    # Honesty diagnostics: how good is the ARCHIVED spread itself? The market's
    # own implied margin is -spread_high; if it correlates with actual only weakly
    # (and the model's MAE beats it), the "edge" is mostly beating a noisy line.
    spread_pred = -b["spread_high"]
    spread_mae = float(np.mean(np.abs(spread_pred - b["actual_margin"]))) if len(b) else np.nan
    model_corr = float(np.corrcoef(b["pred_margin"], b["actual_margin"])[0, 1]) if len(b) > 2 else np.nan
    spread_corr = float(np.corrcoef(spread_pred, b["actual_margin"])[0, 1]) if len(b) > 2 else np.nan
    yrs = sorted(int(y) for y in b["year"].unique())
    return {
        "method": method,
        "odds_years": ",".join(str(y) for y in yrs),
        "n_years": len(yrs), "n_bettable": len(b),
        "stake": STAKE, "spread_odds": SPREAD_ODDS,
        "ats_cover_pct_all": round(cover, 1),
        "margin_mae": round(mae, 2), "spread_mae": round(spread_mae, 2),
        "model_corr": round(model_corr, 3), "spread_corr": round(spread_corr, 3),
        "breakeven_pct": 52.4,
    }


def main():
    ratings = gm.load_ratings()
    games = gm.load_games()
    features = gm.get_features(ratings)

    df = mm.load_data()
    all_rows, tour_years = mm.build_all_matchups(df)
    lookup = bo.load_odds_lookup()
    odds_years = sorted(set(fo.AVAILABLE_YEARS) & set(tour_years))
    print(f"Margin/ATS backtest over {odds_years}\n")

    print("Point-in-time (honest):")
    pit_bets = run_pit(all_rows, odds_years, lookup)
    print("Season-aggregate (leakage-inflated foil):")
    agg_bets = run_agg(all_rows, odds_years, lookup, ratings, games, features)

    bets = pd.DataFrame(pit_bets + agg_bets)
    bets.to_csv(os.path.join(DATA_DIR, "margin_bets.csv"), index=False)

    sweeps = []
    for method in ("pit", "agg"):
        sw = sweep_ats(bets[bets["method"] == method])
        sw.insert(0, "method", method)
        sweeps.append(sw)
    pd.concat(sweeps, ignore_index=True).to_csv(
        os.path.join(DATA_DIR, "margin_model_ats.csv"), index=False)

    meta = pd.DataFrame([_method_meta(bets, m) for m in ("pit", "agg")])
    meta["generated_utc"] = pd.Timestamp.utcnow().isoformat()
    meta.to_csv(os.path.join(DATA_DIR, "margin_model_meta.csv"), index=False)

    for m in ("pit", "agg"):
        row = meta[meta["method"] == m].iloc[0]
        label = "POINT-IN-TIME (honest)" if m == "pit" else "season-aggregate (inflated)"
        print(f"\n{label}: {int(row['n_bettable'])} bets, "
              f"ATS cover {row['ats_cover_pct_all']:.1f}% "
              f"(break-even ~52.4%), margin MAE {row['margin_mae']:.2f} pts")
        sw = sweep_ats(bets[bets["method"] == m])
        print(f"  {'edge>=':>6} {'bets':>5} {'cover%':>7} {'ROI%':>7} {'net$':>10}")
        for r in sw.itertuples(index=False):
            print(f"  {r.edge_pts:>6.0f} {r.placed:>5} {r.cover_pct:>7.1f} "
                  f"{r.roi_pct:>+7.1f} {r.net_pnl:>+10.2f}")
    print("\nWrote margin_model_ats.csv, margin_model_meta.csv, margin_bets.csv")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:            # noqa: BLE001
        pass
    main()

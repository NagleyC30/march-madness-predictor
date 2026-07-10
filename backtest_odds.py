# backtest_odds.py — REAL-odds betting backtest (project checklist items 3 & 3b).
#
# The betting figures in data/betting_simulation.csv are self-referential: they
# pay out at the model's OWN implied moneyline, so they can never show a real
# edge. This script does the honest version — it settles the model's tournament
# picks against REAL closing lines from the sportsbook archive (data/odds.csv,
# produced by fetch_odds.py).
#
# Method (walk-forward, no leakage — identical training to precompute.py):
#   For each training window and each tournament year that has odds:
#     1. Train on prior seasons only (mm_model.train_best_model).
#     2. For every game the tournament actually played, ask the model who wins
#        and how confidently (its implied moneyline).
#     3. At each confidence threshold (-200, -250, -300, -350), place a $10 bet
#        on the model's pick ONLY when it is at least that confident.
#
# Two strategies are settled side by side on the SAME selected bets:
#   * moneyline  — always bet the pick's real closing moneyline (item 3).
#   * flip       — when the pick's real moneyline is SHORTER than the threshold
#                  (heavy chalk that pays pennies), take the real point spread at
#                  standard -110 juice instead; otherwise bet the moneyline
#                  (item 3b — the Kelly "take the points on a huge favorite" idea).
#
# Only games with a real, matchable line are bettable; the rest are reported as
# `no_odds` (and `no_spread` when a flip has no spread) so coverage stays honest.
# Odds exist for tournaments 2008-2019 and 2021 (see fetch_odds.AVAILABLE_YEARS).
#
# Output (committed, consumed by app.py):
#   data/betting_simulation_real.csv       moneyline strategy, per window/year/threshold
#   data/betting_simulation_spreadflip.csv  flip strategy, same shape + spread cols
#   data/betting_real_meta.csv             headline coverage + which years are in.
#
# Usage:  python backtest_odds.py

import os

import pandas as pd

import mm_model as mm
import fetch_odds as fo

DATA_DIR = "data"
STAKE = 10.0
SPREAD_ODDS = -110      # standard point-spread juice for the flip strategy
# NCAA tournament window (month-day). Excludes November-February regular-season
# meetings between two teams that also met in March; conference tourneys can
# start ~March 10, so we anchor on Selection Sunday's neighbourhood and, for any
# pair that met more than once inside the window, keep the LATEST meeting (the
# NCAA game, which follows the conference finals).
TOURNEY_MMDD_MIN = 310


def _mmdd(date_str):
    """'2019-04-06' -> 406."""
    _, m, d = date_str.split("-")
    return int(m) * 100 + int(d)


def load_odds_lookup():
    """Map (year, frozenset{team_key, team_key}) -> per-team real closing lines
    for the latest in-window meeting of each pair:
        {'ml': {key: moneyline}, 'spread': {key: spread}, 'score': {key: score}}
    Missing spread/score cells are simply absent from their sub-dict."""
    odds = pd.read_csv(os.path.join(DATA_DIR, "odds.csv"))
    odds["MMDD"] = odds["DATE"].map(_mmdd)
    odds = odds[odds["MMDD"] >= TOURNEY_MMDD_MIN].sort_values(["YEAR", "DATE"])
    lookup = {}
    for r in odds.itertuples(index=False):
        key = (int(r.YEAR), frozenset((r.AWAY_KEY, r.HOME_KEY)))
        game = {"ml": {r.AWAY_KEY: int(r.ML_AWAY), r.HOME_KEY: int(r.ML_HOME)},
                "spread": {}, "score": {}}
        if pd.notna(r.SPREAD_AWAY) and pd.notna(r.SPREAD_HOME):
            game["spread"] = {r.AWAY_KEY: float(r.SPREAD_AWAY),
                              r.HOME_KEY: float(r.SPREAD_HOME)}
        if pd.notna(r.AWAY_SCORE) and pd.notna(r.HOME_SCORE):
            game["score"] = {r.AWAY_KEY: int(r.AWAY_SCORE),
                             r.HOME_KEY: int(r.HOME_SCORE)}
        lookup[key] = game            # later rows overwrite -> latest meeting
    return lookup


def real_game(lookup, year, team_high, team_low, pick):
    """Return (moneyline, spread, margin) for `pick` in the real game, any of
    which may be None. margin = pick_score - opponent_score."""
    kh, kl = fo.team_key(team_high), fo.team_key(team_low)
    game = lookup.get((year, frozenset((kh, kl))))
    if game is None:
        return None, None, None
    pk = fo.team_key(pick)
    opp = kl if pk == kh else kh
    ml = game["ml"].get(pk)
    spread = game["spread"].get(pk)
    margin = None
    if pk in game["score"] and opp in game["score"]:
        margin = game["score"][pk] - game["score"][opp]
    return ml, spread, margin


def simulate_real_betting(actual_games, year, model, features, lookup,
                          thresholds=mm.BETTING_THRESHOLDS, stake=STAKE):
    """Settle one tournament's games at each confidence threshold under BOTH the
    moneyline and flip-to-spread strategies. Returns {'ml': {thr: stats},
    'flip': {thr: stats}}."""
    # Pre-compute the model's pick, confidence and real lines once per game.
    graded = []
    for g in actual_games:
        th, sh = g["TEAM_HIGH"], int(g["SEED_HIGH"])
        tl, sl = g["TEAM_LOW"], int(g["SEED_LOW"])
        pick, _, p_high = mm.predict_game_proba(th, sh, tl, sl, model_df, model, features)
        p_win = p_high if pick == th else 1 - p_high
        implied = mm.prob_to_american_odds(p_win)
        ml, spread, margin = real_game(lookup, year, th, tl, pick)
        covered = push = None
        if spread is not None and margin is not None:
            edge = margin + spread          # >0 pick covers, ==0 push
            push = edge == 0
            covered = edge > 0
        graded.append({"implied": implied, "ml": ml, "correct": pick == g["WINNER"],
                       "spread": spread, "covered": covered, "push": push})

    ml_res, flip_res = {}, {}
    for thresh in thresholds:
        # ---- moneyline strategy ----
        won = lost = placed = no_odds = 0
        net = 0.0
        ml_sum = 0
        # ---- flip strategy ----
        f_won = f_lost = f_push = f_placed = f_no_odds = f_no_spread = 0
        f_ml_bets = f_spread_bets = 0
        f_net = 0.0
        for gr in graded:
            if gr["implied"] > thresh:       # model not confident enough
                continue
            ml = gr["ml"]
            # moneyline strategy
            if ml is None:
                no_odds += 1
            else:
                placed += 1
                ml_sum += ml
                if gr["correct"]:
                    net += mm.american_odds_payout(ml, stake); won += 1
                else:
                    net -= stake; lost += 1
            # flip strategy
            if ml is None:
                f_no_odds += 1
            elif ml < thresh:                # heavy chalk -> take the spread
                if gr["covered"] is None:    # no spread/score to settle
                    f_no_spread += 1
                else:
                    f_placed += 1; f_spread_bets += 1
                    if gr["push"]:
                        f_push += 1
                    elif gr["covered"]:
                        f_net += mm.american_odds_payout(SPREAD_ODDS, stake); f_won += 1
                    else:
                        f_net -= stake; f_lost += 1
            else:                            # softer line -> keep the moneyline
                f_placed += 1; f_ml_bets += 1
                if gr["correct"]:
                    f_net += mm.american_odds_payout(ml, stake); f_won += 1
                else:
                    f_net -= stake; f_lost += 1

        wagered = placed * stake
        ml_res[thresh] = {
            "placed": placed, "won": won, "lost": lost,
            "net_pnl": round(net, 2), "total_wagered": round(wagered, 2),
            "roi_pct": round(net / wagered * 100, 1) if wagered else 0.0,
            "avg_ml": round(ml_sum / placed) if placed else None,
            "no_odds": no_odds,
        }
        f_wagered = f_placed * stake
        flip_res[thresh] = {
            "placed": f_placed, "won": f_won, "lost": f_lost, "push": f_push,
            "net_pnl": round(f_net, 2), "total_wagered": round(f_wagered, 2),
            "roi_pct": round(f_net / f_wagered * 100, 1) if f_wagered else 0.0,
            "spread_bets": f_spread_bets, "ml_bets": f_ml_bets,
            "no_odds": f_no_odds, "no_spread": f_no_spread,
        }
    return {"ml": ml_res, "flip": flip_res}


# predict_game_proba needs the test year's team-stats frame; set per year in main.
model_df = None


def main():
    global model_df
    df = mm.load_data()
    features = mm.get_features(df)
    all_rows, tour_years = mm.build_all_matchups(df)

    lookup = load_odds_lookup()
    odds_years = sorted(set(fo.AVAILABLE_YEARS) & set(tour_years))
    print(f"Real-odds backtest over {odds_years}")
    print(f"Windows: {list(mm.TRAINING_WINDOWS.keys())}\n")

    ml_rows, flip_rows = [], []
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
            model, model_name, _ = mm.train_best_model(df_train, features)

            model_df = df[df["YEAR"] == test_year]
            actual_games = [r for r in all_rows if r["YEAR"] == test_year]
            sim = simulate_real_betting(
                actual_games, test_year, model, features, lookup)

            m200, f200 = sim["ml"][-200], sim["flip"][-200]
            print(f"[{done}/{total}] {window_name:13} {test_year} | {model_name:17} "
                  f"| ML @-200 ${m200['net_pnl']:+7.2f} ({m200['roi_pct']:+.1f}%) "
                  f"| FLIP ${f200['net_pnl']:+7.2f} ({f200['roi_pct']:+.1f}%, "
                  f"{f200['spread_bets']}sp/{f200['ml_bets']}ml)", flush=True)

            for thresh, s in sim["ml"].items():
                ml_rows.append({"window": window_name, "test_year": test_year,
                                "threshold": thresh, **s})
            for thresh, s in sim["flip"].items():
                flip_rows.append({"window": window_name, "test_year": test_year,
                                  "threshold": thresh, **s})

    out_ml = pd.DataFrame(ml_rows)
    out_flip = pd.DataFrame(flip_rows)
    out_ml.to_csv(os.path.join(DATA_DIR, "betting_simulation_real.csv"), index=False)
    out_flip.to_csv(
        os.path.join(DATA_DIR, "betting_simulation_spreadflip.csv"), index=False)

    def _pooled(out, label):
        ap = out[out["window"] == "all_prior"]
        print(f"\n{label} — all_prior window, pooled across all odds years:")
        print(f"{'thresh':>7} {'bets':>5} {'won':>4} {'lost':>5} {'net_pnl':>10} "
              f"{'roi%':>7}")
        for thresh in mm.BETTING_THRESHOLDS:
            s = ap[ap["threshold"] == thresh]
            placed = int(s["placed"].sum())
            won, lost = int(s["won"].sum()), int(s["lost"].sum())
            net, wag = s["net_pnl"].sum(), s["total_wagered"].sum()
            roi = net / wag * 100 if wag else 0.0
            print(f"{thresh:>7} {placed:>5} {won:>4} {lost:>5} {net:>+10.2f} "
                  f"{roi:>+6.1f}%")

    print(f"\nWrote betting_simulation_real.csv ({len(out_ml)} rows) and "
          f"betting_simulation_spreadflip.csv ({len(out_flip)} rows)")
    _pooled(out_ml, "MONEYLINE")
    _pooled(out_flip, "FLIP-TO-SPREAD")

    # Report the years actually backtested (the first odds year can't be a test
    # year — it has no prior season to train on), not merely odds availability.
    bet_years = sorted(int(y) for y in out_ml["test_year"].unique())
    pd.DataFrame([{
        "odds_years": ",".join(str(y) for y in bet_years),
        "n_years": len(bet_years),
        "stake": STAKE,
        "spread_odds": SPREAD_ODDS,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
    }]).to_csv(os.path.join(DATA_DIR, "betting_real_meta.csv"), index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()

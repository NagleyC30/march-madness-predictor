# backtest_odds.py — REAL-odds betting backtest (project checklist item 3).
#
# The betting figures in data/betting_simulation.csv are self-referential: they
# pay out at the model's OWN implied moneyline, so they can never show a real
# edge. This script does the honest version — it settles the model's tournament
# picks against REAL closing moneylines from the sportsbook archive
# (data/odds.csv, produced by fetch_odds.py).
#
# Method (walk-forward, no leakage — identical training to precompute.py):
#   For each training window and each tournament year that has odds:
#     1. Train on prior seasons only (mm_model.train_best_model).
#     2. For every game the tournament actually played, ask the model who wins
#        and how confidently (its implied moneyline).
#     3. At each confidence threshold (-200, -250, -300, -350), place a $10 bet
#        on the model's pick ONLY when it is at least that confident — but pay
#        out at the game's REAL closing moneyline, and settle on the real result.
#
# Only games with a real, matchable sportsbook line are bettable; the rest are
# reported as `no_odds` so coverage stays honest. Odds exist for tournaments
# 2008-2019 and 2021 (see fetch_odds.AVAILABLE_YEARS).
#
# Output (committed, consumed by app.py):
#   data/betting_simulation_real.csv   per window / year / threshold P&L, plus
#                                       avg_ml, no_odds, and a matched-coverage %.
#   data/betting_real_meta.csv         headline coverage + which years are in.
#
# Usage:  python backtest_odds.py

import os

import pandas as pd

import mm_model as mm
import fetch_odds as fo

DATA_DIR = "data"
STAKE = 10.0
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
    """Map (year, frozenset{team_key, team_key}) -> {team_key: closing_ML} for
    the latest in-window meeting of each pair."""
    odds = pd.read_csv(os.path.join(DATA_DIR, "odds.csv"))
    odds["MMDD"] = odds["DATE"].map(_mmdd)
    odds = odds[odds["MMDD"] >= TOURNEY_MMDD_MIN].sort_values(["YEAR", "DATE"])
    lookup = {}
    for r in odds.itertuples(index=False):
        key = (int(r.YEAR), frozenset((r.AWAY_KEY, r.HOME_KEY)))
        # later rows overwrite earlier ones -> keeps the latest meeting.
        lookup[key] = {r.AWAY_KEY: int(r.ML_AWAY), r.HOME_KEY: int(r.ML_HOME)}
    return lookup


def real_line(lookup, year, team_high, team_low, pick):
    """Closing moneyline for `pick` in the real game team_high vs team_low, or
    None if no matchable real game exists."""
    kh, kl = fo.team_key(team_high), fo.team_key(team_low)
    game = lookup.get((year, frozenset((kh, kl))))
    if game is None:
        return None
    return game.get(fo.team_key(pick))


def simulate_real_betting(actual_games, year, model, features, lookup,
                          thresholds=mm.BETTING_THRESHOLDS, stake=STAKE):
    """Independent, real-odds settlement of one tournament's games at each
    confidence threshold. Returns {threshold: stats}."""
    # Pre-compute the model's pick, confidence and real line once per game.
    graded = []
    for g in actual_games:
        th, sh = g["TEAM_HIGH"], int(g["SEED_HIGH"])
        tl, sl = g["TEAM_LOW"], int(g["SEED_LOW"])
        pick, _, p_high = mm.predict_game_proba(th, sh, tl, sl, model_df, model, features)
        p_win = p_high if pick == th else 1 - p_high
        implied = mm.prob_to_american_odds(p_win)
        ml = real_line(lookup, year, th, tl, pick)
        graded.append((implied, ml, pick == g["WINNER"]))

    results = {}
    for thresh in thresholds:
        won = lost = placed = no_odds = 0
        net = 0.0
        ml_sum = 0
        for implied, ml, correct in graded:
            if implied > thresh:            # model not confident enough
                continue
            if ml is None:                  # no real line -> can't place it
                no_odds += 1
                continue
            placed += 1
            ml_sum += ml
            if correct:
                net += mm.american_odds_payout(ml, stake)
                won += 1
            else:
                net -= stake
                lost += 1
        wagered = placed * stake
        results[thresh] = {
            "placed": placed, "won": won, "lost": lost,
            "net_pnl": round(net, 2),
            "total_wagered": round(wagered, 2),
            "roi_pct": round(net / wagered * 100, 1) if wagered else 0.0,
            "avg_ml": round(ml_sum / placed) if placed else None,
            "no_odds": no_odds,
        }
    return results


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

    rows = []
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

            b200 = sim[-200]
            print(f"[{done}/{total}] {window_name:13} {test_year} | {model_name:17} "
                  f"| @-200: {b200['placed']:2} bets, {b200['won']}-{b200['lost']}, "
                  f"${b200['net_pnl']:+7.2f} ({b200['roi_pct']:+.1f}%)", flush=True)

            for thresh, s in sim.items():
                rows.append({"window": window_name, "test_year": test_year,
                             "threshold": thresh, **s})

    out = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "betting_simulation_real.csv")
    out.to_csv(out_path, index=False)

    # Headline: pooled across all years for the all_prior window at each threshold.
    ap = out[out["window"] == "all_prior"]
    print(f"\nWrote {out_path}: {len(out)} rows\n")
    print("all_prior window, pooled across all odds years:")
    print(f"{'thresh':>7} {'bets':>5} {'won':>4} {'lost':>5} {'net_pnl':>10} "
          f"{'roi%':>7} {'no_odds':>8}")
    meta_rows = []
    for thresh in mm.BETTING_THRESHOLDS:
        s = ap[ap["threshold"] == thresh]
        placed, won, lost = int(s["placed"].sum()), int(s["won"].sum()), int(s["lost"].sum())
        net = s["net_pnl"].sum()
        wag = s["total_wagered"].sum()
        no_odds = int(s["no_odds"].sum())
        roi = net / wag * 100 if wag else 0.0
        print(f"{thresh:>7} {placed:>5} {won:>4} {lost:>5} {net:>+10.2f} "
              f"{roi:>+6.1f}% {no_odds:>8}")
        meta_rows.append({"threshold": thresh, "placed": placed, "won": won,
                          "lost": lost, "net_pnl": round(net, 2),
                          "roi_pct": round(roi, 1), "no_odds": no_odds})

    # Report the years actually backtested (the first odds year can't be a test
    # year — it has no prior season to train on), not merely odds availability.
    bet_years = sorted(int(y) for y in out["test_year"].unique())
    pd.DataFrame([{
        "odds_years": ",".join(str(y) for y in bet_years),
        "n_years": len(bet_years),
        "stake": STAKE,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
    }]).to_csv(os.path.join(DATA_DIR, "betting_real_meta.csv"), index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()

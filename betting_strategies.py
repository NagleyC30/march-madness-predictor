# betting_strategies.py — Betting strategy lab (project checklist item 7).
#
# Items 3/3b showed that flat-betting the model's confident CHALK loses at every
# threshold. This asks the real edge question: are there spots where the model's
# probability beats the market's price (+EV), and does betting only those — with
# smarter staking — do any better?
#
# It post-processes data/bet_games.csv (one settled game-prediction per row,
# written by backtest_odds.py): model win prob + real closing moneylines (both
# sides) + result. No model training here, so it's fast to iterate — the app
# imports these functions to compute strategies live.
#
# Strategies (each = a selection rule + a staking rule):
#   Model chalk (ML)     bet the pick's moneyline when the model is >= -200 sure
#                        (the item-3 baseline), flat stake.
#   Value (+EV, flat)    for each game consider BOTH sides; bet the side whose
#                        model probability makes it positive expected value at
#                        its real price. Flat stake.
#   Value (+EV, Kelly)   same selection, staked by fractional Kelly of a running
#                        bankroll (compounds; equity curve + drawdown reported).
#   Value — underdogs    +EV bets restricted to real underdogs (+ moneyline).
#   Value — edge >= 3%   +EV bets whose de-vigged edge clears a margin.
#
# Output (committed, consumed by app.py as a convenience; the app can also call
# run_all() live):
#   data/betting_strategies_summary.csv   one row per (window, strategy)
#   data/betting_strategies_equity.csv    bankroll path per (window, strategy)
#
# Usage:  python betting_strategies.py

import os
import sys

import pandas as pd

DATA_DIR = "data"
GAMES_FILE = os.path.join(DATA_DIR, "bet_games.csv")

START_BANKROLL = 1000.0     # common starting bankroll for every strategy
FLAT_UNIT = 10.0            # flat stake ($) — 1% of the starting bankroll
KELLY_FRACTION = 0.25       # quarter-Kelly (conservative)
EDGE_MARGIN = 0.03         # "edge >= 3%" strategy threshold

ROUND_ORDER = {r: i for i, r in enumerate(
    ["R64", "R32", "S16", "E8", "F4", "Championship"])}


# ──────────────────────────────────────────────────────────────
# ODDS MATH
# ──────────────────────────────────────────────────────────────

def implied_prob(ml):
    """Raw (with-vig) win probability implied by an American moneyline."""
    ml = float(ml)
    return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)


def net_odds(ml):
    """Net profit per $1 staked on a winning bet (decimal odds minus 1)."""
    ml = float(ml)
    return 100 / abs(ml) if ml < 0 else ml / 100


def devig_prob(ml, ml_opp):
    """The market's fair win probability, with the vig removed using both sides."""
    r, ro = implied_prob(ml), implied_prob(ml_opp)
    tot = r + ro
    return r / tot if tot > 0 else r


# ──────────────────────────────────────────────────────────────
# BET SELECTION
# ──────────────────────────────────────────────────────────────

def _sides(row):
    """Both bettable sides of a game: the model's pick and its opponent. Each is
    a dict with the model prob for that side, its real moneyline, whether it won,
    and the opposing moneyline (for de-vig)."""
    out = [{"p": row["model_p"], "ml": row["ml"], "won": bool(row["correct"]),
            "ml_opp": row["ml_opp"]}]
    if pd.notna(row["ml_opp"]):
        out.append({"p": 1.0 - row["model_p"], "ml": row["ml_opp"],
                    "won": not bool(row["correct"]), "ml_opp": row["ml"]})
    return out


def best_value_side(row):
    """The positive-EV side with the highest per-$1 expected value, or None.

    EV per $1 = p*b - (1-p), where b is the net decimal odds. A side is worth
    betting only when the model's probability beats what the price demands (by
    more than a hair — the epsilon avoids selecting exactly-fair bets that only
    clear zero through floating-point noise)."""
    best, best_ev = None, 1e-9
    for s in _sides(row):
        b = net_odds(s["ml"])
        ev = s["p"] * b - (1.0 - s["p"])
        if ev > best_ev:
            best, best_ev = dict(s, ev=ev, b=b), ev
    return best


def sel_chalk(row):
    """Item-3 baseline: bet the pick's moneyline when the model is at least
    -200 confident (p >= 2/3)."""
    if row["model_p"] >= implied_prob(-200):
        return {"p": row["model_p"], "ml": row["ml"],
                "won": bool(row["correct"]), "ml_opp": row["ml_opp"]}
    return None


def sel_value(row):
    return best_value_side(row)


def sel_value_dog(row):
    s = best_value_side(row)
    return s if (s and s["ml"] > 0) else None


def sel_value_edge(row):
    s = best_value_side(row)
    if s and pd.notna(s.get("ml_opp")):
        if s["p"] - devig_prob(s["ml"], s["ml_opp"]) >= EDGE_MARGIN:
            return s
    return None


STRATEGIES = [
    {"name": "Model chalk (ML ≥ -200)", "select": sel_chalk, "staking": "flat"},
    {"name": "Value (+EV, flat)", "select": sel_value, "staking": "flat"},
    {"name": "Value (+EV, ¼-Kelly)", "select": sel_value, "staking": "kelly"},
    {"name": "Value — underdogs only", "select": sel_value_dog, "staking": "flat"},
    {"name": "Value — edge ≥ 3%", "select": sel_value_edge, "staking": "flat"},
]


# ──────────────────────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────────────────────

def _ordered(games):
    """Games in rough chronological order for a stable bankroll path."""
    g = games.copy()
    g["_ro"] = g["round"].map(ROUND_ORDER).fillna(99)
    return g.sort_values(["year", "_ro", "pick"]).reset_index(drop=True)


def simulate(games, strategy):
    """Run one strategy over one window's games. Returns (summary, equity_df)."""
    g = _ordered(games)
    bank = peak = START_BANKROLL
    max_dd = 0.0
    bets = won = lost = 0
    staked = pnl_total = edge_sum = 0.0
    equity = []
    for row in g.itertuples(index=False):
        side = strategy["select"](row._asdict())
        if side is None:
            continue
        b = net_odds(side["ml"])
        if strategy["staking"] == "kelly":
            f_star = max(0.0, (side["p"] * b - (1.0 - side["p"])) / b)
            stake = min(KELLY_FRACTION * f_star * bank, bank)
        else:
            stake = FLAT_UNIT
        if stake <= 0:
            continue
        bets += 1
        staked += stake
        if side["won"]:
            pnl = stake * b
            won += 1
        else:
            pnl = -stake
            lost += 1
        bank += pnl
        pnl_total += pnl
        if pd.notna(side.get("ml_opp")):
            edge_sum += side["p"] - devig_prob(side["ml"], side["ml_opp"])
        peak = max(peak, bank)
        max_dd = max(max_dd, (peak - bank) / peak if peak > 0 else 0.0)
        equity.append({"bet": bets, "year": int(row.year),
                       "bankroll": round(bank, 2)})
    summary = {
        "bets": bets, "won": won, "lost": lost,
        "staked": round(staked, 2), "net_pnl": round(pnl_total, 2),
        "roi_pct": round(pnl_total / staked * 100, 1) if staked else 0.0,
        "win_rate": round(won / bets * 100, 1) if bets else 0.0,
        "avg_edge_pct": round(edge_sum / bets * 100, 1) if bets else 0.0,
        "final_bankroll": round(bank, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
    }
    return summary, pd.DataFrame(equity)


def run_all(games, window=None, strategies=STRATEGIES):
    """Run every strategy for one window (or every window if None). Returns
    (summary_df, equity_df) tagged by window+strategy."""
    windows = [window] if window else sorted(games["window"].unique())
    srows, erows = [], []
    for w in windows:
        gw = games[games["window"] == w]
        for strat in strategies:
            summ, eq = simulate(gw, strat)
            srows.append({"window": w, "strategy": strat["name"], **summ})
            eq = eq.assign(window=w, strategy=strat["name"])
            erows.append(eq)
    equity = pd.concat(erows, ignore_index=True) if erows else pd.DataFrame()
    return pd.DataFrame(srows), equity


def load_games(path=GAMES_FILE):
    df = pd.read_csv(path)
    df["correct"] = df["correct"].astype(bool)
    return df


def main():
    games = load_games()
    windows = sorted(games["window"].unique())
    print(f"Strategy lab over {len(games):,} game-predictions, "
          f"windows {windows}\n")
    summary, equity = run_all(games)
    summary.to_csv(os.path.join(DATA_DIR, "betting_strategies_summary.csv"),
                   index=False)
    equity.to_csv(os.path.join(DATA_DIR, "betting_strategies_equity.csv"),
                  index=False)

    # Headline: the most-data window.
    w = "all_prior"
    print(f"Strategy results for window '{w}' "
          f"(${START_BANKROLL:.0f} bankroll, ${FLAT_UNIT:.0f} flat unit, "
          f"{KELLY_FRACTION:g}-Kelly):\n")
    sub = summary[summary["window"] == w]
    print(f"{'strategy':26} {'bets':>5} {'win%':>6} {'ROI%':>7} "
          f"{'edge%':>6} {'end$':>8} {'maxDD%':>7}")
    for r in sub.itertuples(index=False):
        print(f"{r.strategy:26} {r.bets:>5} {r.win_rate:>6.1f} {r.roi_pct:>+7.1f} "
              f"{r.avg_edge_pct:>+6.1f} {r.final_bankroll:>8.0f} "
              f"{r.max_drawdown_pct:>7.1f}")
    print(f"\nWrote betting_strategies_summary.csv ({len(summary)} rows) and "
          f"betting_strategies_equity.csv ({len(equity)} rows).")


if __name__ == "__main__":
    try:                       # strategy names use ≥ / ¼ — avoid Windows cp1252
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:          # noqa: BLE001 — best-effort console fix
        pass
    main()

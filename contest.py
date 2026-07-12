# contest.py — "Me vs. the Machine" contest store (project checklist item 14).
#
# An everlasting head-to-head: for each game I predict, I ALSO predict, and we
# track who's better. This module is the integrity-critical persistence + scoring
# layer; the Streamlit page is the UI on top.
#
# Design rules baked in here (see the notes for the full list):
#   * Commit-then-reveal. A pick is written with a UTC timestamp the moment it's
#     made; the model's pick for the same game is captured at the SAME time, so
#     neither side can be edited after the fact.
#   * Blind entry is enforced by the UI (the model's pick isn't shown until after
#     I submit); this layer just stores both picks together.
#   * Settle only after the real result is entered — scores never touch a pick
#     before it's locked.
#
# MANUAL-ENTRY first pass: games are entered by hand (both picks now, result
# later). The auto-schedule feed + real closing lines land in a later phase; the
# betting comparison here is therefore a simple SYMMETRIC flat-stake bet (each
# side flat-bets its own pick at the entered moneyline), which is the only
# apples-to-apples wager when the human gives a pick but not a probability.
#
# Store: data/contest_picks.csv (one row per game). NOTE: on an ephemeral cloud
# filesystem this resets on restart — durable storage is a later upgrade.

import os
from datetime import datetime, timezone

import pandas as pd

DATA_DIR = "data"
PICKS_FILE = os.path.join(DATA_DIR, "contest_picks.csv")
STAKE = 100.0

COLUMNS = ["id", "created_utc", "season", "team_a", "team_b", "location",
           "user_pick", "user_conf", "model_pick", "model_prob_pick",
           "ml_a", "ml_b", "status", "actual_winner", "settled_utc"]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_picks():
    """Read the contest store fresh (never cache — it mutates). Returns an
    empty, correctly-typed frame if the file doesn't exist yet."""
    if not os.path.exists(PICKS_FILE):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(PICKS_FILE)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    # A wholly-empty text column is read as float64 (all-NaN), which then rejects
    # string assignment (e.g. settling writes a timestamp). Force every text
    # column to object/"" so equality checks and later writes both work.
    str_cols = ["created_utc", "team_a", "team_b", "location", "user_pick",
                "model_pick", "status", "actual_winner", "settled_utc"]
    for c in str_cols:
        df[c] = df[c].fillna("").astype(str).replace("nan", "")
    return df[COLUMNS]


def save_picks(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(PICKS_FILE, index=False)


def add_pick(season, team_a, team_b, location, user_pick,
             model_pick, model_prob_pick, user_conf=None, ml_a=None, ml_b=None):
    """Record one game: my pick and the model's pick, captured together and
    timestamped. Returns the new row id."""
    df = load_picks()
    new_id = 1 if df.empty else int(pd.to_numeric(df["id"]).max()) + 1
    row = {
        "id": new_id, "created_utc": _now(), "season": season,
        "team_a": team_a, "team_b": team_b, "location": location,
        "user_pick": user_pick, "user_conf": user_conf,
        "model_pick": model_pick, "model_prob_pick": round(float(model_prob_pick), 4),
        "ml_a": ml_a, "ml_b": ml_b, "status": "pending",
        "actual_winner": "", "settled_utc": "",
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_picks(df)
    return new_id


def settle_pick(pick_id, actual_winner):
    """Record the real winner of a pending game and lock it as settled."""
    df = load_picks()
    m = pd.to_numeric(df["id"]) == int(pick_id)
    if not m.any():
        return False
    df.loc[m, "actual_winner"] = actual_winner
    df.loc[m, "status"] = "settled"
    df.loc[m, "settled_utc"] = _now()
    save_picks(df)
    return True


def delete_pick(pick_id):
    """Remove a pick entirely (e.g. an entry mistake)."""
    df = load_picks()
    df = df[pd.to_numeric(df["id"]) != int(pick_id)]
    save_picks(df)


def _payout(ml, stake=STAKE):
    """Net profit on a winning stake at an American moneyline."""
    ml = float(ml)
    return stake * (100 / abs(ml)) if ml < 0 else stake * (ml / 100)


def _flat_bet_pnl(row, who):
    """Flat-stake P&L for one settled game for 'who' ('user'|'model'), or None if
    that side's pick has no entered moneyline. Each side flat-bets ITS OWN pick."""
    pick = row["user_pick"] if who == "user" else row["model_pick"]
    ml = row["ml_a"] if pick == row["team_a"] else row["ml_b"]
    if pd.isna(ml) or ml in ("", None):
        return None
    won = pick == row["actual_winner"]
    return _payout(ml) if won else -STAKE


def scoreboard(df=None):
    """Aggregate the settled games into the head-to-head record. Returns a dict
    of model vs. user W/L, agreement, the disagreement head-to-head (the key
    'am I adding value?' stat), and flat-bet P&L where lines were entered."""
    if df is None:
        df = load_picks()
    s = df[df["status"] == "settled"].copy()
    out = {"n_settled": len(s), "n_pending": int((df["status"] == "pending").sum())}
    if s.empty:
        return {**out, "model_w": 0, "model_l": 0, "user_w": 0, "user_l": 0,
                "agree": 0, "disagree": 0, "disagree_user_right": 0,
                "disagree_model_right": 0, "model_pnl": 0.0, "user_pnl": 0.0,
                "n_user_bets": 0, "n_model_bets": 0}

    s["model_right"] = s["model_pick"] == s["actual_winner"]
    s["user_right"] = s["user_pick"] == s["actual_winner"]
    s["agree_pick"] = s["model_pick"] == s["user_pick"]
    disagree = s[~s["agree_pick"]]

    user_pnls = [p for p in (_flat_bet_pnl(r, "user") for _, r in s.iterrows())
                 if p is not None]
    model_pnls = [p for p in (_flat_bet_pnl(r, "model") for _, r in s.iterrows())
                  if p is not None]

    return {
        **out,
        "model_w": int(s["model_right"].sum()),
        "model_l": int((~s["model_right"]).sum()),
        "user_w": int(s["user_right"].sum()),
        "user_l": int((~s["user_right"]).sum()),
        "agree": int(s["agree_pick"].sum()),
        "disagree": int(len(disagree)),
        "disagree_user_right": int(disagree["user_right"].sum()),
        "disagree_model_right": int(disagree["model_right"].sum()),
        "user_pnl": round(sum(user_pnls), 2), "model_pnl": round(sum(model_pnls), 2),
        "n_user_bets": len(user_pnls), "n_model_bets": len(model_pnls),
    }

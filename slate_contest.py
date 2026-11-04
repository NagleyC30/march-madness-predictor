# slate_contest.py — Weekly-slate contest logic: lock rules, game states, scoring.
#
# The slate itself (the model's 5 frozen picks) is a CSV in data/slates/, written
# by weekly_slate.py and committed to the repo. YOUR picks for slate games are
# ordinary Me-vs-Machine contest rows (contest.py → Supabase), tagged with the
# slate's game id, so they get the same guarantees as manual picks: the
# database stamps the lock time, the picks can never be edited, a game can be
# settled once, and the unique index allows exactly one pick per slate game.
#
# Rules (decided 2026-10-06, refined 2026-11-01):
#   * A game locks at its tip time (Eastern), from Barttorvik's schedule page.
#     With no listed time it locks at midnight ET at the start of game day.
#   * The model's pick is hidden until you lock yours, or the game locks.
#   * A game you didn't pick before the lock is DROPPED from both records.
#   * After tip-off you record the winner on the card (or on Me vs Machine).

import glob
import os
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd

import contest as ct

DATA_DIR = "data"
SLATE_DIR = os.path.join(DATA_DIR, "slates")
STATS_SUBDIR = "stats"          # data/slates/stats/<slate_id>.csv, from weekly_slate.py
ET = ZoneInfo("America/New_York")


def now_utc():
    return datetime.now(timezone.utc)


# ──────────────────────────────────────────────────────────────
# SLATES  (files in the repo)
# ──────────────────────────────────────────────────────────────

def load_slates(slate_dir=None):
    """Every frozen slate, one row per game, oldest week first."""
    paths = sorted(glob.glob(os.path.join(slate_dir or SLATE_DIR, "*.csv")))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p, keep_default_na=False) for p in paths],
                   ignore_index=True)
    if "tip_et" not in df.columns:
        df["tip_et"] = ""
    df["neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1"])
    return df.sort_values(["week_start", "date", "game_id"]).reset_index(drop=True)


def load_stats(slate_id, slate_dir=None):
    """The team-stats snapshot frozen with a slate (empty if that slate predates
    the stats feature). One row per (game_id, side)."""
    path = os.path.join(slate_dir or SLATE_DIR, STATS_SUBDIR, f"{slate_id}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def lock_time(game_date, tip_et=""):
    """When picks close: the listed tip time (ET) on game day, else midnight ET
    at the start of game day."""
    d = date.fromisoformat(str(game_date))
    t = time(0, 0)
    if isinstance(tip_et, str) and tip_et.strip():
        try:
            t = datetime.strptime(tip_et.strip().upper(), "%I:%M %p").time()
        except ValueError:
            pass
    return datetime.combine(d, t, tzinfo=ET)


def is_locked(game_date, tip_et="", at=None):
    return (at or now_utc()) >= lock_time(game_date, tip_et)


def current_slate_id(slates, today=None):
    """The slate whose week contains today; else the next one ahead; else the
    most recent one. None if there are no slates at all."""
    if slates.empty:
        return None
    today = (today or datetime.now(ET).date()).isoformat()
    weeks = slates.drop_duplicates("slate_id")[["slate_id", "week_start", "week_end"]]
    now = weeks[(weeks.week_start <= today) & (weeks.week_end >= today)]
    if not now.empty:
        return now.slate_id.iloc[-1]
    ahead = weeks[weeks.week_start > today]
    if not ahead.empty:
        return ahead.sort_values("week_start").slate_id.iloc[0]
    return weeks.sort_values("week_start").slate_id.iloc[-1]


# ──────────────────────────────────────────────────────────────
# PICKS  (stored through contest.py: Supabase, or the CSV in local dev)
# ──────────────────────────────────────────────────────────────

def slate_picks():
    """Your contest rows that belong to slate games: game_id, pick_id,
    user_pick, picked_utc, status, actual_winner."""
    p = ct.load_picks()
    p = p[p["slate_game_id"].astype(str) != ""]
    return pd.DataFrame({
        "game_id": p["slate_game_id"].values,
        "pick_id": p["id"].values,
        "user_pick": p["user_pick"].values,
        "picked_utc": p["created_utc"].values,
        "status": p["status"].values,
        "actual_winner": p["actual_winner"].values,
    })


def _check_open(g, user_pick, at):
    if user_pick not in (g["home"], g["away"]):
        raise ValueError(f"{user_pick} isn't playing in this game.")
    if is_locked(g["date"], g.get("tip_et", ""), at):
        raise ValueError("This game is locked — picks close at tip-off.")


def save_draft(slate_row, user_pick, at=None):
    """Save a pick WITHOUT locking it: it can be changed, it never counts, and
    it doesn't reveal the model's pick. Lock it before tip-off to make it count."""
    _check_open(slate_row, user_pick, at)
    ct.save_draft(slate_row["game_id"], user_pick)


def drafts():
    """Saved drafts: game_id, draft_pick, draft_saved_utc."""
    d = ct.load_drafts()
    return d.rename(columns={"user_pick": "draft_pick", "saved_utc": "draft_saved_utc"})


def lock_pick(slate_row, user_pick, at=None):
    """Record your pick for one slate game, with the model's frozen pick from
    the slate. Refuses a team not in the game or a locked game; a second pick
    for the same game raises contest.AlreadyPicked. Returns the contest row id."""
    g = slate_row
    _check_open(g, user_pick, at)
    # contest orientation: team_a = away, team_b = home ('B' = team_b at home)
    pick_id = ct.add_pick(int(g["season"]), g["away"], g["home"],
                          "N" if bool(g["neutral"]) else "B", user_pick,
                          g["model_pick"], float(g["model_prob_pick"]),
                          slate_game_id=g["game_id"])
    ct.delete_draft(g["game_id"])          # the locked pick supersedes the draft
    return pick_id


def lock_saved_drafts(board, at=None):
    """Lock every open game that has a saved draft, as drafted. Returns
    (locked_count, errors)."""
    n, errors = 0, []
    for g in board[(board["state"] == "open") & board["draft_pick"].notna()].to_dict("records"):
        try:
            lock_pick(g, g["draft_pick"], at)
            n += 1
        except ValueError as e:              # incl. contest.AlreadyPicked
            errors.append(f"{g['away']} vs {g['home']}: {e}")
    return n, errors


def record_winner(pick_id, winner):
    """Settle a slate game once it's over (same as Me vs Machine's Settle)."""
    return ct.settle_pick(int(pick_id), winner)


def auto_settle(board, results):
    """Settle every locked, still-pending pick whose winner has been posted in
    the results files (slate_results.py). Returns the number settled. Safe to
    call on every page load: the database only settles a pending pick once, and
    a winner who isn't in the game is ignored."""
    if board.empty or results is None or results.empty:
        return 0
    won = dict(zip(results["game_id"], results["winner"]))
    n = 0
    for g in board[(board["state"] == "picked") & board["locked"]].to_dict("records"):
        w = won.get(g["game_id"])
        if w in (g["home"], g["away"]) and record_winner(g["pick_id"], w):
            n += 1
    return n


# ──────────────────────────────────────────────────────────────
# GAME STATE + SCOREBOARD
# ──────────────────────────────────────────────────────────────

def game_board(slates, picks, at=None, drafts_df=None):
    """One row per slate game with what the page needs:
    state ∈ open | picked | missed | final; reveal (may we show the model's
    pick?); correctness once final; draft_pick (saved, not locked — never
    counts and never reveals anything)."""
    at = at or now_utc()
    if slates.empty:
        return slates
    b = slates.merge(picks, on="game_id", how="left")
    if drafts_df is None or drafts_df.empty:
        drafts_df = pd.DataFrame(columns=["game_id", "draft_pick", "draft_saved_utc"])
    b = b.merge(drafts_df[["game_id", "draft_pick", "draft_saved_utc"]],
                on="game_id", how="left")
    b["draft_pick"] = b["draft_pick"].astype(object).where(b["draft_pick"].notna(), None)
    b["lock_at"] = [lock_time(d, t) for d, t in zip(b["date"], b["tip_et"])]
    # the database stamps created_utc; a pick only counts if it beat the lock
    b["picked_in_time"] = [
        isinstance(pu, str) and pu != "" and datetime.fromisoformat(pu) < lk
        for pu, lk in zip(b["picked_utc"], b["lock_at"])
    ]
    b["user_pick"] = b["user_pick"].astype(object).where(b["picked_in_time"], None)
    locked = pd.Series([at >= lk for lk in b["lock_at"]], index=b.index)
    has_pick = b["user_pick"].notna()

    state = pd.Series("open", index=b.index)
    state[has_pick] = "picked"
    state[locked & ~has_pick] = "missed"
    state[has_pick & b["status"].eq("settled")] = "final"
    b["state"] = state
    b["locked"] = locked
    b["reveal"] = has_pick | locked
    final = b["state"] == "final"
    b["winner"] = b["actual_winner"].astype(object).where(final, None)
    # plain object comparisons: None/NaN never equal a team, and never raise
    b["model_correct"] = final & (b["winner"] == b["model_pick"].astype(object))
    b["user_correct"] = final & (b["winner"] == b["user_pick"])
    return b


def scoreboard(board):
    """Season record over slate games you picked in time that have a result.
    Missed games are excluded from BOTH sides."""
    f = board[board["state"] == "final"] if not board.empty else board
    n = len(f)
    if n == 0:
        return {"n": 0}
    agree = f["user_pick"] == f["model_pick"]
    dis = f[~agree]
    return {
        "n": n,
        "user_w": int(f["user_correct"].sum()),
        "model_w": int(f["model_correct"].sum()),
        "agree": int(agree.sum()),
        "disagree": len(dis),
        "dis_user": int(dis["user_correct"].sum()),
        "dis_model": int(dis["model_correct"].sum()),
        "missed": int((board["state"] == "missed").sum()),
    }

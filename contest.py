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
# Storage (one row per game), chosen at runtime:
#   * Supabase/Postgres when a connection URL is configured (env var
#     SUPABASE_DB_URL, or Streamlit secret [supabase] db_url) — durable, and the
#     lock/settle timestamps are set by the database, not the app. A trigger makes
#     locked fields immutable and settled rows read-only.
#   * data/contest_picks.csv otherwise (local dev / tests). On an ephemeral cloud
#     filesystem the CSV resets on restart, so set the URL for real use.

import os
from datetime import datetime, timezone

import pandas as pd

try:
    import sqlalchemy as sa
except ImportError:      # CSV mode works without the DB stack
    sa = None

DATA_DIR = "data"
PICKS_FILE = os.path.join(DATA_DIR, "contest_picks.csv")
STAKE = 100.0

COLUMNS = ["id", "created_utc", "season", "team_a", "team_b", "location",
           "user_pick", "user_conf", "model_pick", "model_prob_pick",
           "ml_a", "ml_b", "status", "actual_winner", "settled_utc"]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Backend selection ───────────────────────────────────────────

_engine = None
_engine_url = None


def _db_url():
    """The Postgres URL if one is configured, else None (-> CSV mode)."""
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("supabase", {}).get("db_url")
        except Exception:        # no secrets file / not running under Streamlit
            url = None
    if url:                      # pin the psycopg2 driver we ship in requirements
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                url = "postgresql+psycopg2://" + url[len(prefix):]
                break
    return url or None


def storage_backend():
    """'supabase' or 'csv' — shown in the UI so you can see where picks live."""
    return "supabase" if (_db_url() and sa is not None) else "csv"


_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS contest_picks (
        id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        created_utc      TIMESTAMPTZ NOT NULL DEFAULT now(),
        season           INTEGER NOT NULL,
        team_a           TEXT NOT NULL,
        team_b           TEXT NOT NULL,
        location         TEXT NOT NULL CHECK (location IN ('A','B','N')),
        user_pick        TEXT NOT NULL,
        user_conf        DOUBLE PRECISION,
        model_pick       TEXT NOT NULL,
        model_prob_pick  DOUBLE PRECISION NOT NULL,
        ml_a             DOUBLE PRECISION,
        ml_b             DOUBLE PRECISION,
        status           TEXT NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending','settled')),
        actual_winner    TEXT NOT NULL DEFAULT '',
        settled_utc      TIMESTAMPTZ
    )""",
    # Commit-then-reveal, enforced by the database: the picks, teams and lock time
    # can never change; only a pending row can be settled; settled rows are final.
    """CREATE OR REPLACE FUNCTION contest_picks_guard() RETURNS trigger AS $$
    BEGIN
        IF TG_OP = 'DELETE' THEN
            IF OLD.status = 'settled' THEN
                RAISE EXCEPTION 'settled picks are read-only';
            END IF;
            RETURN OLD;
        END IF;
        IF OLD.status = 'settled' THEN
            RAISE EXCEPTION 'settled picks are read-only';
        END IF;
        IF NEW.created_utc IS DISTINCT FROM OLD.created_utc
           OR NEW.season IS DISTINCT FROM OLD.season
           OR NEW.team_a IS DISTINCT FROM OLD.team_a
           OR NEW.team_b IS DISTINCT FROM OLD.team_b
           OR NEW.location IS DISTINCT FROM OLD.location
           OR NEW.user_pick IS DISTINCT FROM OLD.user_pick
           OR NEW.user_conf IS DISTINCT FROM OLD.user_conf
           OR NEW.model_pick IS DISTINCT FROM OLD.model_pick
           OR NEW.model_prob_pick IS DISTINCT FROM OLD.model_prob_pick
           OR NEW.ml_a IS DISTINCT FROM OLD.ml_a
           OR NEW.ml_b IS DISTINCT FROM OLD.ml_b THEN
            RAISE EXCEPTION 'locked pick fields cannot be changed';
        END IF;
        IF NEW.status = 'settled' THEN
            NEW.settled_utc := now();
        END IF;
        RETURN NEW;
    END $$ LANGUAGE plpgsql""",
    "DROP TRIGGER IF EXISTS contest_picks_guard ON contest_picks",
    """CREATE TRIGGER contest_picks_guard BEFORE UPDATE OR DELETE ON contest_picks
       FOR EACH ROW EXECUTE FUNCTION contest_picks_guard()""",
]


def _get_engine():
    """Create (once per URL) the engine and make sure the table + guard exist."""
    global _engine, _engine_url
    url = _db_url()
    if _engine is None or _engine_url != url:
        _engine = sa.create_engine(url, pool_pre_ping=True, pool_recycle=300)
        with _engine.begin() as con:
            for stmt in _SCHEMA:
                con.execute(sa.text(stmt))
        _engine_url = url
    return _engine


def _use_db():
    return storage_backend() == "supabase"


def _normalize(df):
    """Force the canonical column set and text dtypes (see note below)."""
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    # A wholly-empty text column is read as float64 (all-NaN), which then rejects
    # string assignment (e.g. settling writes a timestamp). Force every text
    # column to object/"" so equality checks and later writes both work.
    str_cols = ["created_utc", "team_a", "team_b", "location", "user_pick",
                "model_pick", "status", "actual_winner", "settled_utc"]
    for c in str_cols:
        df[c] = df[c].fillna("").astype(str).replace({"nan": "", "NaT": ""})
    return df[COLUMNS]


def load_picks():
    """Read the contest store fresh (never cache — it mutates). Returns an
    empty, correctly-typed frame if nothing is stored yet."""
    if _use_db():
        df = pd.read_sql(sa.text("SELECT * FROM contest_picks ORDER BY id"),
                         _get_engine())
        for c in ("created_utc", "settled_utc"):       # match the CSV's ISO strings
            df[c] = pd.to_datetime(df[c], utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%S+00:00").where(df[c].notna(), "")
        return _normalize(df)
    if not os.path.exists(PICKS_FILE):
        return pd.DataFrame(columns=COLUMNS)
    return _normalize(pd.read_csv(PICKS_FILE))


def save_picks(df):
    """CSV mode only — the database is written row-by-row, never wholesale."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(PICKS_FILE, index=False)


def add_pick(season, team_a, team_b, location, user_pick,
             model_pick, model_prob_pick, user_conf=None, ml_a=None, ml_b=None):
    """Record one game: my pick and the model's pick, captured together and
    timestamped. Returns the new row id."""
    if _use_db():
        with _get_engine().begin() as con:
            return int(con.execute(sa.text(
                """INSERT INTO contest_picks
                   (season, team_a, team_b, location, user_pick, user_conf,
                    model_pick, model_prob_pick, ml_a, ml_b)
                   VALUES (:season, :team_a, :team_b, :location, :user_pick,
                           :user_conf, :model_pick, :prob, :ml_a, :ml_b)
                   RETURNING id"""),
                {"season": int(season), "team_a": team_a, "team_b": team_b,
                 "location": location, "user_pick": user_pick,
                 "user_conf": user_conf, "model_pick": model_pick,
                 "prob": round(float(model_prob_pick), 4),
                 "ml_a": ml_a, "ml_b": ml_b}).scalar_one())
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
    if _use_db():
        with _get_engine().begin() as con:
            res = con.execute(sa.text(
                """UPDATE contest_picks SET actual_winner = :w, status = 'settled'
                   WHERE id = :id AND status = 'pending'"""),
                {"w": actual_winner, "id": int(pick_id)})
            return res.rowcount > 0
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
    """Remove a pick entirely (e.g. an entry mistake). In the database only a
    still-pending pick can be removed."""
    if _use_db():
        with _get_engine().begin() as con:
            con.execute(sa.text(
                "DELETE FROM contest_picks WHERE id = :id AND status = 'pending'"),
                {"id": int(pick_id)})
        return
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

# weekly_slate.py — Build the weekly "Me vs. the Machine" slate (item 14, auto phase).
#
# Every week the model picks the 5 games it is LEAST sure about and freezes its
# pick for each one. You then pick the same 5 games blind. The selection rule:
#
#   1. Gate. Both teams D1 and rated, game not yet played, date inside the week,
#      at least one team in the national top-60 by BARTHAG and both in the
#      top-75 (so the slate isn't low-major coin flips nobody watches).
#   2. Score. P(home wins) from the point-in-time game model, then the binary
#      entropy H(p) = -p log2 p - (1-p) log2(1-p). H = 1 bit is a pure coin flip
#      and H = 0 is a lock. This is the same entropy the original decision tree
#      split on.
#   3. Select. Take the highest-entropy games greedily, skipping any game whose
#      team is already on the slate. Ties go to the higher-quality matchup
#      (better min BARTHAG of the two teams).
#
# The model's pick, its probability and the entropy are written at generation
# time and never recomputed, so the model can't "learn" from results before its
# picks are scored.
#
# Output: data/slates/<season>_<week_start>.csv (one row per slate game).
#
# Usage:
#   python weekly_slate.py                    # next Mon-Sun week, live Barttorvik data
#   python weekly_slate.py --week 2026-11-09  # a specific week (any date inside it)
#   python weekly_slate.py --k 5 --top 60 --both-top 75   # tune size / quality gate
#
# Needs network access to barttorvik.com (run locally or from GitHub Actions).

import argparse
import csv
import html as htmllib
import io
import math
import os
import re
import time as _time
from datetime import date, datetime, timedelta, timezone
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

import fetch_data as fd
import game_model as gm

SLATE_DIR = os.path.join(gm.DATA_DIR, "slates")
SLATE_SIZE = 5
TOP_N_GATE = 60
# Both teams must also be top-75: replaying 2025 and 2026 this kept slates at
# ~0.99 bits of entropy while lifting the median worse-team rank from ~45-56 to
# ~39-48 (Iowa St. at Kansas instead of UCF at Kansas St.).
BOTH_TOP_GATE = 75

SLATE_COLUMNS = [
    "slate_id", "season", "week_start", "week_end", "game_id", "date",
    "home", "away", "neutral", "game_type",
    "model_p_home", "model_pick", "model_prob_pick", "entropy_bits",
    "home_barthag_rank", "away_barthag_rank", "tip_et",
    "generated_utc", "model_training_data", "ratings_as_of",
]


# ──────────────────────────────────────────────────────────────
# SMALL HELPERS
# ──────────────────────────────────────────────────────────────

def entropy_bits(p):
    """Binary entropy in bits. Works on scalars or arrays; H(0)=H(1)=0."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def season_for(d):
    """Barttorvik season label: the 2026-27 season is 2027 (Nov-Apr)."""
    return d.year + 1 if d.month >= 7 else d.year


def week_bounds(d):
    """Monday..Sunday of the week containing date d."""
    start = d - timedelta(days=d.weekday())
    return start, start + timedelta(days=6)


def next_week_start(today=None):
    """The coming Monday (today if it's Monday). The slate is meant to be built
    on the weekend, before the first game of the week."""
    today = today or date.today()
    return today + timedelta(days=(7 - today.weekday()) % 7)


# ──────────────────────────────────────────────────────────────
# SCHEDULE  (super_sked, including games not yet played)
# ──────────────────────────────────────────────────────────────

def parse_schedule(year, text=None):
    """Every D1-vs-D1 game on the season's schedule, played or not.

    Same column positions as fetch_data.parse_super_sked (which keeps only
    completed games): [1] date, [6] type (99 = non-D1 opponent), [7] neutral,
    [8] away, [14] home, [29] winner (blank until the game is played).
    ``text`` lets tests pass a CSV string instead of downloading.
    """
    if text is None:
        text = fd._download(year, "super_sked")
    out = []
    for r in csv.reader(io.StringIO(text)):
        if len(r) < 15 or r[6].strip() == "99":
            continue
        away, home = r[8].strip(), r[14].strip()
        if not away or not home:
            continue
        try:
            game_date = datetime.strptime(r[1].strip(), "%m/%d/%y").date()
        except ValueError:
            continue
        winner = r[29].strip() if len(r) > 29 else ""
        out.append({
            "YEAR": year,
            "DATE": game_date.isoformat(),
            "AWAY": away,
            "HOME": home,
            "NEUTRAL": r[7].strip() == "1",
            "GAME_TYPE": fd.GAME_TYPES.get(r[6].strip(), "unknown"),
            "PLAYED": winner in (home, away),
        })
    sched = pd.DataFrame(out, columns=["YEAR", "DATE", "AWAY", "HOME", "NEUTRAL",
                                       "GAME_TYPE", "PLAYED"])
    return sched.drop_duplicates(subset=["DATE", "HOME", "AWAY"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# SCHEDULE  (Barttorvik's daily schedule pages — the live source)
# ──────────────────────────────────────────────────────────────
#
# {season}_super_sked.csv isn't published until well into the season (on
# 2026-11-01 the 2027 file was still a redirect to the preseason page), but the
# daily page https://barttorvik.com/schedule.php?date=YYYYMMDD lists every game
# with its tip time (ET), venue and TV. One row per game:
#   <td> <span class="gametime">05:00 PM</span> </td>
#   <td title="Arena — City, ST"><span ...>3</span> <a href="team.php?team=…">Florida</a>
#       vs|at <span ...>12</span> <a href="team.php?team=…">Miami FL</a> … TV … </td>
#   … <td> result link (empty until played) </td>
# "vs" = neutral court (first team listed is just an orientation), "at" = the
# second team is home. A T-Rank of 0 marks a non-D1 opponent.

SCHEDULE_URL = "https://barttorvik.com/schedule.php?date={ymd}&conlimit="
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_TD = re.compile(r"<td([^>]*)>(.*?)</td>", re.S | re.I)
_TEAM = re.compile(r'<a href="team\.php\?team=[^"]*">([^<]+)</a>', re.I)
_RANK = re.compile(r'<span class="mobileout" style="font-size:8px;">\s*(\d+)\s*</span>', re.I)
_TIME = re.compile(r'class="gametime">\s*([0-9]{1,2}:[0-9]{2}\s*[AP]M)', re.I)
_TV = re.compile(r'<span class="mobileout" style="color:gray">\s*([^<]*?)\s*</span>', re.I)
_TITLE = re.compile(r'title="([^"]*)"', re.I)
_TAGS = re.compile(r"<[^>]+>")


def parse_schedule_page(page, game_date, include_non_d1=False):
    """Games listed on one Barttorvik schedule page.

    Returns a frame with YEAR, DATE, AWAY, HOME, NEUTRAL, GAME_TYPE, PLAYED,
    TIP_ET ('07:00 PM', or '' if missing), TV, VENUE and D1. Only D1-vs-D1
    games unless ``include_non_d1`` (rest-day counting needs every game).
    Pure function, so tests can feed saved pages."""
    d = game_date if isinstance(game_date, date) else date.fromisoformat(str(game_date))
    out = []
    for row in _ROW.findall(page):
        tds = _TD.findall(row)
        if len(tds) < 5:
            continue                       # header row
        attrs, cell = tds[1]
        teams = _TEAM.findall(cell)
        ranks = _RANK.findall(cell)
        if len(teams) != 2:
            continue
        d1 = not (len(ranks) == 2 and "0" in [r.strip() for r in ranks])
        if not d1 and not include_non_d1:
            continue                       # non-D1 opponent
        first, second = (htmllib.unescape(t).strip() for t in teams)
        between = _TAGS.sub(" ", cell.split(teams[0], 1)[1].split(teams[1], 1)[0])
        neutral = re.search(r"\bvs\.?\b", between, re.I) is not None
        tm = _TIME.search(tds[0][1])
        tv = _TV.search(cell)
        venue = _TITLE.search(attrs)
        result = _TAGS.sub("", tds[4][1]).strip()
        out.append({
            "YEAR": season_for(d), "DATE": d.isoformat(),
            "AWAY": first, "HOME": second, "NEUTRAL": neutral,
            "GAME_TYPE": "", "PLAYED": bool(result),
            "TIP_ET": tm.group(1).upper().replace("  ", " ") if tm else "",
            "TV": htmllib.unescape(tv.group(1)).strip() if tv else "",
            "VENUE": htmllib.unescape(venue.group(1)).strip() if venue else "",
            "D1": d1,
            "RESULT": result,       # raw result-cell text; see slate_results.py
        })
    cols = ["YEAR", "DATE", "AWAY", "HOME", "NEUTRAL", "GAME_TYPE", "PLAYED",
            "TIP_ET", "TV", "VENUE", "D1", "RESULT"]
    return pd.DataFrame(out, columns=cols)


def _get(url):
    req = Request(url, headers={"User-Agent": fd.USER_AGENT})
    with urlopen(req, timeout=45) as resp:
        return resp.read().decode("utf-8", errors="replace")


class IncompleteSchedule(RuntimeError):
    """A schedule page came back empty where games are certain (a Saturday).
    Barttorvik sometimes serves empty pages to rapid-fire requests; on
    2026-11-01 a fast pull missed every weekend game this way."""


def fetch_days(start, n_days, fetch=None, pause=1.5, retries=2, include_non_d1=False):
    """Schedule pages for ``n_days`` days from ``start``. Requests are spaced
    ``pause`` seconds apart and an empty day is re-fetched ``retries`` more
    times (longer waits each time). Returns (games, empty_days)."""
    fetch = fetch or _get
    frames, empty = [], []
    for i in range(n_days):
        d = start + timedelta(days=i)
        url = SCHEDULE_URL.format(ymd=d.strftime("%Y%m%d"))
        for attempt in range(retries + 1):
            if pause and (i or attempt):
                _time.sleep(pause * (1 + 3 * attempt))
            g = parse_schedule_page(fetch(url), d, include_non_d1=include_non_d1)
            if len(g):
                break
        frames.append(g)
        if g.empty:
            empty.append(d)
    games = pd.concat(frames, ignore_index=True)
    games = games.drop_duplicates(subset=["DATE", "HOME", "AWAY"]).reset_index(drop=True)
    return games, empty


def fetch_week_schedule(week_start, fetch=None, pause=1.5, retries=2, include_non_d1=False):
    """Every game Monday..Sunday from the daily schedule pages (D1 vs D1 only
    unless ``include_non_d1``).

    A Saturday with no D1 games raises IncompleteSchedule: in-season Saturdays
    always have games, so an empty one means the pull failed."""
    games, _ = fetch_days(week_start, 7, fetch=fetch, pause=pause, retries=retries,
                          include_non_d1=include_non_d1)
    d1 = games[games["D1"].astype(bool)]
    empty = []
    for i in range(7):
        d = week_start + timedelta(days=i)
        n = int((d1["DATE"] == d.isoformat()).sum())
        print(f"  {d:%a %b %d}: {n} D1 games listed", flush=True)
        if n == 0:
            empty.append(d)
    if any(d.weekday() == 5 for d in empty):
        raise IncompleteSchedule(
            f"No games came back for Saturday {[d for d in empty if d.weekday() == 5][0]}"
            " — the schedule pull looks incomplete. Wait a few minutes and run again.")
    if empty:
        print(f"  [note] no games listed on: {', '.join(f'{d:%a %b %d}' for d in empty)}",
              flush=True)
    return games


# ──────────────────────────────────────────────────────────────
# TEAM STATS SNAPSHOT  (what the card shows next to each matchup)
# ──────────────────────────────────────────────────────────────
#
# Frozen with the slate, so the card shows the numbers the model saw on build
# day. Everything here is a model input or common fan knowledge — no
# predictions (no lines, win %, or the model's own probability).

STATS_SUBDIR = "stats"           # data/slates/stats/<slate_id>.csv
STAT_COLUMNS = [
    "slate_id", "game_id", "side", "team", "record", "conf_record", "trank",
    "adjoe", "adjoe_rank", "adjde", "adjde_rank", "adjt", "adjt_rank",
    "sos", "sos_rank", "wab", "wab_rank", "rest", "as_of",
]


def team_tables(season, text=None):
    """(ratings, raw) for the season from {season}_team_results.csv: ``ratings``
    is the tidy model-feature frame, ``raw`` keeps record / Con Rec. too."""
    if text is None:
        text = fd._download(season, "team_results")
    raw = pd.read_csv(io.StringIO(text), index_col=False)
    raw.columns = [str(c).strip() for c in raw.columns]
    ratings = fd._tidy_ratings(raw.copy(), season)
    raw = raw.rename(columns={"team": "TEAM"})
    raw["TEAM"] = raw["TEAM"].astype(str).str.strip()
    return ratings, raw


def _rank(series, ascending):
    """National rank (1 = best). All-equal columns (e.g. strength of schedule
    and WAB before anyone has played) get no rank instead of a meaningless #1."""
    if series.nunique(dropna=True) <= 1:
        return pd.Series(pd.NA, index=series.index, dtype="Int64")
    return series.rank(ascending=ascending, method="min").astype("Int64")


def _num(v, nd):
    return "" if pd.isna(v) else round(float(v), nd)


def _int(v):
    return "" if pd.isna(v) else int(v)


def rest_label(team, game_date, schedule, opener_ok, window_ok):
    """Days since the team's previous game before ``game_date``, from every
    game listed (D1 or not, played or scheduled). 'Opener' when the team has
    played no games yet and has none earlier in the week; '7+' when the
    look-back window had no game for it."""
    played = schedule[((schedule["HOME"] == team) | (schedule["AWAY"] == team))
                      & (schedule["DATE"] < game_date)]
    if len(played):
        days = (date.fromisoformat(game_date) - date.fromisoformat(played["DATE"].max())).days
        return f"{days} day{'s' if days != 1 else ''}"
    if opener_ok:
        return "Opener"
    return "7+ days" if window_ok else ""


def team_stats(raw, slate_rows, schedule, as_of, window_ok=True):
    """One row per (slate game, side) with the card's numbers. Ranks are
    national, over every team in ``raw`` (1 = best; for tempo 1 = fastest)."""
    r = raw.copy()
    for c in ("adjoe", "adjde", "adjt", "sos", "WAB", "barthag"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r["trank"] = _rank(r["barthag"], ascending=False)
    r["adjoe_rank"] = _rank(r["adjoe"], ascending=False)
    r["adjde_rank"] = _rank(r["adjde"], ascending=True)      # fewer points allowed
    r["adjt_rank"] = _rank(r["adjt"], ascending=False)
    r["sos_rank"] = _rank(r["sos"], ascending=False)
    r["wab_rank"] = _rank(r["WAB"], ascending=False)
    r = r.set_index("TEAM")
    out = []
    for g in slate_rows.itertuples(index=False):
        for side, team in (("away", g.away), ("home", g.home)):
            if team not in r.index:
                continue
            t = r.loc[team]
            rec = str(t.get("record", "")).strip()
            out.append({
                "slate_id": g.slate_id, "game_id": g.game_id, "side": side, "team": team,
                "record": rec, "conf_record": str(t.get("Con Rec.", "")).strip(),
                "trank": _int(t["trank"]),
                "adjoe": _num(t["adjoe"], 1), "adjoe_rank": _int(t["adjoe_rank"]),
                "adjde": _num(t["adjde"], 1), "adjde_rank": _int(t["adjde_rank"]),
                "adjt": _num(t["adjt"], 1), "adjt_rank": _int(t["adjt_rank"]),
                "sos": _num(t["sos"], 3), "sos_rank": _int(t["sos_rank"]),
                "wab": _num(t["WAB"], 1), "wab_rank": _int(t["wab_rank"]),
                "rest": rest_label(team, g.date, schedule,
                                   opener_ok=rec in ("0-0", ""), window_ok=window_ok),
                "as_of": as_of,
            })
    return pd.DataFrame(out, columns=STAT_COLUMNS)


# ──────────────────────────────────────────────────────────────
# SCORING + SELECTION  (pure functions — no network, easy to test)
# ──────────────────────────────────────────────────────────────

def score_games(games, ratings, model, features):
    """Attach P(home wins), entropy and both teams' BARTHAG ranks.

    ``games``: YEAR/DATE/HOME/AWAY/NEUTRAL (+ GAME_TYPE). ``ratings``: one row per
    team for the season (YEAR/TEAM/<features>), as of the slate date. Games with
    an unrated team are dropped.
    """
    r = ratings.copy()
    r["BARTHAG_RANK"] = r["barthag"].rank(ascending=False, method="min").astype(int)
    feat = r[["TEAM", "BARTHAG_RANK"] + features]

    m = games.merge(feat.add_suffix("_H").rename(columns={"TEAM_H": "HOME"}),
                    on="HOME", how="inner")
    m = m.merge(feat.add_suffix("_A").rename(columns={"TEAM_A": "AWAY"}),
                on="AWAY", how="inner")
    if m.empty:
        return m.assign(P_HOME=[], ENTROPY=[], MIN_BARTHAG=[])

    X = np.column_stack(
        [(m[f"{c}_H"] - m[f"{c}_A"]).values for c in features]
        + [np.where(m["NEUTRAL"].astype(bool).values, 0.0, 1.0)])
    pos = list(model.classes_).index(1)
    m["P_HOME"] = model.predict_proba(X)[:, pos]
    m["ENTROPY"] = entropy_bits(m["P_HOME"].values)
    m["MIN_BARTHAG"] = np.minimum(m["barthag_H"], m["barthag_A"])
    keep = [c for c in games.columns] + ["BARTHAG_RANK_H", "BARTHAG_RANK_A",
                                         "P_HOME", "ENTROPY", "MIN_BARTHAG"]
    return m[keep]


def select_slate(scored, k=SLATE_SIZE, top_n=TOP_N_GATE, both_top=BOTH_TOP_GATE):
    """Gate, rank by entropy, and greedily take k games with no repeated team.

    top_n: at least one team must be ranked <= top_n by BARTHAG.
    both_top: optional stricter floor — BOTH teams ranked <= both_top.
    """
    rh, ra = scored["BARTHAG_RANK_H"], scored["BARTHAG_RANK_A"]
    gate = (rh <= top_n) | (ra <= top_n)
    if both_top is not None:
        gate &= (rh <= both_top) & (ra <= both_top)
    pool = scored[gate].sort_values(["ENTROPY", "MIN_BARTHAG"],
                                    ascending=[False, False])
    picked, used = [], set()
    for row in pool.itertuples(index=False):
        if row.HOME in used or row.AWAY in used:
            continue
        picked.append(row)
        used.update((row.HOME, row.AWAY))
        if len(picked) == k:
            break
    return pd.DataFrame(picked, columns=pool.columns)


def to_slate_rows(slate, season, week_start, meta, ratings_as_of):
    """Freeze the model's picks into the stored slate schema."""
    week_end = week_start + timedelta(days=6)
    slate_id = f"{season}_{week_start.isoformat()}"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for s in slate.itertuples(index=False):
        home_pick = s.P_HOME >= 0.5
        rows.append({
            "slate_id": slate_id,
            "season": season,
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "game_id": f"{s.DATE}|{s.AWAY}@{s.HOME}",
            "date": s.DATE,
            "home": s.HOME,
            "away": s.AWAY,
            "neutral": bool(s.NEUTRAL),
            "game_type": getattr(s, "GAME_TYPE", ""),
            "model_p_home": round(float(s.P_HOME), 4),
            "model_pick": s.HOME if home_pick else s.AWAY,
            "model_prob_pick": round(float(s.P_HOME if home_pick else 1 - s.P_HOME), 4),
            "entropy_bits": round(float(s.ENTROPY), 4),
            "home_barthag_rank": int(s.BARTHAG_RANK_H),
            "away_barthag_rank": int(s.BARTHAG_RANK_A),
            "tip_et": getattr(s, "TIP_ET", "") or "",
            "generated_utc": now,
            "model_training_data": meta.get("training_data", "season-aggregate"),
            "ratings_as_of": ratings_as_of,
        })
    return pd.DataFrame(rows, columns=SLATE_COLUMNS)


# ──────────────────────────────────────────────────────────────
# DRIVER
# ──────────────────────────────────────────────────────────────

def build_slate(week_start, k=SLATE_SIZE, top_n=TOP_N_GATE,
                both_top=BOTH_TOP_GATE, overwrite=False, today=None, dry_run=False):
    """Fetch the schedule + today's ratings, score the week, write the slate."""
    week_start, week_end = week_bounds(week_start)
    season = season_for(week_start)
    out_path = os.path.join(SLATE_DIR, f"{season}_{week_start.isoformat()}.csv")
    if os.path.exists(out_path) and not overwrite and not dry_run:
        # A frozen slate is never regenerated: that would let the model re-pick
        # with newer ratings. Pass --overwrite only to fix a broken slate.
        print(f"Slate already exists, leaving it frozen: {out_path}")
        return pd.read_csv(out_path)

    artifact = gm.load_model()
    if artifact is None:
        raise SystemExit("No data/game_model.joblib — run `python game_model.py` first.")
    model, meta = artifact["model"], artifact["meta"]
    features = meta["features"]

    print(f"Season {season}: week {week_start} → {week_end}", flush=True)
    sched_all = fetch_week_schedule(week_start, include_non_d1=True)
    sched = sched_all[sched_all["D1"].astype(bool)]
    # Only games that haven't been played and aren't today or earlier — built
    # mid-week, the slate skips days whose games may already have tipped.
    today = today or date.today()
    first_day = max(week_start, today + timedelta(days=1)).isoformat()
    in_week = (sched["DATE"] >= first_day) & \
              (sched["DATE"] <= week_end.isoformat()) & ~sched["PLAYED"]
    week_games = sched[in_week]
    print(f"  {len(week_games)} unplayed D1-vs-D1 games from {first_day}", flush=True)

    ratings, raw = team_tables(season)              # live, i.e. point-in-time today
    missing = [c for c in features if c not in ratings.columns]
    if missing:
        raise SystemExit(f"Ratings are missing model features: {missing}")

    scored = score_games(week_games, ratings, model, features)
    rated = set(ratings["TEAM"])
    unmatched = sorted({t for t in pd.concat([week_games.HOME, week_games.AWAY])
                        if t not in rated})
    if unmatched:
        print(f"  [warning] {len(unmatched)} team name(s) not in the ratings, their "
              f"games were skipped: {', '.join(unmatched[:10])}", flush=True)
    slate = select_slate(scored, k=k, top_n=top_n, both_top=both_top)
    if slate.empty:
        # Don't freeze an empty slate — the schedule may just not be posted yet,
        # and a frozen empty file would block the real slate later.
        print("  No games passed the gate; nothing written.", flush=True)
        return slate
    if len(slate) < k:
        print(f"  [warning] only {len(slate)} games passed the gate", flush=True)

    rows = to_slate_rows(slate, season, week_start, meta,
                         ratings_as_of=today.isoformat())
    if dry_run:
        print("Dry run: these would be the slate games (nothing written):")
        print(rows[["date", "tip_et", "away", "home", "neutral"]].to_string(index=False))
        return rows
    os.makedirs(SLATE_DIR, exist_ok=True)
    rows.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    # Card stats, frozen with the slate. Rest days need the week before too.
    try:
        prior, prior_empty = fetch_days(week_start - timedelta(days=7), 7,
                                        retries=1, include_non_d1=True)
        window_ok = len(prior_empty) < 7     # all 7 empty = the pull failed
    except Exception as e:  # noqa: BLE001 — stats are a nice-to-have
        print(f"  [warning] couldn't fetch last week's schedule ({e}); rest days unknown")
        prior, window_ok = sched_all.iloc[0:0], False
    stats = team_stats(raw, rows, pd.concat([prior, sched_all], ignore_index=True),
                       as_of=today.isoformat(), window_ok=window_ok)
    stats_dir = os.path.join(SLATE_DIR, STATS_SUBDIR)
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"{season}_{week_start.isoformat()}.csv")
    stats.to_csv(stats_path, index=False)
    print(f"Wrote {stats_path} ({len(stats)} team rows)")
    print(rows[["date", "tip_et", "away", "home", "neutral",
                "entropy_bits"]].to_string(index=False))
    print("(Model picks are in the file; don't look before you lock yours.)")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the weekly Me-vs-Machine slate (5 highest-entropy games).")
    ap.add_argument("--week", help="any date (YYYY-MM-DD) inside the target week")
    ap.add_argument("--k", type=int, default=SLATE_SIZE)
    ap.add_argument("--top", type=int, default=TOP_N_GATE,
                    help="at least one team ranked <= this by BARTHAG")
    ap.add_argument("--both-top", type=int, default=BOTH_TOP_GATE,
                    help="both teams ranked <= this (0 disables)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild an existing slate (only before any of its games lock)")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and score, print the matchups, write nothing")
    a = ap.parse_args(argv)
    wk = (datetime.strptime(a.week, "%Y-%m-%d").date() if a.week
          else next_week_start())
    rows = build_slate(wk, k=a.k, top_n=a.top, both_top=a.both_top or None,
                       overwrite=a.overwrite, dry_run=a.dry_run)
    # A non-zero exit makes a scheduled run (GitHub Actions / Task Scheduler)
    # show up as failed, so an empty week never passes silently.
    return 0 if len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

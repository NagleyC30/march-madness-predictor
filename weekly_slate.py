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


def parse_schedule_page(page, game_date):
    """Games listed on one Barttorvik schedule page (D1 vs D1 only).

    Returns a frame with YEAR, DATE, AWAY, HOME, NEUTRAL, GAME_TYPE, PLAYED,
    TIP_ET ('07:00 PM', or '' if missing), TV and VENUE. Pure function, so
    tests can feed saved pages."""
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
        if len(ranks) == 2 and "0" in [r.strip() for r in ranks]:
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
        })
    cols = ["YEAR", "DATE", "AWAY", "HOME", "NEUTRAL", "GAME_TYPE", "PLAYED",
            "TIP_ET", "TV", "VENUE"]
    return pd.DataFrame(out, columns=cols)


def _get(url):
    req = Request(url, headers={"User-Agent": fd.USER_AGENT})
    with urlopen(req, timeout=45) as resp:
        return resp.read().decode("utf-8", errors="replace")


class IncompleteSchedule(RuntimeError):
    """A schedule page came back empty where games are certain (a Saturday).
    Barttorvik sometimes serves empty pages to rapid-fire requests; on
    2026-11-01 a fast pull missed every weekend game this way."""


def fetch_week_schedule(week_start, fetch=None, pause=1.5, retries=2):
    """Every D1-vs-D1 game Monday..Sunday from the daily schedule pages.

    Requests are spaced ``pause`` seconds apart, and a day with no games is
    re-fetched ``retries`` more times (with a longer wait) before it's accepted
    as genuinely empty. A Saturday that stays empty raises IncompleteSchedule:
    in-season Saturdays always have games, so it means the pull failed."""
    fetch = fetch or _get
    frames, empty = [], []
    for i in range(7):
        d = week_start + timedelta(days=i)
        url = SCHEDULE_URL.format(ymd=d.strftime("%Y%m%d"))
        for attempt in range(retries + 1):
            if pause and (i or attempt):
                _time.sleep(pause * (1 + 3 * attempt))
            g = parse_schedule_page(fetch(url), d)
            if len(g):
                break
        print(f"  {d:%a %b %d}: {len(g)} D1 games listed", flush=True)
        if g.empty:
            empty.append(d)
        frames.append(g)
    if any(d.weekday() == 5 for d in empty):
        raise IncompleteSchedule(
            f"No games came back for Saturday {[d for d in empty if d.weekday() == 5][0]}"
            " — the schedule pull looks incomplete. Wait a few minutes and run again.")
    if empty:
        print(f"  [note] no games listed on: {', '.join(f'{d:%a %b %d}' for d in empty)}",
              flush=True)
    sched = pd.concat(frames, ignore_index=True)
    return sched.drop_duplicates(subset=["DATE", "HOME", "AWAY"]).reset_index(drop=True)


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
    sched = fetch_week_schedule(week_start)
    # Only games that haven't been played and aren't today or earlier — built
    # mid-week, the slate skips days whose games may already have tipped.
    today = today or date.today()
    first_day = max(week_start, today + timedelta(days=1)).isoformat()
    in_week = (sched["DATE"] >= first_day) & \
              (sched["DATE"] <= week_end.isoformat()) & ~sched["PLAYED"]
    week_games = sched[in_week]
    print(f"  {len(week_games)} unplayed D1-vs-D1 games from {first_day}", flush=True)

    ratings = fd.parse_team_results(season)          # live, i.e. point-in-time today
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

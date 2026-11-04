# slate_results.py — Pull the final result of each weekly-slate game once it's
# over and Barttorvik has posted it.
#
# Barttorvik's daily schedule page (the same one weekly_slate.py reads) fills in
# its Result cell after a game ends. This script walks every slate game whose
# date has arrived, looks the game up on that day's page, and appends the winner
# to data/slates/results/<slate_id>.csv. A scheduled GitHub Action runs it every
# few hours and commits the file; the slate page reads it and settles your locked
# picks automatically (slate_contest.auto_settle). The "Record result" button
# stays as a manual override, e.g. if a result is wrong or never posted.
#
# Rules: a recorded result is never overwritten; a game with no recognisable
# result is simply left for the next run (postponements just stay open).
#
# Usage:
#   python slate_results.py              # fetch results for all slate games due
#   python slate_results.py --dry-run    # print what it sees (raw result text), write nothing
#   python slate_results.py --today 2026-11-09

import argparse
import os
import re
from datetime import date, datetime, timedelta, timezone

import pandas as pd

import weekly_slate as ws

SLATE_DIR = ws.SLATE_DIR
RESULTS_SUBDIR = "results"          # data/slates/results/<slate_id>.csv
RESULT_COLUMNS = ["slate_id", "game_id", "winner", "result_text", "source", "fetched_utc"]

# Result cell, e.g. "W, 78-70". W/L is read from the point of view of the team
# listed FIRST on the schedule page (the AWAY slot in weekly_slate's parse).
# VERIFY ON THE FIRST LIVE RUN (`--dry-run` prints the raw text): if Barttorvik
# uses the other perspective, flip RESULT_PERSPECTIVE.
RESULT_PERSPECTIVE = "first"
_WL = re.compile(r"^\s*([WL])\b[\s,:-]*(\d{1,3})\s*[-–]\s*(\d{1,3})", re.I)
_NAMED = re.compile(r"^\s*(.+?)\s+(\d{1,3}),\s*(.+?)\s+(\d{1,3})\s*$")


def parse_result(text, first, second):
    """Winner's team name from a Result-cell string, or None if it isn't a
    recognised final score. ``first``/``second`` are the page's listed teams
    (away slot, home slot). Refuses ties and a W/L that contradicts its score."""
    text = (text or "").strip()
    if not text:
        return None
    m = _WL.match(text)
    if m:
        wl, a, b = m.group(1).upper(), int(m.group(2)), int(m.group(3))
        if a == b or (wl == "W") != (a > b):      # tie, or "W, 60-70": don't guess
            return None
        viewpoint, other = (first, second) if RESULT_PERSPECTIVE == "first" else (second, first)
        return viewpoint if wl == "W" else other
    m = _NAMED.match(text)                         # "Florida 82, Miami FL 79"
    if m and {m.group(1), m.group(3)} == {first, second}:
        sa, sb = int(m.group(2)), int(m.group(4))
        if sa != sb:
            return m.group(1) if sa > sb else m.group(3)
    return None


def results_path(slate_id, slate_dir=None):
    return os.path.join(slate_dir or SLATE_DIR, RESULTS_SUBDIR, f"{slate_id}.csv")


def load_results(slate_dir=None):
    """Every recorded result, one row per game_id (empty frame if none)."""
    d = os.path.join(slate_dir or SLATE_DIR, RESULTS_SUBDIR)
    paths = sorted(os.path.join(d, f) for f in os.listdir(d)) if os.path.isdir(d) else []
    if not paths:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = pd.concat([pd.read_csv(p, keep_default_na=False, dtype=str) for p in paths],
                   ignore_index=True)
    return df.drop_duplicates("game_id", keep="first").reset_index(drop=True)


def due_games(slates, results, today):
    """Slate games on or before ``today`` that have no recorded result yet."""
    if slates.empty:
        return slates
    done = set(results["game_id"]) if len(results) else set()
    return slates[(slates["date"] <= today.isoformat()) & ~slates["game_id"].isin(done)]


def collect(slates, results, today, fetch=None, pause=1.5, now=None):
    """New result rows for due games. One page fetch per game date; returns
    (rows_df, report) where report lists every due game with the raw text seen."""
    due = due_games(slates, results, today)
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, report, pages = [], [], {}
    for g in due.itertuples(index=False):
        if g.date not in pages:
            pages[g.date], _ = ws.fetch_days(date.fromisoformat(g.date), 1, fetch=fetch,
                                             pause=pause, retries=1, include_non_d1=True)
        p = pages[g.date]
        hit = p[(p["HOME"] == g.home) & (p["AWAY"] == g.away)]
        if hit.empty:                               # orientation can flip on neutral courts
            hit = p[(p["HOME"] == g.away) & (p["AWAY"] == g.home)]
        text = hit.iloc[0]["RESULT"] if len(hit) else None
        winner = None
        if len(hit):
            h = hit.iloc[0]
            winner = parse_result(h["RESULT"], h["AWAY"], h["HOME"])
            if winner not in (g.home, g.away):
                winner = None
        report.append((g.game_id, "not on page" if text is None else (text or "(no result yet)"),
                       winner))
        if winner:
            rows.append({"slate_id": g.slate_id, "game_id": g.game_id, "winner": winner,
                         "result_text": text, "source": "barttorvik-schedule",
                         "fetched_utc": now})
    return pd.DataFrame(rows, columns=RESULT_COLUMNS), report


def save(new_rows, slate_dir=None):
    """Append new rows to their slate's results file; existing rows are kept."""
    for sid, grp in new_rows.groupby("slate_id"):
        path = results_path(sid, slate_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        old = pd.read_csv(path, keep_default_na=False, dtype=str) if os.path.exists(path) \
            else pd.DataFrame(columns=RESULT_COLUMNS)
        out = pd.concat([old, grp], ignore_index=True).drop_duplicates("game_id", keep="first")
        out.to_csv(path, index=False)


def main(argv=None):
    import slate_contest as sc
    ap = argparse.ArgumentParser(description="Record final results for weekly-slate games.")
    ap.add_argument("--today", help="treat this date (YYYY-MM-DD) as today")
    ap.add_argument("--dry-run", action="store_true", help="print only, write nothing")
    a = ap.parse_args(argv)
    # Barttorvik dates are Eastern; late games finish after midnight UTC.
    today = date.fromisoformat(a.today) if a.today else datetime.now(sc.ET).date()
    slates = sc.load_slates()
    res = load_results()
    new, report = collect(slates, res, today)
    for gid, text, winner in report:
        print(f"  {gid}: result cell = {text!r} -> {winner or 'not final yet'}")
    print(f"{len(report)} game(s) awaiting a result, {len(new)} newly final.")
    if len(new) and not a.dry_run:
        save(new)
        print("Wrote results for:", ", ".join(new["game_id"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

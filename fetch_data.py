# fetch_data.py — Download & parse Barttorvik data for the general game predictor.
#
# Produces two tidy tables in ./data, joined by consistent (YEAR, TEAM) names:
#   ratings.csv  — one row per team-season, all D1 teams, numeric efficiency
#                  features (the model's inputs).
#   games.csv    — one row per completed D1-vs-D1 game, with home/away/neutral
#                  and the result (the training + prediction target).
#
# Source files (per season YYYY), fetched with a browser User-Agent because
# barttorvik.com gates automated requests:
#   https://barttorvik.com/{YYYY}_team_results.csv   (clean header)
#   https://barttorvik.com/{YYYY}_super_sked.csv      (NO header — parsed by
#                                                       column position, see below)
#
# super_sked column map (reverse-engineered & statistically verified on 2024):
#   [1]  date (m/d/yy)          [6]  game type: 0 nonconf, 1 conf, 2 conf
#   [7]  neutral flag (1=yes)        tourney, 3 postseason, 99 vs non-D1 (drop)
#   [8]  AWAY team              [14] HOME team   (team1 wins only ~35% of
#   [27] away score            [28] home score   non-neutral games → team2=home)
#   [29] winner name
#
# Usage:
#   python fetch_data.py                # just 2024
#   python fetch_data.py 2024           # one year
#   python fetch_data.py 2015 2024      # inclusive range 2015..2024

import csv
import io
import os
import sys
import time
from datetime import datetime
from urllib.request import Request, urlopen

import pandas as pd

DATA_DIR = "data"
BASE_URL = "https://barttorvik.com/{year}_{kind}.csv"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

GAME_TYPES = {"0": "nonconf", "1": "conf", "2": "conf_tourney", "3": "postseason"}

# Ratings columns to drop (ranks / text) — everything else numeric is a feature.
# "Pro Con L" only exists in newer seasons, so we drop it for a uniform schema.
RATINGS_DROP = {
    "rank", "oe Rank", "de Rank", "rank.1", "WAB Rk", "Fun Rk",
    "record", "Con Rec.", "Pro Con L",
}

# In 2008–2021 the tempo column ships under this mis-quoted header (comma in the
# name); it holds plain adjusted tempo, so we rename it to match newer seasons.
RATINGS_RENAME = {"Fun Rk, adjt": "adjt", "team": "TEAM", "conf": "CONF"}


def _download(year, kind):
    """Fetch one Barttorvik CSV as text, with the browser UA and a short retry."""
    url = BASE_URL.format(year=year, kind=kind)
    last_err = None
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=45) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001 — surface after retries
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last_err}")


# ──────────────────────────────────────────────────────────────
# RATINGS  (team_results.csv — has a clean header)
# ──────────────────────────────────────────────────────────────

def parse_team_results(year):
    """Return a tidy ratings frame: YEAR, TEAM, CONF, <numeric features>."""
    text = _download(year, "team_results")
    # index_col=False stops pandas from promoting the first column to an index
    # when older seasons carry a stray trailing field (which would shift every
    # column, mislabeling conference codes as team names).
    df = pd.read_csv(io.StringIO(text), index_col=False)
    df = df.rename(columns=RATINGS_RENAME)
    df.insert(0, "YEAR", year)

    feature_cols = []
    for c in df.columns:
        if c in ("YEAR", "TEAM", "CONF") or c in RATINGS_DROP:
            continue
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().any():
            df[c] = coerced
            feature_cols.append(c)

    df["TEAM"] = df["TEAM"].astype(str).str.strip()
    return df[["YEAR", "TEAM", "CONF"] + feature_cols]


# ──────────────────────────────────────────────────────────────
# GAMES  (super_sked.csv — headerless, parsed by column position)
# ──────────────────────────────────────────────────────────────

def parse_super_sked(year):
    """Return a tidy games frame of completed D1-vs-D1 games for the season."""
    text = _download(year, "super_sked")
    rows = list(csv.reader(io.StringIO(text)))

    out = []
    for r in rows:
        if len(r) < 30 or r[6] == "99":     # too short, or vs non-D1 opponent
            continue
        away, home, winner = r[8].strip(), r[14].strip(), r[29].strip()
        if not away or not home or not winner:   # unplayed / malformed
            continue
        try:
            away_score = int(round(float(r[27])))
            home_score = int(round(float(r[28])))
            game_date = datetime.strptime(r[1].strip(), "%m/%d/%y").date()
        except (ValueError, TypeError):
            continue
        if winner not in (home, away):
            continue

        out.append({
            "YEAR": year,
            "DATE": game_date.isoformat(),
            "AWAY": away,
            "HOME": home,
            "NEUTRAL": r[7].strip() == "1",
            "AWAY_SCORE": away_score,
            "HOME_SCORE": home_score,
            "WINNER": winner,
            "HOME_WIN": int(winner == home),
            "GAME_TYPE": GAME_TYPES.get(r[6], "unknown"),
        })

    games = pd.DataFrame(out)
    if not games.empty:
        # super_sked lists each game once, but guard against any dupes.
        games = games.drop_duplicates(subset=["DATE", "HOME", "AWAY"])
    return games


# ──────────────────────────────────────────────────────────────
# DRIVER
# ──────────────────────────────────────────────────────────────

def build(years):
    os.makedirs(DATA_DIR, exist_ok=True)
    ratings_parts, games_parts = [], []

    failures = []
    for year in years:
        print(f"[{year}] downloading team_results + super_sked…", flush=True)
        try:
            ratings = parse_team_results(year)
            games = parse_super_sked(year)
        except Exception as e:  # noqa: BLE001 — one bad year shouldn't abort the rest
            print(f"[{year}] SKIPPED: {e}", flush=True)
            failures.append(year)
            continue
        ratings_parts.append(ratings)
        games_parts.append(games)
        # Sanity signal: home teams should win ~64% of non-neutral games. A wildly
        # different rate flags a season whose super_sked column layout differs.
        nn = games[~games["NEUTRAL"]] if not games.empty else games
        hw = f"{nn['HOME_WIN'].mean():.3f}" if len(nn) else "n/a"
        played = "" if games.empty else (
            f", {int(games['NEUTRAL'].sum())} neutral, "
            f"{games['GAME_TYPE'].eq('postseason').sum()} postseason, "
            f"home_win(non-neutral)={hw}")
        print(f"[{year}] {len(ratings)} teams, {len(games)} games{played}",
              flush=True)

    if not ratings_parts:
        raise RuntimeError(f"No years fetched successfully (failures: {failures}).")

    ratings_all = pd.concat(ratings_parts, ignore_index=True)
    games_all = pd.concat(games_parts, ignore_index=True)

    # Safety net: drop any feature column not present (non-null) in every season,
    # so the model always sees a uniform, NaN-free feature matrix.
    meta = {"YEAR", "TEAM", "CONF"}
    dropped = [c for c in ratings_all.columns
               if c not in meta and ratings_all[c].isna().any()]
    if dropped:
        ratings_all = ratings_all.drop(columns=dropped)
        print(f"Dropped non-uniform feature columns: {dropped}")

    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    games_path = os.path.join(DATA_DIR, "games.csv")
    ratings_all.to_csv(ratings_path, index=False)
    games_all.to_csv(games_path, index=False)

    n_feat = len([c for c in ratings_all.columns if c not in ("YEAR", "TEAM", "CONF")])
    print(f"\nWrote {ratings_path}: {len(ratings_all)} rows, {n_feat} feature cols")
    print(f"Wrote {games_path}:   {len(games_all)} rows")
    if failures:
        print(f"Years skipped: {failures}")
    return ratings_all, games_all


def parse_years(argv):
    if not argv:
        return [2024]
    nums = [int(a) for a in argv]
    if len(nums) == 2 and nums[0] < nums[1]:
        return list(range(nums[0], nums[1] + 1))
    return nums


if __name__ == "__main__":
    build(parse_years(sys.argv[1:]))

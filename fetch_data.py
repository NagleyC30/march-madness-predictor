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
#   python fetch_data.py --pit 2024     # point-in-time table (data/games_pit.csv)
#   python fetch_data.py --pit 2011 2026
#
# Point-in-time uses barttorvik.com's daily "time machine" snapshots (from ~2011)
# to attach each game the ratings that existed before it — see the section below.

import csv
import gzip
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

DATA_DIR = "data"
BASE_URL = "https://barttorvik.com/{year}_{kind}.csv"
# Daily "time machine" snapshot of the full ratings table as it stood that
# morning (point-in-time). Available from ~2011. YYYYMMDD, gzipped JSON arrays
# in the same column order as {year}_team_results.csv.
TIMEMACHINE_URL = (
    "https://barttorvik.com/timemachine/team_results/{date}_team_results.json.gz")
TIMEMACHINE_CACHE = os.path.join(DATA_DIR, "timemachine")
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

def _tidy_ratings(df, year):
    """Shared cleanup: rename, stamp YEAR, keep TEAM/CONF + numeric features.
    Used by both the season CSV and the point-in-time JSON snapshots."""
    df = df.rename(columns=RATINGS_RENAME)
    if "YEAR" in df.columns:
        df = df.drop(columns=["YEAR"])
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


def _season_header(year):
    """Column names for a season's team_results (used to label snapshot arrays)."""
    text = _download(year, "team_results")
    return pd.read_csv(io.StringIO(text), index_col=False, nrows=0).columns.tolist()


def parse_team_results(year):
    """Return a tidy ratings frame: YEAR, TEAM, CONF, <numeric features>."""
    text = _download(year, "team_results")
    # index_col=False stops pandas from promoting the first column to an index
    # when older seasons carry a stray trailing field (which would shift every
    # column, mislabeling conference codes as team names).
    df = pd.read_csv(io.StringIO(text), index_col=False)
    return _tidy_ratings(df, year)


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
# POINT-IN-TIME RATINGS  (time machine daily snapshots)
# ──────────────────────────────────────────────────────────────
#
# ratings.csv is season-aggregate, so a November game "knows" the team's final
# rating. The time machine gives each team's rating as it stood on any morning,
# letting us attach each game the ratings that existed *before* it was played.
# build_point_in_time_games() writes data/games_pit.csv, an as-of-date version
# of the model matrix (feature diffs + HOME_COURT + HOME_WIN) for honest
# backtesting. Snapshots are cached locally so re-runs are cheap.

def _download_timemachine(date_str):
    """Fetch one snapshot ('YYYYMMDD'). Returns a list of team records, or None
    if that date has no snapshot (404). Caches the gzip locally."""
    os.makedirs(TIMEMACHINE_CACHE, exist_ok=True)
    cache_path = os.path.join(TIMEMACHINE_CACHE, f"{date_str}.json.gz")
    if os.path.exists(cache_path):
        try:
            with gzip.open(cache_path) as f:
                return json.load(f)
        except Exception:            # corrupt cache — refetch
            os.remove(cache_path)

    url = TIMEMACHINE_URL.format(date=date_str)
    last_err = None
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=45) as resp:
                raw = resp.read()
            data = json.loads(gzip.decompress(raw))   # validate before caching
            with open(cache_path, "wb") as f:
                f.write(raw)
            return data
        except HTTPError as e:
            if e.code == 404:
                return None
            last_err = e
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:  # noqa: BLE001 — transient network/gzip; retry
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    # A single corrupt/unreachable snapshot shouldn't abort a multi-thousand-file
    # pull — treat it as missing so the nearest-snapshot walk-back handles it.
    print(f"  ! skipping {date_str}: {last_err}", flush=True)
    return None


def _snapshot_ratings(date_str, year, header):
    """Tidy ratings frame (indexed by TEAM) as of date_str, or None if missing.
    ``header`` labels the snapshot's positional arrays (extra trailing field, if
    any, is dropped by zip)."""
    records = _download_timemachine(date_str)
    if records is None:
        return None
    df = pd.DataFrame([dict(zip(header, r)) for r in records])
    if "team" not in df.columns and "TEAM" not in df.columns:
        return None
    tidy = _tidy_ratings(df, year)
    return tidy.set_index("TEAM")


def _nearest_snapshot(date_obj, year, header, cache, max_back=7):
    """Ratings as of date_obj, walking back up to max_back days to the nearest
    available snapshot. Results (incl. misses) are memoised in ``cache``."""
    for back in range(max_back + 1):
        key = (date_obj - timedelta(days=back)).strftime("%Y%m%d")
        if key not in cache:
            cache[key] = _snapshot_ratings(key, year, header)
        if cache[key] is not None:
            return cache[key]
    return None


def build_point_in_time_games(years, out_path=None):
    """Attach each game the ratings that existed before it, writing
    data/games_pit.csv (feature diffs + HOME_COURT + HOME_WIN). Needs
    data/games.csv and data/ratings.csv (run the main fetch first)."""
    out_path = out_path or os.path.join(DATA_DIR, "games_pit.csv")
    games_all = pd.read_csv(os.path.join(DATA_DIR, "games.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    features = [c for c in ratings.columns if c not in ("YEAR", "TEAM", "CONF")]
    diff_cols = [f"{c}_DIFF" for c in features]

    parts = []
    for year in years:
        gy = games_all[games_all["YEAR"] == year]
        if gy.empty:
            print(f"[{year}] no games in games.csv — skipping", flush=True)
            continue
        header = _season_header(year)
        cache = {}
        rows, unmatched = [], 0
        for g in gy.itertuples(index=False):
            date_obj = datetime.strptime(g.DATE, "%Y-%m-%d").date()
            # Barttorvik's YYYYMMDD snapshot already includes that day's games, so
            # we anchor on the DAY BEFORE — ratings reflecting results only up to
            # (not including) the game we're about to predict.
            snap = _nearest_snapshot(date_obj - timedelta(days=1), year, header, cache)
            if snap is None or g.HOME not in snap.index or g.AWAY not in snap.index:
                unmatched += 1
                continue
            h, a = snap.loc[g.HOME], snap.loc[g.AWAY]
            row = {"YEAR": year, "DATE": g.DATE, "HOME": g.HOME, "AWAY": g.AWAY,
                   "GAME_TYPE": g.GAME_TYPE, "NEUTRAL": g.NEUTRAL}
            for c, dc in zip(features, diff_cols):
                row[dc] = float(h[c]) - float(a[c])
            row["HOME_COURT"] = 0 if g.NEUTRAL else 1
            row["HOME_WIN"] = int(g.HOME_WIN)
            rows.append(row)
        parts.append(pd.DataFrame(rows))
        n_snaps = sum(1 for v in cache.values() if v is not None)
        print(f"[{year}] {len(rows):,} PIT games, {unmatched} unmatched, "
              f"{n_snaps} snapshots", flush=True)

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(out):,} rows, {len(diff_cols)} feature diffs")
    return out


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
    args = sys.argv[1:]
    if args and args[0] == "--pit":
        # Point-in-time table (needs data/games.csv + data/ratings.csv first).
        build_point_in_time_games(parse_years(args[1:]))
    else:
        build(parse_years(args))

# fetch_cbb_lines.py — Fetch REAL sportsbook lines from CollegeBasketballData.com
# (CBBD), the free basketball sibling of CollegeFootballData. Replaces the noisy
# SBR-parsed spreads (fetch_odds.py) with actual book numbers and extends coverage
# past 2021 — and pulls TOTALS too, for the planned over/under model.
#
# Why: the margin model showed our archived SBR spreads correlate with actual
# margins at only ~0.53 (a real close is ~0.65), i.e. a noisy proxy. CBBD gives
# the actual posted spread, moneyline and over/under per game, current + historical.
#
# Auth: a FREE API key (Bearer token) from https://collegebasketballdata.com/key.
# The key NEVER lands in the repo — set it in the CBBD_API_KEY env var or a
# gitignored `.cbbd_key` file. Like the other fetchers this runs locally and
# writes a committed CSV, so the Streamlit app never needs the key.
#
# Output:
#   data/lines_cbbd.csv — one row per game, a SUPERSET of odds.csv's schema so it
#   can drop into backtest_odds.py, plus real spread prices and a total:
#       YEAR, DATE, AWAY, HOME, NEUTRAL, ML_AWAY, ML_HOME,
#       SPREAD_AWAY, SPREAD_HOME, SPREAD_PRICE_AWAY, SPREAD_PRICE_HOME,
#       TOTAL, TOTAL_PRICE_OVER, TOTAL_PRICE_UNDER,
#       AWAY_SCORE, HOME_SCORE, PROVIDER, AWAY_KEY, HOME_KEY
#   (YEAR = CBBD `season` = the year the season ends, matching odds.csv's tourney
#   year. *_KEY reuse fetch_odds.team_key so the file joins to the model's teams.)
#
# Usage:
#   python fetch_cbb_lines.py                 # default range (2022..CURRENT_SEASON)
#   python fetch_cbb_lines.py 2024            # one season (2023-24)
#   python fetch_cbb_lines.py 2022 2026       # inclusive range
#   python fetch_cbb_lines.py --probe 2024    # dump the raw JSON shape of one
#                                             # game+line, to confirm field names

import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from fetch_odds import team_key            # reuse the team-name reconciler

DATA_DIR = "data"
OUT_PATH = os.path.join(DATA_DIR, "lines_cbbd.csv")
API_URL = "https://api.collegebasketballdata.com/lines"
CURRENT_SEASON = 2026                       # the in-progress 2025-26 season
DEFAULT_START = 2022                         # first year past the SBR archive (2021)
# CBBD sits behind Cloudflare, which 403s the default urllib User-Agent
# (Error 1010). A normal browser UA passes the bot check, same as fetch_odds.py.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# When a game has lines from several books, pick ONE representative line in this
# order of preference; fall back to the first provider present.
PROVIDER_PREF = ["consensus", "DraftKings", "Bovada", "ESPN Bet",
                 "William Hill (US)", "Caesars"]


# ──────────────────────────────────────────────────────────────
# AUTH
# ──────────────────────────────────────────────────────────────

def get_key():
    """CBBD API key from the CBBD_API_KEY env var or a gitignored `.cbbd_key`
    file (repo root or home). Never hard-code or commit it."""
    key = os.environ.get("CBBD_API_KEY")
    if key and key.strip():
        return key.strip()
    for p in (".cbbd_key", os.path.join(os.path.expanduser("~"), ".cbbd_key")):
        if os.path.exists(p):
            with open(p) as f:
                k = f.read().strip()
            if k:
                return k
    raise SystemExit(
        "No CBBD API key found. Get a free one at "
        "https://collegebasketballdata.com/key, then either:\n"
        "  set CBBD_API_KEY=<key>            (environment variable), or\n"
        "  echo <key> > .cbbd_key            (gitignored file in the repo root)")


# ──────────────────────────────────────────────────────────────
# FETCH
# ──────────────────────────────────────────────────────────────

def fetch_season(year, key):
    """GET /lines?season=YEAR -> parsed JSON list of games. Retries transient
    errors; surfaces a clear message on 401 (bad/missing key) or 429 (quota)."""
    url = f"{API_URL}?season={year}"
    req = Request(url, headers={"Authorization": f"Bearer {key}",
                                "Accept": "application/json",
                                "User-Agent": USER_AGENT})
    last = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except HTTPError as e:               # noqa: PERF203
            if e.code == 401:
                raise SystemExit("CBBD returned 401 Unauthorized — the API key is "
                                 "missing or invalid.")
            if e.code == 429:
                raise SystemExit("CBBD returned 429 — monthly request quota "
                                 "exceeded. Try again next cycle or upgrade the tier.")
            last = e
        except URLError as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last}")


# ──────────────────────────────────────────────────────────────
# PARSE  (defensive field access — CBBD mirrors the CFBD schema, but confirm the
# exact key names with `--probe` on the first authenticated run and adjust here.)
# ──────────────────────────────────────────────────────────────

def _g(d, *names, default=None):
    """First present, non-null value among several candidate key spellings."""
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def pick_line(lines):
    """Choose one representative book line for a game."""
    if not lines:
        return None
    by_prov = {}
    for ln in lines:
        by_prov.setdefault(_g(ln, "provider", "sportsbook", default="?"), ln)
    for p in PROVIDER_PREF:
        if p in by_prov:
            return by_prov[p]
    return lines[0]


def _iso_date(v):
    """ISO datetime/string -> 'YYYY-MM-DD' (or '' if unparseable)."""
    if not v:
        return ""
    return str(v)[:10]


def to_row(game, year):
    """One CBBD game -> our odds.csv-superset row, or None if unusable."""
    line = pick_line(_g(game, "lines", default=[]) or [])
    if line is None:
        return None
    home = _g(game, "homeTeam", "home_team", "home")
    away = _g(game, "awayTeam", "away_team", "away")
    if not home or not away:
        return None
    # CBBD/CFBD spread is from the HOME perspective (negative => home favored).
    sp_home = _g(line, "spread")
    sp_home = None if sp_home is None else float(sp_home)
    sp_away = None if sp_home is None else -sp_home
    total = _g(line, "overUnder", "over_under", "total")
    ml_home = _g(line, "homeMoneyline", "moneylineHome", "home_moneyline")
    ml_away = _g(line, "awayMoneyline", "moneylineAway", "away_moneyline")
    hs = _g(game, "homeScore", "home_score", "homePoints")
    as_ = _g(game, "awayScore", "away_score", "awayPoints")
    return {
        "YEAR": int(_g(game, "season", default=year)),
        "DATE": _iso_date(_g(game, "startDate", "start_date", "startDateTime")),
        "AWAY": away, "HOME": home,
        "NEUTRAL": bool(_g(game, "neutralSite", "neutral_site", default=False)),
        "ML_AWAY": None if ml_away is None else int(ml_away),
        "ML_HOME": None if ml_home is None else int(ml_home),
        "SPREAD_AWAY": sp_away, "SPREAD_HOME": sp_home,
        "SPREAD_PRICE_AWAY": _g(line, "awaySpreadPrice", "spreadPriceAway"),
        "SPREAD_PRICE_HOME": _g(line, "homeSpreadPrice", "spreadPriceHome"),
        "TOTAL": None if total is None else float(total),
        "TOTAL_PRICE_OVER": _g(line, "overPrice", "overUnderOverPrice"),
        "TOTAL_PRICE_UNDER": _g(line, "underPrice", "overUnderUnderPrice"),
        "AWAY_SCORE": None if as_ is None else int(as_),
        "HOME_SCORE": None if hs is None else int(hs),
        "PROVIDER": _g(line, "provider", "sportsbook", default=""),
        "AWAY_KEY": team_key(away), "HOME_KEY": team_key(home),
    }


def build(years, key):
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []
    for y in years:
        games = fetch_season(y, key)
        n0 = len(rows)
        for g in games:
            r = to_row(g, y)
            if r is not None and (r["SPREAD_HOME"] is not None
                                  or r["ML_HOME"] is not None
                                  or r["TOTAL"] is not None):
                rows.append(r)
        print(f"[{y}] {len(games):,} games returned, "
              f"{len(rows) - n0:,} with a usable line", flush=True)
        time.sleep(0.5)                       # be polite to the free API
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}: {len(df):,} games, "
          f"{df['YEAR'].min()}-{df['YEAR'].max()}"
          if len(df) else f"\nWrote {OUT_PATH}: 0 games (check the API/key)")
    if len(df):
        with_spread = df["SPREAD_HOME"].notna().sum()
        with_total = df["TOTAL"].notna().sum()
        print(f"  {with_spread:,} have a spread, {with_total:,} have a total. "
              f"Providers: {', '.join(sorted(df['PROVIDER'].dropna().unique())[:6])}")
    return df


# ──────────────────────────────────────────────────────────────
# PROBE — confirm the real JSON field names on the first keyed run
# ──────────────────────────────────────────────────────────────

def probe(year, key):
    games = fetch_season(year, key)
    print(f"{len(games)} games returned for {year}.")
    if not games:
        return
    g = next((x for x in games if _g(x, "lines")), games[0])
    print("\nGame object keys:\n ", sorted(g.keys()))
    lines = _g(g, "lines", default=[]) or []
    print(f"\nFirst game has {len(lines)} line(s).")
    if lines:
        print("Line object keys:\n ", sorted(lines[0].keys()))
    print("\nSample game (trimmed):")
    trimmed = {k: g[k] for k in list(g.keys()) if k != "lines"}
    trimmed["lines"] = lines[:1]
    print(json.dumps(trimmed, indent=2, default=str)[:1600])


# ──────────────────────────────────────────────────────────────

def parse_years(argv):
    if not argv:
        return list(range(DEFAULT_START, CURRENT_SEASON + 1))
    nums = [int(a) for a in argv]
    if len(nums) == 2 and nums[0] < nums[1]:
        return list(range(nums[0], nums[1] + 1))
    return nums


def main(argv):
    key = get_key()
    if argv and argv[0] == "--probe":
        year = int(argv[1]) if len(argv) > 1 else CURRENT_SEASON
        probe(year, key)
        return
    build(parse_years(argv), key)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                         # noqa: BLE001
        pass
    main(sys.argv[1:])

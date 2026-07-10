# fetch_odds.py — Download & parse REAL historical sportsbook moneylines for the
# betting backtest (project checklist item 3).
#
# Source: Sportsbook Reviews Online's free NCAA basketball odds archive, one xlsx
# per season. Each game is two rows (visitor then home); we fold them into one
# row per game and keep the closing moneyline (the "ML" column) for both sides.
#
#   https://www.sportsbookreviewsonline.com/.../ncaa-basketball-YYYY-YY.xlsx
#
# The archive currently covers seasons 2007-08 .. 2020-21, i.e. tournaments
# 2008-2019 and 2021 (2020 had no tournament; 2022+ are not published there).
#
# Output (committed, consumed by backtest_odds.py):
#   data/odds.csv  — one row per game:
#       YEAR, DATE, AWAY, HOME, NEUTRAL, ML_AWAY, ML_HOME,
#       SPREAD_AWAY, SPREAD_HOME, AWAY_SCORE, HOME_SCORE, AWAY_KEY, HOME_KEY
#   where *_KEY is a canonical team key that joins to the model's team names
#   (see team_key()), the signed SPREAD_* let spread bets be settled (favorite
#   negative), and the scores give the actual margin. Rows with a missing/
#   unusable moneyline are dropped; spread/score fields may be blank if that
#   game had no closing spread or score in the archive.
#
# Usage:  python fetch_odds.py            # all available tournament years
#         python fetch_odds.py 2019       # one tournament year
#         python fetch_odds.py 2011 2019  # inclusive range

import os
import re
import sys
import time
from urllib.request import Request, urlopen

import pandas as pd

DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "odds_cache")
BASE_URL = ("https://www.sportsbookreviewsonline.com/wp-content/uploads/"
            "sportsbookreviewsonline_com_737/ncaa-basketball-{label}.xlsx")
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Tournaments with published odds (2020 skipped — no tournament).
AVAILABLE_YEARS = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
                   2018, 2019, 2021]


def season_label(tour_year):
    """2019 -> '2018-19' (the season that ends in that tournament)."""
    return f"{tour_year - 1}-{str(tour_year)[-2:]}"


# ──────────────────────────────────────────────────────────────
# TEAM NAME CANONICALIZATION
# ──────────────────────────────────────────────────────────────
#
# SBR and the model (Barttorvik) spell teams differently and SBR's own spelling
# drifts year to year ("Indiana" vs "IndianaU", "NorfolkSt" vs "NorfolkState").
# team_key() collapses both sides to one canonical string. Most teams fall out
# of a deterministic normalizer (punctuation-strip + trailing "St"->"State");
# the irregular ones are pinned by ALIAS below (verified against the archive).

# canonical model TEAM name -> set of SBR spellings (as produced by _base()).
ALIAS = {
    'VCU': {'vacommonwealth', 'virginiacommonwealth', 'vcu'},
    'UCF': {'centralflorida', 'ucf'},
    'UMBC': {'mdbaltimoreco', 'marylandbc', 'umbc'},
    'UTSA': {'texsanantonio', 'texassanantonio', 'utsa'},
    'Miami FL': {'miamiflorida', 'miamifl'},
    'Miami OH': {'miamiohio', 'miamioh'},
    'Green Bay': {'wiscgreenbay', 'greenbay', 'wisconsingreenbay'},
    'Milwaukee': {'wiscmilwaukee', 'milwaukee', 'wisconsinmilwaukee'},
    'Little Rock': {'arkansaslr', 'arkansaslittlerock', 'littlerock'},
    'Cal Poly': {'calpolyslo', 'calpoly'},
    'Loyola MD': {'loyolamaryland', 'loyolamd'},
    'Loyola Chicago': {'loyolachicago', 'loyolaill', 'loyolaillinois'},
    'Louisiana Lafayette': {'ullafayette', 'louisianalafayette', 'louisiana'},
    'Penn': {'pennsylvania', 'penn'},
    'Albany': {'albanyny', 'albany'},
    'Boston University': {'bostonu', 'bostonuniversity'},
    'Arkansas Pine Bluff': {'arkpinebluff', 'arkansaspinebluff'},
    'College of Charleston': {'collofcharleston', 'collcharleston',
                              'collegeofcharleston', 'charleston'},
    'East Tennessee St.': {'etennesseest', 'etennst', 'easttennst',
                           'easttennesseestate'},
    'Middle Tennessee': {'middletennst', 'middletennessee', 'middletenn'},
    'Cal St. Bakersfield': {'csbakersfield', 'calstatebakersfield', 'bakersfield'},
    'Cal St. Fullerton': {'csfullerton', 'fullertonst', 'calstatefullerton'},
    'Cal St. Northridge': {'csnorthridge', 'calstatenorthridge', 'northridge'},
    'UC Irvine': {'calirvine', 'ucirvine'},
    'UC Santa Barbara': {'calsantabarbara', 'calsantabarb', 'ucsantabarbara', 'ucsb'},
    'UC Davis': {'ucdavis', 'caldavis'},
    'UNC Asheville': {'ncasheville', 'uncasheville'},
    'UNC Greensboro': {'ncgreensboro', 'uncgreensboro'},
    'UNC Wilmington': {'ncwilmington', 'uncwilmington'},
    'UT Arlington': {'texasarlington', 'utarlington'},
    'Stephen F. Austin': {'stephenaustin', 'stephenfaustin'},
    'Florida Gulf Coast': {'flagulfcoast', 'floridagulfcoast'},
    "Mount St. Mary's": {'mtstmarys', 'mountstmarys'},
    "Saint Mary's": {'saintmarysca', 'stmarysca', 'saintmarys', 'stmarys'},
    "Saint Joseph's": {'stjosephs', 'saintjosephs', 'stjosephspa'},
    "Saint Peter's": {'stpeters', 'saintpeters'},
    'North Dakota St.': {'ndakotast', 'northdakotast', 'northdakotastate'},
    'South Dakota St.': {'sdakotast', 'southdakotast', 'southdakotastate'},
    'North Carolina St.': {'ncstate', 'northcarolinast', 'northcarolinastate'},
    'North Carolina Central': {'nccentral', 'northcarolinacentral'},
    'North Carolina A&T': {'ncarolinaandt', 'ncat', 'northcarolinaat',
                           'northcarolinaandt', 'ncarolinaaandt',
                           'northcarolinaaandt'},
    'Norfolk St.': {'norfolkst', 'norfolkstate'},
    'Northwestern St.': {'northwesternst', 'northwesternstate'},
    'Mississippi Valley St.': {'missvalleyst', 'mississippivalleyst',
                               'mississippivalleystate', 'mvsu'},
    'Chattanooga': {'tennesseechat', 'chattanooga', 'tennchattanooga'},
    'Detroit': {'detroit', 'detroitu', 'detroitmercy'},
    'Eastern Washington': {'easternwashington', 'ewashington'},
    'Northern Colorado': {'nocolorado', 'northerncolorado', 'nocolo'},
    'George Washington': {'geowashington', 'georgewashington'},
    'LIU Brooklyn': {'liubrooklyn', 'liu', 'longisland'},
    'Fairleigh Dickinson': {'fairleighdickinson', 'fairdickinson', 'fdu'},
    'American': {'american', 'americanu'},
    'Utah': {'utah', 'utahu'},
    'Sam Houston St.': {'samhouston', 'samhoustonst', 'samhoustonstate'},
}

# Reverse index: SBR base spelling -> canonical model key (base of the model name).
_SBR_TO_CANON = {}


def _base(s):
    """Punctuation-insensitive lowercase key. Collapses the spelling drift that
    separates SBR from Barttorvik: 'St'->'State' (both the standalone word and a
    concatenated 'NorfolkSt' suffix) and a trailing 'University' 'U' that SBR
    tacks on some seasons ('MemphisU', 'IndianaU'). The length guard keeps real
    acronyms (VCU, LSU, BYU, SMU, TCU, LIU) intact."""
    s = str(s).lower().replace('&', ' and ').replace('.', ' ')
    s = re.sub(r'\bst\b', 'state', s)          # standalone 'St' word -> State
    s = re.sub(r'[^a-z0-9]', '', s)
    if s.endswith('st'):                         # concatenated 'NorfolkSt' etc.
        s += 'ate'
    if s.endswith('u') and len(s) >= 5:          # 'MemphisU' -> 'memphis'
        s = s[:-1]
    return s


for _canon_team, _forms in ALIAS.items():
    _canon = _base(_canon_team)
    for _f in _forms:
        _SBR_TO_CANON[_f] = _canon
        _SBR_TO_CANON[_base(_f)] = _canon


def team_key(name):
    """Canonical join key for a team name from either source."""
    b = _base(name)
    if b in _SBR_TO_CANON:
        return _SBR_TO_CANON[b]
    raw = re.sub(r'[^a-z0-9]', '', str(name).lower().replace('&', 'and'))
    return _SBR_TO_CANON.get(raw, b)


# ──────────────────────────────────────────────────────────────
# DOWNLOAD & PARSE
# ──────────────────────────────────────────────────────────────

def _download(tour_year):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"ncaab-{season_label(tour_year)}.xlsx")
    if os.path.exists(path):
        return path
    url = BASE_URL.format(label=season_label(tour_year))
    last_err = None
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=60) as resp:
                data = resp.read()
            with open(path, "wb") as f:
                f.write(data)
            return path
        except Exception as e:  # noqa: BLE001 — surface after retries
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def _clean_ml(v):
    """Parse a moneyline cell to int, or None. Blanks/'NL'/'pk' -> None; a lone
    0 (SBR's 'no line') -> None."""
    if pd.isna(v):
        return None
    if isinstance(v, str):
        v = v.strip().replace('+', '')
        if v == '' or not re.fullmatch(r'-?\d+', v):
            return None
        v = int(v)
    v = int(v)
    return None if v == 0 else v


def _iso_date(mmdd, tour_year):
    """MMDD int + tournament year -> 'YYYY-MM-DD'. Nov/Dec belong to the prior
    calendar year; Jan-Apr to the tournament year."""
    mmdd = int(mmdd)
    month, day = mmdd // 100, mmdd % 100
    cal_year = tour_year - 1 if month >= 8 else tour_year
    return f"{cal_year:04d}-{month:02d}-{day:02d}"


def _num(v):
    """Parse a numeric line cell to float, or None. 'pk'/'pick' (pick'em) -> 0."""
    if pd.isna(v):
        return None
    if isinstance(v, str):
        v = v.strip().lower()
        if v in ("pk", "pick", "p"):
            return 0.0
        try:
            return float(v)
        except ValueError:
            return None
    return float(v)


def _spreads(close_a, close_b, ml_away, ml_home):
    """Split the two Close cells into (spread_away, spread_home).

    SBR interleaves the point spread and the game total across a game's two rows;
    the spread is the smaller magnitude, the total the larger. The favorite (more
    negative moneyline) lays the spread, so its spread is negative and the
    underdog's is the mirror. Returns (None, None) if either cell is unusable."""
    ca, cb = _num(close_a), _num(close_b)
    if ca is None or cb is None:
        return None, None
    mag = min(abs(ca), abs(cb))               # spread magnitude (< total)
    if ml_away <= ml_home:                     # away is the favorite
        return -mag, mag
    return mag, -mag


def parse_season(tour_year):
    df = pd.read_excel(_download(tour_year))
    recs = df.to_dict('records')
    out = []
    for i in range(0, len(recs) - 1, 2):
        a, h = recs[i], recs[i + 1]
        # Rows must pair as visitor/neutral then home/neutral.
        vh_a, vh_h = str(a.get('VH', '')).strip(), str(h.get('VH', '')).strip()
        if vh_a not in ('V', 'N') or vh_h not in ('H', 'N'):
            continue
        ml_away, ml_home = _clean_ml(a.get('ML')), _clean_ml(h.get('ML'))
        if ml_away is None or ml_home is None:
            continue
        try:
            date = _iso_date(a['Date'], tour_year)
        except (ValueError, TypeError):
            continue
        away, home = str(a['Team']).strip(), str(h['Team']).strip()
        spread_away, spread_home = _spreads(
            a.get('Close'), h.get('Close'), ml_away, ml_home)
        score_away, score_home = _num(a.get('Final')), _num(h.get('Final'))
        out.append({
            "YEAR": tour_year,
            "DATE": date,
            "AWAY": away,
            "HOME": home,
            "NEUTRAL": vh_a == 'N',
            "ML_AWAY": ml_away,
            "ML_HOME": ml_home,
            "SPREAD_AWAY": spread_away,
            "SPREAD_HOME": spread_home,
            "AWAY_SCORE": None if score_away is None else int(score_away),
            "HOME_SCORE": None if score_home is None else int(score_home),
            "AWAY_KEY": team_key(away),
            "HOME_KEY": team_key(home),
        })
    return pd.DataFrame(out)


def build(years):
    os.makedirs(DATA_DIR, exist_ok=True)
    parts = []
    for y in years:
        g = parse_season(y)
        print(f"[{y}] {len(g):,} games with moneylines "
              f"(season {season_label(y)})", flush=True)
        parts.append(g)
    odds = pd.concat(parts, ignore_index=True)
    out_path = os.path.join(DATA_DIR, "odds.csv")
    odds.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(odds):,} games, "
          f"{odds['YEAR'].min()}-{odds['YEAR'].max()}")
    return odds


def parse_years(argv):
    if not argv:
        return AVAILABLE_YEARS
    nums = [int(a) for a in argv]
    if len(nums) == 2 and nums[0] < nums[1]:
        rng = list(range(nums[0], nums[1] + 1))
        return [y for y in rng if y in AVAILABLE_YEARS]
    return [y for y in nums if y in AVAILABLE_YEARS]


if __name__ == "__main__":
    build(parse_years(sys.argv[1:]))

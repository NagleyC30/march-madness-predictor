# update_current_season.py — Pull the in-progress season's Barttorvik ratings
# into data/ratings.csv so the Game Predictor can forecast the upcoming year's
# games (project checklist item 5).
#
# The general Game Predictor is trained on completed seasons (2008-2026). This
# adds the CURRENT season (2026-27 = year 2027) using Barttorvik's live/preseason
# T-Rank ratings, which refresh as games are played — re-run this any time to
# pull the latest numbers. The trained model doesn't change (the new season has
# no completed games to learn from); it simply gains a set of ratings to predict
# from, so head-to-head predictions for the new season work immediately.
#
# Barttorvik publishes preseason ratings months before tip-off, but the schedule
# (super_sked) only appears closer to the season. When it does, scheduled-game
# prediction can be layered on; until then this enables any 2026-27 matchup.
#
# Usage:
#   python update_current_season.py          # auto-detect next season, refresh
#   python update_current_season.py 2027     # a specific season
#   python update_current_season.py --remove 2027   # drop it again

import os
import sys

import pandas as pd

import fetch_data as fd

DATA_DIR = "data"
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")
META = ("YEAR", "TEAM", "CONF")


def _base_features(ratings):
    return [c for c in ratings.columns if c not in META]


def _write(ratings):
    ratings.sort_values(["YEAR", "TEAM"]).to_csv(RATINGS_FILE, index=False)


def remove_season(year):
    ratings = pd.read_csv(RATINGS_FILE)
    before = len(ratings)
    ratings = ratings[ratings["YEAR"] != year]
    _write(ratings)
    print(f"Removed {before - len(ratings)} rows for {year}. "
          f"ratings.csv now {sorted(ratings['YEAR'].unique())}")


def schedule_available(year):
    """Whether Barttorvik has published this season's schedule yet."""
    try:
        fd.parse_super_sked(year)
        return True
    except Exception:
        return False


def update(year=None):
    ratings = pd.read_csv(RATINGS_FILE)
    base = _base_features(ratings)
    completed_max = int(ratings["YEAR"].max())
    if year is None:
        year = completed_max + 1

    print(f"Fetching Barttorvik ratings for {year} "
          f"(season {year - 1}-{str(year)[-2:]})…", flush=True)
    try:
        new = fd.parse_team_results(year)
    except Exception as e:  # noqa: BLE001 — surface a clean message
        print(f"Could not fetch {year} ratings: {e}")
        print("Barttorvik may not have published this season yet — try again later.")
        return

    # Align to the committed schema: every model feature must be present; keep
    # exactly those columns so ratings.csv stays uniform for the pickled model.
    missing = [c for c in base if c not in new.columns]
    if missing:
        print(f"Schema drift — {year} is missing model features {missing}. "
              "Not updating ratings.csv (investigate before forcing).")
        return
    new = new[list(META) + base]
    n_null = int(new[base].isna().sum().sum())
    if n_null:
        print(f"Warning: {n_null} null feature values in {year} ratings.")

    # Refresh: drop any prior copy of this season, then append the fresh pull.
    ratings = pd.concat([ratings[ratings["YEAR"] != year], new], ignore_index=True)
    _write(ratings)

    has_sked = schedule_available(year)
    print(f"\nMerged {len(new)} teams for {year}. "
          f"ratings.csv now covers {sorted(ratings['YEAR'].unique())}.")
    print(f"Schedule (super_sked) published for {year}? "
          f"{'yes — scheduled-game prediction is possible' if has_sked else 'not yet'}.")
    print("\nThe Game Predictor page will now offer this season. Re-run any time "
          "to pull refreshed ratings as the season progresses.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--remove":
        remove_season(int(args[1]))
    else:
        update(int(args[0]) if args else None)

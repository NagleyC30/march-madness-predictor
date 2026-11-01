# game_model.py — Core for the general (any-matchup) game predictor.
#
# Unlike mm_model.py (tournament-only, neutral-site, seed-framed), this predicts
# ANY college-basketball game. Each game is encoded as the HOME-minus-AWAY
# difference across Barttorvik team-season ratings, plus a HOME_COURT indicator
# (+1 home / 0 neutral), and the model predicts P(home team wins).
#
# Data comes from fetch_data.py:  data/ratings.csv  and  data/games.csv.

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")
GAMES_FILE = os.path.join(DATA_DIR, "games.csv")
PIT_FILE = os.path.join(DATA_DIR, "games_pit.csv")
# Point-in-time training starts here: earlier seasons only have time-machine
# snapshots from mid-season on (partial, late-biased coverage).
PIT_MIN_YEAR = 2015
MODEL_FILE = os.path.join(DATA_DIR, "game_model.joblib")

META_COLS = {"YEAR", "TEAM", "CONF"}
HOME_COURT = "HOME_COURT"   # extra model feature, not a rating difference
TARGET = "HOME_WIN"


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────

def load_ratings(path=RATINGS_FILE):
    return pd.read_csv(path)


def load_games(path=GAMES_FILE):
    return pd.read_csv(path)


def get_features(ratings):
    """Numeric rating columns (everything that isn't metadata)."""
    return [c for c in ratings.columns if c not in META_COLS]


# ──────────────────────────────────────────────────────────────
# DATASET CONSTRUCTION
# ──────────────────────────────────────────────────────────────

def build_game_dataset(games, ratings, features, keep_meta=False):
    """Turn games + ratings into a model matrix.

    Each row: {feature}_DIFF = home_rating - away_rating for every feature, plus
    HOME_COURT (0 on neutral courts, else 1) and the HOME_WIN label. Games where
    either team lacks a rating row are dropped by the inner joins.

    keep_meta=True also carries DATE/HOME/AWAY/GAME_TYPE/NEUTRAL through for the
    backtest to group by. Training ignores them (it selects feature_columns).
    """
    feat = ratings[["YEAR", "TEAM"] + features]

    m = games.merge(feat.rename(columns={"TEAM": "HOME"}),
                    on=["YEAR", "HOME"], how="inner")
    m = m.merge(feat.rename(columns={"TEAM": "AWAY"}),
                on=["YEAR", "AWAY"], how="inner", suffixes=("_H", "_A"))

    out = pd.DataFrame({"YEAR": m["YEAR"].values})
    if keep_meta:
        for col in ("DATE", "HOME", "AWAY", "GAME_TYPE", "NEUTRAL"):
            if col in m.columns:
                out[col] = m[col].values
    for c in features:
        out[f"{c}_DIFF"] = (m[f"{c}_H"] - m[f"{c}_A"]).values
    out[HOME_COURT] = np.where(m["NEUTRAL"].values, 0.0, 1.0)
    out[TARGET] = m[TARGET].astype(int).values
    return out


def feature_columns(features):
    """Ordered model columns: every rating diff, then the home-court flag."""
    return [f"{c}_DIFF" for c in features] + [HOME_COURT]


# ──────────────────────────────────────────────────────────────
# MODEL TRAINING
# ──────────────────────────────────────────────────────────────

# Monotonic constraints: a better offense, defense, BARTHAG or WAB can never
# make a team LESS likely to win, and home court only helps. Without these the
# boosted trees produce noisy, sometimes backwards calls in sparse regions (elite
# vs elite early in the season). Walk-forward 2022-2026 they improve Brier
# 0.1827 -> 0.1818 and halve disagreement with Barttorvik's own log5 on top-25
# preseason matchups (16% -> 8%).
MONO_INCREASING = {"adjoe", "barthag", "WAB", "Qual Barthag", "proj. W", "Pro Con W"}
MONO_DECREASING = {"adjde", "Proj. L"}


def monotonic_constraints(features):
    """+1/-1/0 per model column (rating diffs, then HOME_COURT=+1)."""
    return [1 if c in MONO_INCREASING else -1 if c in MONO_DECREASING else 0
            for c in features] + [1]


def build_pipeline(random_state=0, features=None):
    """StandardScaler + HistGradientBoosting. On the ~100k-game log it matches
    or beats a RandomForest on accuracy while pickling to <1 MB (a full RF is
    tens of MB), which keeps the committed artifact small enough for GitHub.

    Passing ``features`` turns on the monotonic constraints above (StandardScaler
    is a positive rescale, so directions survive it)."""
    cst = monotonic_constraints(features) if features is not None else None
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(random_state=random_state,
                                               monotonic_cst=cst)),
    ])


def train_model(df_train, features, cv=3, do_cv=True, random_state=0):
    """Fit the pipeline on a game-dataset frame. Returns (model, cv_accuracy).

    cv_accuracy is a mean k-fold accuracy for reporting (None if do_cv=False).
    """
    cols = feature_columns(features)
    X = df_train[cols].values
    y = df_train[TARGET].values

    pipe = build_pipeline(random_state=random_state, features=features)
    cv_acc = None
    if do_cv:
        cv_acc = float(cross_val_score(pipe, X, y, cv=cv, n_jobs=-1).mean())
    pipe.fit(X, y)
    return pipe, cv_acc


# ──────────────────────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────────────────────

def predict_game(home_team, away_team, neutral, ratings_year, model, features):
    """Predict P(home team wins) for a single matchup in one season.

    ``ratings_year`` is the ratings frame filtered to the relevant YEAR. On a
    neutral court, pass neutral=True (HOME_COURT=0); the ``home_team`` label is
    then just an orientation and carries no venue advantage.
    Returns a float probability, or None if a team has no ratings row.
    """
    hr = ratings_year[ratings_year["TEAM"] == home_team]
    ar = ratings_year[ratings_year["TEAM"] == away_team]
    if hr.empty or ar.empty:
        return None

    h, a = hr.iloc[0], ar.iloc[0]
    diff = [float(h[c]) - float(a[c]) for c in features]
    x = np.array([diff + [0.0 if neutral else 1.0]])

    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)
    idx = classes.index(1) if 1 in classes else len(classes) - 1
    return float(proba[idx])


def win_probabilities(team_a, team_b, location, ratings_year, model, features):
    """Convenience wrapper for the UI. ``location`` is 'A' (team A home), 'B'
    (team B home) or 'N' (neutral). Returns (p_a, p_b) or None if unrated.

    Orienting by the actual home team keeps HOME_COURT attached to the right
    side; on a neutral court team A is oriented as home with HOME_COURT=0.
    """
    if location == "B":
        p_home = predict_game(team_b, team_a, False, ratings_year, model, features)
        p_a = None if p_home is None else 1.0 - p_home
    else:
        p_home = predict_game(team_a, team_b, location == "N",
                              ratings_year, model, features)
        p_a = p_home
    if p_a is None:
        return None
    return p_a, 1.0 - p_a


# ──────────────────────────────────────────────────────────────
# PERSISTENCE  (train once locally, load on the app / cloud)
# ──────────────────────────────────────────────────────────────

def load_pit_dataset(path=PIT_FILE, min_year=PIT_MIN_YEAR):
    """Point-in-time model matrix: every game carries the ratings as they stood
    the morning before tip (from fetch_data.py --pit). Same columns as
    build_game_dataset(..., keep_meta=True)."""
    df = pd.read_csv(path)
    return df[df["YEAR"] >= min_year].reset_index(drop=True)


def save_model(path=MODEL_FILE, cv=3, source="pit"):
    """Train on the full game log and persist model + metadata with joblib.

    source="pit" (default) trains on point-in-time ratings (data/games_pit.csv),
    which is what the model actually sees when it predicts a live game: the
    ratings *as of that day*, not the season's final numbers. Walk-forward on
    2022-2026 this beats season-aggregate training on every metric (Brier
    0.191 -> 0.183, calibration error 0.028 -> 0.007), and the gap is widest in
    November, when early ratings are noisiest. source="agg" keeps the old
    season-aggregate behaviour (games.csv + ratings.csv) for comparison.

    Run locally after fetch_data.py. The app loads the artifact, so the cloud
    never needs games.csv or a live training pass.
    """
    ratings = load_ratings()
    features = get_features(ratings)
    if source == "pit":
        ds = load_pit_dataset()
    elif source == "agg":
        ds = build_game_dataset(load_games(), ratings, features)
    else:
        raise ValueError(f"source must be 'pit' or 'agg', got {source!r}")

    model, cv_acc = train_model(ds, features, cv=cv, do_cv=True)
    years = sorted(ds["YEAR"].unique().tolist())
    meta = {
        "features": features,
        "cv_accuracy": cv_acc,
        "n_games": int(len(ds)),
        "year_min": int(years[0]),
        "year_max": int(years[-1]),
        "home_win_base": float(ds[TARGET].mean()),
        "sklearn_version": sklearn.__version__,
        "training_data": "point-in-time" if source == "pit" else "season-aggregate",
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, path, compress=3)
    return meta


def load_model(path=MODEL_FILE):
    """Load the persisted {'model', 'meta'} artifact, or None if it's missing."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)


if __name__ == "__main__":
    import sys
    src = "agg" if "--agg" in sys.argv else "pit"
    print(f"Training game model ({src}) …", flush=True)
    m = save_model(source=src)
    print(f"Saved {MODEL_FILE}")
    print(f"  {m['n_games']:,} games, {m['year_min']}–{m['year_max']}, "
          f"{len(m['features'])} features")
    print(f"  CV accuracy: {m['cv_accuracy']:.4f} "
          f"(baseline home-win rate {m['home_win_base']:.4f})")
    print(f"  training data: {m['training_data']}")
    print(f"  sklearn {m['sklearn_version']}, "
          f"file size {os.path.getsize(MODEL_FILE) / 1e6:.1f} MB")

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

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")
GAMES_FILE = os.path.join(DATA_DIR, "games.csv")

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

def build_game_dataset(games, ratings, features):
    """Turn games + ratings into a model matrix.

    Each row: {feature}_DIFF = home_rating - away_rating for every feature, plus
    HOME_COURT (0 on neutral courts, else 1) and the HOME_WIN label. Games where
    either team lacks a rating row are dropped by the inner joins.
    """
    feat = ratings[["YEAR", "TEAM"] + features]

    m = games.merge(feat.rename(columns={"TEAM": "HOME"}),
                    on=["YEAR", "HOME"], how="inner")
    m = m.merge(feat.rename(columns={"TEAM": "AWAY"}),
                on=["YEAR", "AWAY"], how="inner", suffixes=("_H", "_A"))

    out = pd.DataFrame({"YEAR": m["YEAR"].values})
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

def build_pipeline(n_estimators=200, max_depth=None, random_state=0):
    """StandardScaler + RandomForest. RF handles the ~35 correlated efficiency
    diffs well and, unlike Bagging, exposes feature importances."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            n_jobs=-1, random_state=random_state)),
    ])


def train_model(df_train, features, cv=3, do_cv=True, **pipe_kwargs):
    """Fit the pipeline on a game-dataset frame. Returns (model, cv_accuracy).

    cv_accuracy is a mean k-fold accuracy for reporting (None if do_cv=False).
    """
    cols = feature_columns(features)
    X = df_train[cols].values
    y = df_train[TARGET].values

    pipe = build_pipeline(**pipe_kwargs)
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

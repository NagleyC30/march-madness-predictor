# model_explain.py — teaching helpers for the "How the Models Work" page (item 15).
#
# The learning page explains what each classifier is doing. To keep it grounded
# in THIS project (not generic textbook figures), these helpers train quick
# models on the real tournament data and expose two concrete artifacts:
#
#   * feature_importance() — which of the 34 KenPom/Barttorvik feature-diffs a
#     Random Forest actually leans on.
#   * sample_tree()        — a real, shallow decision tree rendered as text
#     (export_text, so no matplotlib/graphviz dependency).
#
# Everything the model sees is a HIGH-minus-LOW seed *difference* (build_model_
# dataset), and the target is HIGH_SEED_WINS — so a positive split threshold means
# "the better-seeded team has more of this stat."

import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

import mm_model as mm

DATA_DIR = "data"
IMPORTANCE_FILE = os.path.join(DATA_DIR, "model_explain_importance.csv")
TREE_FILE = os.path.join(DATA_DIR, "model_explain_tree.txt")


def _dataset(df=None, features=None):
    """All historical tournament matchups as (X feature-diffs, y HIGH_SEED_WINS)."""
    if df is None:
        df = mm.load_data()
    if features is None:
        features = mm.get_features(df)
    rows, _ = mm.build_all_matchups(df)
    ds = mm.build_model_dataset(rows, df, features)
    diffs = [f"{c}_DIFF" for c in features]
    return ds[diffs].values, ds["HIGH_SEED_WINS"].values, list(features)


def feature_importance(df=None, features=None, top_n=15, n_estimators=300):
    """Impurity-based feature importance from a Random Forest trained on every
    historical matchup. Returns a DataFrame [feature, importance] (top_n rows,
    descending). Feature names are shown without the '_DIFF' suffix — each is the
    higher-seed-minus-lower-seed difference of that stat."""
    X, y, features = _dataset(df, features)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=0,
                                n_jobs=-1).fit(X, y)
    imp = (pd.DataFrame({"feature": features, "importance": rf.feature_importances_})
           .sort_values("importance", ascending=False)
           .head(top_n).reset_index(drop=True))
    return imp


def sample_tree(df=None, features=None, max_depth=3):
    """A real, shallow decision tree (max_depth) trained on every historical
    matchup, rendered as text. Illustrates how a single tree splits on the
    feature-diffs before forests/boosting layer many of them together."""
    X, y, features = _dataset(df, features)
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X, y)
    return export_text(dt, feature_names=features)


def teaching_artifacts(df=None, features=None, top_n=15, max_depth=3,
                       n_estimators=300):
    """Both learning-page artifacts from a SINGLE dataset build (the matchup
    reconstruction is the slow part, so we don't want to do it twice). Returns
    {'importance': DataFrame, 'tree': str}."""
    X, y, features = _dataset(df, features)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=0,
                                n_jobs=-1).fit(X, y)
    importance = (pd.DataFrame({"feature": features,
                                "importance": rf.feature_importances_})
                  .sort_values("importance", ascending=False)
                  .head(top_n).reset_index(drop=True))
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X, y)
    return {"importance": importance,
            "tree": export_text(dt, feature_names=features)}


def load_artifacts():
    """Read the precomputed teaching artifacts if present (instant), else compute
    them live (slow — the matchup rebuild dominates). Returns the same dict shape
    as teaching_artifacts()."""
    if os.path.exists(IMPORTANCE_FILE) and os.path.exists(TREE_FILE):
        with open(TREE_FILE, encoding="utf-8") as fh:
            return {"importance": pd.read_csv(IMPORTANCE_FILE),
                    "tree": fh.read()}
    return teaching_artifacts()


def main():
    """Precompute the learning-page artifacts to disk so the page loads instantly.
    Re-run whenever KenPom Barttorvik.csv changes."""
    art = teaching_artifacts(top_n=15, max_depth=3)
    os.makedirs(DATA_DIR, exist_ok=True)
    art["importance"].to_csv(IMPORTANCE_FILE, index=False)
    with open(TREE_FILE, "w", encoding="utf-8") as fh:
        fh.write(art["tree"])
    print(f"Wrote {IMPORTANCE_FILE} ({len(art['importance'])} features) and "
          f"{TREE_FILE} ({art['tree'].count(chr(10))} lines).")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:          # noqa: BLE001
        pass
    main()

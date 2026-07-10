# mm_model.py — Importable core for the B.O.B. tournament predictor.
#
# Side-effect-free: no stdout teeing, no file writes on import. precompute.py,
# the Streamlit app, and the analysis scripts all import from here so the
# tournament-modeling logic lives in exactly one place.

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

DATA_FILE = 'KenPom Barttorvik.csv'

# ──────────────────────────────────────────────────────────────
# DATA LOAD & FEATURE CONFIG
# ──────────────────────────────────────────────────────────────

META_COLS = {'YEAR', 'CONF', 'QUAD NO', 'TEAM', 'TEAM ID', 'SEED', 'ROUND'}


def load_data(path=DATA_FILE):
    """Load the KenPom/Barttorvik feature table."""
    return pd.read_csv(path)


def get_features(df):
    """Numeric feature columns (everything that isn't metadata)."""
    return [c for c in df.columns if c not in META_COLS]


# ──────────────────────────────────────────────────────────────
# BRACKET METADATA
# ──────────────────────────────────────────────────────────────

ALL_REGION_MAPS = {
    2026: {69: 'South', 70: 'East', 71: 'West', 72: 'Midwest'},
    2025: {65: 'South', 66: 'East', 67: 'West', 68: 'Midwest'},
    2024: {61: 'East',  62: 'South', 63: 'West', 64: 'Midwest'},
    2023: {57: 'South', 58: 'Midwest', 59: 'East', 60: 'West'},
    2022: {53: 'West',  54: 'South', 55: 'East', 56: 'Midwest'},
    2021: {49: 'West',  50: 'South', 51: 'East', 52: 'Midwest'},
    2019: {45: 'East',  46: 'South', 47: 'West', 48: 'Midwest'},
    2018: {41: 'South', 42: 'East',  43: 'West', 44: 'Midwest'},
    2017: {37: 'East',  38: 'Midwest', 39: 'West', 40: 'South'},
    2016: {33: 'South', 34: 'East',  35: 'West', 36: 'Midwest'},
    2015: {29: 'Midwest', 30: 'East', 31: 'West', 32: 'South'},
    2014: {25: 'South', 26: 'West',  27: 'East', 28: 'Midwest'},
    2013: {21: 'Midwest', 22: 'South', 23: 'West', 24: 'East'},
    2012: {17: 'South', 18: 'East',  19: 'West', 20: 'Midwest'},
    2011: {13: 'East',  14: 'Southwest', 15: 'West', 16: 'Southeast'},
    2010: {9: 'Midwest', 10: 'East', 11: 'West', 12: 'South'},
    2009: {5: 'Midwest', 6: 'East',  7: 'West',  8: 'South'},
    2008: {1: 'Region 1', 2: 'Region 2', 3: 'Region 3', 4: 'Region 4'},
}

SEED_PAIRS         = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
BRACKET_SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
ROUND_NAMES        = ['R64', 'R32', 'S16', 'E8']
ALL_ROUND_NAMES    = ['R64', 'R32', 'S16', 'E8', 'F4', 'Championship']

ROUND_LABELS = {
    'R64': 'Round of 64',
    'R32': 'Round of 32',
    'S16': 'Sweet 16',
    'E8': 'Elite Eight',
    'F4': 'Final Four',
    'Championship': 'Championship',
}


# ──────────────────────────────────────────────────────────────
# MATCHUP BUILDER  (reconstructs actual tournament results)
# ──────────────────────────────────────────────────────────────

def build_matchups_for_year(df_year, year):
    """Return list of matchup dicts reconstructed from ROUND values."""
    rows               = []
    regional_champions = {}
    REGION_NAMES       = ALL_REGION_MAPS.get(year, {})

    for quad_no in sorted(df_year['QUAD NO'].unique()):
        region_df   = df_year[df_year['QUAD NO'] == quad_no]
        region_name = REGION_NAMES.get(quad_no, str(quad_no))

        seed_to_info = {}
        for s_high, s_low in SEED_PAIRS:
            for seed in (s_high, s_low):
                if seed in seed_to_info:
                    continue
                cands = region_df[region_df['SEED'] == seed]
                if len(cands) >= 1:
                    r = cands.iloc[0] if len(cands) == 1 else cands.loc[cands['ROUND'].idxmin()]
                    seed_to_info[seed] = (r['TEAM'], int(r['ROUND']))

        bracket = [(seed_to_info[s][0], s, seed_to_info[s][1])
                   for s in BRACKET_SEED_ORDER if s in seed_to_info]
        if len(bracket) != 16:
            continue

        current = bracket
        for rnd_name in ROUND_NAMES:
            next_round = []
            for i in range(0, len(current), 2):
                t1, s1, r1 = current[i]
                t2, s2, r2 = current[i + 1]
                if r1 == r2:
                    continue
                winner_name = t1 if r1 < r2 else t2
                winner_seed = s1 if r1 < r2 else s2
                winner_rnd  = min(r1, r2)
                th, sh = (t1, s1) if s1 <= s2 else (t2, s2)
                tl, sl = (t2, s2) if s1 <= s2 else (t1, s1)
                rows.append({
                    'YEAR': year, 'REGION': region_name, 'ROUND': rnd_name,
                    'TEAM_HIGH': th, 'SEED_HIGH': sh,
                    'TEAM_LOW':  tl, 'SEED_LOW':  sl,
                    'WINNER': winner_name, 'HIGH_SEED_WINS': winner_name == th,
                })
                next_round.append((winner_name, winner_seed, winner_rnd))
            current = next_round

        if current:
            regional_champions[quad_no] = current[0]

    # Final Four
    quad_order = sorted(regional_champions.keys())
    f4_pairs   = [(quad_order[0], quad_order[2]), (quad_order[1], quad_order[3])] \
                 if len(quad_order) == 4 else []
    f4_winners = []

    for qa, qb in f4_pairs:
        if qa not in regional_champions or qb not in regional_champions:
            continue
        t1, s1, r1 = regional_champions[qa]
        t2, s2, r2 = regional_champions[qb]
        if r1 == r2:
            continue
        winner_name = t1 if r1 < r2 else t2
        winner_seed = s1 if r1 < r2 else s2
        winner_rnd  = min(r1, r2)
        th, sh = (t1, s1) if s1 <= s2 else (t2, s2)
        tl, sl = (t2, s2) if s1 <= s2 else (t1, s1)
        ra = REGION_NAMES.get(qa, str(qa))
        rb = REGION_NAMES.get(qb, str(qb))
        rows.append({
            'YEAR': year, 'REGION': f'{ra} / {rb}', 'ROUND': 'F4',
            'TEAM_HIGH': th, 'SEED_HIGH': sh,
            'TEAM_LOW':  tl, 'SEED_LOW':  sl,
            'WINNER': winner_name, 'HIGH_SEED_WINS': winner_name == th,
        })
        f4_winners.append((winner_name, winner_seed, winner_rnd))

    if len(f4_winners) == 2:
        t1, s1, r1 = f4_winners[0]
        t2, s2, r2 = f4_winners[1]
        if r1 != r2:
            winner_name = t1 if r1 < r2 else t2
            th, sh = (t1, s1) if s1 <= s2 else (t2, s2)
            tl, sl = (t2, s2) if s1 <= s2 else (t1, s1)
            rows.append({
                'YEAR': year, 'REGION': 'Championship', 'ROUND': 'Championship',
                'TEAM_HIGH': th, 'SEED_HIGH': sh,
                'TEAM_LOW':  tl, 'SEED_LOW':  sl,
                'WINNER': winner_name, 'HIGH_SEED_WINS': winner_name == th,
            })

    return rows


def build_all_matchups(df, exclude_years=(2026, 2027)):
    """Reconstruct every historical matchup across all completed tournaments."""
    rows = []
    years = sorted(y for y in df['YEAR'].unique() if y not in exclude_years)
    for year in years:
        rows.extend(build_matchups_for_year(df[df['YEAR'] == year], year))
    return rows, years


def build_model_dataset(matchup_rows, df_stats, features):
    """Convert matchup rows to feature-diff rows for ML training."""
    feature_diffs = [f'{c}_DIFF' for c in features]
    model_rows = []
    for row in matchup_rows:
        yr_df = df_stats[df_stats['YEAR'] == row['YEAR']]
        th_r  = yr_df[yr_df['TEAM'] == row['TEAM_HIGH']]
        tl_r  = yr_df[yr_df['TEAM'] == row['TEAM_LOW']]
        if th_r.empty or tl_r.empty:
            continue
        th, tl = th_r.iloc[0], tl_r.iloc[0]
        diff_row = {'YEAR': row['YEAR']}
        for col, cdiff in zip(features, feature_diffs):
            diff_row[cdiff] = th[col] - tl[col]
        diff_row['HIGH_SEED_WINS'] = int(row['HIGH_SEED_WINS'])
        model_rows.append(diff_row)
    return pd.DataFrame(model_rows)


# ──────────────────────────────────────────────────────────────
# MODEL TRAINING
# ──────────────────────────────────────────────────────────────

PIPE = Pipeline([
    ('scaler',     StandardScaler()),
    ('selector',   VarianceThreshold()),
    ('classifier', BaggingClassifier()),
])

BC_PARAMS = {
    'scaler':                   [None, StandardScaler(), MinMaxScaler()],
    'classifier__n_estimators': [5, 10, 15, 20, 25],
    'classifier__random_state': [0],
}

RF_PARAMS = {
    'scaler':                   [None, StandardScaler(), MinMaxScaler()],
    'classifier':               [RandomForestClassifier()],
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth':    [None, 10, 20],
    'classifier__random_state': [0],
}


def train_best_model(df_train, features):
    """Train BaggingClassifier and RandomForest, return the better one."""
    feature_diffs = [f'{c}_DIFF' for c in features]
    X = df_train[feature_diffs].values
    y = df_train['HIGH_SEED_WINS'].values

    bc_grid = GridSearchCV(PIPE, BC_PARAMS, cv=5, n_jobs=-1).fit(X, y)
    rf_grid = GridSearchCV(PIPE, RF_PARAMS, cv=5, n_jobs=-1).fit(X, y)

    if rf_grid.best_score_ >= bc_grid.best_score_:
        return rf_grid.best_estimator_, 'RandomForest', rf_grid.best_score_
    return bc_grid.best_estimator_, 'BaggingClassifier', bc_grid.best_score_


# ──────────────────────────────────────────────────────────────
# PREDICTION UTILITIES
# ──────────────────────────────────────────────────────────────

def predict_game_proba(team_a, seed_a, team_b, seed_b, df_features, model, features):
    """
    Returns (winner_name, winner_seed, prob_high_seed_wins).
    prob_high_seed_wins is the model's confidence that the lower seed number wins.
    """
    if seed_a <= seed_b:
        th, sh, tl, sl = team_a, seed_a, team_b, seed_b
    else:
        th, sh, tl, sl = team_b, seed_b, team_a, seed_a

    th_r = df_features[df_features['TEAM'] == th]
    tl_r = df_features[df_features['TEAM'] == tl]

    if th_r.empty or tl_r.empty:
        return th, sh, 0.6   # fallback: favor higher seed

    th_row, tl_row = th_r.iloc[0], tl_r.iloc[0]
    diff = [[
        float(th_row[c]) - float(tl_row[c])
        if not (pd.isna(th_row[c]) or pd.isna(tl_row[c])) else 0.0
        for c in features
    ]]

    try:
        proba = model.predict_proba(diff)[0]
        classes = list(model.classes_) if hasattr(model, 'classes_') else [0, 1]
        idx1 = classes.index(1) if 1 in classes else 1
        p_high = proba[idx1]
    except Exception:
        p_high = 0.6

    winner = (th, sh) if p_high >= 0.5 else (tl, sl)
    return winner[0], winner[1], p_high


def simulate_full_bracket(df_year_stats, model, year, features):
    """Simulate a complete bracket for a given year's team stats."""
    results      = []
    reg_champs   = {}
    REGION_NAMES = ALL_REGION_MAPS.get(year, {})

    for quad_no in sorted(df_year_stats['QUAD NO'].unique()):
        region_df   = df_year_stats[df_year_stats['QUAD NO'] == quad_no]
        region_name = REGION_NAMES.get(quad_no, str(quad_no))

        seed_to_team = {}
        for sh, sl in SEED_PAIRS:
            for seed in (sh, sl):
                if seed in seed_to_team:
                    continue
                cands = region_df[region_df['SEED'] == seed]
                if len(cands) >= 1:
                    r = cands.iloc[0] if len(cands) == 1 \
                        else cands.loc[cands['ROUND'].idxmin()]
                    seed_to_team[seed] = (r['TEAM'], int(r['SEED']))

        current = [seed_to_team[s] for s in BRACKET_SEED_ORDER if s in seed_to_team]
        if len(current) != 16:
            continue

        for rnd_name in ROUND_NAMES:
            next_round = []
            for i in range(0, len(current), 2):
                t1, s1 = current[i]
                t2, s2 = current[i + 1]
                w_name, w_seed, p_high = predict_game_proba(
                    t1, s1, t2, s2, df_year_stats, model, features)
                th, sh2 = (t1, s1) if s1 <= s2 else (t2, s2)
                tl, sl2 = (t2, s2) if s1 <= s2 else (t1, s1)
                results.append({
                    'YEAR': year, 'REGION': region_name, 'ROUND': rnd_name,
                    'TEAM_HIGH': th, 'SEED_HIGH': sh2,
                    'TEAM_LOW':  tl, 'SEED_LOW':  sl2,
                    'PRED_WINNER': w_name, 'PRED_SEED': w_seed,
                    'PROB_HIGH_WINS': p_high,
                })
                next_round.append((w_name, w_seed))
            current = next_round

        if current:
            reg_champs[quad_no] = current[0]

    # Final Four — inferred from regional champs
    quads    = sorted(reg_champs.keys())
    f4_pairs = [(quads[0], quads[2]), (quads[1], quads[3])] if len(quads) == 4 else []
    f4_winners = []

    for qa, qb in f4_pairs:
        if qa not in reg_champs or qb not in reg_champs:
            continue
        t1, s1 = reg_champs[qa]
        t2, s2 = reg_champs[qb]
        w_name, w_seed, p_high = predict_game_proba(
            t1, s1, t2, s2, df_year_stats, model, features)
        th, sh2 = (t1, s1) if s1 <= s2 else (t2, s2)
        tl, sl2 = (t2, s2) if s1 <= s2 else (t1, s1)
        ra = REGION_NAMES.get(qa, str(qa))
        rb = REGION_NAMES.get(qb, str(qb))
        results.append({
            'YEAR': year, 'REGION': f'{ra} / {rb}', 'ROUND': 'F4',
            'TEAM_HIGH': th, 'SEED_HIGH': sh2,
            'TEAM_LOW':  tl, 'SEED_LOW':  sl2,
            'PRED_WINNER': w_name, 'PRED_SEED': w_seed,
            'PROB_HIGH_WINS': p_high,
        })
        f4_winners.append((w_name, w_seed))

    if len(f4_winners) == 2:
        t1, s1 = f4_winners[0]
        t2, s2 = f4_winners[1]
        w_name, w_seed, p_high = predict_game_proba(
            t1, s1, t2, s2, df_year_stats, model, features)
        th, sh2 = (t1, s1) if s1 <= s2 else (t2, s2)
        tl, sl2 = (t2, s2) if s1 <= s2 else (t1, s1)
        results.append({
            'YEAR': year, 'REGION': 'Championship', 'ROUND': 'Championship',
            'TEAM_HIGH': th, 'SEED_HIGH': sh2,
            'TEAM_LOW':  tl, 'SEED_LOW':  sl2,
            'PRED_WINNER': w_name, 'PRED_SEED': w_seed,
            'PROB_HIGH_WINS': p_high,
        })

    return results


# ──────────────────────────────────────────────────────────────
# ACCURACY SCORING
# ──────────────────────────────────────────────────────────────

def score_predictions_cascade(pred_rows, actual_matchups_df, year):
    """Bracket/cascade scoring: only counts a game if the same two teams
    appeared in both the simulation and reality."""
    actual = actual_matchups_df[actual_matchups_df['YEAR'] == year]
    actual_key = {}
    for _, row in actual.iterrows():
        key = (row['ROUND'], frozenset([row['TEAM_HIGH'], row['TEAM_LOW']]))
        actual_key[key] = row['WINNER']

    scores = {r: {'correct': 0, 'total': 0} for r in ALL_ROUND_NAMES}
    for game in pred_rows:
        rnd = game['ROUND']
        key = (rnd, frozenset([game['TEAM_HIGH'], game['TEAM_LOW']]))
        if key in actual_key:
            scores[rnd]['total'] += 1
            if game['PRED_WINNER'] == actual_key[key]:
                scores[rnd]['correct'] += 1

    for rnd in scores:
        t = scores[rnd]['total']
        scores[rnd]['accuracy'] = scores[rnd]['correct'] / t if t > 0 else None
    return scores


def score_predictions_independent(actual_matchups_df, year, df_year_stats, model, features):
    """Independent game scoring: for every game that actually occurred, ask the
    model directly who would win, regardless of bracket cascade."""
    actual = actual_matchups_df[actual_matchups_df['YEAR'] == year]
    scores = {r: {'correct': 0, 'total': 0} for r in ALL_ROUND_NAMES}

    for _, row in actual.iterrows():
        rnd = row['ROUND']
        th, sh = row['TEAM_HIGH'], int(row['SEED_HIGH'])
        tl, sl = row['TEAM_LOW'],  int(row['SEED_LOW'])
        actual_winner = row['WINNER']
        w_name, _, _ = predict_game_proba(th, sh, tl, sl, df_year_stats, model, features)
        scores[rnd]['total'] += 1
        if w_name == actual_winner:
            scores[rnd]['correct'] += 1

    for rnd in scores:
        t = scores[rnd]['total']
        scores[rnd]['accuracy'] = scores[rnd]['correct'] / t if t > 0 else None
    return scores


# ──────────────────────────────────────────────────────────────
# BETTING SIMULATION
# ──────────────────────────────────────────────────────────────

def prob_to_american_odds(p):
    """Convert win probability to American moneyline odds."""
    p = max(0.001, min(0.999, p))
    if p >= 0.5:
        return -round(100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def american_odds_payout(odds, stake=1.0):
    """Net profit on a winning $stake bet at given American odds."""
    if odds < 0:
        return stake * (100 / abs(odds))
    return stake * (odds / 100)


def simulate_betting(pred_rows, actual_matchups_df, year,
                     thresholds=(-200, -250, -300, -350), stake=10.0):
    """For each threshold, bet only on games where the model's implied moneyline
    is at least as confident as the threshold. Returns dict keyed by threshold."""
    actual = actual_matchups_df[actual_matchups_df['YEAR'] == year]
    actual_key = {}
    for _, row in actual.iterrows():
        key = (row['ROUND'], frozenset([row['TEAM_HIGH'], row['TEAM_LOW']]))
        actual_key[key] = row['WINNER']

    results = {}
    for thresh in thresholds:
        bets_won = bets_lost = bets_placed = 0
        net_pnl = 0.0

        for game in pred_rows:
            p_high = game.get('PROB_HIGH_WINS', 0.5)
            pred = game['PRED_WINNER']
            p_win = p_high if pred == game['TEAM_HIGH'] else 1 - p_high
            implied_line = prob_to_american_odds(p_win)

            if implied_line > thresh:   # not confident enough
                continue

            bets_placed += 1
            rnd = game['ROUND']
            key = (rnd, frozenset([game['TEAM_HIGH'], game['TEAM_LOW']]))
            if key not in actual_key:
                continue

            actual_winner = actual_key[key]
            payout = american_odds_payout(implied_line, stake)
            if pred == actual_winner:
                net_pnl += payout
                bets_won += 1
            else:
                net_pnl -= stake
                bets_lost += 1

        total_wagered = bets_placed * stake
        roi = (net_pnl / total_wagered * 100) if total_wagered > 0 else 0.0
        results[thresh] = {
            'placed': bets_placed, 'won': bets_won, 'lost': bets_lost,
            'net_pnl': round(net_pnl, 2),
            'total_wagered': round(total_wagered, 2),
            'roi_pct': round(roi, 1),
        }
    return results


# ──────────────────────────────────────────────────────────────
# WALK-FORWARD HELPERS
# ──────────────────────────────────────────────────────────────

TRAINING_WINDOWS = {
    'all_prior':     None,
    'last_1_year':   1,
    'last_3_years':  3,
    'last_5_years':  5,
    'last_10_years': 10,
}

BETTING_THRESHOLDS = (-200, -250, -300, -350)


def get_train_years_for_window(all_years, target_year, window_size):
    prior = [y for y in all_years if y < target_year]
    return prior if window_size is None else prior[-window_size:]


# ──────────────────────────────────────────────────────────────
# CUSTOM USER METRICS
# ──────────────────────────────────────────────────────────────
#
# Lets an app user upload their own metric (e.g. coaching tenure, tournament
# history) keyed by (YEAR, TEAM), then measures how much it helps the model:
# permutation importance ranked against the existing features, plus an accuracy
# delta from training with vs. without the metric on one target season.

def prepare_custom_metric(custom_df):
    """Normalize an uploaded metric table to canonical (YEAR, TEAM, <numeric…>).

    Returns (clean_df, metric_cols, error). ``error`` is None on success.
    Key columns are matched case-insensitively; every other column that parses
    as numeric is treated as a metric. Duplicate (YEAR, TEAM) rows are averaged.
    """
    df = custom_df.copy()
    lookup = {str(c).lower().strip(): c for c in df.columns}
    if 'year' not in lookup or 'team' not in lookup:
        return None, [], "Upload must contain 'YEAR' and 'TEAM' columns."

    df = df.rename(columns={lookup['year']: 'YEAR', lookup['team']: 'TEAM'})
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    df = df.dropna(subset=['YEAR', 'TEAM'])
    if df.empty:
        return None, [], "No rows with a valid YEAR and TEAM were found."
    df['YEAR'] = df['YEAR'].astype(int)
    df['TEAM'] = df['TEAM'].astype(str).str.strip()

    metric_cols = []
    for c in list(df.columns):
        if c in ('YEAR', 'TEAM'):
            continue
        coerced = pd.to_numeric(df[c], errors='coerce')
        if coerced.notna().any():
            df[c] = coerced
            metric_cols.append(c)

    if not metric_cols:
        return None, [], "No numeric metric columns found besides YEAR/TEAM."

    df = (df[['YEAR', 'TEAM'] + metric_cols]
          .groupby(['YEAR', 'TEAM'], as_index=False).mean())
    return df, metric_cols, None


def custom_metric_coverage(df_stats, clean_df):
    """How well the upload's (YEAR, TEAM) keys cover the tournament team-seasons.

    Returns a dict with the base count, matched count, coverage fraction, and the
    upload keys that matched nothing (usually team-name spelling mismatches).
    """
    base_keys = set(map(tuple, df_stats[['YEAR', 'TEAM']].itertuples(index=False)))
    up_keys   = set(map(tuple, clean_df[['YEAR', 'TEAM']].itertuples(index=False)))
    matched   = base_keys & up_keys
    return {
        'n_base':           len(base_keys),
        'n_matched':        len(matched),
        'coverage':         len(matched) / len(base_keys) if base_keys else 0.0,
        'unmatched_upload': sorted(up_keys - base_keys),
    }


def augment_with_custom(df_stats, clean_df, metric_cols):
    """Left-join the custom metric columns onto the stats table.

    Uncovered team-seasons are filled with the per-column mean so that a matchup
    where the metric is missing contributes ~0 to the feature difference (the
    'neutral fill' policy) instead of being dropped.
    """
    merged = df_stats.merge(clean_df, on=['YEAR', 'TEAM'], how='left')
    for c in metric_cols:
        col_mean = merged[c].mean()
        merged[c] = merged[c].fillna(0.0 if pd.isna(col_mean) else col_mean)
    return merged


def evaluate_custom_metric(df_stats, clean_df, metric_cols, target_year, window_name):
    """Train the best model with and without the uploaded metric(s) for one
    target season + training window, and compute permutation importance on that
    season's actual games.

    Returns a results dict (or ``{'error': …}`` when the selection can't be run).
    """
    metric_cols = list(metric_cols)

    # Guard against a metric column that collides with an existing feature.
    existing = set(df_stats.columns)
    renames  = {c: f'{c} (user)' for c in metric_cols if c in existing}
    if renames:
        clean_df = clean_df.rename(columns=renames)
        metric_cols = [renames.get(c, c) for c in metric_cols]

    base_features = get_features(df_stats)
    merged        = augment_with_custom(df_stats, clean_df, metric_cols)
    aug_features  = base_features + [c for c in metric_cols if c not in base_features]

    rows, years = build_all_matchups(merged)
    if target_year not in years:
        return {'error': f'{target_year} has no completed results to score against.'}

    window_size = TRAINING_WINDOWS[window_name]
    train_years = get_train_years_for_window(years, target_year, window_size)
    if not train_years:
        return {'error': f'No prior seasons before {target_year} for this window.'}

    train_rows = [r for r in rows if r['YEAR'] in train_years]
    test_rows  = [r for r in rows if r['YEAR'] == target_year]

    df_train_aug  = build_model_dataset(train_rows, merged, aug_features)
    df_test_aug   = build_model_dataset(test_rows,  merged, aug_features)
    df_train_base = build_model_dataset(train_rows, merged, base_features)
    df_test_base  = build_model_dataset(test_rows,  merged, base_features)

    if df_test_aug.empty or df_train_aug.empty:
        return {'error': f'Not enough matchup data for {target_year} with this window.'}

    model_aug,  name_aug,  cv_aug  = train_best_model(df_train_aug,  aug_features)
    model_base, name_base, cv_base = train_best_model(df_train_base, base_features)

    diffs_aug  = [f'{c}_DIFF' for c in aug_features]
    diffs_base = [f'{c}_DIFF' for c in base_features]
    X_aug,  y_aug  = df_test_aug[diffs_aug].values,   df_test_aug['HIGH_SEED_WINS'].values
    X_base, y_base = df_test_base[diffs_base].values, df_test_base['HIGH_SEED_WINS'].values

    acc_aug  = accuracy_score(y_aug,  model_aug.predict(X_aug))
    acc_base = accuracy_score(y_base, model_base.predict(X_base))

    perm = permutation_importance(
        model_aug, X_aug, y_aug,
        n_repeats=15, random_state=0, scoring='accuracy')
    importance = pd.DataFrame({
        'feature':    aug_features,
        'importance': perm.importances_mean,
        'std':        perm.importances_std,
        'is_custom':  [c in metric_cols for c in aug_features],
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    importance['rank'] = importance.index + 1

    return {
        'metric_cols':   metric_cols,
        'train_years':   train_years,
        'target_year':   target_year,
        'n_test_games':  int(len(y_aug)),
        'n_features':    len(aug_features),
        'acc_with':      float(acc_aug),
        'acc_without':   float(acc_base),
        'acc_delta':     float(acc_aug - acc_base),
        'model_with':    name_aug,  'cv_with':    float(cv_aug),
        'model_without': name_base, 'cv_without': float(cv_base),
        'importance':    importance,
    }

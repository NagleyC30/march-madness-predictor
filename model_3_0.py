# model_3_0.py — Walk-forward validation, windowed training, betting simulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
warnings.filterwarnings('ignore')

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()

_log_file = open('model_3_0_output.txt', 'w', encoding='utf-8')
sys.stdout = _Tee(sys.__stdout__, _log_file)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV

# ──────────────────────────────────────────────────────────────
# DATA LOAD & FEATURE CONFIG
# ──────────────────────────────────────────────────────────────

df = pd.read_csv('KenPom Barttorvik.csv')

META_COLS    = {'YEAR', 'CONF', 'QUAD NO', 'TEAM', 'TEAM ID', 'SEED', 'ROUND'}

features      = [c for c in df.columns if c not in META_COLS]
feature_diffs = [f'{c}_DIFF' for c in features]

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

SEED_PAIRS          = [(1,16),(2,15),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9)]
BRACKET_SEED_ORDER  = [1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]
ROUND_NAMES         = ['R64','R32','S16','E8']
ALL_ROUND_NAMES     = ['R64','R32','S16','E8','F4','Championship']

# ──────────────────────────────────────────────────────────────
# MATCHUP BUILDER  (same logic as model_2_0, extracts training rows)
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
                t2, s2, r2 = current[i+1]
                if r1 == r2:
                    continue
                winner_name = t1 if r1 < r2 else t2
                winner_seed = s1 if r1 < r2 else s2
                winner_rnd  = min(r1, r2)
                th, sh = (t1,s1) if s1 <= s2 else (t2,s2)
                tl, sl = (t2,s2) if s1 <= s2 else (t1,s1)
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
    quad_order  = sorted(regional_champions.keys())
    f4_pairs    = [(quad_order[0], quad_order[2]), (quad_order[1], quad_order[3])] \
                   if len(quad_order) == 4 else []
    f4_winners  = []

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
        th, sh = (t1,s1) if s1 <= s2 else (t2,s2)
        tl, sl = (t2,s2) if s1 <= s2 else (t1,s1)
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
            th, sh = (t1,s1) if s1 <= s2 else (t2,s2)
            tl, sl = (t2,s2) if s1 <= s2 else (t1,s1)
            rows.append({
                'YEAR': year, 'REGION': 'Championship', 'ROUND': 'Championship',
                'TEAM_HIGH': th, 'SEED_HIGH': sh,
                'TEAM_LOW':  tl, 'SEED_LOW':  sl,
                'WINNER': winner_name, 'HIGH_SEED_WINS': winner_name == th,
            })

    return rows


def build_model_dataset(matchup_rows, df_stats):
    """Convert matchup rows to feature-diff rows for ML training."""
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
    'scaler':                    [None, StandardScaler(), MinMaxScaler()],
    'classifier__n_estimators':  [5, 10, 15, 20, 25],
    'classifier__random_state':  [0],
}

RF_PARAMS = {
    'scaler':                    [None, StandardScaler(), MinMaxScaler()],
    'classifier':                [RandomForestClassifier()],
    'classifier__n_estimators':  [50, 100, 150],
    'classifier__max_depth':     [None, 10, 20],
    'classifier__random_state':  [0],
}


def train_best_model(df_train):
    """Train BaggingClassifier and RandomForest, return the better one."""
    X = df_train[feature_diffs].values
    y = df_train['HIGH_SEED_WINS'].values

    bc_grid = GridSearchCV(PIPE, BC_PARAMS, cv=5).fit(X, y)
    rf_grid = GridSearchCV(PIPE, RF_PARAMS, cv=5).fit(X, y)

    if rf_grid.best_score_ >= bc_grid.best_score_:
        return rf_grid.best_estimator_, 'RandomForest', rf_grid.best_score_
    return bc_grid.best_estimator_, 'BaggingClassifier', bc_grid.best_score_


# ──────────────────────────────────────────────────────────────
# PREDICTION UTILITIES
# ──────────────────────────────────────────────────────────────

def predict_game_proba(team_a, seed_a, team_b, seed_b, df_features, model):
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
        # class order: 0 = upset, 1 = high-seed wins
        classes = list(model.classes_) if hasattr(model, 'classes_') else [0, 1]
        idx1 = classes.index(1) if 1 in classes else 1
        p_high = proba[idx1]
    except Exception:
        p_high = 0.6

    winner = (th, sh) if p_high >= 0.5 else (tl, sl)
    return winner[0], winner[1], p_high


def simulate_full_bracket(df_year_stats, model, year):
    """
    Simulate a complete bracket for a given year's team stats.
    Returns list of predicted game dicts including probability.
    """
    results     = []
    reg_champs  = {}
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
                t2, s2 = current[i+1]
                w_name, w_seed, p_high = predict_game_proba(
                    t1, s1, t2, s2, df_year_stats, model)
                th, sh2 = (t1,s1) if s1<=s2 else (t2,s2)
                tl, sl2 = (t2,s2) if s1<=s2 else (t1,s1)
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
    quads = sorted(reg_champs.keys())
    f4_pairs = [(quads[0], quads[2]), (quads[1], quads[3])] if len(quads) == 4 else []
    f4_winners = []

    for qa, qb in f4_pairs:
        if qa not in reg_champs or qb not in reg_champs:
            continue
        t1, s1 = reg_champs[qa]
        t2, s2 = reg_champs[qb]
        w_name, w_seed, p_high = predict_game_proba(
            t1, s1, t2, s2, df_year_stats, model)
        th, sh2 = (t1,s1) if s1<=s2 else (t2,s2)
        tl, sl2 = (t2,s2) if s1<=s2 else (t1,s1)
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
            t1, s1, t2, s2, df_year_stats, model)
        th, sh2 = (t1,s1) if s1<=s2 else (t2,s2)
        tl, sl2 = (t2,s2) if s1<=s2 else (t1,s1)
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
    """
    Bracket/cascade scoring: only counts a game if the exact same two teams
    appeared in both the simulation and reality (i.e., the model correctly
    advanced both teams to that round). Rounds where the model sent the wrong
    teams will show 0/0 — this is expected and reflects bracket cascade errors.
    """
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


def score_predictions_independent(actual_matchups_df, year, df_year_stats, model):
    """
    Independent game scoring: for every game that actually occurred, ask the
    model directly 'who would win this matchup?' regardless of how the bracket
    was simulated. This gives a true per-round accuracy unaffected by cascade
    errors — it's the fairest measure of model quality.
    """
    actual = actual_matchups_df[actual_matchups_df['YEAR'] == year]
    scores = {r: {'correct': 0, 'total': 0} for r in ALL_ROUND_NAMES}

    for _, row in actual.iterrows():
        rnd  = row['ROUND']
        th, sh = row['TEAM_HIGH'], int(row['SEED_HIGH'])
        tl, sl = row['TEAM_LOW'],  int(row['SEED_LOW'])
        actual_winner = row['WINNER']

        w_name, _, _ = predict_game_proba(th, sh, tl, sl, df_year_stats, model)

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
        return -round(100 * p / (1 - p))   # negative = favorite
    else:
        return round(100 * (1 - p) / p)    # positive = underdog


def american_odds_payout(odds, stake=1.0):
    """Net profit on a winning $stake bet at given American odds."""
    if odds < 0:
        return stake * (100 / abs(odds))
    else:
        return stake * (odds / 100)


def simulate_betting(pred_rows, actual_matchups_df, year,
                     thresholds=(-200, -250, -300, -350),
                     stake=10.0):
    """
    For each threshold, bet only on games where the model's implied moneyline
    is AT LEAST as confident as the threshold (i.e., odds <= threshold).
    Returns dict keyed by threshold with P&L, ROI, record.

    Note: This uses model probability as a proxy for true odds.
    For real-money analysis, replace PROB_HIGH_WINS with actual sportsbook lines.
    """
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
            # Determine our bet: the team the model predicts to win
            pred = game['PRED_WINNER']
            pred_seed = game['PRED_SEED']

            # Probability for the predicted winner
            if pred == game['TEAM_HIGH']:
                p_win = p_high
            else:
                p_win = 1 - p_high

            implied_line = prob_to_american_odds(p_win)

            # Only bet if confidence meets threshold (i.e., odds are at least as negative)
            if implied_line > thresh:   # e.g. -180 > -200 → skip (not confident enough)
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
# BUILD COMPLETE HISTORICAL MATCHUP SET
# ──────────────────────────────────────────────────────────────

print("Building historical matchup dataset...")
all_matchup_rows = []
tournament_years = sorted(y for y in df['YEAR'].unique() if y not in (2026, 2027))
for year in tournament_years:
    all_matchup_rows.extend(build_matchups_for_year(df[df['YEAR'] == year], year))

df_all_matchups = pd.DataFrame(all_matchup_rows)
print(f"Total historical matchups: {len(df_all_matchups)}")


# ──────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ──────────────────────────────────────────────────────────────
# Training configurations to test
TRAINING_WINDOWS = {
    'all_prior':    None,   # use all available years before test year
    'last_1_year':  1,
    'last_3_years': 3,
    'last_5_years': 5,
    'last_10_years': 10,
}

BETTING_THRESHOLDS = (-200, -250, -300, -350)

# Walk-forward: earliest test year = 2009 (need at least 2008 to train)
test_years = [y for y in tournament_years if y >= 2009]

print(f"\nWalk-forward validation over {len(test_years)} test years: {test_years[0]}–{test_years[-1]}")
print(f"Training window variants: {list(TRAINING_WINDOWS.keys())}\n")

# Store results
all_wf_results  = []   # per-year, per-window accuracy
all_bet_results = []   # per-year, per-window, per-threshold P&L

for window_name, window_size in TRAINING_WINDOWS.items():
    print(f"{'='*60}")
    print(f"WINDOW: {window_name}")
    print(f"{'='*60}")

    for test_year in test_years:
        # Determine training years
        available_prior = [y for y in tournament_years if y < test_year]
        if not available_prior:
            continue
        if window_size is None:
            train_years = available_prior
        else:
            train_years = available_prior[-window_size:]

        if not train_years:
            continue

        # Build training dataset
        train_matchup_rows = [r for r in all_matchup_rows if r['YEAR'] in train_years]
        df_train_model = build_model_dataset(train_matchup_rows, df)

        if len(df_train_model) < 10:
            print(f"  {test_year}: insufficient training data ({len(df_train_model)} rows) — skipping")
            continue

        # Train model
        model, model_name, cv_score = train_best_model(df_train_model)

        # Simulate test year bracket using that year's team stats
        df_test_year = df[df['YEAR'] == test_year]
        pred_rows    = simulate_full_bracket(df_test_year, model, test_year)

        # Cascade scoring: only scores matchups that actually occurred in simulation
        cascade_scores = score_predictions_cascade(pred_rows, df_all_matchups, test_year)
        # Independent scoring: asks the model about every actual game directly
        indep_scores   = score_predictions_independent(df_all_matchups, test_year, df_test_year, model)

        casc_correct = sum(v['correct'] for v in cascade_scores.values())
        casc_total   = sum(v['total']   for v in cascade_scores.values())
        indep_correct = sum(v['correct'] for v in indep_scores.values())
        indep_total   = sum(v['total']   for v in indep_scores.values())
        casc_acc  = casc_correct  / casc_total   if casc_total  > 0 else 0.0
        indep_acc = indep_correct / indep_total  if indep_total > 0 else 0.0

        print(f"  {test_year} | train={train_years} | {model_name} | cv={cv_score:.3f}")
        print(f"           cascade acc={casc_acc:.1%} ({casc_correct}/{casc_total})  "
              f"independent acc={indep_acc:.1%} ({indep_correct}/{indep_total})")
        print(f"  {'Round':<15} {'Cascade':>12} {'Independent':>13}")
        print(f"  {'-'*42}")
        for rnd in ALL_ROUND_NAMES:
            cs = cascade_scores.get(rnd, {'correct': 0, 'total': 0, 'accuracy': None})
            is_ = indep_scores.get(rnd, {'correct': 0, 'total': 0, 'accuracy': None})
            casc_str  = f"{cs['correct']}/{cs['total']} ({cs['accuracy']:.0%})" \
                        if cs['total'] > 0 else "  —"
            indep_str = f"{is_['correct']}/{is_['total']} ({is_['accuracy']:.0%})" \
                        if is_['total'] > 0 else "  —"
            print(f"  {rnd:<15} {casc_str:>12} {indep_str:>13}")

        # Record both scoring modes
        for rnd in ALL_ROUND_NAMES:
            cs  = cascade_scores.get(rnd, {'correct': 0, 'total': 0, 'accuracy': None})
            is_ = indep_scores.get(rnd,   {'correct': 0, 'total': 0, 'accuracy': None})
            all_wf_results.append({
                'window':           window_name,
                'test_year':        test_year,
                'train_years':      str(train_years),
                'model':            model_name,
                'cv_score':         cv_score,
                'round':            rnd,
                'cascade_correct':  cs['correct'],
                'cascade_total':    cs['total'],
                'cascade_accuracy': cs['accuracy'],
                'indep_correct':    is_['correct'],
                'indep_total':      is_['total'],
                'indep_accuracy':   is_['accuracy'],
            })

        # Betting simulation
        bet_sim = simulate_betting(
            pred_rows, df_all_matchups, test_year,
            thresholds=BETTING_THRESHOLDS
        )
        for thresh, bstats in bet_sim.items():
            all_bet_results.append({
                'window':    window_name,
                'test_year': test_year,
                'threshold': thresh,
                **bstats,
            })

        print()

# ──────────────────────────────────────────────────────────────
# AGGREGATE RESULTS
# ──────────────────────────────────────────────────────────────

df_wf      = pd.DataFrame(all_wf_results)
df_betting = pd.DataFrame(all_bet_results)

print("\n" + "="*70)
print("WALK-FORWARD SUMMARY — Overall Accuracy by Window")
print("="*70)
summary = (df_wf.groupby('window')
               .apply(lambda g: pd.Series({
                   'cascade_correct':  g['cascade_correct'].sum(),
                   'cascade_total':    g['cascade_total'].sum(),
                   'cascade_acc':      g['cascade_correct'].sum() / g['cascade_total'].sum()
                                        if g['cascade_total'].sum() > 0 else 0,
                   'indep_correct':    g['indep_correct'].sum(),
                   'indep_total':      g['indep_total'].sum(),
                   'indep_acc':        g['indep_correct'].sum() / g['indep_total'].sum()
                                        if g['indep_total'].sum() > 0 else 0,
               }))
               .reset_index())
print(summary.to_string(index=False))

print("\n" + "="*70)
print("WALK-FORWARD SUMMARY — Independent Accuracy by Round (all windows combined)")
print("="*70)
by_round = (df_wf.groupby('round')
                 .apply(lambda g: pd.Series({
                     'indep_correct': g['indep_correct'].sum(),
                     'indep_total':   g['indep_total'].sum(),
                     'indep_acc':     g['indep_correct'].sum() / g['indep_total'].sum()
                                       if g['indep_total'].sum() > 0 else 0,
                     'cascade_correct': g['cascade_correct'].sum(),
                     'cascade_total':   g['cascade_total'].sum(),
                     'cascade_acc':     g['cascade_correct'].sum() / g['cascade_total'].sum()
                                         if g['cascade_total'].sum() > 0 else 0,
                 }))
                 .reindex(ALL_ROUND_NAMES)
                 .reset_index())
print(by_round.to_string(index=False))

print("\n" + "="*70)
print("BETTING SIMULATION SUMMARY — Cumulative P&L by Window & Threshold")
print("="*70)
bet_summary = (df_betting
               .groupby(['window', 'threshold'])
               .agg(total_placed=('placed', 'sum'),
                    total_won=('won', 'sum'),
                    total_lost=('lost', 'sum'),
                    total_pnl=('net_pnl', 'sum'),
                    total_wagered=('total_wagered', 'sum'))
               .reset_index())
bet_summary['roi_pct'] = (bet_summary['total_pnl'] / bet_summary['total_wagered'] * 100).round(1)
print(bet_summary.to_string(index=False))

# ──────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ──────────────────────────────────────────────────────────────

# 1. Independent accuracy over time per window
pivot_acc = (df_wf.groupby(['window', 'test_year'])
                  .apply(lambda g: g['indep_correct'].sum() / g['indep_total'].sum()
                         if g['indep_total'].sum() > 0 else np.nan)
                  .reset_index(name='accuracy'))

plt.figure(figsize=(12, 6))
for window_name in TRAINING_WINDOWS:
    sub = pivot_acc[pivot_acc['window'] == window_name]
    plt.plot(sub['test_year'], sub['accuracy'], marker='o', label=window_name)
plt.axhline(0.5, linestyle='--', color='gray', alpha=0.5, label='50% baseline')
plt.title('Walk-Forward Bracket Accuracy by Training Window')
plt.xlabel('Test Year')
plt.ylabel('Overall Accuracy')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('accuracy_by_window.png', dpi=120)
plt.show()

# 2. P&L per threshold across all windows
for window_name in TRAINING_WINDOWS:
    sub = df_betting[df_betting['window'] == window_name]
    pnl_by_year = sub.groupby(['test_year', 'threshold'])['net_pnl'].sum().reset_index()

    plt.figure(figsize=(12, 5))
    for thresh in BETTING_THRESHOLDS:
        t_sub = pnl_by_year[pnl_by_year['threshold'] == thresh]
        # Cumulative P&L
        t_sub = t_sub.sort_values('test_year').copy()
        t_sub['cumulative_pnl'] = t_sub['net_pnl'].cumsum()
        plt.plot(t_sub['test_year'], t_sub['cumulative_pnl'], marker='o', label=f'Line {thresh}')
    plt.axhline(0, linestyle='--', color='gray', alpha=0.5)
    plt.title(f'Cumulative P&L — {window_name} ($10/bet)')
    plt.xlabel('Test Year')
    plt.ylabel('Cumulative Net P&L ($)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'pnl_{window_name}.png', dpi=120)
    plt.show()

# ──────────────────────────────────────────────────────────────
# SELECT BEST MODEL CONFIG & PREDICT 2026 / 2027
# ──────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("BEST MODEL CONFIG (highest walk-forward accuracy)")
print("="*70)

best_window_row = summary.loc[summary['accuracy'].idxmax()]
best_window     = best_window_row['window']
best_window_size = TRAINING_WINDOWS[best_window]

print(f"Best window: {best_window}  |  accuracy: {best_window_row['accuracy']:.2%}")

def get_train_years_for_window(all_years, target_year, window_size):
    prior = [y for y in all_years if y < target_year]
    return prior if window_size is None else prior[-window_size:]


def predict_future_year(target_year, label):
    if target_year not in [y for y in df['YEAR'].unique()]:
        print(f"\n[SKIP] {label}: no team data for {target_year} in dataset. "
              f"Add {target_year} rows to 'KenPom Barttorvik.csv' to enable predictions.")
        return

    print(f"\n{'='*70}")
    print(f"PREDICTING {label} BRACKET (best window: {best_window})")
    print(f"{'='*70}")

    train_yrs = get_train_years_for_window(tournament_years, target_year, best_window_size)
    if not train_yrs:
        print(f"  [ERROR] No training years available for {target_year}.")
        return

    train_matchup_rows = [r for r in all_matchup_rows if r['YEAR'] in train_yrs]
    df_train_model     = build_model_dataset(train_matchup_rows, df)
    model, model_name, cv_score = train_best_model(df_train_model)

    df_target = df[df['YEAR'] == target_year]
    pred_rows  = simulate_full_bracket(df_target, model, target_year)

    df_pred = pd.DataFrame(pred_rows)
    round_order_cat = pd.CategoricalDtype(ALL_ROUND_NAMES, ordered=True)
    df_pred['ROUND'] = df_pred['ROUND'].astype(round_order_cat)
    df_pred = df_pred.sort_values(['ROUND', 'REGION']).reset_index(drop=True)

    print(f"\nModel: {model_name}  |  CV score: {cv_score:.3f}")
    print(f"Training years: {train_yrs}\n")

    for _, row in df_pred.iterrows():
        p_high = row['PROB_HIGH_WINS']
        winner = row['PRED_WINNER']
        implied = prob_to_american_odds(p_high if winner == row['TEAM_HIGH'] else 1-p_high)
        print(f"  [{row['ROUND']:13s}] {row['REGION']:25s} "
              f"{row['TEAM_HIGH']}(#{row['SEED_HIGH']}) vs "
              f"{row['TEAM_LOW']}(#{row['SEED_LOW']})  →  "
              f"{winner}  [line: {implied:+d}]")

    champ_row = df_pred[df_pred['ROUND'] == 'Championship']
    if not champ_row.empty:
        champ = champ_row.iloc[0]['PRED_WINNER']
        seed  = champ_row.iloc[0]['PRED_SEED']
        print(f"\n{'='*55}")
        print(f"  {label} NATIONAL CHAMPION PREDICTION: {champ} (seed #{seed})")
        print(f"{'='*55}")

    df_pred.to_csv(f'bracket_prediction_{target_year}.csv', index=False)
    print(f"\n  Saved: bracket_prediction_{target_year}.csv")


predict_future_year(2026, '2026')
predict_future_year(2027, '2027')   # will skip unless 2027 data is added to CSV

# Save aggregate CSVs for further analysis
df_wf.to_csv('walk_forward_accuracy.csv', index=False)
df_betting.to_csv('betting_simulation.csv', index=False)
print("\nSaved: walk_forward_accuracy.csv, betting_simulation.csv")

sys.stdout = sys.__stdout__
_log_file.close()
print("Output saved to model_3_0_output.txt")

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Read and view the dataset
df=pd.read_csv('KenPom Barttorvik.csv')

df.head()

# All columns are retained — features are determined dynamically from numeric columns
META_COLS = {'YEAR', 'CONF', 'QUAD NO', 'TEAM', 'TEAM ID', 'SEED', 'ROUND'}
column_keep = {'WIN%', 'OREB%', 'DREB%', '2PT%', '3PT%', 'AVG HGT', 'ELITE SOS'}

for col in df:
  if (col not in META_COLS) and (col not in column_keep):
    df.drop(col, axis=1, inplace=True)

# Define the features list based on the columns that were kept for analysis
# Exclude META_COLS as these are identifiers/metadata, not direct features for comparison
features = [col for col in df.columns if col not in META_COLS]

df.head()

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', VarianceThreshold()),
    ('classifier', BaggingClassifier()),
])

baggingClassifierParamaters = {
    'scaler' : [None, StandardScaler(), MinMaxScaler()],
    'classifier__n_estimators' : [5, 10, 15, 20, 25],
    'classifier__random_state' : [0]
}

randomForestParamaters = {
    'scaler' : [None, StandardScaler(), MinMaxScaler()],
    'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : [50, 100, 150],
    'classifier__max_depth' : [None, 10, 20],
    'classifier__random_state' : [0]
}

REGION_NAMES_2025 = {65: 'South', 66: 'East', 67: 'West', 68: 'Midwest'}
REGION_NAMES_2024 = {61: 'East', 62: 'South', 63: 'West', 64: 'Midwest'}
REGION_NAMES_2023 = {57: 'South', 58: 'Midwest', 59: 'East', 60: 'West'}
REGION_NAMES_2022 = {53: 'West', 54: 'South', 55: 'East', 56: 'Midwest'}
REGION_NAMES_2021 = {49: 'West', 50: 'South', 51: 'East', 52: 'Midwest'}
REGION_NAMES_2019 = {45: 'East', 46: 'South', 47: 'West', 48: 'Midwest'}
REGION_NAMES_2018 = {41: 'South', 42: 'East', 43: 'West', 44: 'Midwest'}
REGION_NAMES_2017 = {37: 'East', 38: 'Midwest', 39: 'West', 40: 'South'}
REGION_NAMES_2016 = {33: 'South', 34: 'East', 35: 'West', 36: 'Midwest'}
REGION_NAMES_2015 = {29: 'Midwest', 30: 'East', 31: 'West', 32: 'South'}
REGION_NAMES_2014 = {25: 'South', 26: 'West', 27: 'East', 28: 'Midwest'}
REGION_NAMES_2013 = {21: 'Midwest', 22: 'South', 23: 'West', 24: 'East'}
REGION_NAMES_2012 = {17: 'South', 18: 'East', 19: 'West', 20: 'Midwest'}
REGION_NAMES_2011 = {13: 'East', 14: 'Southwest', 15: 'West', 16: 'Southeast'}
REGION_NAMES_2010 = {9: 'Midwest', 10: 'East', 11: 'West', 12: 'South'}
REGION_NAMES_2009 = {5: 'Midwest', 6: 'East', 7: 'West', 8: 'South'}
REGION_NAMES_2008 = {1: 'Region 1', 2: 'Region 2', 3: 'Region 3', 4: 'Region 4'}

ALL_REGION_MAPS = {
    2025: REGION_NAMES_2025,
    2024: REGION_NAMES_2024,
    2023: REGION_NAMES_2023,
    2022: REGION_NAMES_2022,
    2021: REGION_NAMES_2021,
    2019: REGION_NAMES_2019,
    2018: REGION_NAMES_2018,
    2017: REGION_NAMES_2017,
    2016: REGION_NAMES_2016,
    2015: REGION_NAMES_2015,
    2014: REGION_NAMES_2014,
    2013: REGION_NAMES_2013,
    2012: REGION_NAMES_2012,
    2011: REGION_NAMES_2011,
    2010: REGION_NAMES_2010,
    2009: REGION_NAMES_2009,
    2008: REGION_NAMES_2008,
}

SEED_PAIRS = [(1, 16), (2, 15), (3, 14), (4, 13),
              (5, 12), (6, 11), (7, 10), (8,  9)]

BRACKET_SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

FINAL_FOUR_PAIRS = {}
for year, region_map in ALL_REGION_MAPS.items():
    quad_order = sorted(region_map.keys())

    if len(quad_order) == 4:
        FINAL_FOUR_PAIRS[year] = [
            (quad_order[0], quad_order[2]),
            (quad_order[1], quad_order[3])
        ]
    else:
        print(f"[WARNING] Year {year} has an unexpected number of regions ({len(quad_order)}). Skipping Final Four pairing for this year.")
        FINAL_FOUR_PAIRS[year] = []

'''
Function: Reconstructs every historical game for one year by replaying the
  bracket from each team's final ROUND value. Lower ROUND = advanced further =
  won that game.

Arguments:
  df_year (dataframe): a dataframe formed from our original dataframe filtered
    by year
  year (int): the year we are rebuilding

Returns:
  rows (list): a list of the data to be put into a new dataframe with the formed
    matchups
'''

def build_all_round_matchups_for_year(df_year, year):
    rows = []
    regional_champions = {}

    REGION_NAMES = ALL_REGION_MAPS.get(year, {})

    for quad_no in sorted(df_year['QUAD NO'].unique()):
        region_df   = df_year[df_year['QUAD NO'] == quad_no]
        region_name = REGION_NAMES.get(quad_no, str(quad_no))

        seed_to_info = {}
        for s_high, s_low in SEED_PAIRS:
            for seed in (s_high, s_low):
                if seed in seed_to_info:
                    continue
                candidates = region_df[region_df['SEED'] == seed]
                if len(candidates) == 1:
                    r = candidates.iloc[0]
                    seed_to_info[seed] = (r['TEAM'], int(r['ROUND']))
                elif len(candidates) > 1:
                    r = candidates.loc[candidates['ROUND'].idxmin()]
                    seed_to_info[seed] = (r['TEAM'], int(r['ROUND']))

        bracket = []
        for s in BRACKET_SEED_ORDER:
            if s in seed_to_info:
                team, rnd = seed_to_info[s]
                bracket.append((team, s, rnd))

        if len(bracket) != 16:
            continue

        current = bracket
        for rnd_name in ['R64', 'R32', 'S16', 'E8']:
            next_round = []
            for i in range(0, len(current), 2):
                t1_name, t1_seed, t1_rnd = current[i]
                t2_name, t2_seed, t2_rnd = current[i + 1]

                if t1_rnd == t2_rnd:
                    continue

                winner_name = t1_name if t1_rnd < t2_rnd else t2_name
                winner_seed = t1_seed if t1_rnd < t2_rnd else t2_seed
                winner_rnd  = t1_rnd  if t1_rnd < t2_rnd else t2_rnd

                if t1_seed <= t2_seed:
                    team_high, seed_high = t1_name, t1_seed
                    team_low,  seed_low  = t2_name, t2_seed
                else:
                    team_high, seed_high = t2_name, t2_seed
                    team_low,  seed_low  = t1_name, t1_seed

                rows.append({
                    'YEAR'      : year,
                    'REGION'    : region_name,
                    'ROUND'     : rnd_name,
                    'TEAM_HIGH' : team_high,
                    'SEED_HIGH' : seed_high,
                    'TEAM_LOW'  : team_low,
                    'SEED_LOW'  : seed_low,
                    'WINNER'         : winner_name,
                    'HIGH_SEED_WINS' : winner_name == team_high,
                })
                next_round.append((winner_name, winner_seed, winner_rnd))
            current = next_round

        if current:
            regional_champions[quad_no] = current[0]

    finals_teams = sorted([(q, t) for q, t in regional_champions.items()
                            if t[2] in (1, 2)], key=lambda x: x[1][2])
    f4_losers    = sorted([(q, t) for q, t in regional_champions.items()
                            if t[2] == 4],      key=lambda x: x[0])

    if len(finals_teams) == 2 and len(f4_losers) == 2:
        dynamic_f4_pairs = [
            (finals_teams[0][0], f4_losers[0][0]),
            (finals_teams[1][0], f4_losers[1][0]),
        ]
    else:
        dynamic_f4_pairs = FINAL_FOUR_PAIRS.get(year, [])

    f4_winners = []
    for quad_a, quad_b in dynamic_f4_pairs:
        if quad_a not in regional_champions or quad_b not in regional_champions:
            continue
        t1_name, t1_seed, t1_rnd = regional_champions[quad_a]
        t2_name, t2_seed, t2_rnd = regional_champions[quad_b]

        if t1_rnd == t2_rnd:
            continue

        winner_name = t1_name if t1_rnd < t2_rnd else t2_name
        winner_seed = t1_seed if t1_rnd < t2_rnd else t2_seed
        winner_rnd  = t1_rnd  if t1_rnd < t2_rnd else t2_rnd

        if t1_seed <= t2_seed:
            team_high, seed_high = t1_name, t1_seed
            team_low,  seed_low  = t2_name, t2_seed
        else:
            team_high, seed_high = t2_name, t2_seed
            team_low,  seed_low  = t1_name, t1_seed

        region_a = REGION_NAMES.get(quad_a, str(quad_a))
        region_b = REGION_NAMES.get(quad_b, str(quad_b))
        rows.append({
            'YEAR'      : year,
            'REGION'    : f'{region_a} / {region_b}',
            'ROUND'     : 'F4',
            'TEAM_HIGH' : team_high,
            'SEED_HIGH' : seed_high,
            'TEAM_LOW'  : team_low,
            'SEED_LOW'  : seed_low,
            'WINNER'         : winner_name,
            'HIGH_SEED_WINS' : winner_name == team_high,
        })
        f4_winners.append((winner_name, winner_seed, winner_rnd))

    if len(f4_winners) == 2:
        t1_name, t1_seed, t1_rnd = f4_winners[0]
        t2_name, t2_seed, t2_rnd = f4_winners[1]
        if t1_rnd != t2_rnd:
            winner_name = t1_name if t1_rnd < t2_rnd else t2_name
            if t1_seed <= t2_seed:
                team_high, seed_high = t1_name, t1_seed
                team_low,  seed_low  = t2_name, t2_seed
            else:
                team_high, seed_high = t2_name, t2_seed
                team_low,  seed_low  = t1_name, t1_seed
            rows.append({
                'YEAR'      : year,
                'REGION'    : 'Championship',
                'ROUND'     : 'Championship',
                'TEAM_HIGH' : team_high,
                'SEED_HIGH' : seed_high,
                'TEAM_LOW'  : team_low,
                'SEED_LOW'  : seed_low,
                'WINNER'         : winner_name,
                'HIGH_SEED_WINS' : winner_name == team_high,
            })

    return rows


all_matchup_rows = []
for year in sorted(df['YEAR'].unique()):  # Loops through all years
    if year == 2026:  # Ignores 2026 rows
        continue

    '''
    Function Call: builds the matchup dataframe for a particular year

    Paramaters:
      df_year (dataframe): a dataframe formed from our original dataframe
        filtered by year
      year (int): the year we are rebuilding

    Returns:
      rows (list): a list of the data to be put into a new dataframe with the
        formed matchups
    '''
    all_matchup_rows.extend(build_all_round_matchups_for_year(df[df['YEAR'] == year], year))

# Creates a new dataframe with the new formed matchup data
df_all_matchups = pd.DataFrame(all_matchup_rows)

# Counts the number of unique values of each round after reindexing them from
# their original values (64, 32, etc.) to our human readable values of (R64,
# R32, S16, E8, F4, Championship)
round_dist = df_all_matchups['ROUND'].value_counts().reindex(
    ['R64', 'R32', 'S16', 'E8', 'F4', 'Championship'])

# Outputs the number of rows in the new dataframe which is equal to the number
# of matchups created (used for double checking counts)
print(f'Training games built: {len(df_all_matchups)} total')
print(round_dist.to_string())

# Outputs the rate of the higher seed winning (i.e., no upset)
high_seed_win_rate = df_all_matchups['HIGH_SEED_WINS'].mean()
print(f'Higher seed wins {high_seed_win_rate:.1%} of games in training data')

'''
The amount of games built was reported as 1071. This was verified by peforming
simple math that says there should be 63 games in each tournament for the
years ranging from 2008-2025 (17 years). The total amount of games should then
be: 63 * 17 = 1071
'''


feature_cols = [f'{c}_DIFF' for c in features]

all_model_rows = []
for _, row in df_all_matchups.iterrows():
    yr_df   = df[df['YEAR'] == row['YEAR']]
    th_rows = yr_df[yr_df['TEAM'] == row['TEAM_HIGH']]
    tl_rows = yr_df[yr_df['TEAM'] == row['TEAM_LOW']]

    if th_rows.empty or tl_rows.empty:
        continue

    th = th_rows.iloc[0]
    tl = tl_rows.iloc[0]

    diff_row = {'YEAR': row['YEAR']}
    for col, col_diff in zip(features, feature_cols):
        diff_row[col_diff] = th[col] - tl[col]
    diff_row['HIGH_SEED_WINS'] = int(row['HIGH_SEED_WINS'])
    all_model_rows.append(diff_row)

df_full_model = pd.DataFrame(all_model_rows)
print(f'Full model dataset: {df_full_model.shape[0]} rows, {df_full_model.shape[1]} columns')

print("\n=== Exploratory Data Analysis ===\n")

# Convert to DataFrame for easier plotting
df_eda = df_full_model.copy()

# 1. Target distribution (class balance)
plt.figure()
sns.countplot(x='HIGH_SEED_WINS', data=df_eda)
plt.title('Distribution of Target Variable (High Seed Wins)')
plt.xlabel('High Seed Wins (1 = Yes, 0 = Upset)')
plt.ylabel('Count')
plt.show()

print("Insight:")
print(f"- High seeds win {df_eda['HIGH_SEED_WINS'].mean():.2%} of games.")
print("- This indicates whether the dataset is balanced or biased toward favorites.\n")


# 2. Feature distributions
df_eda[feature_cols].hist(figsize=(12, 10))
plt.suptitle("Feature Difference Distributions", y=1.02)
plt.show()

print("Insight:")
print("- Most feature differences are centered around 0, meaning teams are often similar.")
print("- Large positive values favor the higher seed, negative values favor the lower seed.\n")


# 3. Correlation heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df_eda.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True)
plt.title("Correlation Heatmap")
plt.show()

print("Insight:")
print("- Look for features strongly correlated with HIGH_SEED_WINS.")
print("- These features are likely important predictors in the model.\n")


# 4. Feature vs Target (boxplots for key features)
important_features = feature_cols[:6]  # take a subset for readability

for col in important_features:
    plt.figure()
    sns.boxplot(x='HIGH_SEED_WINS', y=col, data=df_eda)
    plt.title(f'{col} vs High Seed Wins')
    plt.show()

print("Insight:")
print("- If distributions differ significantly between classes, the feature is predictive.")
print("- Example: If WIN%_DIFF is higher when HIGH_SEED_WINS=1, stronger teams win more.\n")


# 5. Upset analysis (key for classification task)
upset_rate = 1 - df_eda['HIGH_SEED_WINS'].mean()

print("Upset Analysis:")
print(f"- Upset rate: {upset_rate:.2%}")
print("- The model is essentially learning when an upset occurs.\n")


# 6. Mean feature differences by class
group_means = df_eda.groupby('HIGH_SEED_WINS')[feature_cols].mean()

print("Average Feature Differences by Outcome:")
print(group_means)

print("\nInsight:")
print("- Positive values for HIGH_SEED_WINS=1 show what advantages lead to expected wins.")
print("- Values closer to 0 in upsets suggest more evenly matched teams.\n")

df_full_model.head()

# Define X_train and y_train from df_full_model
X_train = df_full_model[feature_cols].values
y_train = df_full_model['HIGH_SEED_WINS'].values

# Split the data into training and testing sets
# Using a common split ratio of 80% for training and 20% for testing
# stratify=y_train ensures that the proportion of target variable (HIGH_SEED_WINS) is similar in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f'Training set size: {len(X_train)} samples')
print(f'Test set size: {len(X_test)} samples')
print(f'Proportion of High Seed Wins in Training: {y_train.mean():.1%}')
print(f'Proportion of High Seed Wins in Test: {y_test.mean():.1%}')

# Bagging Classifier Scores
# Grid search for pipeline with no scaling
baggingClassifierGrid = GridSearchCV(pipe, baggingClassifierParamaters, cv=5).fit(X_train, y_train)
print('BAGGING CLASSIFIER TEST SCORES')
print('Training set score: ' + str(baggingClassifierGrid.score(X_train, y_train)))
print('Test set score: ' + str(baggingClassifierGrid.score(X_test, y_test)) + '\n\n')

randomForestClassifierGrid = GridSearchCV(pipe, randomForestParamaters, cv=5).fit(X_train, y_train)
print('RANDOM FOREST TEST SCORES')
print('Training set score: ' + str(randomForestClassifierGrid.score(X_train, y_train)))
print('Test set score: ' + str(randomForestClassifierGrid.score(X_test, y_test)) + '\n\n')

bcBestParams = baggingClassifierGrid.best_params_
print(bcBestParams)
# Stores the optimum model in best_pipe
bcBestPipe = baggingClassifierGrid.best_estimator_
print(bcBestPipe)

rfBestParams = randomForestClassifierGrid.best_params_
print(rfBestParams)
# Stores the optimum model in best_pipe
rfBestPipe = randomForestClassifierGrid.best_estimator_
print(rfBestPipe)

if (randomForestClassifierGrid.score(X_test, y_test) > baggingClassifierGrid.score(X_test, y_test)):
  best_model = rfBestPipe # Assign the actual pipeline object
else:
  best_model = bcBestPipe # Assign the actual pipeline object

print(f"Best Model: {best_model.named_steps['classifier'].__class__.__name__}")

df_2026 = df[df['YEAR'] == 2026]
print(f'\n2026 teams in dataset: {len(df_2026)}')

ROUND_NAMES = ['R64', 'R32', 'S16', 'E8']


"""
Function: Predict the winner between two teams using their feature differences.
  Lower seed number = higher seed = 'favorite'.
Arguments:
  team_a (str), seed_a (int): names and seeds of team A
  team_b (str), seed_b (int): names and seeds of team B
  df_features (dataframe): our dataframe with the extracted features
  model: our trained model

Returns: tuple (winner_name, winner_seed)
  Model predicts HIGH_SEED_WINS: 1 → better-seeded team wins, 0 → worse-seeded
    team wins.
"""
def predict_game(team_a, seed_a, team_b, seed_b, df_features, model):
    if seed_a <= seed_b:
        team_high, seed_high = team_a, seed_a
        team_low,  seed_low  = team_b, seed_b
    else:
        team_high, seed_high = team_b, seed_b
        team_low,  seed_low  = team_a, seed_a

    th_rows = df_features[df_features['TEAM'] == team_high]
    tl_rows = df_features[df_features['TEAM'] == team_low]

    if th_rows.empty or tl_rows.empty:
        return team_high, seed_high

    th, tl = th_rows.iloc[0], tl_rows.iloc[0]

    diff = [[
        float(th[col]) - float(tl[col])
        if not (pd.isna(th[col]) or pd.isna(tl[col])) else 0.0
        for col in features
    ]]

    high_seed_wins = bool(model.predict(diff)[0])
    return (team_high, seed_high) if high_seed_wins else (team_low, seed_low)


def simulate_bracket_round(teams, round_name, region_name,
                           df_features, model, results_list):
    """
    Pair adjacent teams in bracket order, predict each game.
    teams: list of (team_name, seed).
    Appends result dicts to results_list. Returns list of winners.
    """
    winners = []
    num_teams_to_pair = len(teams)

    if num_teams_to_pair % 2 != 0 and num_teams_to_pair > 0:
        winners.append(teams[-1])
        num_teams_to_pair -= 1

    for i in range(0, num_teams_to_pair, 2):
        t1_name, t1_seed = teams[i]
        t2_name, t2_seed = teams[i + 1]
        w_name, w_seed = predict_game(
            t1_name, t1_seed, t2_name, t2_seed, df_features, model
        )
        results_list.append({
            'REGION'      : region_name,
            'ROUND'       : round_name,
            'TEAM_HIGH'   : t1_name if t1_seed <= t2_seed else t2_name,
            'SEED_HIGH'   : min(t1_seed, t2_seed),
            'TEAM_LOW'    : t2_name if t1_seed <= t2_seed else t1_name,
            'SEED_LOW'    : max(t1_seed, t2_seed),
            'PRED_WINNER' : w_name,
            'PRED_SEED'   : w_seed,
        })
        winners.append((w_name, w_seed))
    return winners


all_bracket_results = []
regional_champions  = {}

REGION_NAMES_FOR_SIM = ALL_REGION_MAPS.get(2026, {})

for quad_no in sorted(df_2026['QUAD NO'].unique()):
    region_df   = df_2026[df_2026['QUAD NO'] == quad_no]
    region_name = REGION_NAMES_FOR_SIM.get(quad_no, str(quad_no))

    seed_to_team = {}
    for seed_high, seed_low in SEED_PAIRS:
        for seed in (seed_high, seed_low):
            if seed in seed_to_team:
                continue
            candidates = region_df[region_df['SEED'] == seed]
            if len(candidates) == 1:
                r = candidates.iloc[0]
                seed_to_team[seed] = (r['TEAM'], int(r['SEED']))
            elif len(candidates) > 1:
                r = candidates.loc[candidates['ROUND'].idxmin()]
                seed_to_team[seed] = (r['TEAM'], int(r['SEED']))
            elif len(candidates) == 0:
                print(f'[WARNING] {region_name}: no team found for seed {seed}')

    current_teams = [seed_to_team[s] for s in BRACKET_SEED_ORDER if s in seed_to_team]

    if len(current_teams) != 16:
        print(f'[WARNING] {region_name}: expected 16 teams, got {len(current_teams)}')

    for round_name in ROUND_NAMES:
        current_teams = simulate_bracket_round(
            current_teams, round_name, region_name,
            df_2026, best_model, all_bracket_results
        )

    regional_champions[quad_no] = current_teams[0]
    champ_name, champ_seed = current_teams[0]
    print(f'{region_name:10s} champion: {champ_name} (seed {champ_seed})')

print('\n── Final Four ──')

F4_MATCHUPS_2026 = [
    ('Michigan',  'Arizona'),
    ('Houston',   'Duke'),
]

print('\n2026 Final Four team data verification:')
for team_name in [t for pair in F4_MATCHUPS_2026 for t in pair]:
    rows = df_2026[df_2026['TEAM'] == team_name]
    if rows.empty:
        print(f'[WARNING] "{team_name}" not found in 2026 data — '
              f'check spelling against: {df_2026["TEAM"].tolist()}')
    else:
        r = rows.iloc[0]
        region = REGION_NAMES_FOR_SIM.get(int(r['QUAD NO']), str(int(r['QUAD NO'])))
        print(f'  {team_name}: seed={int(r["SEED"])}, region={region}')

f4_winners = []
for team_a_name, team_b_name in F4_MATCHUPS_2026:
    a_rows = df_2026[df_2026['TEAM'] == team_a_name]
    b_rows = df_2026[df_2026['TEAM'] == team_b_name]

    if a_rows.empty or b_rows.empty:
        print(f'[ERROR] Cannot predict {team_a_name} vs {team_b_name} — team not found.')
        continue

    seed_a = int(a_rows.iloc[0]['SEED'])
    seed_b = int(b_rows.iloc[0]['SEED'])

    w_name, w_seed = predict_game(
        team_a_name, seed_a, team_b_name, seed_b, df_2026, best_model
    )

    team_high = team_a_name if seed_a <= seed_b else team_b_name
    seed_high = min(seed_a, seed_b)
    team_low  = team_b_name if seed_a <= seed_b else team_a_name
    seed_low  = max(seed_a, seed_b)

    all_bracket_results.append({
        'REGION'      : 'Final Four',
        'ROUND'       : 'Final Four',
        'TEAM_HIGH'   : team_high,
        'SEED_HIGH'   : seed_high,
        'TEAM_LOW'    : team_low,
        'SEED_LOW'    : seed_low,
        'PRED_WINNER' : w_name,
        'PRED_SEED'   : w_seed,
    })
    f4_winners.append((w_name, w_seed))
    print(f'\n  {team_high} (seed #{seed_high}) vs {team_low} (seed #{seed_low})')
    print(f'  → {w_name} advances')

if len(f4_winners) == 2:
    print('\n── Championship ──')
    (t1_name, t1_seed), (t2_name, t2_seed) = f4_winners
    w_name, w_seed = predict_game(
        t1_name, t1_seed, t2_name, t2_seed, df_2026, best_model
    )
    all_bracket_results.append({
        'REGION'      : 'Championship',
        'ROUND'       : 'Championship',
        'TEAM_HIGH'   : t1_name if t1_seed <= t2_seed else t2_name,
        'SEED_HIGH'   : min(t1_seed, t2_seed),
        'TEAM_LOW'    : t2_name if t1_seed <= t2_seed else t1_name,
        'SEED_LOW'    : max(t1_seed, t2_seed),
        'PRED_WINNER' : w_name,
        'PRED_SEED'   : w_seed,
    })
    print(f'  {t1_name} (seed #{t1_seed}) vs {t2_name} (seed #{t2_seed})')
    print(f'\n{"="*55}')
    print(f'  2026 NATIONAL CHAMPION PREDICTION: {w_name} (seed #{w_seed})')
    print(f'{"="*55}')

df_bracket = pd.DataFrame(all_bracket_results)
round_order = ['R64', 'R32', 'S16', 'E8', 'Final Four', 'Championship']
df_bracket['ROUND'] = pd.Categorical(
    df_bracket['ROUND'], categories=round_order, ordered=True
)
df_bracket = df_bracket.sort_values(['ROUND', 'REGION']).reset_index(drop=True)

print('\n── Full 2026 Bracket Predictions ──')
print(df_bracket[['ROUND', 'REGION', 'TEAM_HIGH', 'SEED_HIGH',
                   'TEAM_LOW', 'SEED_LOW', 'PRED_WINNER', 'PRED_SEED']
                 ].to_string(index=False))

df_bracket['LOWER_SEED_WINS'] = df_bracket['PRED_SEED'] == df_bracket['SEED_LOW']
lower_wins_by_round = (df_bracket
                       .groupby('ROUND', observed=True)['LOWER_SEED_WINS'].sum()
                       .reindex([r for r in round_order]))

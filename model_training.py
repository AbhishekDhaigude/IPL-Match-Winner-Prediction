import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
import warnings
warnings.filterwarnings('ignore')

# Verify scikit-learn version
import sklearn
print(f"Using scikit-learn version: {sklearn.__version__}")

# Load preprocessed data
print("Loading preprocessed data...")
batsman_full = pd.read_csv('processed_batsman_stats.csv')
bowler_full = pd.read_csv('processed_bowler_stats.csv')
venue_stats = pd.read_csv('processed_venue_stats.csv')
team_stats = pd.read_csv('processed_team_stats.csv')
h2h_stats = pd.read_csv('processed_head_to_head_stats.csv')
player_roles = pd.read_csv('processed_player_roles.csv')
match_results = pd.read_csv('processed_match_results.csv')  # Already preprocessed

# Merge batsman and bowler data with roles
batsman_full = batsman_full.merge(player_roles, on='player_name', how='left')
bowler_full = bowler_full.merge(player_roles, on='player_name', how='left')

# Combine batsman and bowler data for all-rounders, including venue and team info
player_data = pd.concat([
    batsman_full[['match_id', 'player_name', 'role', 'dream11_points', 'recent_avg_runs', 'most_recent_strike_rate', 'venue', 'player_team', 'opposing_team']],
    bowler_full[['match_id', 'player_name', 'role', 'dream11_points', 'recent_avg_wickets', 'recent_economy', 'venue', 'player_team', 'opposing_team']]
]).drop_duplicates(subset=['match_id', 'player_name'], keep='last')

# Prepare features and target for player points prediction
features = [
    'recent_avg_runs', 'most_recent_strike_rate', 'recent_avg_wickets', 'recent_economy',
    'avg_first_innings_score', 'avg_second_innings_score', 'team_win_rate',
    'opponent_win_rate', 'team_h2h_win_rate'
]

# Merge with venue_stats, team_stats, and h2h_stats
X = player_data.merge(venue_stats, on='venue', how='left').merge(team_stats.rename(columns={'win_rate': 'team_win_rate'}),
    left_on='player_team', right_on='team', how='left').merge(team_stats.rename(columns={'win_rate': 'opponent_win_rate'}), 
    left_on='opposing_team', right_on='team', how='left', 
    suffixes=('', '_opponent')).merge(h2h_stats.rename(columns={'win_rate': 'team_h2h_win_rate'}), 
    left_on=['player_team', 'opposing_team'], right_on=['team', 'opponent'], how='left')[features].fillna(0)
y = player_data['dream11_points'].fillna(0)

# Reset indices to ensure alignment
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Create train_mask after merges
train_mask = X.index.map(lambda i: int(str(player_data['match_id'].iloc[i])[:4]) < 2023)

# Split data into training (2008-2022) and testing (2023)
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

# Train models for player points prediction
print("Training player points prediction models...")

# XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_error = np.sqrt(mean_squared_error(y_test, xgb_pred))
print(f"XGBoost RMSE: {xgb_error}")

# CatBoost
cat_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_seed=42, verbose=0)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
cat_error = np.sqrt(mean_squared_error(y_test, cat_pred))
print(f"CatBoost RMSE: {cat_error}")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_error = np.sqrt(mean_squared_error(y_test, rf_pred))
print(f"Random Forest RMSE: {rf_error}")

# Ensemble Model
ensemble_pred = (0.5 * xgb_pred + 0.25 * cat_pred + 0.25 * rf_pred)
ensemble_error = np.sqrt(mean_squared_error(y_test, ensemble_pred))
print(f"Ensemble RMSE: {ensemble_error}")

# Save player points prediction models
import joblib
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(cat_model, 'cat_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')

# ===== WINNING PERCENTAGE PREDICTION MODEL =====
print("\nPreparing data for match win prediction...")

# Prepare features and target for match win prediction using match_results
match_results = match_results.merge(venue_stats, on='venue', how='left').merge(
    team_stats.rename(columns={'win_rate': 'team1_win_rate'}), 
    left_on='team1', right_on='team', how='left'
).merge(
    team_stats.rename(columns={'win_rate': 'team2_win_rate'}), 
    left_on='team2', right_on='team', how='left', suffixes=('', '_team2')
).merge(
    h2h_stats.rename(columns={'win_rate': 'h2h_win_rate'}), 
    left_on=['team1', 'team2'], right_on=['team', 'opponent'], how='left'
)

# Fill missing values with defaults
match_results = match_results.fillna({
    'avg_first_innings_score': match_results['avg_first_innings_score'].mean(),
    'avg_second_innings_score': match_results['avg_second_innings_score'].mean(),
    'team1_win_rate': 0.5,
    'team2_win_rate': 0.5,
    'h2h_win_rate': 0.5
})

# Features and target
X_win = match_results[['avg_first_innings_score', 'avg_second_innings_score', 'team1_win_rate', 'team2_win_rate', 'h2h_win_rate']]
y_win = match_results['team1_won']

# Split data based on date (training: 2008-2022, testing: 2023)
match_results['year'] = pd.to_datetime(match_results['date']).dt.year
train_mask_win = match_results['year'] < 2023
X_win_train, X_win_test = X_win[train_mask_win], X_win[~train_mask_win]
y_win_train, y_win_test = y_win[train_mask_win], y_win[~train_mask_win]

# Train models for match win prediction
print("Training match win prediction models...")

# XGBoost for win prediction
xgb_win_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
xgb_win_model.fit(X_win_train, y_win_train)
xgb_win_pred = xgb_win_model.predict(X_win_test)
xgb_win_acc = accuracy_score(y_win_test, xgb_win_pred)
print(f"XGBoost Win Prediction Accuracy: {xgb_win_acc:.4f}")

# Random Forest for win prediction
rf_win_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf_win_model.fit(X_win_train, y_win_train)
rf_win_pred = rf_win_model.predict(X_win_test)
rf_win_acc = accuracy_score(y_win_test, rf_win_pred)
print(f"Random Forest Win Prediction Accuracy: {rf_win_acc:.4f}")

# Save win prediction models
joblib.dump(xgb_win_model, 'xgb_win_model.pkl')
joblib.dump(rf_win_model, 'rf_win_model.pkl')

# Function to assign credit points based on player performance
def assign_player_credits(player_name, predicted_points, role):
    base_credit = 8.0
    points_factor = min(max(0.5, predicted_points / 30), 2.5)
    role_factors = {
        'Batsman': 1.0,
        'Bowler': 0.95,
        'All-Rounder': 1.1,
        'Wicket-Keeper': 0.9
    }
    role_factor = role_factors.get(role, 1.0)
    credit = base_credit * points_factor * role_factor
    credit = min(max(8.0, credit), 10.5)
    return round(credit, 1)

# Function to fetch playing XI (placeholder)
def fetch_playing_xi(team1, team2):
    return {team1: [], team2: []}

# Team selection using Linear Programming
def select_dream11_team(team1, team2, model):
    match_players = player_data[player_data['player_team'].isin([team1, team2])].copy()
    
    print(f"Players available from {team1} and {team2}: {len(match_players)}")
    print(f"Role distribution: {match_players['role'].value_counts().to_dict()}")
    
    if len(match_players) < 11:
        print("Insufficient players to form a team of 11. Returning available players.")
        return match_players[['player_name', 'role', 'player_team', 'dream11_points']]
    
    # Predict points
    match_features = match_players.merge(venue_stats, on='venue', how='left').merge(
        team_stats.rename(columns={'win_rate': 'team_win_rate'}), 
        left_on='player_team', right_on='team', how='left'
    ).merge(
        team_stats.rename(columns={'win_rate': 'opponent_win_rate'}), 
        left_on='opposing_team', right_on='team', how='left', suffixes=('', '_opponent')
    ).merge(
        h2h_stats.rename(columns={'win_rate': 'team_h2h_win_rate'}), 
        left_on=['player_team', 'opposing_team'], right_on=['team', 'opponent'], how='left'
    )[features].fillna(0)
    match_players['predicted_points'] = model.predict(match_features)
    
    # Assign credits
    match_players['credits'] = match_players.apply(lambda x: assign_player_credits(
        x['player_name'], x['predicted_points'], x['role']), axis=1)
    
    # Linear Programming problem
    prob = LpProblem("Dream11_Team_Selection", LpMaximize)
    player_vars = {row['player_name']: LpVariable(f"p_{i}", cat='Binary') 
                  for i, row in match_players.iterrows()}
    captain_var = {row['player_name']: LpVariable(f"c_{i}", cat='Binary') 
                  for i, row in match_players.iterrows()}
    vice_captain_var = {row['player_name']: LpVariable(f"vc_{i}", cat='Binary') 
                      for i, row in match_players.iterrows()}
    
    prob += lpSum([
        (player_vars[p] * match_players.loc[match_players['player_name'] == p, 'predicted_points'].values[0] +
         captain_var[p] * match_players.loc[match_players['player_name'] == p, 'predicted_points'].values[0] +
         vice_captain_var[p] * match_players.loc[match_players['player_name'] == p, 'predicted_points'].values[0] * 0.5)
        for p in match_players['player_name']
    ])
    
    prob += lpSum(player_vars.values()) == 11
    prob += lpSum([player_vars[p] * match_players.loc[match_players['player_name'] == p, 'credits'].values[0] 
                  for p in match_players['player_name']]) <= 100
    prob += lpSum(captain_var.values()) == 1
    prob += lpSum(vice_captain_var.values()) == 1
    for p in match_players['player_name']:
        prob += captain_var[p] <= player_vars[p]
        prob += vice_captain_var[p] <= player_vars[p]
        prob += captain_var[p] + vice_captain_var[p] <= 1
    
    # Group by role and get player names instead of indices
    roles = match_players.groupby('role')['player_name'].apply(list).to_dict()

    # Role constraints with checks
    if 'Batsman' in roles and len(roles['Batsman']) >= 3:
        prob += lpSum([player_vars[p] for p in roles['Batsman']]) >= 3
        prob += lpSum([player_vars[p] for p in roles['Batsman']]) <= 6
    else:
        print("Warning: Insufficient Batsmen. Relaxing constraint.")
        prob += lpSum([player_vars[p] for p in roles.get('Batsman', [])]) >= min(3, len(roles.get('Batsman', [])))

    if 'Bowler' in roles and len(roles['Bowler']) >= 3:
        prob += lpSum([player_vars[p] for p in roles['Bowler']]) >= 3
        prob += lpSum([player_vars[p] for p in roles['Bowler']]) <= 6
    else:
        print("Warning: Insufficient Bowlers. Relaxing constraint.")
        prob += lpSum([player_vars[p] for p in roles.get('Bowler', [])]) >= min(3, len(roles.get('Bowler', [])))

    if 'All-Rounder' in roles and len(roles['All-Rounder']) >= 1:
        prob += lpSum([player_vars[p] for p in roles['All-Rounder']]) >= 1
        prob += lpSum([player_vars[p] for p in roles['All-Rounder']]) <= 4
    else:
        print("Warning: Insufficient All-Rounders. Relaxing constraint.")
        prob += lpSum([player_vars[p] for p in roles.get('All-Rounder', [])]) >= min(1, len(roles.get('All-Rounder', [])))

    if 'Wicket-Keeper' in roles and len(roles['Wicket-Keeper']) >= 1:
        prob += lpSum([player_vars[p] for p in roles['Wicket-Keeper']]) >= 1
        prob += lpSum([player_vars[p] for p in roles['Wicket-Keeper']]) <= 4
    else:
        print("Warning: Insufficient Wicket-Keepers. Relaxing constraint.")
        prob += lpSum([player_vars[p] for p in roles.get('Wicket-Keeper', [])]) >= min(1, len(roles.get('Wicket-Keeper', [])))
        
    team1_players = match_players[match_players['player_team'] == team1]['player_name']
    team2_players = match_players[match_players['player_team'] == team2]['player_name']
    prob += lpSum([player_vars[p] for p in team1_players]) <= 7
    prob += lpSum([player_vars[p] for p in team2_players]) <= 7
    
    status = prob.solve()
    print(f"Status: {LpStatus[status]}")
    
    selected_team = []
    for p in match_players['player_name']:
        if player_vars[p].varValue == 1:
            role = 'Player'
            if captain_var[p].varValue == 1:
                role = 'Captain'
            elif vice_captain_var[p].varValue == 1:
                role = 'Vice-Captain'
            selected_team.append({
                'player_name': p,
                'role': match_players.loc[match_players['player_name'] == p, 'role'].values[0],
                'team_role': role,
                'team': match_players.loc[match_players['player_name'] == p, 'player_team'].values[0],
                'predicted_points': match_players.loc[match_players['player_name'] == p, 'predicted_points'].values[0],
                'credits': match_players.loc[match_players['player_name'] == p, 'credits'].values[0]
            })
    
    return pd.DataFrame(selected_team)

# Function to predict match winning percentages
def predict_match_result(team1, team2, venue):
    venue_row = venue_stats[venue_stats['venue'] == venue].iloc[0] if len(venue_stats[venue_stats['venue'] == venue]) > 0 else pd.Series({
        'avg_first_innings_score': venue_stats['avg_first_innings_score'].mean(),
        'avg_second_innings_score': venue_stats['avg_second_innings_score'].mean()
    })
    team1_row = team_stats[team_stats['team'] == team1].iloc[0] if len(team_stats[team_stats['team'] == team1]) > 0 else pd.Series({
        'win_rate': 0.5
    })
    team2_row = team_stats[team_stats['team'] == team2].iloc[0] if len(team_stats[team_stats['team'] == team2]) > 0 else pd.Series({
        'win_rate': 0.5
    })
    h2h_row = h2h_stats[(h2h_stats['team'] == team1) & (h2h_stats['opponent'] == team2)].iloc[0] if len(h2h_stats[(h2h_stats['team'] == team1) & (h2h_stats['opponent'] == team2)]) > 0 else pd.Series({
        'win_rate': 0.5
    })
    
    X_pred = pd.DataFrame({
        'avg_first_innings_score': [venue_row['avg_first_innings_score']],
        'avg_second_innings_score': [venue_row['avg_second_innings_score']],
        'team1_win_rate': [team1_row['win_rate']],
        'team2_win_rate': [team2_row['win_rate']],
        'h2h_win_rate': [h2h_row['win_rate']]
    })
    
    xgb_win_prob = xgb_win_model.predict_proba(X_pred)[0][1]
    rf_win_prob = rf_win_model.predict_proba(X_pred)[0][1]
    team1_win_prob = (xgb_win_prob + rf_win_prob) / 2
    team2_win_prob = 1 - team1_win_prob
    
    return {team1: team1_win_prob * 100, team2: team2_win_prob * 100}

if __name__ == "__main__":
    team1 = "Mumbai Indians"
    team2 = "Chennai Super Kings"
    venue = "Wankhede Stadium"
    
    playing_xi = fetch_playing_xi(team1, team2)
    team = select_dream11_team(team1, team2, xgb_model)
    print(team)
    
    win_probs = predict_match_result(team1, team2, venue)
    print(f"\nMatch Win Prediction:")
    print(f"{team1}: {win_probs[team1]:.1f}%")
    print(f"{team2}: {win_probs[team2]:.1f}%")
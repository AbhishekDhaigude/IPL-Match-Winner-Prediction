import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
print("Loading datasets...")
matches_df = pd.read_csv('IPL_Matches_2008_2023.csv')
deliveries_df = pd.read_csv('deliveries_updated_mens_ipl.csv')
ball_by_ball_df = pd.read_csv('IPL_ball_by_ball_2008_2023.csv')

# Standardize column names
matches_df = matches_df.rename(columns={'matchId': 'match_id'})
deliveries_df = deliveries_df.rename(columns={'matchId': 'match_id'})
ball_by_ball_df = ball_by_ball_df.rename(columns={'match_id': 'match_id'})

# Convert date columns to datetime
def convert_dates(df, date_columns):
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                print(f"Could not convert {col} to datetime")
    return df

matches_df = convert_dates(matches_df, ['date', 'date1', 'date2'])
deliveries_df = convert_dates(deliveries_df, ['date'])
ball_by_ball_df = convert_dates(ball_by_ball_df, ['start_date'])

# Handle missing values in matches_df
matches_df = matches_df.fillna({
    'winner': 'no result',
    'winner_runs': 0,
    'winner_wickets': 0,
    'city': 'unknown',
    'player_of_match': 'unknown',
    'venue': 'unknown'
})

# Function to create batsman statistics
def create_batsman_stats(deliveries, ball_by_ball):
    batsman_stats = ball_by_ball.groupby(['match_id', 'striker']).agg({
        'runs_off_bat': 'sum',
        'ball': 'count'
    }).reset_index().rename(columns={'striker': 'player_name', 'runs_off_bat': 'runs', 'ball': 'balls_faced'})
    
    # Calculate strike rate
    batsman_stats['strike_rate'] = (batsman_stats['runs'] / batsman_stats['balls_faced']) * 100
    batsman_stats['strike_rate'] = batsman_stats['strike_rate'].fillna(0)
    
    return batsman_stats

# Function to create bowler statistics
def create_bowler_stats(deliveries, ball_by_ball):
    bowler_stats = ball_by_ball.groupby(['match_id', 'bowler']).agg({
        'runs_off_bat': 'sum',
        'extras': 'sum',
        'ball': 'count',
        'wicket_type': lambda x: x.notna().sum()
    }).reset_index().rename(columns={'wicket_type': 'wickets'})
    
    # Calculate total runs conceded
    bowler_stats['runs_conceded'] = bowler_stats['runs_off_bat'] + bowler_stats['extras']
    bowler_stats['overs'] = bowler_stats['ball'] / 6
    bowler_stats['economy'] = bowler_stats['runs_conceded'] / bowler_stats['overs']
    bowler_stats['economy'] = bowler_stats['economy'].fillna(0)
    
    return bowler_stats.drop(columns=['runs_off_bat', 'extras'])

# Function to classify player roles
def classify_player_roles(batsman_stats, bowler_stats):
    all_players = set(batsman_stats['player_name']).union(set(bowler_stats['bowler']))
    
    batsman_avg = batsman_stats.groupby('player_name')['runs'].mean().reset_index()
    bowler_avg = bowler_stats.groupby('bowler')['wickets'].mean().reset_index()
    
    player_roles = []
    for player in all_players:
        batting_avg = batsman_avg[batsman_avg['player_name'] == player]['runs'].values
        bowling_avg = bowler_avg[bowler_avg['bowler'] == player]['wickets'].values
        
        bat_avg = batting_avg[0] if len(batting_avg) > 0 else 0
        bowl_avg = bowling_avg[0] if len(bowling_avg) > 0 else 0
        
        if bat_avg > 20 and bowl_avg > 1:
            role = 'All-Rounder'
        elif bat_avg > 20:
            role = 'Batsman'
        elif bowl_avg > 1:
            role = 'Bowler'
        else:
            role = 'Batsman'
        
        player_roles.append({'player_name': player, 'role': role})
    
    player_roles_df = pd.DataFrame(player_roles)
    wicketkeepers = [
        'MS Dhoni', 'Dinesh Karthik', 'Rishabh Pant', 'Wriddhiman Saha',
        'KL Rahul', 'Jos Buttler', 'Quinton de Kock', 'Ishan Kishan',
        'Sanju Samson', 'Jonny Bairstow', 'Nicholas Pooran'
    ]
    player_roles_df.loc[player_roles_df['player_name'].isin(wicketkeepers), 'role'] = 'Wicket-Keeper'
    
    return player_roles_df

# Function to create venue statistics
def create_venue_stats(matches_df, ball_by_ball_df):
    venue_scores = []
    for venue in matches_df['venue'].unique():
        venue_matches = matches_df[matches_df['venue'] == venue]['match_id'].unique()
        venue_innings = ball_by_ball_df[ball_by_ball_df['match_id'].isin(venue_matches)]
        if not venue_innings.empty:
            avg_score = venue_innings.groupby(['match_id', 'innings']).agg({
                'runs_off_bat': 'sum',
                'extras': 'sum'
            }).reset_index()
            avg_score['total'] = avg_score['runs_off_bat'] + avg_score['extras']
            avg_first_innings = avg_score[avg_score['innings'] == 1]['total'].mean()
            avg_second_innings = avg_score[avg_score['innings'] == 2]['total'].mean()
        else:
            avg_first_innings = np.nan
            avg_second_innings = np.nan
        venue_scores.append({
            'venue': venue,
            'avg_first_innings_score': avg_first_innings,
            'avg_second_innings_score': avg_second_innings
        })
    
    venue_stats_df = pd.DataFrame(venue_scores)
    venue_stats_df['avg_first_innings_score'] = venue_stats_df['avg_first_innings_score'].fillna(
        venue_stats_df['avg_first_innings_score'].mean())
    venue_stats_df['avg_second_innings_score'] = venue_stats_df['avg_second_innings_score'].fillna(
        venue_stats_df['avg_second_innings_score'].mean())
    
    return venue_stats_df

# Function to create team statistics
def create_team_stats(matches_df):
    teams = set(matches_df['team1'].dropna()).union(set(matches_df['team2'].dropna()))
    team_stats = []
    for team in teams:
        team1_matches = matches_df[matches_df['team1'] == team]
        team2_matches = matches_df[matches_df['team2'] == team]
        total_matches = len(team1_matches) + len(team2_matches)
        wins = len(team1_matches[team1_matches['winner'] == team]) + len(team2_matches[team2_matches['winner'] == team])
        win_rate = wins / total_matches if total_matches > 0 else 0
        team_stats.append({'team': team, 'total_matches': total_matches, 'total_wins': wins, 'win_rate': win_rate})
    
    return pd.DataFrame(team_stats)

# Function to create head-to-head statistics
def create_head_to_head(matches_df):
    head_to_head = []
    for _, match in matches_df.iterrows():
        team1, team2, winner = match['team1'], match['team2'], match['winner']
        if pd.isna(team1) or pd.isna(team2) or winner == 'no result':
            continue
        head_to_head.append({'team': team1, 'opponent': team2, 'result': 'win' if winner == team1 else 'loss'})
        head_to_head.append({'team': team2, 'opponent': team1, 'result': 'win' if winner == team2 else 'loss'})
    
    h2h_df = pd.DataFrame(head_to_head)
    h2h_stats = h2h_df.groupby(['team', 'opponent']).agg({'result': lambda x: (x == 'win').mean()}).reset_index()
    h2h_stats.rename(columns={'result': 'win_rate'}, inplace=True)
    return h2h_stats

# Function to create match results dataset
def create_match_results(matches_df):
    match_results = matches_df[['match_id', 'team1', 'team2', 'winner', 'venue', 'date', 'season']].copy()
    match_results['team1_won'] = (match_results['winner'] == match_results['team1']).astype(int)
    match_results = match_results[match_results['winner'] != 'no result']  # Exclude no-result matches
    return match_results[['match_id', 'team1', 'team2', 'winner', 'venue', 'date', 'season', 'team1_won']]

# Function to calculate player form
def calculate_player_form(batsman_stats, bowler_stats, matches_df):
    match_order = {row['match_id']: i for i, row in matches_df.sort_values('date').iterrows()}
    batsman_stats['match_order'] = batsman_stats['match_id'].map(match_order)
    bowler_stats['match_order'] = bowler_stats['match_id'].map(match_order)
    
    batsman_form = batsman_stats.sort_values(['player_name', 'match_order']).groupby('player_name').apply(
        lambda x: x.assign(recent_avg_runs=x['runs'].rolling(window=5, min_periods=1).mean(),most_recent_strike_rate=x['strike_rate'].rolling(window=5, min_periods=1).mean())).reset_index(drop=True)
    
    bowler_form = bowler_stats.sort_values(['bowler', 'match_order']).groupby('bowler').apply(
        lambda x: x.assign(
            recent_avg_wickets=x['wickets'].rolling(window=5, min_periods=1).mean(),
            recent_economy=x['economy'].rolling(window=5, min_periods=1).mean()
        )
    ).reset_index(drop=True).rename(columns={'bowler': 'player_name'})
    
    return batsman_form, bowler_form

# Function to calculate Dream11 points
def calculate_dream11_points(batsman_stats, bowler_stats):
    batsman_stats['batting_points'] = (
        batsman_stats['runs'] +
        (batsman_stats['runs'] >= 30).astype(int) * 4 +
        (batsman_stats['runs'] >= 50).astype(int) * 8 +
        (batsman_stats['runs'] >= 100).astype(int) * 16
    )
    batsman_stats['sr_bonus'] = 0
    mask = batsman_stats['balls_faced'] >= 10
    batsman_stats.loc[mask & (batsman_stats['strike_rate'] >= 170), 'sr_bonus'] = 6
    batsman_stats.loc[mask & (batsman_stats['strike_rate'] >= 150) & (batsman_stats['strike_rate'] < 170), 'sr_bonus'] = 4
    batsman_stats.loc[mask & (batsman_stats['strike_rate'] >= 130) & (batsman_stats['strike_rate'] < 150), 'sr_bonus'] = 2
    batsman_stats.loc[mask & (batsman_stats['strike_rate'] >= 110) & (batsman_stats['strike_rate'] < 130), 'sr_bonus'] = 1
    batsman_stats.loc[mask & (batsman_stats['strike_rate'] < 70), 'sr_bonus'] = -2
    
    bowler_stats['bowling_points'] = (
        bowler_stats['wickets'] * 25 +
        (bowler_stats['wickets'] >= 3).astype(int) * 8 +
        (bowler_stats['wickets'] >= 4).astype(int) * 4 +
        (bowler_stats['wickets'] >= 5).astype(int) * 4
    )
    bowler_stats['eco_bonus'] = 0
    mask = bowler_stats['overs'] >= 2
    bowler_stats.loc[mask & (bowler_stats['economy'] < 5), 'eco_bonus'] = 6
    bowler_stats.loc[mask & (bowler_stats['economy'] >= 5) & (bowler_stats['economy'] < 6), 'eco_bonus'] = 4
    bowler_stats.loc[mask & (bowler_stats['economy'] >= 6) & (bowler_stats['economy'] < 7), 'eco_bonus'] = 2
    bowler_stats.loc[mask & (bowler_stats['economy'] >= 7) & (bowler_stats['economy'] < 8), 'eco_bonus'] = 1
    bowler_stats.loc[mask & (bowler_stats['economy'] >= 10) & (bowler_stats['economy'] < 11), 'eco_bonus'] = -1
    bowler_stats.loc[mask & (bowler_stats['economy'] >= 11) & (bowler_stats['economy'] < 12), 'eco_bonus'] = -2
    bowler_stats.loc[mask & (bowler_stats['economy'] >= 12), 'eco_bonus'] = -4
    
    batsman_stats['dream11_points'] = batsman_stats['batting_points'] + batsman_stats['sr_bonus']
    bowler_stats['dream11_points'] = bowler_stats['bowling_points'] + bowler_stats['eco_bonus']
    
    return batsman_stats, bowler_stats

# Function to assign player teams
def assign_player_teams(batsman_stats, bowler_stats, matches_df, ball_by_ball_df):
    player_team = ball_by_ball_df.groupby(['match_id', 'batting_team']).agg({
        'striker': lambda x: list(set(x)),
        'bowler': lambda x: list(set(x))
    }).reset_index()
    
    player_team = player_team.melt(id_vars=['match_id', 'batting_team'], 
                                  value_vars=['striker', 'bowler'], 
                                  var_name='role', 
                                  value_name='player_name').explode('player_name')
    
    player_team = player_team.drop(columns=['role']).drop_duplicates()
    player_team = player_team.merge(matches_df[['match_id', 'team1', 'team2']], on='match_id', how='left')
    player_team['player_team'] = player_team['batting_team']
    player_team['opposing_team'] = player_team.apply(
        lambda row: row['team2'] if row['player_team'] == row['team1'] else row['team1'], axis=1)
    
    batsman_full = batsman_stats.merge(player_team[['match_id', 'player_name', 'player_team', 'opposing_team']], 
                                      on=['match_id', 'player_name'], how='left')
    bowler_full = bowler_stats.merge(player_team[['match_id', 'player_name', 'player_team', 'opposing_team']], 
                                    on=['match_id', 'player_name'], how='left')
    
    return batsman_full, bowler_full

# Execute preprocessing pipeline
print("Starting preprocessing pipeline...")

batsman_stats = create_batsman_stats(deliveries_df, ball_by_ball_df)
bowler_stats = create_bowler_stats(deliveries_df, ball_by_ball_df)
player_roles = classify_player_roles(batsman_stats, bowler_stats)
venue_stats = create_venue_stats(matches_df, ball_by_ball_df)
team_stats = create_team_stats(matches_df)
h2h_stats = create_head_to_head(matches_df)
match_results = create_match_results(matches_df)  # New function for match results
batsman_form, bowler_form = calculate_player_form(batsman_stats, bowler_stats, matches_df)
batsman_points, bowler_points = calculate_dream11_points(batsman_form, bowler_form)
batsman_full, bowler_full = assign_player_teams(batsman_points, bowler_points, matches_df, ball_by_ball_df)

# Add venue to player data
batsman_full = batsman_full.merge(matches_df[['match_id', 'venue']], on='match_id', how='left')
bowler_full = bowler_full.merge(matches_df[['match_id', 'venue']], on='match_id', how='left')

# Save processed data
print("Saving processed data...")
batsman_full.to_csv('processed_batsman_stats.csv', index=False)
bowler_full.to_csv('processed_bowler_stats.csv', index=False)
venue_stats.to_csv('processed_venue_stats.csv', index=False)
team_stats.to_csv('processed_team_stats.csv', index=False)
h2h_stats.to_csv('processed_head_to_head_stats.csv', index=False)
player_roles.to_csv('processed_player_roles.csv', index=False)
match_results.to_csv('processed_match_results.csv', index=False)  # Save new file

print("Preprocessing complete!")
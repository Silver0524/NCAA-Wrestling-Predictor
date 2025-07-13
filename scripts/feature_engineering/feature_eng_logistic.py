import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def duration_to_seconds(duration_str):
    try:
        minutes, seconds = map(int, duration_str.split(":"))
        return minutes * 60 + seconds
    except:
        return None

def double_matches(df):
    df_b = df.copy()
    df_b['wrestler_id'] = df['opponent_id']
    df_b['opponent_id'] = df['wrestler_id']
    df_b['wrestler'] = df['opponent']
    df_b['opponent'] = df['wrestler']
    df_b['wrestler_school'] = df['opponent_school']
    df_b['opponent_school'] = df['wrestler_school']
    df_b['wrestler_score'] = df['opponent_score']
    df_b['opponent_score'] = df['wrestler_score']
    df_b['result'] = np.where(df['result'] == 'W', 'L', 'W')

    result = pd.concat([df, df_b])
    return result.sort_values(['date', 'wrestler_id']).reset_index()

def create_base_features(df):
    df[['wrestler_score', 'opponent_score']] = df['score'].str.split(' - ', expand=True)
    df.loc[df['result_type'] == 'FALL', ['wrestler_score', 'opponent_score']] = 0
    df = df.drop(['score'], axis=1)
    
    df['is_dual_meet'] = [1 if x == 'Dual' else 0 for x in df['event']]

    df = df[~df['result_type'].isin(['MFOR', 'INJ', 'CMFF', 'DQ', 'DEF', 'FOR'])].copy()

    # Convert to seconds
    df["duration_seconds"] = df["match_duration"].apply(duration_to_seconds)
    # Create is_overtime column (True if > 7 minutes)
    df["is_overtime"] = (df["duration_seconds"] > 420).astype(int)
    df = df.drop('match_duration', axis=1)

    df['wrestler_score'] = pd.to_numeric(df['wrestler_score'], errors='coerce')
    df['opponent_score'] = pd.to_numeric(df['opponent_score'], errors='coerce')

    df = double_matches(df)

    df['is_win'] = (df['result'] == 'W').astype(int)
    df['point_differential'] = df['wrestler_score'] - df['opponent_score']

def calculate_win_rate_last_5(df):
    """Calculate win rate for last 5 matches for each wrestler"""
    df = df.sort_values(['wrestler_id', 'date'])
    
    # Calculate rolling win rate for last 5 matches
    df['win_rate_last_5'] = df.groupby('wrestler_id')['is_win'].rolling(
        window=5, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    # Shift by 1 to avoid data leakage (use data up to previous match)
    df['win_rate_last_5'] = df.groupby('wrestler_id')['win_rate_last_5'].shift(1)
    
    return df

def calculate_opponent_career_win_rate(df):
    """Calculate opponent's career win rate at time of match"""
    df = df.sort_values(['wrestler_id', 'date'])
    
    # Calculate cumulative win rate for each wrestler
    df['matches_wrestled'] = df.groupby('wrestler_id').cumcount() + 1
    df['cumulative_wins'] = df.groupby('wrestler_id')['is_win'].cumsum()
    df['career_win_rate'] = df['cumulative_wins'] / df['matches_wrestled']
    
    # Shift to get career win rate before current match
    df['career_win_rate'] = df.groupby('wrestler_id')['career_win_rate'].shift(1)
    
    # Create a mapping of wrestler career win rates by match date
    wrestler_rates = df.groupby(['wrestler_id', 'date'])['career_win_rate'].first().reset_index()
    
    # Map opponent win rates
    df = df.merge(
        wrestler_rates.rename(columns={'wrestler_id': 'opponent_id', 'career_win_rate': 'opponent_career_win_rate'}),
        on=['opponent_id', 'date'],
        how='left'
    )
    
    return df

def calculate_form_differential_5(df):
    """Calculate form differential (own form - opponent form) over last 5 matches"""
    df = df.sort_values(['wrestler_id', 'date'])
    
    # Calculate form (win rate) for last 5 matches
    df['form_last_5'] = df.groupby('wrestler_id')['is_win'].rolling(
        window=5, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    # Shift to avoid data leakage
    df['form_last_5'] = df.groupby('wrestler_id')['form_last_5'].shift(1)
    
    # Create opponent form mapping
    form_mapping = df.groupby(['wrestler_id', 'date'])['form_last_5'].first().reset_index()
    
    # Map opponent form
    df = df.merge(
        form_mapping.rename(columns={'wrestler_id': 'opponent_id', 'form_last_5': 'opponent_form_last_5'}),
        on=['opponent_id', 'date'],
        how='left'
    )
    
    # Calculate form differential
    df['form_differential_5'] = df['form_last_5'] - df['opponent_form_last_5']
    
    return df

def calculate_avg_point_differential_last_5(df):
    """Calculate average point differential over last 5 matches"""
    df = df.sort_values(['wrestler_id', 'date'])
    
    # Calculate rolling average point differential
    df['avg_point_differential_last_5'] = df.groupby('wrestler_id')['point_differential'].rolling(
        window=5, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    # Shift to avoid data leakage
    df['avg_point_differential_last_5'] = df.groupby('wrestler_id')['avg_point_differential_last_5'].shift(1)
    
    return df

def calculate_h2h_win_rate(df):
    """Calculate head-to-head win rate against specific opponent"""
    df = df.sort_values(['wrestler_id', 'date'])
    
    # Create head-to-head combinations
    df['h2h_key'] = df.apply(lambda x: tuple(sorted([x['wrestler_id'], x['opponent_id']])), axis=1)
    
    # Calculate cumulative H2H record
    df['h2h_matches'] = df.groupby(['wrestler_id', 'opponent_id']).cumcount()
    df['h2h_wins'] = df.groupby(['wrestler_id', 'opponent_id'])['is_win'].cumsum()
    
    # Calculate H2H win rate (shift to avoid data leakage)
    df['h2h_win_rate'] = df['h2h_wins'] / (df['h2h_matches'] + 1)
    df['h2h_win_rate'] = df.groupby(['wrestler_id', 'opponent_id'])['h2h_win_rate'].shift(1)
    
    # Fill NaN with 0.5 (no previous history)
    df['h2h_win_rate'] = df['h2h_win_rate'].fillna(0.5)
    
    return df

def calculate_experience_differential(df):
    """Calculate experience differential (total matches played difference)"""
    df = df.sort_values(['wrestler_id', 'date'])

    # Shift to get experience before current match
    df['experience'] = df.groupby('wrestler_id')['matches_wrestled'].shift(1).fillna(0)
    
    # Map opponent experience
    exp_mapping = df.groupby(['wrestler_id', 'date'])['experience'].first().reset_index()
    
    df = df.merge(
        exp_mapping.rename(columns={'wrestler_id': 'opponent_id', 'experience': 'opponent_experience'}),
        on=['opponent_id', 'date'],
        how='left'
    )
    
    # Calculate experience differential
    df['experience_differential'] = df['experience'] - df['opponent_experience']
    
    return df

def calculate_scoring_advantage_5(df):
    """Calculate scoring advantage (avg points scored - league avg) over last 5 matches"""
    df = df.sort_values(['wrestler_id', 'date'])
    
    # Calculate league average points scored by date
    df['league_avg_points'] = df.groupby('date')['wrestler_score'].transform('mean')
    
    # Calculate rolling average points scored
    df['avg_points_scored_5'] = df.groupby('wrestler_id')['wrestler_score'].rolling(
        window=5, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    # Shift to avoid data leakage
    df['avg_points_scored_5'] = df.groupby('wrestler_id')['avg_points_scored_5'].shift(1)
    
    # Calculate scoring advantage
    df['scoring_advantage_5'] = df['avg_points_scored_5'] - df['league_avg_points']
    
    return df

def calculate_defensive_advantage_5(df):
    """Calculate defensive advantage (league avg points allowed - avg points allowed) over last 5 matches"""
    df = df.sort_values(['wrestler_id', 'date'])
    
    # Calculate league average points allowed by date
    df['league_avg_points_allowed'] = df.groupby('date')['opponent_score'].transform('mean')
    
    # Calculate rolling average points allowed
    df['avg_points_allowed_5'] = df.groupby('wrestler_id')['opponent_score'].rolling(
        window=5, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    # Shift to avoid data leakage
    df['avg_points_allowed_5'] = df.groupby('wrestler_id')['avg_points_allowed_5'].shift(1)
    
    # Calculate defensive advantage (lower points allowed = better defense)
    df['defensive_advantage_5'] = df['league_avg_points_allowed'] - df['avg_points_allowed_5']
    
    return df

def calculate_dominant_win_rate_last_5(df, dominant_threshold=8):
    """Calculate dominant win rate (wins by significant margin) over last 5 matches"""
    df = df.sort_values(['wrestler_id', 'date'])
    
    # Define dominant wins (e.g., winning by 10+ points)
    df['dominant_win'] = ((df['is_win'] == 1) & 
                         (df['wrestler_score'] - df['opponent_score'] >= dominant_threshold)).astype(int)
    
    # Calculate rolling dominant win rate
    df['dominant_win_rate_last_5'] = df.groupby('wrestler_id')['dominant_win'].rolling(
        window=5, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    # Shift to avoid data leakage
    df['dominant_win_rate_last_5'] = df.groupby('wrestler_id')['dominant_win_rate_last_5'].shift(1)
    
    return df

def engineer_all_features(df):
    """Apply all feature engineering functions"""
    print("Engineering features...")
    
    # Apply all feature engineering functions
    df = calculate_win_rate_last_5(df)
    df = calculate_opponent_career_win_rate(df)
    df = calculate_form_differential_5(df)
    df = calculate_avg_point_differential_last_5(df)
    df = calculate_h2h_win_rate(df)
    df = calculate_experience_differential(df)
    df = calculate_scoring_advantage_5(df)
    df = calculate_defensive_advantage_5(df)
    df = calculate_dominant_win_rate_last_5(df)
    
    return df

def prepare_data_for_modeling(df):
    """Prepare data for logistic regression modeling"""
    priority_features = [
        'win_rate_last_5',
        'opponent_career_win_rate', 
        'form_differential_5',
        'avg_point_differential_last_5',
        'h2h_win_rate',
        'experience_differential',
        'scoring_advantage_5',
        'defensive_advantage_5',
        'dominant_win_rate_last_5'
    ]
    
    # Remove rows with NaN values in priority features
    df_clean = df.dropna(subset=priority_features + ['is_win'])
    
    # Separate features and target
    X = df_clean[priority_features]
    y = df_clean['is_win']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=priority_features, index=X.index)
    
    return X_scaled, y, scaler

if __name__ == '__main__':
    raw = pd.read_csv('../data/clean/d1_results_unique.csv')
    df = create_base_features(raw)
    df_engineered = engineer_all_features(df)
    X, y, scaler = prepare_data_for_modeling(df_engineered)
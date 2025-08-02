"""
Feature Engineering for Decision Tree Modeling

This module generates engineered features from cleaned NCAA Division I wrestling match data
for the purpose of modeling with a decision tree.

The script is designed to be run as a standalone program. When executed directly,
it reads a deduplicated match dataset, applies all transformations, and saves
the resulting feature set to disk for use in machine learning pipelines.

It includes functions to:
- Convert and clean match metrics (e.g., score, duration)
- Create base features like match outcomes and point differential
- Generate advanced features such as rolling win rates, form differentials,
  scoring/defensive advantages, experience gaps, and head-to-head history

Expected input:
- Cleaned and deduplicated match-level dataset: `data/clean/d1_results_unique.csv`

Output:
- Feature-enhanced dataset: `data/features/decision_tree_features.csv`

Usage:
    python feature_eng_dt.py
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import os

def duration_to_seconds(duration_str):
    """
    Converts a match duration string (e.g., "2:30") to total seconds.
    
    Parameters:
        duration_str (str): Match duration in "mm:ss" format.
        
    Returns:
        int or None: Duration in seconds, or None if parsing fails.
    """

    try:
        minutes, seconds = map(int, duration_str.split(":"))
        return minutes * 60 + seconds
    except:
        return None

def double_matches(df):
    """
    Creates a mirrored version of each match with roles reversed
    (opponent becomes primary wrestler and vice versa).
    
    Parameters:
        df (DataFrame): Original match data.
        
    Returns:
        DataFrame: Dataset with both original and mirrored matches included.
    """

    # Create copy for reversal
    df_b = df.copy()

    # Reverse wrestler and opponent in copy df
    df_b['wrestler_id'] = df['opponent_id']
    df_b['opponent_id'] = df['wrestler_id']
    df_b['wrestler'] = df['opponent']
    df_b['opponent'] = df['wrestler']
    df_b['wrestler_school'] = df['opponent_school']
    df_b['opponent_school'] = df['wrestler_school']
    df_b['wrestler_score'] = df['opponent_score']
    df_b['opponent_score'] = df['wrestler_score']
    df_b['result'] = np.where(df['result'] == 'W', 'L', 'W')

    # Combine and return df and copy
    result = pd.concat([df, df_b])
    return result.sort_values(['date', 'wrestler_id']).reset_index()

def create_base_features(df):
    """
    Generates base match-level features from cleaned match data.
    
    Steps:
    - Splits score into individual values
    - Flags dual meets
    - Removes forfeits/disqualifications
    - Converts durations to seconds
    - Identifies overtime matches
    - Adds mirrored matches using `double_matches`
    - Creates win/loss indicators and point differential
    
    Parameters:
        df (DataFrame): Cleaned match dataset.
        
    Returns:
        DataFrame: Base features with mirrored matches and core metrics.
    """

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

    return df

def calculate_streak(wins):
    """Calculate current win/loss streak for each match"""
    streaks = [0]
    current_streak = 0
    
    for i in range(1, len(wins)):
        # FIXED: Look at the actual previous match result
        prev_is_win = wins.iloc[i-1]
        
        if i == 1:
            # First streak is just 1 (win or loss)
            current_streak = 1
        else:
            # Check if previous match had same result as match before that
            if wins.iloc[i-2] == prev_is_win:
                current_streak += 1
            else:
                current_streak = 1
        
        # Positive for win streaks, negative for loss streaks
        streak_value = current_streak if prev_is_win == 1 else -1 * current_streak
        streaks.append(streak_value)
    
    return streaks

def calculate_historical_features(df):
    """
    Computes rolling and cumulative historical performance metrics for each wrestler.

    Features include:
    - Win rate over last 5 matches and 10 matches.
    - Career win rate and total match count.
    - Season-to-date win rate and match count.
    - Streak of consecutive wins or losses.

    Args:
        df (pd.DataFrame): DataFrame containing base match data.

    Returns:
        pd.DataFrame: DataFrame with added historical performance features.
    """
     
    for window in [3, 5, 10, 15]:
            df[f'win_rate_last_{window}'] = df.groupby('wrestler_id')['is_win'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
        
    # FIXED: Win/loss streaks
    streaks = df.groupby('wrestler_id')['is_win'].transform(
        lambda x: calculate_streak(x.reset_index(drop=True))
    )
    df['streak'] = streaks

    # FIXED: Career totals (up to current match) - ensure first match has 0s
    df['career_wins'] = df.groupby('wrestler_id')['is_win'].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    df['career_losses'] = df.groupby('wrestler_id')['is_win'].transform(
        lambda x: (1 - x).cumsum().shift(1).fillna(0)
    )
    df['career_matches'] = df['career_wins'] + df['career_losses']

    # FIXED: Seasonal performance - ensure first match of season has 0 wins
    df['season_wins'] = df.groupby(['wrestler_id', 'season'])['is_win'].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    df['season_matches'] = df.groupby(['wrestler_id', 'season']).cumcount()
    df['season_win_rate'] = np.where(df['season_matches'] > 0, 
                                    df['season_wins'] / df['season_matches'], 0)
    
    return df

def calculate_match_features(df):
    """
    Adds indicators of match quality and dominance using recent results.

    Features include:
    - Bonus win rate over last 5 matches.
    - Rate of close matches (±3 points) over last 5.
    - Dominant win rate over last 5 matches (wins by 8+ points or fall).
    - Opponent scoring metrics: points allowed and differential.

    Args:
        df (pd.DataFrame): Match data with historical features.

    Returns:
        pd.DataFrame: Updated DataFrame with match quality indicators.
    """

    bonus_results = ['FALL', 'TF', 'MD']
    df['bonus_win'] = ((df['result'] == 'W') & 
                            (df['result_type'].isin(bonus_results))).astype(int)

    # Close matches (within 3 points)
    df['close_match'] = (abs(df['point_differential']) <= 3).astype(int)

    # Rolling percentages for outcome quality
    for window in [5, 10]:
        df[f'bonus_win_rate_last_{window}'] = df.groupby('wrestler_id')['bonus_win'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        
        # Close match win rate
        df['close_match_win'] = df['is_win'] * df['close_match']
        df[f'close_match_win_rate_last_{window}'] = df.groupby('wrestler_id')['close_match_win'].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum().shift(1)
        ) / df.groupby('wrestler_id')['close_match'].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum().shift(1)
        ).replace(0, np.nan)
        df[f'close_match_win_rate_last_{window}'] = df[f'close_match_win_rate_last_{window}'].fillna(0)
    
    return df

def calculate_score_competition_features(df):
    """
    Computes scoring trends, match durations, and dual/tournament breakdowns.

    Adds:
    - Average point differential, wrestler score, and opponent score (last 5 matches).
    - Mean match duration (last 5 matches).
    - Win rates split by dual meets and tournaments.
    - Win rates by weight class and event type.

    Args:
        df (pd.DataFrame): DataFrame with match and bonus-related stats.

    Returns:
        pd.DataFrame: Enhanced DataFrame with scoring and format-based features.
    """

    # Rolling scoring averages
    for window in [3, 5, 10]:
        df[f'avg_points_scored_last_{window}'] = df.groupby('wrestler_id')['wrestler_score'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        df[f'avg_points_allowed_last_{window}'] = df.groupby('wrestler_id')['opponent_score'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        df[f'avg_point_differential_last_{window}'] = (df[f'avg_points_scored_last_{window}'] - 
                                                        df[f'avg_points_allowed_last_{window}'])

    # Overtime frequency
    for window in [5, 10]:
        df[f'overtime_rate_last_{window}'] = df.groupby('wrestler_id')['is_overtime'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )

    # Match duration averages
    for window in [5, 10]:
        df[f'avg_duration_last_{window}'] = df.groupby('wrestler_id')['duration_seconds'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )

    # Competition format performance
    # Dual meet performance
    df['dual_meet_wins'] = df.groupby('wrestler_id')[['is_win', 'is_dual_meet']].apply(
        lambda x: (x['is_win'] * x['is_dual_meet']).cumsum().shift(1).fillna(0)
    ).values
    df['dual_meet_matches'] = df.groupby('wrestler_id')['is_dual_meet'].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    df['dual_meet_win_rate'] = np.where(df['dual_meet_matches'] > 0,
                                        df['dual_meet_wins'] / df['dual_meet_matches'], 0.5)

    # Tournament performance
    df['tournament_wins'] = df.groupby('wrestler_id')[['is_win', 'is_dual_meet']].apply(
        lambda x: (x['is_win'] * (1 - x['is_dual_meet'])).cumsum().shift(1).fillna(0)
    ).values
    df['tournament_matches'] = df.groupby('wrestler_id')['is_dual_meet'].apply(
        lambda x: (1 - x).cumsum().shift(1).fillna(0)
    ).values
    df['tournament_win_rate'] = np.where(df['tournament_matches'] > 0,
                                        df['tournament_wins'] / df['tournament_matches'], 0.5)
    
    # Weight class experience
    df['weight_class_matches'] = df.groupby(['wrestler_id', 'weight_class']).cumcount()
    df['weight_class_wins'] = df.groupby(['wrestler_id', 'weight_class'])['is_win'].cumsum().shift(1).fillna(0)
    df.loc[df['weight_class_matches'] == 0, 'weight_class_wins'] = 0
    df['weight_class_win_rate'] = np.where(df['weight_class_matches'] > 0,
                                            df['weight_class_wins'] / df['weight_class_matches'], 0.5)
    
    return df

def calculate_opponent_strength_features(df):
    """
    Aggregates and merges opponent-level rolling statistics for comparative analysis.

    Features merged from opponent’s recent performance:
    - Win rate over last 5 and 10 matches.
    - Bonus win rate, dominant win rate.
    - Match experience.
    - Average point differential.

    Opponent stats are prefixed with `opponent_` and joined using match metadata.

    Args:
        df (pd.DataFrame): Match data with wrestler stats.

    Returns:
        pd.DataFrame: Dataset including opponent strength features.
    """

    # Calculate each wrestler's overall performance for opponent strength metrics
    wrestler_performance = df.groupby('wrestler_id').agg({
        'is_win': ['count', 'sum'],
        'wrestler_score': 'mean',
        'opponent_score': 'mean',
        'bonus_win': 'mean'
    }).reset_index()

    wrestler_performance.columns = ['wrestler_id', 'total_matches', 'total_wins', 
                                    'avg_score', 'avg_allowed', 'bonus_rate']
    wrestler_performance['overall_win_rate'] = wrestler_performance['total_wins'] / wrestler_performance['total_matches']

    # Merge opponent stats
    df = df.merge(wrestler_performance[['wrestler_id', 'overall_win_rate', 'avg_score', 'bonus_rate']], 
                    left_on='opponent_id', right_on='wrestler_id', 
                    how='left', suffixes=('', '_opp'))
    df = df.drop('wrestler_id_opp', axis=1)
    df = df.rename(columns={'overall_win_rate': 'opponent_career_win_rate',
                            'avg_score': 'opponent_avg_score',
                            'bonus_rate': 'opponent_bonus_rate'})

    # Rolling opponent strength
    for window in [3, 5]:
        df[f'avg_opponent_win_rate_last_{window}'] = df.groupby('wrestler_id')['opponent_career_win_rate'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )

    # Quality opponent record (opponents with >65% win rate)
    df['quality_opponent'] = (df['opponent_career_win_rate'] > 0.65).astype(int)
    for window in [5, 10]:
        df[f'quality_opponent_win_rate_last_{window}'] = df.groupby('wrestler_id')[['is_win', 'quality_opponent']].apply(
            lambda x: (x['is_win'] * x['quality_opponent']).rolling(window=window, min_periods=1).sum().shift(1) /
                        x['quality_opponent'].rolling(window=window, min_periods=1).sum().shift(1)
        ).fillna(0.5).values

    # Map opponent win rates
    for window in [5, 10]:
        rate_mapping = df.groupby(['wrestler_id', 'date'])[f'win_rate_last_{window}'].first().reset_index()

        df = df.merge(
            rate_mapping.rename(columns={'wrestler_id': 'opponent_id', f'win_rate_last_{window}': f'opponent_win_rate_last_{window}'}),
            on=['opponent_id', 'date'],
            how='left'
        )

        # Calculate experience differential
        df[f'form_differential_{window}'] = df[f'win_rate_last_{window}'] - df[f'opponent_win_rate_last_{window}']

    # Map opponent scoring rate
    point_mapping = df.groupby(['wrestler_id', 'date'])['avg_points_scored_last_5'].first().reset_index()

    df = df.merge(
        point_mapping.rename(columns={'wrestler_id': 'opponent_id', 'avg_points_scored_last_5': 'opponent_avg_points_scored_last_5'}),
        on=['opponent_id', 'date'],
        how='left'
    )

    # Calculate scoring rate differential
    df['scoring_advantage_last_5'] = df['avg_points_scored_last_5'] - df['opponent_avg_points_scored_last_5']

    # Map opponent scoring allowed rate
    point_mapping = df.groupby(['wrestler_id', 'date'])['avg_points_allowed_last_5'].first().reset_index()

    df = df.merge(
        point_mapping.rename(columns={'wrestler_id': 'opponent_id', 'avg_points_allowed_last_5': 'opponent_avg_points_allowed_last_5'}),
        on=['opponent_id', 'date'],
        how='left'
    )

    # Calculate scoring rate differential
    df['defensive_advantage_last_5'] = df['avg_points_allowed_last_5'] - df['opponent_avg_points_allowed_last_5']

    # Map opponent rest
    rest_mapping = df.groupby(['wrestler_id', 'date'])['days_since_last_match'].first().reset_index()

    df = df.merge(
        rest_mapping.rename(columns={'wrestler_id': 'opponent_id', 'days_since_last_match': 'opponent_days_since_last_match'}),
        on=['opponent_id', 'date'],
        how='left'
    )

    # Calculate rest differential
    df['rest_differential'] = df['days_since_last_match'] - df['opponent_days_since_last_match']
    
    return df

def calculate_match_frequency(dates, window_days=30):
    """Calculate matches per week over rolling window"""
    frequencies = []
    
    for i in range(len(dates)):
        if i == 0:
            frequencies.append(0)
        else:
            window_start = dates.iloc[i] - timedelta(days=window_days)
            recent_matches = sum(1 for j in range(i) if dates.iloc[j] >= window_start)
            matches_per_week = recent_matches * 7 / window_days
            frequencies.append(matches_per_week)
    
    return pd.Series(frequencies)

def calculate_time_based_features(df):
    """
    Adds time-driven features to account for rest, season experience, and match spacing.

    Features include:
    - Days since last match and rolling average days between matches.
    - Number of matches wrestled so far in the season.
    - Match number for the wrestler and opponent within the season.
    - Experience differential and rest differential between wrestler and opponent.

    Args:
        df (pd.DataFrame): DataFrame with opponent features and temporal columns.

    Returns:
        pd.DataFrame: DataFrame with added time-based features.
    """

    df['date'] = pd.to_datetime(df['date'])
    
    df['days_since_last_match'] = df.groupby('wrestler_id')['date'].diff().dt.days
    df['days_since_last_match'] = df['days_since_last_match'].fillna(0)

    # Match frequency (matches per week over rolling periods)
    df['matches_per_week_last_30_days'] = df.groupby('wrestler_id')['date'].apply(
        lambda x: calculate_match_frequency(x.reset_index(drop=True), window_days=30)
    ).values

    # Yearly performance
    df['year'] = df['date'].dt.year
    yearly_performance = df.groupby(['wrestler_id', 'year'])['is_win'].mean().reset_index()
    yearly_performance['year'] = yearly_performance['year'] + 1
    yearly_performance.columns = ['wrestler_id', 'year', 'prev_yearly_win_rate']
    df = df.merge(yearly_performance, on=['wrestler_id', 'year'], how='left')
    df['prev_yearly_win_rate'] = df['prev_yearly_win_rate'].fillna(0.5)

    df = df.sort_values(['wrestler_id', 'date'])

    # Shift to get experience before current match
    df['experience'] = df.groupby('wrestler_id').cumcount().fillna(0)

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

def calculate_h2h_record(df, wrestler_id, opponent_id, current_date):
    """Calculate head-to-head record between two wrestlers before current date"""
    # Get all matches between these two wrestlers before current date
    h2h_matches = df[
        (((df['wrestler_id'] == wrestler_id) & (df['opponent_id'] == opponent_id)) |
         ((df['wrestler_id'] == opponent_id) & (df['opponent_id'] == wrestler_id))) &
        (df['date'] < current_date)
    ]
    
    if len(h2h_matches) == 0:
        return 0, 0
    
    # Count wins for the wrestler in question
    wrestler_wins = len(h2h_matches[
        (h2h_matches['wrestler_id'] == wrestler_id) & (h2h_matches['result'] == 'W')
    ])
    
    return wrestler_wins, len(h2h_matches)

def calculate_h2h_features(df):
    """
    Tracks wrestler-specific history against each unique opponent.

    Features:
    - Head-to-head win rate (rolling).
    - Number of prior matches between wrestler and opponent.

    Uses a grouped rolling window based on the `h2h_key` (wrestler-opponent pair).

    Args:
        df (pd.DataFrame): DataFrame with time and performance features.

    Returns:
        pd.DataFrame: DataFrame augmented with head-to-head matchup features.
    """
    
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
    df.loc[df['is_win'] == True, 'h2h_wins'] = df['h2h_wins'] - 1

    return df

def engineer_all_features(df):
    """
    Applies all feature engineering functions to the dataset.
    
    Returns:
        DataFrame: Fully featured dataset.
    """
    
    # Apply all feature engineering functions
    df = calculate_historical_features(df)
    df = calculate_match_features(df)
    df = calculate_score_competition_features(df)
    df = calculate_time_based_features(df)
    df = calculate_opponent_strength_features(df)
    df = calculate_h2h_features(df)
    
    return df

if __name__ == '__main__':
    # Load cleaned, deduplicated match results
    raw = pd.read_csv('data/clean/d1_results_unique.csv')
    df = create_base_features(raw).drop(columns=['index'])

    # Apply full feature engineering pipeline
    df_engineered = engineer_all_features(df)

    # Save output to disk
    os.makedirs('data/features/', exist_ok=True)
    df_engineered.to_csv('data/features/decision_tree_features.csv', index=False)
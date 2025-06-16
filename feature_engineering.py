import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

class WrestlingFeatureEngineer:
    
    def __init__(self):
        self.wrestler_stats = defaultdict(dict)
    
    def engineer_all_features(self, df):
        # Initialize feature columns
        feature_cols = [
            'Win_Streak', 'Loss_Streak',
            'Recent_Win_Rate_5', 'Recent_Win_Rate_10', 'Recent_Bonus_Rate',
            'Weighted_Recent_Performance', 'Days_Since_Last_Match', 
            'Matches_This_Season', 'Season_Win_Rate', 'Career_Win_Rate',
            'Opponent_Avg_Win_Rate', 'Strength_of_Schedule',
            'Performance_vs_Strong', 'Opponent_Recent_Win_Rate', 'Form_Differential',
            'H2H_Matches', 'H2H_Win_Rate', 'H2H_Last_Result',
            'Recent_Momentum'
        ]
        
        for col in feature_cols:
            df[col] = 0.0
            
        # Process each match chronologically
        for idx, row in df.iterrows():
            self._calculate_match_features(df, idx, row)
            
        return df
    
    
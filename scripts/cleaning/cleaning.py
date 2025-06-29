import pandas as pd
import numpy as np
import hashlib

class WrestlingDataCleaner:
    def __init__(self):
        pass

    def generate_match_id(self, row):
        key = f"{row['season']}_{row['date']}_{row['wrestler_id']}_{row['opponent_id']}_{row['weight_class']}"
        return hashlib.md5(key.encode()).hexdigest()

    def clean_data(self, df):
        # Combine season and date to create a datetime column
        df['temp_date'] = pd.to_datetime(df['season'].astype(str) + '/' + df['date'], format='%Y/%m/%d')
        df['month'] = df['temp_date'].dt.month

        # Adjust year for months August-December (assumed to be previous calendar year)
        df['datetime'] = df.apply(
            lambda row: row['temp_date'].replace(year=int(row['season']) - 1) if row['month'] >= 8 else row['temp_date'],
            axis=1
        )

        # Clean up temporary columns and move datetime column to 'date'
        df = df.drop(['temp_date', 'month', 'date'], axis=1)
        datetime_col = df.pop('datetime')
        df.insert(1, 'date', datetime_col)

        # Format season as "previous/current"
        df['season'] = (df['season'] - 1).astype(str) + '/' + df['season'].astype(str)

        # Generate unique match IDs
        df['match_id'] = df.apply(self.generate_match_id, axis=1)

        # Extract match duration or default to '7:00'
        df['match_duration'] = df['score'].str.extract(r'(\d{1,2}:\d{2})')[0].fillna('7:00')

        # Clean up score for technical falls
        if df['result_type'].str.contains('TF').any():
            df['score'] = df['score'].replace(r' \d{1,2}:\d{2}', '', regex=True)

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Ensure correct data types
        df['wrestler_id'] = df['wrestler_id'].astype(int)
        df['opponent_id'] = df['opponent_id'].astype(int)
        df['weight_class'] = df['weight_class'].astype(str)

        # Sort by date and wrestler_id
        df = df.sort_values(['date', 'wrestler_id']).copy()

        # Reordering columns
        raw = raw.iloc[:, [13, 0, 1, 2, 3, 10, 11, 12, 7, 8, 9, 4, 5, 6, 14]]

        return df

if __name__ == "__main__":
    raw = pd.read_csv('../data/raw/d1_all_match_results.csv')
    cleaner = WrestlingDataCleaner()
    cleaned_data = cleaner.clean_data(raw)
    cleaned_data.to_csv('../data/cleaned/d1_results_clean.csv', index=False)
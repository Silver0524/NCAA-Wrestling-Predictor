"""
Wrestling Match Data Cleaner

This module defines a `WrestlingDataCleaner` class that provides methods to clean and deduplicate 
NCAA Division I wrestling match data collected from WrestleStat. The cleaner standardizes scores, 
infers match durations, corrects weight class anomalies, formats dates, and generates stable unique 
match identifiers to remove duplicate records.

Intended use:
- Process scraped match CSV files.
- Generate clean and deduplicated datasets for analysis or modeling.
- Save output to organized directories.

Main Methods:
- clean_data(df): Cleans and transforms the raw match data.
- deduplicate_matches(df): Removes duplicate matches using a generated hash ID.
- generate_match_id(row): Generates a stable ID for each match to aid in deduplication.
- parse_score(row): Extracts the match score and duration from raw input strings.

Example:
    python cleaning.py
"""

import pandas as pd
import hashlib
import os
import re
import tqdm
import glob

class WrestlingDataCleaner:
    """A class to clean and deduplicate wrestling match data scraped from WrestleStat."""

    def __init__(self):
        pass

    def generate_match_id(self, row):
        """
        Generate a unique match identifier (MD5 hash) for deduplication.

        The ID is based on season, date, event, weight class, wrestler/opponent IDs (ordered),
        match duration, and normalized score. This allows duplicate matches (i.e., same match
        recorded from both wrestlers' perspectives) to be identified.

        Args:
            row (pd.Series): A row of the DataFrame representing one match.

        Returns:
            str: A hexadecimal hash string uniquely identifying the match.
        """

        # If there is a score present, split the two values
        if ' - ' in str(row['score']):
            score1, score2 = row['score'].split(' - ')
            parts = [
                str(row["season"]),
                str(row["date"]),
                row["event"],
                str(row["weight_class"]),
                str(min(row["wrestler_id"], row["opponent_id"])),
                str(max(row["wrestler_id"], row["opponent_id"])),
                str(row["match_duration"]),
                str(min(int(score1), int(score2))), 
                str(max(int(score1), int(score2)))
            ]
        # Otherwise just use the string
        else:
            parts = [
                str(row["season"]),
                str(row["date"]),
                row["event"],
                str(row["weight_class"]),
                str(min(row["wrestler_id"], row["opponent_id"])),
                str(max(row["wrestler_id"], row["opponent_id"])),
                str(row["match_duration"]),
                str(row['score'])
            ]
            
        key = "_".join(parts).lower().replace(" ", "")
        return hashlib.md5(key.encode()).hexdigest()

    def parse_score(self, row):
        """
        Parses the 'score' column to extract the normalized score and match duration.

        Args:
            row (pd.Series): A match row from the dataset.

        Returns:
            pd.Series: A Series with two values:
                - str: Cleaned score (e.g. '8 - 3')
                - str or None: Match duration (e.g. '7:00', or None if not applicable)
        """

        val = row['score']
        
        # Normalize spacing: "26 -10 6:40" → "26 - 10 6:40"
        val = re.sub(r'\s*-\s*', ' - ', val.strip())        # normalize dashes
        val = re.sub(r'\s+', ' ', val)                      # collapse multiple spaces

        # Case 1: score + time (e.g. '15 - 0 :42', '20 - 3 0:0')
        match = re.match(r'^(\d+)\s-\s(\d+)\s+(:?\d{1,2}:\d{1,2})$', val)
        if match:
            score = f"{match.group(1)} - {match.group(2)}"
            time = match.group(3)
            return pd.Series([score, time])
        
        # Case 2: only score (e.g., '7-5')
        match = re.match(r'^(\d+)\s*-\s*(\d+)$', val)
        if match:
            return pd.Series([f"{match.group(1)} - {match.group(2)}", None])
        
        # Case 3: only time (e.g., '2:04', ':42')
        if re.match(r'^:?\d{1,2}:\d{1,2}$', val):
            return pd.Series([row['result_type'], val])
    
        # Fallback
        return pd.Series([val, None])

    def clean_data(self, df):
        """
        Cleans and standardizes raw WrestleStat match data.

        Steps include:
        - Removing invalid weight classes
        - Standardizing scores and match durations
        - Converting match dates into full datetime objects
        - Adjusting seasons to reflect academic years
        - Normalizing perspective of scores (wrestler’s view)
        - Reordering columns

        Args:
            df (pd.DataFrame): Raw wrestling match data.

        Returns:
            pd.DataFrame: Cleaned match data ready for analysis.
        """

        # Removing rows labeled as incorrect weight class
        df = df[~((df['weight_class'] == 0) | (df['weight_class'] == 6))].copy()

        # Manually fixing a few rows that I could find online
        df.loc[df['score'] == '4:12', 'score'] = '16 - 0 4:12'
        df.loc[df['score'] == '17 - 1 :42', 'score'] = '17 - 1 1:42'
        df.loc[df['score'] == '15 - 0 :00', 'score'] = '15 - 0 7:00'

        # Creating match_duration column from the score column and adjusting for special cases
        df[['score', 'match_duration']] = df.apply(self.parse_score, axis=1)
        df.loc[df['result_type'].isin(['MD', 'DEC']), 'match_duration'] = '7:00'
        df.loc[df['result_type'] == 'SV-1', 'match_duration'] = '9:00'
        df.loc[df['result_type'] == 'TB-1', 'match_duration'] = '10:00'
        df.loc[df['result_type'] == 'SV-2', 'match_duration'] = '11:00'
        df.loc[df['result_type'] == 'TB-2', 'match_duration'] = '12:00'
        df.loc[df['result_type'] == 'SV-3', 'match_duration'] = '13:00'
        df.loc[df['result_type'] == 'TB-3', 'match_duration'] = '14:00'
        df.loc[df['result_type'].isin(['MFOR', 'FOR', 'CMFF', 'DQ']), 'match_duration'] = '0:00'
        df['match_duration'] = df['match_duration'].fillna('0:00')

        # Removing time from score for technical falls
        df.loc[df['result_type'].str.contains('TF'), 'score'] = df['score'].replace(r' \d{1}:\d{1,2}', '', regex=True)

        # Adding datetime column that captures full date
        df['temp_date'] = pd.to_datetime(df['season'].astype('str') + '/' + df['date'], format='%Y/%m/%d')
            
        # Extract month to identify which dates need adjustment
        df['month'] = df['temp_date'].dt.month

        # Adjust year for months that are typically in the previous calendar year
        # Assuming August-December (months 8-12) belong to the previous calendar year
        df['datetime'] = df.apply(lambda row: 
            row['temp_date'].replace(year=int(row['season']) - 1) 
            if row['month'] >= 8  # August through December
            else row['temp_date'], axis=1)

        # Clean up temporary columns and move datetime column
        df = df.drop(['temp_date', 'month', 'date'], axis=1)
        datetime = df.pop('datetime')
        df.insert(1, 'date', datetime)

        # Editing season column to reflect full span
        df['season'] = (df['season'] - 1).astype('str') + '/' + df['season'].astype('str')

        df = df.reset_index(drop=True)

        # Swapping scores for matches where the wrestler lost to ensure the score reflects the wrestlers's perspective
        df[['score1', 'score2']] = df['score'].str.split(' - ', expand=True)
        df.loc[df['result'] == 'L', 'score'] = df.score2 + ' - ' + df.score1
        df.drop(['score1', 'score2'], axis=1, inplace=True)

        # Fill score column for matches where there is no score
        df.loc[df['result_type'].isin(['FALL', 'INJ', 'DEF']), 'score'] = df['result_type']

        # Standardizing technical fall notation
        df.loc[df['result_type'].str.contains('TF'), 'result_type'] = 'TF'

        # Reorder columns to match desired output
        df = df.iloc[:, [0, 1, 2, 3, 10, 11, 12, 7, 8, 9, 4, 5, 6, 13]].copy()

        return df
    
    def deduplicate_matches(self, df):
        """
        Removes duplicate matches by computing a match hash and filtering unique entries.

        Args:
            df (pd.DataFrame): Cleaned wrestling match data.

        Returns:
            pd.DataFrame: Deduplicated dataset, sorted by date and weight class.
        """

        # Generate a unique match identifier for each match
        df['match_id'] = df.apply(self.generate_match_id, axis=1)

        # Drop duplicates based on match_id, keeping the first occurrence
        df.drop_duplicates(subset='match_id', keep='first', inplace=True)

        # Drop the match_id column as it's no longer needed
        df = df.drop(columns=['match_id'])

        # Sort rows by date and weight class
        df = df.sort_values(['date', 'weight_class']).copy()

        return df

if __name__ == "__main__":
    # Think about implementing team/year specific cleaning
    '''# Folder containing your raw CSV files in subfolders
    input_folder = 'data/raw/'
    output_folder = 'data/clean/'
    os.makedirs(output_folder, exist_ok=True)

    # Initialize the cleaner
    cleaner = WrestlingDataCleaner()

    # Loop through all CSV files in team_results and year_results subfolders
    pattern = os.path.join(input_folder, '*', '*.csv')
    for file_path in tqdm(glob.glob(pattern), desc=f'Cleaning files in {input_folder}'):
        # Load data
        raw = pd.read_csv(file_path)
        # Clean data
        cleaned_data = cleaner.clean_data(raw)
        # Deduplicate matches
        data_unique = cleaner.deduplicate_matches(cleaned_data)

        # Create output file names
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        cleaned_file = os.path.join(output_folder, f"{base_name}_clean.csv")
        unique_file = os.path.join(output_folder, f"{base_name}_unique.csv")

        # Save cleaned and deduplicated data
        cleaned_data.to_csv(cleaned_file, index=False)
        data_unique.to_csv(unique_file, index=False)'''
    
    # Load in raw data
    raw = pd.read_csv('data/raw/d1_results_raw.csv')

    # Initialize the cleaner
    cleaner = WrestlingDataCleaner()

    # Ensure directory exists
    os.makedirs('data/clean/', exist_ok=True)

    # Clean and save data
    cleaned_data = cleaner.clean_data(raw)
    cleaned_file = 'data/clean/d1_results_clean.csv'
    cleaned_data.to_csv(cleaned_file, index=False)
    
    # Deduplicate and save data
    data_unique = cleaner.deduplicate_matches(cleaned_data)
    unique_file = 'data/clean/d1_results_unique.csv'
    data_unique.to_csv(unique_file, index=False)

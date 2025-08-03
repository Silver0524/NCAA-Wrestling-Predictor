"""
decision_tree.py

This module prepares engineered NCAA Division I wrestling data for decision tree modeling,
performs a temporal train-test split to avoid data leakage and trains a
decision tree classifier to predict match outcomes.

Functions included:
- prepare_data_for_modeling: filters and formats features/target for modeling
- temporal_train_test_split: splits data into train and test sets based on match date
- train_logistic_regression: executes the full modeling pipeline and returns predictions

Expected input:
- A feature-enhanced and deduplicated dataset including a 'date' column and binary 'is_win' label.

Output:
- Trained scikit-learn DecisionTreeClassifier model serialized via joblib

Usage:
    python decision_tree.py
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

def prepare_data_for_modeling(df):
    """
    Cleans and selects features from the dataset for logistic regression modeling.

    Filters out rows with missing values in priority features and the target column.
    Separates the selected features and target variable ('is_win'), while retaining
    the 'date' column for temporal splitting.

    Args:
        df (pd.DataFrame): The raw feature DataFrame including match-level statistics.

    Returns:
        X (pd.DataFrame): Feature DataFrame with selected features and 'date'.
        y (pd.Series): Target variable (1 if win, 0 if loss).
        priority_features (list): List of feature column names used in modeling.
    """

    non_feature_cols = [
        'duration_seconds', 'point_differential',
        'wrestler_score', 'opponent_score', 
        'result', 'result_type', 'is_overtime', 
        'bonus_win', 'close_match', 'h2h_key',
        'season', 'year', 'wrestler', 'opponent', 
        'event', 'wrestler_school', 'opponent_school',
        'close_match_win'
    ]

    priority_features = [col for col in df.columns if col not in non_feature_cols + ['is_win', 'date']]
    
    # Remove rows with NaN values in priority features
    df_clean = df.dropna(subset=priority_features + ['is_win'])
    
    # Separate features and target, keep date column
    X = df_clean[priority_features + ['date']]
    y = df_clean['is_win']
    
    return X, y, priority_features

def temporal_train_test_split(X, y, test_size=0.2, split_date=None):
    """
    Splits the dataset into training and testing sets based on match date.

    This function prevents data leakage by ensuring the test set contains only
    matches that occur chronologically after the training set.

    Args:
        X (pd.DataFrame): Features with a 'date' column.
        y (pd.Series): Binary target variable.
        test_size (float, optional): Proportion of data to assign to the test set.
        split_date (str or pd.Timestamp, optional): Manual override for the date to split on.

    Returns:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Testing feature set.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        split_date (pd.Timestamp): Date used to split the data.
    """

    if split_date is None:
        # Calculate split date based on test_size
        sorted_dates = X['date'].sort_values()
        split_idx = int(len(sorted_dates) * (1 - test_size))
        split_date = sorted_dates.iloc[split_idx]
    
    # Create temporal split
    train_mask = X['date'] < split_date
    test_mask = X['date'] >= split_date
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Split date: {split_date}")
    print(f"Train set: {len(X_train)} matches ({X_train['date'].min()} to {X_train['date'].max()})")
    print(f"Test set: {len(X_test)} matches ({X_test['date'].min()} to {X_test['date'].max()})")
    
    return X_train, X_test, y_train, y_test, split_date

def train_decision_tree(X, y, feature_columns, max_depth=10, min_samples_split=20, min_samples_leaf=10):
    """
    Trains a decision tree model on wrestling match data.

    Performs temporal splitting, model training, and prediction
    on the test set. Returns all components needed for evaluation or saving.

    Args:
        X (pd.DataFrame): Full feature set including 'date'.
        y (pd.Series): Binary match outcome labels (1 = win, 0 = loss).
        feature_columns (list): List of features to include in the model.
        max_depth (int, optional): Maximum depth of the decision tree.
        min_samples_split (int, optional): Minimum number of samples required to split an internal node.
        min_samples_leaf (int, optional): Minimum number of samples required to be at a leaf node.

    Returns:
        model (DecisionTreeClassifier): Trained decision tree model.
        X_test_scaled (pd.DataFrame): Test features with date.
        y_test (pd.Series): True labels for test set.
        y_pred (np.ndarray): Binary predictions on test set.
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class.
    """
    
    # Temporal split
    X_train, X_test, y_train, y_test, split_date = temporal_train_test_split(X, y)
    
    # Decision trees don't require feature scaling, but we'll keep dates separate
    X_train_features = X_train[feature_columns]
    X_test_features = X_test[feature_columns]
    
    # Train model
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train_features, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_features)
    y_pred_proba = model.predict_proba(X_test_features)[:, 1]
    
    return model, X_test, y_test, y_pred, y_pred_proba

if __name__ == "__main__":
    # Load the feature dataset
    df = pd.read_csv('data/features/decision_tree_features.csv')

    # Prepare data and train model
    X, y, feature_cols = prepare_data_for_modeling(df)
    model, X_test, y_test, y_pred, y_pred_proba = train_decision_tree(X, y, feature_cols)

    # Save the trained model
    joblib.dump(model, 'models/decision_tree_model.pkl')
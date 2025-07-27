"""
logistic_regression_model.py

This module prepares engineered NCAA Division I wrestling data for logistic regression modeling,
performs a temporal train-test split to avoid data leakage, scales feature values, and trains a
logistic regression classifier to predict match outcomes.

Functions included:
- prepare_data_for_modeling: filters and formats features/target for modeling
- temporal_train_test_split: splits data into train and test sets based on match date
- scale_features: standardizes numerical features using training set statistics
- train_logistic_regression: executes the full modeling pipeline and returns predictions

Expected input:
- A feature-enhanced and deduplicated dataset including a 'date' column and binary 'is_win' label.

Output:
- Trained scikit-learn LogisticRegression model serialized via joblib

Usage:
    python logistic_regression.py
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

def scale_features(X_train, X_test, feature_columns):
    """
    Standardizes the feature columns using z-score normalization based on training data.

    Scales both training and testing features using the mean and standard deviation
    from the training set. The 'date' column is retained unscaled.

    Args:
        X_train (pd.DataFrame): Training features, including 'date'.
        X_test (pd.DataFrame): Testing features, including 'date'.
        feature_columns (list): List of column names to scale.

    Returns:
        X_train_scaled (pd.DataFrame): Scaled training features with date preserved.
        X_test_scaled (pd.DataFrame): Scaled testing features with date preserved.
    """

    # Separate features from date
    X_train_features = X_train[feature_columns]
    X_test_features = X_test[feature_columns]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)
    
    # Add back the date column
    X_train_scaled['date'] = X_train['date']
    X_test_scaled['date'] = X_test['date']
    
    return X_train_scaled, X_test_scaled

def train_logistic_regression(X, y, feature_columns):
    """
    Trains a logistic regression model on wrestling match data.

    Performs temporal splitting, feature scaling, model training, and prediction
    on the test set. Returns all components needed for evaluation or saving.

    Args:
        X (pd.DataFrame): Full feature set including 'date'.
        y (pd.Series): Binary match outcome labels (1 = win, 0 = loss).
        feature_columns (list): List of features to include in the model.

    Returns:
        model (LogisticRegression): Trained logistic regression model.
        X_test_scaled (pd.DataFrame): Scaled test features with date.
        y_test (pd.Series): True labels for test set.
        y_pred (np.ndarray): Binary predictions on test set.
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class.
    """

    # Temporal split
    X_train, X_test, y_train, y_test, split_date = temporal_train_test_split(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, feature_columns)
    
    # Train model (only on feature columns, not date)
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled[feature_columns], y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled[feature_columns])
    y_pred_proba = model.predict_proba(X_test_scaled[feature_columns])[:, 1]
    
    return model, X_test_scaled, y_test, y_pred, y_pred_proba

if __name__ == "__main__":
    # Load the feature dataset
    df = pd.read_csv('data/features/logistic_reg_features.csv')

    # Prepare data and train model
    X, y, feature_cols = prepare_data_for_modeling(df)
    model, X_test, y_test, y_pred, y_pred_proba = train_logistic_regression(X, y, feature_cols)

    # Save the trained model
    joblib.dump(model, 'models/logistic_regression_model.pkl')
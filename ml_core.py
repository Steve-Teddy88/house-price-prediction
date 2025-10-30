"""
Machine Learning Core Module for House Price Prediction
===================================================

This module contains the core machine learning functionality for the house price prediction system.
It handles data preparation, model training, evaluation, and predictions.

Author: ML Engineer
Version: 2.0.0
"""

import logging
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelValidationError(Exception):
    """Raised when model validation fails."""


class DataValidationError(Exception):
    """Raised when data validation fails."""


def load_and_prepare_data() -> (
    Tuple[
        pd.DataFrame,
        pd.Series,
        np.ndarray,
        np.ndarray,
        pd.Series,
        pd.Series,
        StandardScaler,
        List[str],
    ]
):
    """Load and prepare the California Housing dataset for model training.

    Returns:
        tuple: Contains the following elements:
            - X (pd.DataFrame): Original features
            - y (pd.Series): Original targets
            - X_train_scaled (np.ndarray): Scaled training features
            - X_test_scaled (np.ndarray): Scaled testing features
            - y_train (pd.Series): Training targets
            - y_test (pd.Series): Testing targets
            - scaler (StandardScaler): Fitted scaler for feature normalization
            - feature_names (List[str]): List of feature names
    """
    logging.info("Loading and preparing data...")
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # Handle missing values (if any, though California Housing is clean)
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler, data.feature_names


def validate_input_data(X: np.ndarray, feature_names: List[str]) -> bool:
    """Validate input data format and features.

    Args:
        X: Input features array
        feature_names: Expected feature names

    Returns:
        bool: True if validation passes

    Raises:
        DataValidationError: If validation fails
    """
    if X.shape[1] != len(feature_names):
        raise DataValidationError(
            f"Expected {len(feature_names)} features, got {X.shape[1]}"
        )
    return True


def train_model(
    X_train: np.ndarray, y_train: np.ndarray, model_name: str = "Random Forest"
) -> BaseEstimator:
    """Train a machine learning model.

    Args:
        X_train: Training features
        y_train: Training targets
        model_name: The name of the model to train

    Returns:
        Trained model
    """
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate a model and compute performance metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets

    Returns:
        Dict containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # For proper CV, it should be done on X_train_scaled or within a pipeline.
    # For simplicity in this example, we'll use X_test for CV as well.
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring="r2")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "CV R² Mean": cv_mean,
        "CV R² Std": cv_std,
        "y_pred": y_pred,
    }


def predict_price(model: BaseEstimator, scaler: StandardScaler, features: List[float]) -> float:
    """Predict house price from input features.

    Args:
        model: Trained model
        scaler: Fitted scaler
        features: Input features for prediction

    Returns:
        Predicted price
    """
    # Ensure features is a 2D array for scaling
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return float(pred)


def save_artifacts(
    model: BaseEstimator,
    scaler: StandardScaler,
    feature_names: List[str],
    path: str = ".",
):
    """Saves the trained model, scaler, and feature names to disk."""
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "model.joblib"))
    joblib.dump(scaler, os.path.join(path, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(path, "feature_names.joblib"))
    print(f"✅ Model, scaler, and feature names saved to {path}")


def load_artifacts(path: str = ".") -> Tuple[BaseEstimator, StandardScaler, List[str]]:
    """Loads the trained model, scaler, and feature names from disk."""
    model = joblib.load(os.path.join(path, "model.joblib"))
    scaler = joblib.load(os.path.join(path, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(path, "feature_names.joblib"))
    print(f"✅ Model, scaler, and feature names loaded from {path}")
    return model, scaler, feature_names

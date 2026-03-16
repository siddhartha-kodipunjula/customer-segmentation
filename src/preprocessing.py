"""Preprocessing utilities for customer segmentation.

This module contains functions for cleaning the dataset, encoding categorical
features, and preparing feature matrices for clustering.
"""

from __future__ import annotations

import pandas as pd


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset.

    For this dataset, we drop rows with missing values. If the dataset grows
    in complexity, this function can be updated to impute or otherwise fill values.
    """

    missing_before = df.isna().sum().sum()
    if missing_before > 0:
        df = df.dropna().reset_index(drop=True)
    return df


def encode_gender(df: pd.DataFrame, column: str = "Gender") -> pd.DataFrame:
    """Encode the Gender column as a numeric feature.

    We use simple label encoding (Male=0, Female=1) because the feature is binary.
    """

    df = df.copy()
    df[column] = df[column].map({"Male": 0, "Female": 1})
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select important features for clustering.

    Returns a DataFrame containing the features:
    - Age
    - Annual Income (k$)
    - Spending Score (1-100)
    """

    required_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for clustering: {missing_cols}")

    return df[required_cols].copy()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline.

    This pipeline:
     - Handles missing values
     - Encodes categorical variables
     - Selects features for clustering

    Returns:
        A cleaned DataFrame with the selected features.
    """

    df = handle_missing_values(df)
    if "Gender" in df.columns:
        df = encode_gender(df)
    df_features = select_features(df)
    return df_features

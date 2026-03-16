"""Data loading utilities for the Customer Segmentation project.

This module provides functions to load the Mall Customers dataset and
perform an initial inspection.

If the dataset file is missing, a small synthetic dataset is generated
and saved to the expected location.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _generate_synthetic_data(path: Path, n_customers: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic Mall Customers dataset.

    The generated data has the same schema as the original Mall Customers dataset.
    """

    rng = np.random.default_rng(seed=random_state)

    # Create synthetic clusters to mimic real-world segmentation
    ages = rng.integers(18, 70, size=n_customers)
    incomes = rng.normal(loc=60, scale=26, size=n_customers).clip(15, 137)
    spending = rng.normal(loc=50, scale=28, size=n_customers).clip(1, 99)

    # Introduce some structure for demonstration purposes
    # High income & high spenders
    high_income_idx = rng.choice(n_customers, size=n_customers // 5, replace=False)
    incomes[high_income_idx] = rng.normal(loc=95, scale=12, size=len(high_income_idx)).clip(60, 137)
    spending[high_income_idx] = rng.normal(loc=80, scale=10, size=len(high_income_idx)).clip(55, 99)

    # Low income & low spenders
    low_income_idx = rng.choice(n_customers, size=n_customers // 5, replace=False)
    incomes[low_income_idx] = rng.normal(loc=25, scale=8, size=len(low_income_idx)).clip(15, 45)
    spending[low_income_idx] = rng.normal(loc=25, scale=10, size=len(low_income_idx)).clip(1, 45)

    genders = rng.choice(["Male", "Female"], size=n_customers, p=[0.5, 0.5])

    df = pd.DataFrame(
        {
            "CustomerID": np.arange(1, n_customers + 1),
            "Gender": genders,
            "Age": ages.astype(int),
            "Annual Income (k$)": incomes.round(1),
            "Spending Score (1-100)": spending.round(1),
        }
    )

    df.to_csv(path, index=False)
    return df


def load_data(csv_path: str | Path = None) -> pd.DataFrame:
    """Load the Mall Customers dataset and print a quick summary.

    If the file is missing, a synthetic dataset is generated and saved.

    Args:
        csv_path: Path to the CSV file. Defaults to "data/mall_customers.csv".

    Returns:
        Loaded pandas DataFrame.
    """

    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "mall_customers.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print(f"Dataset not found at {csv_path}. Generating synthetic dataset...")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = _generate_synthetic_data(csv_path)
    else:
        df = pd.read_csv(csv_path)

    print("\n=== Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nInfo:")
    print(df.info())

    missing = df.isna().sum()
    print("\nMissing values per column:")
    print(missing)

    return df

"""KMeans clustering utilities.

This module contains functions to scale features, determine the optimal number of
clusters using the Elbow method, and train a KMeans clustering model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def scale_features(df: pd.DataFrame, feature_columns: Optional[list[str]] = None) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale features using StandardScaler.

    Args:
        df: Feature dataframe to scale.
        feature_columns: Optional list to select specific columns.

    Returns:
        Tuple of (scaled_dataframe, scaler)
    """

    if feature_columns is not None:
        df = df[feature_columns]

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    return scaled_df, scaler


def compute_wcss(X: pd.DataFrame, k_values: Iterable[int]) -> list[float]:
    """Compute Within-Cluster Sum of Squares (WCSS) for a range of k values."""

    wcss = []
    for k in k_values:
        model = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        model.fit(X)
        wcss.append(model.inertia_)
    return wcss


def plot_elbow_curve(
    k_values: Iterable[int],
    wcss: list[float],
    out_path: str | Path = "outputs/charts/elbow_method.png",
    title: str = "Elbow Method for Optimal k",
) -> None:
    """Plot and save the Elbow curve.

    Args:
        k_values: Sequence of k values.
        wcss: Corresponding WCSS values.
        out_path: File path to save the plot.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.xticks(k_values)
    plt.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_kmeans(
    X: pd.DataFrame,
    n_clusters: int = 5,
    init: str = "k-means++",
    random_state: int = 42,
) -> KMeans:
    """Train a KMeans model and return the fitted estimator."""

    model = KMeans(n_clusters=n_clusters, init=init, random_state=random_state, n_init=10)
    model.fit(X)
    return model


def assign_cluster_labels(df: pd.DataFrame, model: KMeans, label_column: str = "Cluster") -> pd.DataFrame:
    """Add cluster labels to a DataFrame."""

    df = df.copy()
    df[label_column] = model.predict(df.select_dtypes(include=[float, int]))
    return df

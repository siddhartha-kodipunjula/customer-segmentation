"""Visualization utilities for customer segmentation.

This module provides functions to create and save charts that help interpret
customer clusters.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_income_vs_spending(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    income_col: str = "Annual Income (k$)",
    spending_col: str = "Spending Score (1-100)",
    out_path: str | Path = "outputs/charts/income_vs_spending.png",
    centroids: pd.DataFrame | None = None,
) -> None:
    """Scatter plot Income vs Spending Score with cluster coloring."""

    out_path = Path(out_path)
    _ensure_dir(out_path)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=income_col,
        y=spending_col,
        hue=cluster_col,
        palette="tab10",
        s=60,
        alpha=0.85,
        edgecolor="w",
    )

    if centroids is not None:
        plt.scatter(
            centroids[income_col],
            centroids[spending_col],
            c="black",
            s=200,
            marker="X",
            label="Centroids",
            edgecolor="white",
        )

    plt.title("Customer Segments: Income vs Spending Score")
    plt.xlabel(income_col)
    plt.ylabel(spending_col)
    plt.legend(title=cluster_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pairplot(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    out_path: str | Path = "outputs/charts/pairplot.png",
) -> None:
    """Create a seaborn pairplot for the main features colored by cluster."""

    out_path = Path(out_path)
    _ensure_dir(out_path)

    pairplot = sns.pairplot(
        df,
        hue=cluster_col,
        palette="tab10",
        height=2.5,
        corner=True,
        diag_kind="kde",
    )
    pairplot.fig.suptitle("Feature Pairplot by Cluster", y=1.02)
    pairplot.savefig(out_path, dpi=150)
    plt.close("all")


def plot_cluster_distribution(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    out_path: str | Path = "outputs/charts/cluster_distribution.png",
) -> None:
    """Plot the number of customers in each cluster."""

    out_path = Path(out_path)
    _ensure_dir(out_path)

    counts = df[cluster_col].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=counts.index, y=counts.values, palette="tab10")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.title("Customer Count per Cluster")
    for i, v in enumerate(counts.values):
        plt.text(i, v + max(counts.values) * 0.01, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def render_plotly_scatter(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    income_col: str = "Annual Income (k$)",
    spending_col: str = "Spending Score (1-100)",
):
    """Return a Plotly scatter plot figure for use in dashboards."""

    import plotly.express as px

    fig = px.scatter(
        df,
        x=income_col,
        y=spending_col,
        color=cluster_col,
        color_continuous_scale=None,
        labels={
            income_col: "Annual Income (k$)",
            spending_col: "Spending Score (1-100)",
        },
        title="Customer Clusters: Income vs Spending Score",
        hover_data=df.columns,
    )
    fig.update_layout(legend_title_text=cluster_col)
    return fig

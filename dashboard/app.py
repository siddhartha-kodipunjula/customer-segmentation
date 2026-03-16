"""Streamlit dashboard for customer segmentation using K-Means clustering."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure the project root is on sys.path so we can import src/ modules when running via Streamlit
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.clustering_model import (
    compute_wcss,
    plot_elbow_curve,
    scale_features,
    train_kmeans,
)
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.visualization import render_plotly_scatter


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_raw = load_data()
    df_clean = preprocess(df_raw)
    df_scaled, _ = scale_features(df_clean)
    return df_raw, df_clean, df_scaled


def compute_cluster_stats(df: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    stats = df.groupby(cluster_col).agg(
        customers=(cluster_col, "count"),
        avg_income=("Annual Income (k$)", "mean"),
        avg_spending=("Spending Score (1-100)", "mean"),
        avg_age=("Age", "mean"),
    )
    stats = stats.round(1).reset_index()
    return stats


def render_business_insights(stats: pd.DataFrame) -> None:
    st.markdown("### Marketing Insights")

    for _, row in stats.iterrows():
        cluster = int(row["Cluster"])
        income = float(row["avg_income"])
        spending = float(row["avg_spending"])

        if income > 70 and spending > 70:
            label = "Premium customers"
            advice = "Focus on loyalty programs, exclusive offers, and premium experiences."
        elif income < 40 and spending < 40:
            label = "Budget customers"
            advice = "Use promotions, discounts, and value bundles to drive engagement."
        elif income > 70 and spending < 50:
            label = "High income, low spending"
            advice = "Target with personalized promotions and upsell campaigns."
        elif income < 40 and spending > 60:
            label = "Low income, high spending"
            advice = "Offer flexible payment plans and value-driven bundles."
        else:
            label = "Moderate spenders"
            advice = "Use tailored recommendations and occasional promotions."

        st.markdown(
            f"**Cluster {cluster}: {label}**  \
            - Average income: {income:.1f}k$  \
            - Average spending: {spending:.1f}"
        )
        st.write(advice)


def main() -> None:
    st.set_page_config(
        page_title="Customer Segmentation Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Customer Segmentation Dashboard")
    st.write(
        "Use K-Means clustering to group customers by purchasing behavior and explore actionable insights."
    )

    df_raw, df_clean, df_scaled = load_and_prepare_data()

    st.sidebar.header("Controls")
    st.sidebar.markdown("### Dataset Overview")
    st.sidebar.write(f"Total customers: **{len(df_raw):,}**")
    st.sidebar.write(f"Average age: **{df_raw['Age'].mean():.1f}**")
    st.sidebar.write(f"Average income: **{df_raw['Annual Income (k$)'].mean():.1f} k$**")

    n_clusters = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed with Python, scikit-learn, and Streamlit.")

    # Elbow method
    st.header("Elbow Method")
    k_values = list(range(1, 11))
    wcss = compute_wcss(df_scaled, k_values)
    plot_elbow_curve(k_values, wcss, out_path="outputs/charts/elbow_method.png")

    st.image("outputs/charts/elbow_method.png", caption="Elbow method: WCSS vs number of clusters", use_column_width=True)
    st.write(
        "Select the number of clusters using the slider in the sidebar. The elbow point indicates a good tradeoff "
        "between model complexity and explained variance."
    )

    # Train clustering model
    kmeans = train_kmeans(df_scaled, n_clusters=n_clusters)
    df_clusters = df_clean.copy()
    df_clusters["Cluster"] = kmeans.labels_

    # Cluster overview
    st.header("Customer Clusters")
    scatter_fig = render_plotly_scatter(df_clusters)
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Cluster statistics
    st.header("Cluster Statistics")
    stats = compute_cluster_stats(df_clusters)
    st.dataframe(stats)

    # Business insights
    render_business_insights(stats)


if __name__ == "__main__":
    main()

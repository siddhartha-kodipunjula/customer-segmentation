# Customer Segmentation using K-Means Clustering

## Project Overview
This project demonstrates a complete customer segmentation pipeline using the Mall Customers dataset. It applies K-Means clustering to group customers based on purchasing behavior and provides interactive visualizations and business insights via a Streamlit dashboard.

## Features
- Data loading and inspection
- Data preprocessing and feature selection
- Feature scaling using StandardScaler
- Elbow method for choosing the optimal number of clusters
- K-Means clustering and cluster labeling
- Cluster visualization (scatter plots, pairplots, distribution plots)
- Interactive Streamlit dashboard with insights

## Folder Structure
```
Customer_Segmentation_KMeans/
│
├── data/
│   └── mall_customers.csv
│
├── notebooks/
│   └── segmentation_analysis.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── clustering_model.py
│   ├── visualization.py
│
├── dashboard/
│   └── app.py
│
├── outputs/
│   ├── charts/
│   └── reports/
│
├── requirements.txt
│
└── README.md
```

## Installation

1. Create a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate     # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

## Notes
- If `data/mall_customers.csv` is missing, the project will generate a synthetic dataset automatically.
- Charts generated during analysis are stored under `outputs/charts/`.

## Next Steps (Optional)
- Add clustering validation metrics (silhouette score).
- Persist model artifacts with `joblib`.
- Add more interactive filtering to the dashboard.

import pandas as pd
import joblib

kmeans_pipeline = joblib.load("../models/kmeans_customer_pipeline.joblib")

def segment_customer(data):
    df = pd.DataFrame([data])
    categorical_cols = [
        "Gender", "Marital_Status", "Region",
        "Customer_Type", "Channel_Preference", "Category_Pref"
    ]
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    cluster = kmeans_pipeline.predict(df)[0]
    CLUSTER_LABELS_KMEANS = {
        0: "Value-Focused Shoppers",
        1: "High-Spending Loyal Customers",
        2: "Occasional Deal Seekers"
    }
    return {
        "cluster_label": int(cluster),
        "segment": CLUSTER_LABELS_KMEANS.get(cluster, f"Cluster {cluster}")
    }

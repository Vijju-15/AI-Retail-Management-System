from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from fastapi.middleware.cors import CORSMiddleware

# ---------- INIT ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MODEL LOADING ----------
# 1. Prophet Sales Forecasting Model
prophet_model = joblib.load("models/prophet_sales_model.pkl")

# 2. XGBoost Pricing Model
xgb_model = joblib.load("models/xgb_pricing_model.joblib")
xgb_encoders = joblib.load("models/label_encoders.pkl")

kmeans_pipeline = joblib.load("models/kmeans_customer_pipeline.joblib")  # adjust path if needed


# ---------- SCHEMAS ----------
class ForecastRequest(BaseModel):
    periods: int = 30

class PricingInput(BaseModel):
    Product_ID: str
    Brand: str
    Category: str
    Region: str
    Price_INR: float
    Discount_percent: float = Field(..., alias="Discount_percent")
    Competitor_Pricing_INR: float
    Holiday_Promotion: str
    Weather_Condition: str
    Customer_Type: str
    Loyalty_Score: float
    Inventory_Level: int

class CustomerInput(BaseModel):
    Age: int
    Gender: int
    Marital_Status: int
    Region: int
    Loyalty_Score: float
    Frequency: int
    Recency: int
    Discount_Response_Rate: float
    Average_Basket_Value: float
    Annual_Expenditure: float
    Customer_Type: int
    Channel_Preference: int
    Tenure_Years: float
    Returns_Rate: float
    Satisfaction_Score: int
    Category_Pref: int

CLUSTER_LABELS = {
    -1: "Outlier / Unclustered",
    0: "High-value Repeat Customer",
    1: "Occasional Discount Seeker",
    2: "Low-engagement Shopper"
}

# ---------- ROUTES ----------
@app.get("/")
def root():
    return {"message": "Retail AI Backend is Live!"}

@app.post("/forecast")
def forecast(request: ForecastRequest):
    future = prophet_model.make_future_dataframe(periods=request.periods)
    forecast = prophet_model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(request.periods).to_dict(orient="records")

@app.post("/optimize-price")
def optimize_price(data: PricingInput):
    df = pd.DataFrame([data.dict(by_alias=True)])

    # Label encode safely
    for col in xgb_encoders:
        value = df[col].iloc[0]
        if value not in xgb_encoders[col].classes_:
            return {
                "error": f"Invalid value for '{col}': '{value}' not seen during training."
            }
        df[col] = xgb_encoders[col].transform(df[col])

    df["Discount_%"] = df["Discount_percent"]
    df["Price_Gap"] = df["Price_INR"] - df["Competitor_Pricing_INR"]
    df["Price_Per_Unit"] = df["Price_INR"]
    df["Discount_Effectiveness"] = 1 / (df["Discount_%"] + 1)
    df["Price_Discount_Interaction"] = df["Price_INR"] * (1 - df["Discount_%"] / 100)

    features = [
        'Product_ID', 'Brand', 'Category', 'Region', 'Price_INR', 'Discount_%',
        'Competitor_Pricing_INR', 'Price_Gap', 'Price_Per_Unit',
        'Discount_Effectiveness', 'Holiday_Promotion', 'Weather_Condition',
        'Customer_Type', 'Loyalty_Score', 'Inventory_Level', 'Price_Discount_Interaction'
    ]

    prediction = xgb_model.predict(df[features])
    units = float(np.expm1(prediction[0]))
    revenue = float(df["Price_INR"].iloc[0] * units)

    return {
        "Units_Predicted": round(units, 2),
        "Expected_Revenue": round(revenue, 2)
    }
@app.post("/segment")
def segment_customer(data: CustomerInput):
    df = pd.DataFrame([data.dict()])

    # Cast categorical columns to string for OneHotEncoder compatibility
    categorical_cols = [
        "Gender", "Marital_Status", "Region",
        "Customer_Type", "Channel_Preference", "Category_Pref"
    ]
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    # Predict cluster
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

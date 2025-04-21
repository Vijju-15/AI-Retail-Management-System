from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Union, Dict
from pymongo import MongoClient
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from fastapi.middleware.cors import CORSMiddleware
from .kafka_producer import send_to_kafka

# ---------- INIT ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MONGO DB ----------
client = MongoClient("mongodb://localhost:27017/")
db = client["retail_ai_db"]
forecast_collection = db["forecast_results"]
pricing_collection = db["pricing_results"]
segment_collection = db["segment_results"]

# ---------- MODEL LOADING ----------
prophet_model = joblib.load("models/prophet_sales_model.pkl")
xgb_model = joblib.load("models/xgb_pricing_model.joblib")
xgb_encoders = joblib.load("models/label_encoders.pkl")
kmeans_pipeline = joblib.load("models/kmeans_customer_pipeline.joblib")

# ---------- SCHEMAS ----------
class ForecastRequest(BaseModel):
    shop_id: str
    periods: int = 30

class PricingInput(BaseModel):
    shop_id: str
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
    shop_id: str
    Age: int
    Gender: str
    Marital_Status: str
    Region: str
    Loyalty_Score: float
    Frequency: int
    Recency: int
    Discount_Response_Rate: float
    Average_Basket_Value: float
    Annual_Expenditure: float
    Customer_Type: str
    Channel_Preference: str
    Tenure_Years: float
    Returns_Rate: float
    Satisfaction_Score: int
    Category_Pref: str

class KafkaRequest(BaseModel):
    topic: str
    message: Union[str, Dict]

# ---------- ROUTES ----------
@app.get("/")
def root():
    return {"message": "Retail AI Backend is Live!"}

@app.post("/forecast")
def forecast(request: ForecastRequest):
    future = prophet_model.make_future_dataframe(periods=request.periods)
    forecast = prophet_model.predict(future)
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(request.periods).to_dict(orient="records")

    forecast_collection.insert_one({
        "shop_id": request.shop_id,
        "timestamp": datetime.utcnow(),
        "input": request.dict(),
        "output": result
    })

    return result

@app.post("/optimize-price")
def optimize_price(data: PricingInput):
    df = pd.DataFrame([data.dict(by_alias=True, exclude={"shop_id"})])

    for col in xgb_encoders:
        value = df[col].iloc[0]
        if value not in xgb_encoders[col].classes_:
            return {"error": f"Invalid value for '{col}': '{value}' not seen during training."}
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

    result = {
        "Units_Predicted": round(units, 2),
        "Expected_Revenue": round(revenue, 2)
    }

    pricing_collection.insert_one({
        "shop_id": data.shop_id,
        "timestamp": datetime.utcnow(),
        "input": data.dict(by_alias=True, exclude={"shop_id"}),
        "output": result
    })

    return result
@app.post("/segment")
def segment_customer(data: CustomerInput):
    df = pd.DataFrame([data.dict(exclude={"shop_id"})])

    categorical_cols = [
        "Gender", "Marital_Status", "Region",
        "Customer_Type", "Channel_Preference", "Category_Pref"
    ]
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    cluster = kmeans_pipeline.predict(df)[0]

    # ✅ Corrected Cluster-to-Segment Mapping (Based on MongoDB Analysis)
    CLUSTER_LABELS_KMEANS = {
        0: "Value-Focused Shoppers",
        1: "Occasional Deal Seekers",
        2: "High-Spending Loyal Customers"
    }

    result = {
        "cluster_label": int(cluster),
        "segment": CLUSTER_LABELS_KMEANS.get(cluster, f"Cluster {cluster}")
    }

    # ⏺️ Save result to MongoDB
    segment_collection.insert_one({
        "shop_id": data.shop_id,
        "timestamp": datetime.utcnow(),
        "input": data.dict(exclude={"shop_id"}),
        "output": result
    })

    return result


@app.post("/send-kafka/")
def send_kafka_data(data: KafkaRequest):
    send_to_kafka(data.topic, data.message)
    return {"status": "Message sent to Kafka"}

@app.get("/records/forecast")
def get_forecast_records():
    return list(forecast_collection.find({}, {"_id": 0}))

@app.get("/records/pricing")
def get_pricing_records():
    return list(pricing_collection.find({}, {"_id": 0}))

@app.get("/records/segment")
def get_segment_records():
    return list(segment_collection.find({}, {"_id": 0}))

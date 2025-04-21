import pandas as pd
import joblib
import numpy as np

# Load XGBoost model and encoders
xgb_model = joblib.load("../models/xgb_pricing_model.joblib")
xgb_encoders = joblib.load("../models/label_encoders.pkl")

def make_pricing_prediction(data):
    df = pd.DataFrame([data])

    for col in xgb_encoders:
        value = df[col].iloc[0]
        if value not in xgb_encoders[col].classes_:
            return {"error": f"Invalid value for '{col}': '{value}'"}
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

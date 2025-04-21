import pandas as pd
import joblib  # ✅ instead of pickle

# Load your Prophet model (already trained)
model = joblib.load("../models/prophet_sales_model.pkl")  # ✅ correct loader

def make_forecast(input_data):
    df = pd.DataFrame([input_data])
    forecast = model.predict(df)
    return forecast[['ds', 'yhat']].to_dict(orient="records")

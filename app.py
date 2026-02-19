from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import datetime
import os

# =========================
# INITIALIZE APP
# =========================

app = FastAPI(title="Sales Prediction API")

# =========================
# LOAD MODEL & COLUMNS
# =========================

model = joblib.load("model/model.pkl")
columns = joblib.load("model/columns.pkl")

# =========================
# INPUT SCHEMA
# =========================

class SalesInput(BaseModel):
    QUANTITYORDERED: int
    PRICEEACH: float
    QTR_ID: int
    MONTH_ID: int
    YEAR_ID: int
    PRODUCTLINE: str
    COUNTRY: str
    DEALSIZE: str

# =========================
# HOME ROUTE
# =========================

@app.get("/")
def home():
    return {"message": "Sales Prediction API is running 🚀"}

# =========================
# PREDICTION ROUTE
# =========================

@app.post("/predict")
def predict(data: SalesInput):

    # Convert request to dictionary
    input_dict = data.dict()

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # Apply same encoding as training
    df = pd.get_dummies(df)

    # Align with training columns
    df = df.reindex(columns=columns, fill_value=0)

    # Make prediction
    prediction = model.predict(df)[0]

    # =========================
    # LOGGING PREDICTIONS
    # =========================

    log_entry = input_dict.copy()
    log_entry["Predicted_Sales"] = float(prediction)
    log_entry["Timestamp"] = str(datetime.datetime.now())

    log_df = pd.DataFrame([log_entry])

    log_file = "prediction_logs.csv"

    if os.path.exists(log_file):
        existing = pd.read_csv(log_file)
        updated = pd.concat([existing, log_df], ignore_index=True)
        updated.to_csv(log_file, index=False)
    else:
        log_df.to_csv(log_file, index=False)

    return {
        "Predicted_Sales": float(prediction)
    }

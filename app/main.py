from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import HeartDiseaseInput


# Create FastAPI application
app = FastAPI(
    title="Heart Disease Prediction API",
    description="FastAPI service for predicting the presence of heart disease.",
    version="1.0.0"
)


# Path to trained model
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "heart_model.joblib"

# Load trained model
model = joblib.load(MODEL_PATH)


# Feature order expected by the model
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/info")
def info():
    return {
        "model_type": type(model).__name__,
        "features": FEATURES
    }


@app.post("/predict")
def predict(data: HeartDiseaseInput):

    input_data = pd.DataFrame(
        [[
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal
        ]],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    return {
        "heart_disease": bool(prediction)
    }
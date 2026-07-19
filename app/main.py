"""
FastAPI service for breast cancer diagnosis prediction.

Run locally:
    uvicorn app.main:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for interactive Swagger UI (great for demo videos).
"""
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Predicts whether a breast tumor is Benign or Malignant from cell nuclei measurements "
                "(Wisconsin Breast Cancer dataset). Trained with Logistic Regression, SVM, Random Forest "
                "and XGBoost; the best-performing model is served here.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load artifacts at startup ----
model = joblib.load(MODELS_DIR / "best_model.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")
with open(MODELS_DIR / "feature_names.json") as f:
    FEATURE_NAMES = json.load(f)
with open(MODELS_DIR / "model_info.json") as f:
    MODEL_INFO = json.load(f)


def to_field_name(col: str) -> str:
    """Turn 'concave points_mean' into a valid Python identifier 'concave_points_mean'."""
    return col.replace(" ", "_")


FIELD_TO_COLUMN = {to_field_name(c): c for c in FEATURE_NAMES}


class TumorFeatures(BaseModel):
    """All 30 measurements from the FNA (fine needle aspirate) digitized image."""
    model_config = ConfigDict(populate_by_name=True)

    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float = Field(alias="concave points_mean")
    symmetry_mean: float
    fractal_dimension_mean: float

    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float = Field(alias="concave points_se")
    symmetry_se: float
    fractal_dimension_se: float

    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float = Field(alias="concave points_worst")
    symmetry_worst: float
    fractal_dimension_worst: float


class PredictionResponse(BaseModel):
    diagnosis: str
    malignant_probability: float
    benign_probability: float
    model_used: str


@app.get("/")
def root():
    return {
        "message": "Breast Cancer Prediction API is running.",
        "docs": "/docs",
        "model_used": MODEL_INFO["best_model_name"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """Return training/evaluation metrics for all models that were compared."""
    return MODEL_INFO


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TumorFeatures):
    try:
        data = features.model_dump(by_alias=True)
        ordered = np.array([[data[col] for col in FEATURE_NAMES]])
        scaled = scaler.transform(ordered)
        proba = model.predict_proba(scaled)[0]
        pred = model.predict(scaled)[0]

        return PredictionResponse(
            diagnosis="Malignant" if pred == 1 else "Benign",
            malignant_probability=round(float(proba[1]), 4),
            benign_probability=round(float(proba[0]), 4),
            model_used=MODEL_INFO["best_model_name"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/sample")
def predict_sample():
    """Convenience endpoint: runs a prediction on a built-in example so you don't
    need to type all 30 values by hand during a demo."""
    sample = {
        "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8, "area_mean": 1001,
        "smoothness_mean": 0.1184, "compactness_mean": 0.2776, "concavity_mean": 0.3001,
        "concave points_mean": 0.1471, "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871,
        "radius_se": 1.095, "texture_se": 0.9053, "perimeter_se": 8.589, "area_se": 153.4,
        "smoothness_se": 0.006399, "compactness_se": 0.04904, "concavity_se": 0.05373,
        "concave points_se": 0.01587, "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
        "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.6, "area_worst": 2019,
        "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119,
        "concave points_worst": 0.2654, "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.1189,
    }
    features = TumorFeatures(**sample)
    return predict(features)

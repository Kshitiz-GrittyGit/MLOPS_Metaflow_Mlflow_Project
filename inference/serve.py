from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import joblib
import mlflow.pyfunc
import pandas as pd
import os
from pathlib import Path

# -------------------------------
# Load artifacts at startup
# -------------------------------

ARTIFACT_DIR = Path("/Users/kshitiztiwari/my_ml_school/mlartifacts/0/17ba0b9d56aa41aa9e891617ad7bcbb0/artifacts/preprocessing")

# You can also fetch from MLflow registry directly if preferred
MODEL_URI = Path("/Users/kshitiztiwari/my_ml_school/mlartifacts/0/17ba0b9d56aa41aa9e891617ad7bcbb0/artifacts/model")

try:
    model = mlflow.xgboost.load_model(MODEL_URI)
    feature_transformer = joblib.load(ARTIFACT_DIR / "features.joblib")
    target_transformer = joblib.load(ARTIFACT_DIR / "target.joblib")
except Exception as e:
    raise RuntimeError(f"Error loading model or transformers: {e}")

# -------------------------------
# Define FastAPI app and schema
# -------------------------------

app = FastAPI(title="Penguin Classifier API", version="1.0")

class PenguinInput(BaseModel):
    island: str
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: Literal["Male", "Female"]

@app.get("/")
def home():
    return {"message": "Penguin Classifier API is live!"}

@app.post("/predict")
def predict(penguin: PenguinInput):
    try:
        input_df = pd.DataFrame([penguin.dict()])
        X = feature_transformer.transform(input_df)
        pred = model.predict(X)

        target_encoder = target_transformer.named_transformers_['target']
        species = target_encoder.inverse_transform(pred.reshape(-1, 1))[0][0]
        return {"prediction": species}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

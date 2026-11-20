
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from config import MODEL_DIR

INFO = json.loads((MODEL_DIR / "model_info.json").read_text())

if INFO["model"] == "MLP":
    from tensorflow import keras
    _model = keras.models.load_model(MODEL_DIR / "model.keras")
    _scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    def _predict(X):
        return _model.predict(_scaler.transform(X), verbose=0).ravel()
else:
    _model = joblib.load(MODEL_DIR / "model.joblib")
    def _predict(X):
        return _model.predict(X)


class WeatherInput(BaseModel):
    time: datetime = Field(description="Start of the hour, same clock as the training data")
    temperature: float = Field(description="Air temperature, deg C")
    humidity: float = Field(ge=0, le=100, description="Relative humidity, %")
    pressure: float = Field(gt=0, description="Air pressure, hPa")
    precipitation: float = Field(0, ge=0, description="Rain in the hour, mm")
    wind_speed: float = Field(ge=0, description="m/s")
    wind_direction: float = Field(ge=0, le=360, description="degrees")


app = FastAPI(title="Narok GTI predictor", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": INFO["model"]}


@app.post("/predict")
def predict(x: WeatherInput):
    row = x.model_dump()
    t = row.pop("time")
    row["hour"] = t.hour
    row["day_of_year"] = t.timetuple().tm_yday
    X = pd.DataFrame([row])[INFO["features"]]
    gti = max(float(_predict(X)[0]), 0.0)
    return {"gti_wm2": round(gti, 1), "model": INFO["model"]}


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

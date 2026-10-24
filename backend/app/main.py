"""FastAPI service exposing the trained credit-risk model.

Run from the project root:
    .venv/bin/uvicorn backend.app.main:app --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.app.limits import BodySizeLimitMiddleware, max_body_bytes
from backend.app.schemas import CustomerFeatures, PredictionResponse
from backend.app.store import (
    ModelStore,
    PredictionLog,
    artifact_dir,
    history_enabled,
    history_retention_days,
    predictions_db_path,
)

DISCLAIMER = (
    "probability_bad is the model's score for the 'bad credit risk' class of the 1994 German Credit "
    "dataset. It is not calibrated (class-weighted training overstates the bad rate), so read it as a "
    "risk score compared against the decision threshold, not as a literal default probability. It is "
    "not a complete real-world credit score and must not be the sole basis for a lending decision."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = ModelStore(artifact_dir()).load()
    app.state.log = PredictionLog(predictions_db_path(), history_retention_days()) if history_enabled() else None
    yield


app = FastAPI(title="Credit Risk Scoring API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if o.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
# Added last so it runs first: rejects oversized bodies before anything reads them.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_body_bytes())


def _require_model():
    store = app.state.store
    if not store.ready:
        raise HTTPException(status_code=503, detail=store.error)
    return store


@app.get("/health")
def health():
    store = app.state.store
    return {
        "status": "ok",
        "model_loaded": store.ready,
        "model_name": store.bundle["model_name"] if store.ready else None,
        "error": store.error,
    }


@app.get("/metrics")
def metrics():
    return _require_model().metrics


@app.get("/model-info")
def model_info():
    return _require_model().model_info


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    store = _require_model()
    features = customer.model_dump()
    result = store.predict(features)
    prediction_id = app.state.log.add(features, result) if app.state.log else None
    return {**result, "prediction_id": prediction_id, "disclaimer": DISCLAIMER}


@app.get("/predictions")
def predictions(limit: int = Query(20, ge=1, le=200)):
    return app.state.log.recent(limit) if app.state.log else []

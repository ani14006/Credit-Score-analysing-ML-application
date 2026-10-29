"""FastAPI service exposing the trained credit-risk model.

Run from the project root:
    .venv/bin/uvicorn backend.app.main:app --port 8000
"""

import hmac
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.app.store import PROJECT_ROOT

# Backend-only secrets (ANTHROPIC_API_KEY) may live in a git-ignored .env; real env vars win.
load_dotenv(PROJECT_ROOT / ".env", override=False)

from backend.app.agent.llm import build_provider  # noqa: E402
from backend.app.agent.ratelimit import RateLimiter  # noqa: E402
from backend.app.agent.routes import router as agent_router  # noqa: E402
from backend.app.agent.storage import AgentStore  # noqa: E402
from backend.app.limits import BodySizeLimitMiddleware, max_body_bytes  # noqa: E402
from backend.app.schemas import CustomerFeatures, PredictionLookup, PredictionResponse  # noqa: E402
from backend.app.store import (  # noqa: E402
    ModelStore,
    PredictionLog,
    artifact_dir,
    hash_token,
    history_enabled,
    history_retention_days,
    new_access_token,
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
    app.state.agent_store = AgentStore(predictions_db_path()) if app.state.log else None
    app.state.agent_provider = build_provider()
    app.state.agent_rate_limiter = RateLimiter(int(os.environ.get("CREDIT_AGENT_RATE_LIMIT_PER_MINUTE", "10")))
    yield


app = FastAPI(title="Credit Risk Scoring API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if o.strip()
    ],
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type", "X-Prediction-Token"],
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
    prediction_id, token = None, None
    if app.state.log:
        token = new_access_token()
        prediction_id = app.state.log.add(features, result, token_hash=hash_token(token))
    return {**result, "prediction_id": prediction_id, "access_token": token, "disclaimer": DISCLAIMER}


@app.post("/predictions/lookup")
def lookup_predictions(body: PredictionLookup):
    """History for the caller's own predictions: only rows whose access token is presented."""
    if not app.state.log:
        return []
    return app.state.log.lookup([item.model_dump() for item in body.items])


@app.get("/predictions")
def predictions(limit: int = Query(20, ge=1, le=200), authorization: str | None = Header(None)):
    """Operator-only listing of all stored predictions (requires CREDIT_ADMIN_TOKEN)."""
    admin_token = os.environ.get("CREDIT_ADMIN_TOKEN", "")
    if not admin_token:
        raise HTTPException(403, "Listing all predictions is disabled. Set CREDIT_ADMIN_TOKEN to enable "
                                 "operator access, or use POST /predictions/lookup with prediction tokens.")
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, admin_token):
        raise HTTPException(401, "Invalid or missing operator token.", headers={"WWW-Authenticate": "Bearer"})
    return app.state.log.recent(limit) if app.state.log else []


app.include_router(agent_router)

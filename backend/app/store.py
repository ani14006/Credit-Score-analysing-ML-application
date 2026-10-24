"""Model artifact loading and prediction-history persistence."""

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from ml.features import MODEL_FEATURES, NUMERIC_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Numeric ranges seen in the training data; inputs outside them are accepted
# but flagged, because the model has to extrapolate.
TRAINING_RANGES = {
    "duration_months": (4, 72),
    "credit_amount": (250, 18424),
    "installment_rate_pct": (1, 4),
    "residence_since": (1, 4),
    "age_years": (19, 75),
    "existing_credits": (1, 4),
    "num_dependents": (1, 2),
}


def artifact_dir():
    return Path(os.environ.get("CREDIT_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))


def predictions_db_path():
    return Path(os.environ.get("CREDIT_PREDICTIONS_DB", PROJECT_ROOT / "var" / "predictions.db"))


class ModelStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.bundle = None
        self.metrics = None
        self.model_info = None
        self.error = None

    def load(self):
        files = {name: self.directory / name for name in ("model.joblib", "metrics.json", "model_info.json")}
        missing = [str(p) for p in files.values() if not p.exists()]
        if missing:
            self.error = f"Missing model artifacts: {', '.join(missing)}. Run `python -m ml.train` first."
            return self
        try:
            bundle = joblib.load(files["model.joblib"])
            if list(bundle["features"]) != MODEL_FEATURES:
                raise ValueError("Artifact feature list does not match ml.features.MODEL_FEATURES")
            self.bundle = bundle
            self.metrics = json.loads(files["metrics.json"].read_text())
            self.model_info = json.loads(files["model_info.json"].read_text())
        except Exception as exc:
            self.bundle = None
            self.error = f"Failed to load model artifacts: {exc}"
        return self

    @property
    def ready(self):
        return self.bundle is not None

    def predict(self, features: dict):
        frame = pd.DataFrame([features], columns=MODEL_FEATURES)
        frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(float)
        probability_bad = float(self.bundle["pipeline"].predict_proba(frame)[0, 1])
        threshold = float(self.bundle["threshold"])
        predicted_class = int(probability_bad >= threshold)
        warnings = [
            f"{name}={features[name]} is outside the training range {lo}-{hi}; the model is extrapolating."
            for name, (lo, hi) in TRAINING_RANGES.items()
            if not lo <= features[name] <= hi
        ]
        return {
            "predicted_class": predicted_class,
            "prediction": "bad_credit_risk" if predicted_class else "good_credit_risk",
            "probability_bad": probability_bad,
            "probability_good": 1.0 - probability_bad,
            "decision_threshold": threshold,
            "model_name": self.bundle["model_name"],
            "model_trained_at": self.bundle["trained_at"],
            "warnings": warnings,
        }


class PredictionLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_trained_at TEXT NOT NULL,
                    features TEXT NOT NULL,
                    probability_bad REAL NOT NULL,
                    predicted_class INTEGER NOT NULL,
                    decision_threshold REAL NOT NULL
                )"""
            )

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, features, result):
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO predictions (created_at, model_name, model_trained_at, features, "
                "probability_bad, predicted_class, decision_threshold) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    result["model_name"],
                    result["model_trained_at"],
                    json.dumps(features),
                    result["probability_bad"],
                    result["predicted_class"],
                    result["decision_threshold"],
                ),
            )
            return cur.lastrowid

    def recent(self, limit=20):
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{**dict(r), "features": json.loads(r["features"])} for r in rows]

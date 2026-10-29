"""Model artifact loading and prediction-history persistence."""

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd

from ml.features import MODEL_FEATURES, NUMERIC_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

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


def history_enabled():
    return os.environ.get("CREDIT_STORE_HISTORY", "true").strip().lower() not in {"0", "false", "no", "off"}


def history_retention_days():
    return int(os.environ.get("CREDIT_HISTORY_RETENTION_DAYS", "30"))


class ModelStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.bundle = None
        self.metrics = None
        self.model_info = None
        self.error = None

    def load(self):
        files = {name: self.directory / name for name in ("model.joblib", "metrics.json", "model_info.json")}
        missing = [name for name, path in files.items() if not path.exists()]
        if missing:
            logger.error("Missing model artifacts in %s: %s", self.directory, missing)
            self.error = (f"Missing model artifacts: {', '.join(missing)}. "
                          "Run `python -m ml.train` first.")
            return self
        try:
            # joblib/pickle executes code on load: only ever load artifacts produced by
            # ml/train.py from the operator-configured directory, never user uploads.
            bundle = joblib.load(files["model.joblib"])
            if list(bundle["features"]) != MODEL_FEATURES:
                self.error = "Failed to load model artifacts: feature list does not match the API schema."
                logger.error(self.error)
                return self
            self.bundle = bundle
            self.metrics = json.loads(files["metrics.json"].read_text())
            self.model_info = json.loads(files["model_info.json"].read_text())
        except Exception as exc:
            self.bundle = None
            logger.exception("Failed to load model artifacts from %s", self.directory)
            self.error = f"Failed to load model artifacts ({type(exc).__name__}). See the server log."
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


def new_access_token():
    """Unguessable per-prediction capability token; only its SHA-256 is stored."""
    return secrets.token_urlsafe(32)


def hash_token(token):
    return hashlib.sha256(token.encode()).hexdigest()


def connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # agent tables cascade-delete with their prediction
    return conn


PUBLIC_COLUMNS = ("id, created_at, model_name, model_trained_at, features, probability_bad, "
                  "predicted_class, decision_threshold")


class PredictionLog:
    def __init__(self, path, retention_days=30):
        self.path = Path(path)
        self.retention_days = retention_days
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
            columns = {r["name"] for r in conn.execute("PRAGMA table_info(predictions)")}
            if "token_hash" not in columns:  # databases created before access tokens existed
                conn.execute("ALTER TABLE predictions ADD COLUMN token_hash TEXT")
        self.purge_expired()

    def purge_expired(self):
        if self.retention_days <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat(timespec="seconds")
        with closing(self._connect()) as conn, conn:
            return conn.execute("DELETE FROM predictions WHERE created_at < ?", (cutoff,)).rowcount

    def _connect(self):
        return connect(self.path)

    def add(self, features, result, token_hash=None):
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO predictions (created_at, model_name, model_trained_at, features, "
                "probability_bad, predicted_class, decision_threshold, token_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    result["model_name"],
                    result["model_trained_at"],
                    json.dumps(features),
                    result["probability_bad"],
                    result["predicted_class"],
                    result["decision_threshold"],
                    token_hash,
                ),
            )
            row_id = cur.lastrowid
        self.purge_expired()
        return row_id

    def recent(self, limit=20):
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {PUBLIC_COLUMNS} FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row(r) for r in rows]

    def get_authorized(self, prediction_id, token):
        """The prediction, only if `token` is the one issued for it; otherwise None."""
        if not token:
            return None
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT {PUBLIC_COLUMNS}, token_hash FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
        if row is None or row["token_hash"] is None:
            return None
        if not hmac.compare_digest(row["token_hash"], hash_token(token)):
            return None
        return _row(row)

    def lookup(self, items):
        """Predictions for which the caller holds the access token (missing or wrong tokens are skipped)."""
        found = [self.get_authorized(i["id"], i["token"]) for i in items]
        return sorted((r for r in found if r), key=lambda r: -r["id"])


def _row(r):
    d = dict(r)
    d.pop("token_hash", None)
    d["features"] = json.loads(d["features"])
    return d

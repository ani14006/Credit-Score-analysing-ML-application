"""Security-sensitive behaviour: error disclosure, body limits, CORS, retention, history toggle."""

import json
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.store import PredictionLog


@pytest.fixture
def env(trained, tmp_path, monkeypatch):
    monkeypatch.setenv("CREDIT_ARTIFACT_DIR", str(trained["artifact_dir"]))
    monkeypatch.setenv("CREDIT_PREDICTIONS_DB", str(tmp_path / "p.db"))
    return monkeypatch


def test_missing_artifact_errors_do_not_disclose_server_paths(tmp_path, monkeypatch, sample_customer):
    secret_dir = tmp_path / "private-server-location"
    monkeypatch.setenv("CREDIT_ARTIFACT_DIR", str(secret_dir))
    monkeypatch.setenv("CREDIT_PREDICTIONS_DB", str(tmp_path / "p.db"))
    with TestClient(app) as c:
        bodies = [c.get("/health").text, c.get("/metrics").text, c.post("/predict", json=sample_customer).text]
    for body in bodies:
        assert "private-server-location" not in body
        assert str(tmp_path) not in body
    assert "model.joblib" in bodies[0]


def test_load_failure_does_not_echo_exception_text(trained, tmp_path, monkeypatch):
    broken = tmp_path / "artifacts"
    shutil.copytree(trained["artifact_dir"], broken)
    (broken / "model.joblib").write_bytes(b"garbage")
    monkeypatch.setenv("CREDIT_ARTIFACT_DIR", str(broken))
    monkeypatch.setenv("CREDIT_PREDICTIONS_DB", str(tmp_path / "p.db"))
    with TestClient(app) as c:
        error = c.get("/health").json()["error"]
    assert str(tmp_path) not in error
    assert error.startswith("Failed to load model artifacts (")


def test_oversized_body_rejected_with_413(env, sample_customer):
    with TestClient(app) as c:
        big = json.dumps({**sample_customer, "padding": "x" * 100_000})
        r = c.post("/predict", content=big, headers={"Content-Type": "application/json"})
        assert r.status_code == 413
        # a normal request still works afterwards
        assert c.post("/predict", json=sample_customer).status_code == 200


def test_chunked_body_without_content_length_is_limited(env, sample_customer):
    def chunks():
        yield b'{"padding": "'
        for _ in range(50):
            yield b"x" * 1024
        yield b'"}'
    with TestClient(app) as c:
        r = c.post("/predict", content=chunks(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_cors_rejects_unknown_origin(env):
    with TestClient(app) as c:
        allowed = c.options("/predict", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
        foreign = c.options("/predict", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
        simple = c.get("/predictions", headers={"Origin": "https://evil.example"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in foreign.headers
    assert "access-control-allow-origin" not in simple.headers
    assert "access-control-allow-credentials" not in allowed.headers


def test_retention_purges_old_rows(tmp_path):
    db = tmp_path / "r.db"
    log = PredictionLog(db, retention_days=30)
    result = {"model_name": "m", "model_trained_at": "t", "probability_bad": 0.5,
              "predicted_class": 1, "decision_threshold": 0.36}
    log.add({"x": 1}, result)
    old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat(timespec="seconds")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE predictions SET created_at = ?", (old,))
    conn.close()
    new_id = log.add({"x": 2}, result)
    assert [r["id"] for r in log.recent()] == [new_id]
    # a fresh instance (backend restart) also purges on startup
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE predictions SET created_at = ?", (old,))
    conn.close()
    assert PredictionLog(db, retention_days=30).recent() == []


def test_history_can_be_disabled(env, sample_customer, tmp_path):
    env.setenv("CREDIT_STORE_HISTORY", "false")
    env.setenv("CREDIT_PREDICTIONS_DB", str(tmp_path / "disabled.db"))
    with TestClient(app) as c:
        r = c.post("/predict", json=sample_customer)
        assert r.status_code == 200 and r.json()["prediction_id"] is None and r.json()["access_token"] is None
        assert c.post("/predictions/lookup", json={"items": []}).json() == []
    assert not (tmp_path / "disabled.db").exists()


def test_sql_injection_in_limit_is_rejected(env):
    env.setenv("CREDIT_ADMIN_TOKEN", "operator-test-token-123456")
    auth = {"Authorization": "Bearer operator-test-token-123456"}
    with TestClient(app) as c:
        assert c.get("/predictions", params={"limit": "1; DROP TABLE predictions"}, headers=auth).status_code == 422
        assert c.get("/predictions", headers=auth).status_code == 200


def test_history_listing_requires_operator_token(env, sample_customer):
    with TestClient(app) as c:
        c.post("/predict", json=sample_customer)
        disabled = c.get("/predictions")
    assert disabled.status_code == 403 and "probability_bad" not in disabled.text
    env.setenv("CREDIT_ADMIN_TOKEN", "operator-test-token-123456")
    with TestClient(app) as c:
        assert c.get("/predictions").status_code == 401
        assert c.get("/predictions", headers={"Authorization": "Bearer wrong-token"}).status_code == 401
        rows = c.get("/predictions", headers={"Authorization": "Bearer operator-test-token-123456"}).json()
    assert rows and "token_hash" not in rows[0]


def test_lookup_returns_only_predictions_with_matching_token(env, sample_customer):
    with TestClient(app) as c:
        a = c.post("/predict", json=sample_customer).json()
        b = c.post("/predict", json={**sample_customer, "credit_amount": 9000}).json()
        own = c.post("/predictions/lookup", json={"items": [{"id": a["prediction_id"], "token": a["access_token"]}]}).json()
        forged = c.post("/predictions/lookup", json={"items": [{"id": b["prediction_id"], "token": a["access_token"]}]}).json()
    assert [r["id"] for r in own] == [a["prediction_id"]]
    assert forged == []
    assert "token_hash" not in own[0] and "access_token" not in own[0]


def test_only_token_hash_is_stored(env, sample_customer, tmp_path):
    with TestClient(app) as c:
        r = c.post("/predict", json=sample_customer).json()
    with sqlite3.connect(tmp_path / "p.db") as conn:
        stored = conn.execute("SELECT token_hash FROM predictions WHERE id = ?", (r["prediction_id"],)).fetchone()[0]
    conn.close()
    assert r["access_token"] not in stored and len(stored) == 64

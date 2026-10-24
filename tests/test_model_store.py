import json
import shutil

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.store import ModelStore
from ml.features import MODEL_FEATURES


def test_store_loads_real_artifact(trained):
    store = ModelStore(trained["artifact_dir"]).load()
    assert store.ready and store.error is None
    assert type(store.bundle["pipeline"].named_steps["model"]).__name__ in {
        "RandomForestClassifier", "LogisticRegression"}
    assert store.bundle["features"] == MODEL_FEATURES
    assert store.metrics == trained["metrics"]


def test_store_predict_calls_pipeline_predict_proba(trained, sample_customer, monkeypatch):
    store = ModelStore(trained["artifact_dir"]).load()
    calls = []
    original = store.bundle["pipeline"].predict_proba
    def spy(X):
        calls.append(X.copy())
        return original(X)
    monkeypatch.setattr(store.bundle["pipeline"], "predict_proba", spy)
    result = store.predict(sample_customer)
    assert len(calls) == 1
    assert list(calls[0].columns) == MODEL_FEATURES
    assert result["probability_bad"] == pytest.approx(original(calls[0])[0, 1])


def test_corrupted_model_file_is_reported(trained, tmp_path):
    broken = tmp_path / "artifacts"
    shutil.copytree(trained["artifact_dir"], broken)
    (broken / "model.joblib").write_bytes(b"not a joblib file")
    store = ModelStore(broken).load()
    assert not store.ready
    assert "Failed to load model artifacts" in store.error


def test_feature_mismatch_is_rejected(trained, tmp_path):
    bad = tmp_path / "artifacts"
    shutil.copytree(trained["artifact_dir"], bad)
    bundle = joblib.load(bad / "model.joblib")
    bundle["features"] = list(reversed(bundle["features"]))
    joblib.dump(bundle, bad / "model.joblib")
    store = ModelStore(bad).load()
    assert not store.ready
    assert "does not match" in store.error


def test_api_reports_corrupted_artifact(trained, tmp_path, monkeypatch, sample_customer):
    broken = tmp_path / "artifacts"
    shutil.copytree(trained["artifact_dir"], broken)
    (broken / "model.joblib").write_bytes(b"garbage")
    monkeypatch.setenv("CREDIT_ARTIFACT_DIR", str(broken))
    monkeypatch.setenv("CREDIT_PREDICTIONS_DB", str(tmp_path / "p.db"))
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        r = c.post("/predict", json=sample_customer)
        assert r.status_code == 503
        assert "Failed to load" in r.json()["detail"]


def test_metrics_include_calibration_diagnostics(trained):
    for result in trained["metrics"]["models"].values():
        cal = result["test"]["calibration"]
        assert 0 <= cal["brier_score"] <= 1
        assert cal["observed_bad_rate"] == pytest.approx(trained["y_test"].mean())
        assert len(cal["reliability_bins"]) >= 3
    assert "Not calibrated" in trained["model_info"]["probability_calibration"]

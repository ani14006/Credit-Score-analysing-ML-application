import json

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from ml.features import MODEL_FEATURES


@pytest.fixture
def client(trained, tmp_path, monkeypatch):
    monkeypatch.setenv("CREDIT_ARTIFACT_DIR", str(trained["artifact_dir"]))
    monkeypatch.setenv("CREDIT_PREDICTIONS_DB", str(tmp_path / "predictions.db"))
    with TestClient(app) as c:
        yield c


def test_health_reports_model_loaded(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["error"] is None


def test_metrics_and_model_info_come_from_training_artifacts(client, trained):
    assert client.get("/metrics").json() == trained["metrics"]
    info = client.get("/model-info").json()
    assert info["selected_model"] == trained["model_info"]["selected_model"]
    assert set(info["numeric_features"] + info["categorical_features"]) == set(MODEL_FEATURES)


def test_predict_matches_saved_model_output(client, trained, sample_customer):
    """The API's probability must equal the joblib pipeline's own predict_proba for the same row."""
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    body = response.json()

    expected = trained["bundle"]["pipeline"].predict_proba(trained["X_test"].iloc[[0]].astype(
        {f: float for f in trained["model_info"]["numeric_features"]}))[0, 1]
    assert body["probability_bad"] == pytest.approx(expected)
    assert body["predicted_class"] == int(expected >= trained["bundle"]["threshold"])
    assert body["probability_good"] == pytest.approx(1 - expected)


def test_predictions_vary_with_input_and_agree_with_model_on_test_set(client, trained):
    X = trained["X_test"].head(25)
    expected = trained["bundle"]["pipeline"].predict_proba(X)[:, 1]
    got = []
    for _, row in X.iterrows():
        payload = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        got.append(client.post("/predict", json=payload).json()["probability_bad"])
    assert got == pytest.approx(list(expected))
    assert len(set(round(p, 6) for p in got)) > 10  # not a constant / canned response


def test_riskier_profile_gets_higher_probability(client, sample_customer):
    safe = {**sample_customer, "checking_status": "no_checking_account", "credit_history": "critical_or_other_credits",
            "duration_months": 6, "credit_amount": 1000, "savings_status": "ge_1000_dm"}
    risky = {**sample_customer, "checking_status": "lt_0_dm", "credit_history": "no_credits_or_all_paid",
             "duration_months": 60, "credit_amount": 15000, "savings_status": "lt_100_dm"}
    p_safe = client.post("/predict", json=safe).json()["probability_bad"]
    p_risky = client.post("/predict", json=risky).json()["probability_bad"]
    assert p_risky > p_safe


@pytest.mark.parametrize("change, field", [
    ({"duration_months": 0}, "duration_months"),
    ({"credit_amount": "lots"}, "credit_amount"),
    ({"purpose": "yacht"}, "purpose"),
    ({"age_years": 12}, "age_years"),
])
def test_invalid_input_returns_422(client, sample_customer, change, field):
    response = client.post("/predict", json={**sample_customer, **change})
    assert response.status_code == 422
    assert any(err["loc"][-1] == field for err in response.json()["detail"])


def test_missing_and_unknown_fields_rejected(client, sample_customer):
    missing = {k: v for k, v in sample_customer.items() if k != "housing"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**sample_customer, "personal_status_sex": "male_single"}).status_code == 422


def test_out_of_range_input_is_flagged(client, sample_customer):
    body = client.post("/predict", json={**sample_customer, "credit_amount": 50000}).json()
    assert any("credit_amount" in w for w in body["warnings"])


def test_prediction_history_is_persisted(client, sample_customer, tmp_path):
    first = client.post("/predict", json=sample_customer).json()
    ref = {"id": first["prediction_id"], "token": first["access_token"]}
    history = client.post("/predictions/lookup", json={"items": [ref]}).json()
    assert history[0]["id"] == first["prediction_id"]
    assert history[0]["probability_bad"] == pytest.approx(first["probability_bad"])

    from backend.app.store import PredictionLog
    assert PredictionLog(tmp_path / "predictions.db").recent()[0]["id"] == first["prediction_id"]


def test_missing_artifacts_return_503(tmp_path, monkeypatch, sample_customer):
    monkeypatch.setenv("CREDIT_ARTIFACT_DIR", str(tmp_path / "nothing-here"))
    monkeypatch.setenv("CREDIT_PREDICTIONS_DB", str(tmp_path / "p.db"))
    with TestClient(app) as c:
        health = c.get("/health").json()
        assert health["model_loaded"] is False
        assert "Missing model artifacts" in health["error"]
        for path in ("/metrics", "/model-info"):
            assert c.get(path).status_code == 503
        assert c.post("/predict", json=sample_customer).status_code == 503


def test_metrics_endpoint_content_matches_artifact_file(client, trained):
    on_disk = json.loads((trained["artifact_dir"] / "metrics.json").read_text())
    body = client.get("/metrics").json()
    assert body == on_disk
    selected = body["models"][body["selected_model"]]["test"]
    cm = selected["confusion_matrix"]
    tp, fp = cm["tp"], cm["fp"]
    assert selected["precision"] == pytest.approx(tp / (tp + fp))
    assert len(selected["roc_curve"]) > 10


def test_model_info_endpoint_matches_artifact_file(client, trained):
    on_disk = json.loads((trained["artifact_dir"] / "model_info.json").read_text())
    assert client.get("/model-info").json() == on_disk


@pytest.mark.parametrize("body", [
    {},
    {"checking_status": 3},
    "not json",
])
def test_malformed_bodies_return_422(client, sample_customer, body):
    if isinstance(body, dict) and body:
        body = {**sample_customer, **body}
    if isinstance(body, str):
        response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    else:
        response = client.post("/predict", json=body)
    assert response.status_code == 422


def test_invalid_history_limit_returns_422(client):
    assert client.get("/predictions?limit=0").status_code == 422
    assert client.get("/predictions?limit=abc").status_code == 422


def test_history_survives_app_restart(trained, tmp_path, monkeypatch, sample_customer):
    db = tmp_path / "persist.db"
    monkeypatch.setenv("CREDIT_ARTIFACT_DIR", str(trained["artifact_dir"]))
    monkeypatch.setenv("CREDIT_PREDICTIONS_DB", str(db))
    with TestClient(app) as c:
        saved = c.post("/predict", json=sample_customer).json()
    with TestClient(app) as c:  # new lifespan = backend restart
        history = c.post("/predictions/lookup",
                         json={"items": [{"id": saved["prediction_id"], "token": saved["access_token"]}]}).json()
    assert [r["id"] for r in history] == [saved["prediction_id"]]
    assert history[0]["probability_bad"] == pytest.approx(saved["probability_bad"])
    assert history[0]["features"] == sample_customer

    import sqlite3
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT id, probability_bad FROM predictions").fetchall()
    assert rows == [(saved["prediction_id"], pytest.approx(saved["probability_bad"]))]

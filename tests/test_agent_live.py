"""LIVE LLM integration test - calls the real provider (Gemini or Anthropic, per .env) through the
app code. Opt-in: excluded from normal runs by pytest.ini; run it with
    .venv/bin/python -m pytest -m live -v
It makes three real requests (explanation, follow-up, 30-day plan) and uses a little API quota.
"""

import os

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not (os.environ.get("GEMINI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
                       reason="no GEMINI_API_KEY / ANTHROPIC_API_KEY set: live LLM test not run"),
]


def test_live_agent_end_to_end(trained, tmp_path, monkeypatch, sample_customer):
    monkeypatch.setenv("CREDIT_ARTIFACT_DIR", str(trained["artifact_dir"]))
    monkeypatch.setenv("CREDIT_PREDICTIONS_DB", str(tmp_path / "live.db"))
    with TestClient(app) as c:
        assert c.get("/agent/status").json()["configured"] is True
        pred = c.post("/predict", json=sample_customer).json()
        h = {"X-Prediction-Token": pred["access_token"]}

        r = c.post("/agent/sessions", json={"prediction_id": pred["prediction_id"],
                                            "user_context": {"goal": "avoid missing EMI payments"}}, headers=h)
        assert r.status_code == 201, r.text
        session = r.json()
        exp = session["explanation"]
        assert exp["prioritized_steps"] and exp["uncertainty_and_limitations"]
        text = " ".join([exp["prediction_meaning"], *exp["uncertainty_and_limitations"]]).lower()
        assert any(w in text for w in ("1994", "german", "not a credit", "bureau", "cibil")), text

        q = c.post(f"/agent/sessions/{session['session_id']}/messages",
                   json={"question": "What information would you need to give more specific advice?"}, headers=h)
        assert q.status_code == 200, q.text
        assert len(q.json()["answer"]) > 40

        p = c.post(f"/agent/sessions/{session['session_id']}/plan", json={"horizon_days": 30}, headers=h)
        assert p.status_code == 201, p.text
        plan = p.json()
        assert len(plan["tasks"]) >= 3 and all(1 <= t["week"] <= 5 for t in plan["tasks"])
        print("\nLIVE model:", session["model"], "| headline:", exp["headline"])
        print("LIVE follow-up answer:", q.json()["answer"][:300].replace("\n", " "))
        print("LIVE plan:", plan["title"], "|", [(t["week"], t["title"]) for t in plan["tasks"]][:6])

        t = c.post("/agent/text", json={"prediction_id": pred["prediction_id"],
                                        "question": "Explain credit utilization in simple terms."}, headers=h)
        assert t.status_code == 200 and t.headers["content-type"].startswith("text/plain"), t.text
        print("LIVE /agent/text:", t.text[:300].replace("\n", " "))

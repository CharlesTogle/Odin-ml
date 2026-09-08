from __future__ import annotations

from tests.conftest import load_transactions


def test_pfp_standard_classify(client):
    txns = load_transactions()
    payload = {"user_id": "test-user-1", "classification_mode": "STANDARD",
               "payload": {"historical_transactions": txns}}
    resp = client.post("/api/v1/pfp/classify", json=payload)
    # Interim scope: PFP training is deferred; STANDARD mode needs the model.
    assert resp.status_code == 503


def test_pfp_questionnaire_classify(client):
    payload = {
        "user_id": "test-user-2",
        "classification_mode": "QUESTIONNAIRE",
        "payload": {
            "questionnaire_answers": {
                "income_variability": "variable",
                "obligation_level": "high",
                "emergency_runway": "low",
            }
        },
    }
    resp = client.post("/api/v1/pfp/classify", json=payload)
    assert resp.status_code == 200
    classification = resp.json()["classification"]
    assert classification["prediction"] == "VARIABLE_OBLIGATED_AT_RISK"
    assert classification["status"] == "SUCCESS"


def test_pfp_rejects_invalid_mode(client):
    payload = {"user_id": "test-user-3", "classification_mode": "ENSEMBLE",
               "payload": {"historical_transactions": []}}
    resp = client.post("/api/v1/pfp/classify", json=payload)
    assert resp.status_code == 422

from __future__ import annotations

from tests.conftest import load_transactions


def test_anomaly_detect(client):
    txns = load_transactions()
    payload = {"user_id": "test-user-1", "transactions": txns}
    resp = client.post("/api/v1/anomaly/detect", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "anomalous_transactions" in body
    assert "overspending_transactions" in body
    assert isinstance(body["anomalous_transactions"], list)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["status"] in ("SUCCESS", "FALLBACK")


def test_anomaly_detect_batch(client):
    txns = load_transactions(n=90)
    payload = {
        "requests": [
            {"user_id": "test-user-1", "transactions": txns[:45]},
            {"user_id": "test-user-2", "transactions": txns[45:]},
        ]
    }
    resp = client.post("/api/v1/anomaly/detect/batch", json=payload)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert all("anomalous_transactions" in r for r in results)

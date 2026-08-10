from __future__ import annotations

from tests.conftest import load_transactions


def test_anomaly_detect(client):
    txns = load_transactions()
    payload = {"user_id": "test-user-1", "transactions": txns}
    resp = client.post("/api/v1/anomaly/detect", json=payload)
    assert resp.status_code == 200
    anomaly = resp.json()["anomaly"]
    assert "is_anomalous" in anomaly
    assert 0 <= anomaly["score"] <= 1
    assert anomaly["explanation"]


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

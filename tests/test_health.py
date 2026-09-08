from __future__ import annotations


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_metrics(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    # Interim scope: forecaster + anomaly are trained; pfp is deferred.
    assert {"forecaster", "anomaly"} <= set(body["modules"])

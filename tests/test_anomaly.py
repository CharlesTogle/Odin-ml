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


def test_anomaly_overspending_detects_excess(client):
    txns = load_transactions(n=60)
    low_budget = 1.0
    payload = {
        "user_id": "test-user-3",
        "detection_type": "OVERSPENDING",
        "transactions": txns,
        "budget_allocations": [{"category_id": "food", "budget_amount": low_budget}],
    }
    resp = client.post("/api/v1/anomaly/detect", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert any(o["category"] == "food" for o in body["overspending_transactions"])
    assert body["status"] in ("SUCCESS", "FALLBACK")


def test_adaptive_threshold_detector_scores():
    """AdaptiveThresholdDetector (awarded Tier 2 candidate) is an in-scope,
    app-unpicklable scorer: it must live in app.ml.models and behave like the
    other statistical detectors (fit -> [0, )-bounded score)."""
    import numpy as np

    from app.ml.models import AdaptiveThresholdDetector, IQRDetector

    rng = np.random.RandomState(42)
    X = rng.normal(0.0, 1.0, size=(500, 4))

    adaptive = AdaptiveThresholdDetector(iqr_multiplier=1.5).fit(X)
    scores = adaptive.score(X)
    assert scores.shape == (500,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()

    outlier = X.max(axis=0) * 5.0
    assert adaptive.score(outlier[None, :])[0] > scores.mean()

    baseline = IQRDetector(iqr_multiplier=1.5).fit(X)
    assert np.isfinite(baseline.score(X)).all()

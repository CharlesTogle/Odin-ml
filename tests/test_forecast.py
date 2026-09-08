from __future__ import annotations

from tests.conftest import load_transactions


def test_forecast_predict(client):
    txns = load_transactions()
    payload = {"user_id": "test-user-1", "historical_transactions": txns,
               "forecast_horizon": "MONTHLY", "forecast_level": "TOTAL"}
    resp = client.post("/api/v1/forecast/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("SUCCESS", "FALLBACK")
    assert len(body["forecasts"]) == 1
    assert body["forecasts"][0]["amount"] >= 0
    assert body["confidence_intervals"]["lower_95"] <= body["confidence_intervals"]["upper_95"]


def test_forecast_predict_arima_success(client):
    """ARIMA (current winner) must serve a real prediction: SUCCESS, not fallback."""
    txns = load_transactions()
    payload = {"user_id": "test-user-1", "historical_transactions": txns,
               "forecast_horizon": "MONTHLY", "forecast_level": "TOTAL"}
    resp = client.post("/api/v1/forecast/predict", json=payload)
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "SUCCESS"
    assert body["forecasts"][0]["amount"] > 0


def test_forecast_rejects_short_history(client):
    txns = load_transactions(n=1)
    payload = {"user_id": "test-user-2", "historical_transactions": txns,
               "forecast_horizon": "MONTHLY", "forecast_level": "TOTAL"}
    resp = client.post("/api/v1/forecast/predict", json=payload)
    # Falls back (FALLBACK) or errors gracefully; never a 5xx.
    assert resp.status_code in (200, 422)


def test_forecast_rejects_empty_transactions(client):
    payload = {"user_id": "test-user-3", "historical_transactions": []}
    resp = client.post("/api/v1/forecast/predict", json=payload)
    assert resp.status_code == 422

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


class _StubPooled:
    def forecast(self, steps):
        return _StubSeries()


class _StubSeries:
    @property
    def iloc(self):
        # Mirror pandas: result.iloc[0] indexes the first forecast point.
        return self

    def __getitem__(self, _):
        return 5.0


def test_forecast_sarima_branch_served(client):
    """SARIMA-kind artifacts resolve through the pooled forecast branch."""
    from app.models.registry import ModuleModel
    from app.services.forecast_service import _predict_monthly_total

    model = ModuleModel(
        module="forecaster",
        model={"kind": "sarima", "model": _StubPooled(), "pool_level": 1.0},
        evaluation={},
        feature_columns=[],
    )
    txns = [
        {"date": "2023-01-05", "amount": 100.0, "category": "food", "transaction_type": "debit"},
        {"date": "2023-01-20", "amount": 200.0, "category": "transport", "transaction_type": "debit"},
    ]
    pred, intervals = _predict_monthly_total(model, txns)

    import numpy as np

    assert np.isfinite(pred)
    assert pred > 0
    assert intervals["lower_95"] <= intervals["upper_95"]


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

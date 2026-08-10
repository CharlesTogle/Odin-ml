from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.registry import ModelRegistry


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as c:
        assert c.get("/ready").json()["status"] == "ready"
        yield c


@pytest.fixture(scope="session")
def registry() -> ModelRegistry:
    return app.state.registry


def load_transactions(n: int = 120, seed: int = 7) -> list[dict]:
    import pandas as pd

    txn = pd.read_parquet("training/synth/transactions.parquet",
                          columns=["date", "amount", "category", "transaction_type"])
    txn = txn.sample(n, random_state=seed)
    return [
        {
            "date": str(r.date),
            "amount": float(r.amount),
            "category": r.category,
            "transaction_type": r.transaction_type,
        }
        for _, r in txn.iterrows()
    ]

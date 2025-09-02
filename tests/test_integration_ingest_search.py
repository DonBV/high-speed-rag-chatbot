import os

import pytest
from fastapi.testclient import TestClient

from app.main import app

skip_if_no_integration = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="RUN_INTEGRATION != 1: run docker compose and set the variable",
)

client = TestClient(app)


@skip_if_no_integration
def test_ingest_and_search_roundtrip():
    payload = {"id": 101, "content": "Postgres with pgvector enables vector search"}
    r1 = client.post("/ingest", json=payload)
    assert r1.status_code == 200, r1.text

    r2 = client.post("/search", json={"query": "vector search", "k": 3})
    assert r2.status_code == 200, r2.text
    data = r2.json()
    assert "hits" in data and isinstance(data["hits"], list)

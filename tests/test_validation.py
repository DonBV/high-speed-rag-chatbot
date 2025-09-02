from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_docs_alive():
    assert client.get("/docs").status_code == 200

def test_ingest_requires_content():
    resp = client.post("/ingest", json={"id": 1})
    assert resp.status_code == 422  # FastAPI schema validation

def test_search_requires_query():
    resp = client.post("/search", json={"k": 3})
    assert resp.status_code == 422

def test_search_wrong_types():
    resp = client.post("/search", json={"query": "hello", "k": "3"})  # k must be int
    assert resp.status_code == 422

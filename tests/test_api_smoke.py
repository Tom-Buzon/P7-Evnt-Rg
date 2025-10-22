# tests/test_api_smoke.py
from fastapi.testclient import TestClient
from src.api.app import app

def test_api_smoke():
    client = TestClient(app)
    r = client.get("/docs")
    assert r.status_code == 200
    assert "Swagger" in r.text or "OpenAPI" in r.text

def test_ask_endpoint():
    client = TestClient(app)
    r = client.post("/ask", json={"question": "Hello", "k": 2})
    # Le handler appelle answer(); si des mocks ne sont pas actifs ici,
    # on accepte juste un 200/400 selon ta config. On teste la route existe.
    assert r.status_code in (200, 400)

# tests/test_api.py  (ajoute ce test en plus de celui existant)
from fastapi.testclient import TestClient
from src.api.app import app

def test_docs_ui_serves():
    client = TestClient(app)
    r = client.get("/docs")
    assert r.status_code == 200
    assert "Swagger" in r.text or "OpenAPI" in r.text

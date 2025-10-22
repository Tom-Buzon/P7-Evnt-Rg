# tests/test_api.py
from __future__ import annotations
from fastapi.testclient import TestClient

def test_api_ask_endpoint(monkeypatch):
    """
    Teste l’API FastAPI sans lancer de vrai LLM : on monkeypatch `answer`.
    """
    from src.api.app import app

    def _fake_answer(q, k=5, **kwargs):
        return {"answer": "OK", "contexts": ["ctx"], "ranking": [], "k": k}

    # remplace la fonction réelle
    import src.api.app as api_mod
    monkeypatch.setattr(api_mod, "answer", _fake_answer, raising=True)

    client = TestClient(app)
    r = client.post("/ask", json={"question": "ping", "k": 3})
    assert r.status_code == 200
    payload = r.json()
    assert payload["answer"] == "OK"
    assert payload["k"] == 3

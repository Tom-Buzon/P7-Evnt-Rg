# tests/conftest.py
from __future__ import annotations
import sys
import types
import pytest

# FakeEmbeddings selon la version
try:
    from langchain_core.embeddings import FakeEmbeddings
except Exception:
    from langchain.embeddings import FakeEmbeddings  # type: ignore

# FAISS peut être à deux emplacements
try:
    from langchain_community.vectorstores import FAISS
except ModuleNotFoundError:
    from langchain.vectorstores import FAISS  # type: ignore


@pytest.fixture(scope="session")
def tmp_repo(tmp_path_factory):
    return tmp_path_factory.mktemp("repo_tmp")


@pytest.fixture(autouse=True)
def patch_settings(tmp_repo, monkeypatch):
    from src.config import settings
    clean_dir = tmp_repo / "data" / "clean"
    index_dir = tmp_repo / "artifacts" / "index"
    clean_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "clean_dir", clean_dir, raising=False)
    monkeypatch.setattr(settings, "index_dir", index_dir, raising=False)
    return {"clean_dir": clean_dir, "index_dir": index_dir}


@pytest.fixture
def tiny_index(patch_settings):
    """Crée un mini index FAISS sur disque avec FakeEmbeddings."""
    docs = [
        ("Concert à Toulouse le 30 juin 2024 au Rex (rock)", {
            "uid": "u1", "title": "Kraken Fest 2024", "city": "Toulouse",
            "start": "2024-06-30T14:30:00Z", "url": "https://exemple/kraken"
        }),
        ("Exposition à Montpellier en septembre 2024 (photo)", {
            "uid": "u2", "title": "Expo Photo Montpellier", "city": "Montpellier",
            "start": "2024-09-10T09:00:00Z", "url": "https://exemple/expo"
        }),
        ("Festival de jazz à Albi début juillet 2024", {
            "uid": "u3", "title": "Jazz à Albi", "city": "Albi",
            "start": "2024-07-02T18:00:00Z", "url": "https://exemple/jazz"
        }),
    ]
    emb = FakeEmbeddings(size=10)
    from src.config import settings as _s
    vs = FAISS.from_texts([t for t, _ in docs], embedding=emb, metadatas=[m for _, m in docs])
    vs.save_local(str(_s.index_dir))
    return {"embeddings": emb, "docs": docs}


@pytest.fixture
def patch_ollama_module(monkeypatch, tiny_index):
    """
    Injecte un module factice 'langchain_ollama' dans sys.modules :
      - OllamaEmbeddings -> wrappe FakeEmbeddings
      - ChatOllama -> renvoie toujours une réponse fixe
    Ainsi, `from langchain_ollama import OllamaEmbeddings` dans answer() fonctionne sans vrai Ollama.
    """
    fake_mod = types.ModuleType("langchain_ollama")
    _emb = tiny_index["embeddings"]

    class _OllamaEmbeddings:
        def __init__(self, *a, **k): pass
        def embed_query(self, text: str): return _emb.embed_query(text)
        def embed_documents(self, texts): return _emb.embed_documents(texts)

    class _ChatOllama:
        def __init__(self, **kwargs): pass
        class _Resp:
            content = "Réponse factice (Ollama mock)."
        def invoke(self, text: str):
            return self._Resp()

    fake_mod.OllamaEmbeddings = _OllamaEmbeddings
    fake_mod.ChatOllama = _ChatOllama

    sys.modules["langchain_ollama"] = fake_mod
    return fake_mod


@pytest.fixture
def patch_embeddings(patch_ollama_module):
    """
    Fixture attendue par tes tests existants. Elle garantit juste que
    le faux module 'langchain_ollama' est injecté (via patch_ollama_module).
    """
    return True


@pytest.fixture
def patch_llm_ollama(patch_ollama_module):
    """
    Fixture attendue par tes tests existants. Expose la classe factice ChatOllama si besoin.
    """
    return patch_ollama_module.ChatOllama


@pytest.fixture
def patch_llm_mistral(monkeypatch):
    """
    Mock Mistral pour ne pas appeler l'API réelle : structure .choices[0].message.content
    """
    import src.scripts.query_rag as qr
    msg = types.SimpleNamespace(content="Réponse factice (Mistral mock).")
    choice = types.SimpleNamespace(message=msg)
    resp = types.SimpleNamespace(choices=[choice])

    class _DummyChat:
        @staticmethod
        def complete(**kwargs):
            return resp

    class _DummyClient:
        chat = _DummyChat()
        def __init__(self, api_key: str | None = None): pass

    monkeypatch.setattr(qr, "Mistral", _DummyClient, raising=True)
    return _DummyClient

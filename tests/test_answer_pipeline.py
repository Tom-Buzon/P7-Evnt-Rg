# tests/test_answer_pipeline.py
from __future__ import annotations

def test_answer_with_mistral_backend(tiny_index, patch_embeddings, patch_llm_mistral):
    from src.scripts.query_rag import answer
    res = answer(
        "Quels concerts à Toulouse en juin 2024 ?",
        k=3,
        use_mmr=True,
        temperature=0.2,
        max_tokens=120,
        chat_backend="mistral",
        mistral_model="mistral-medium-2508",
    )
    assert "Réponse factice (Mistral mock)" in res["answer"]
    assert len(res["contexts"]) >= 1
    assert len(res["ranking"]) >= 1


def test_answer_with_ollama_backend(tiny_index, patch_embeddings, patch_llm_ollama):
    from src.scripts.query_rag import answer
    res = answer(
        "Expositions à Montpellier en septembre 2024 ?",
        k=2,
        use_mmr=True,
        chat_backend="ollama",
        ollama_chat_model="llama2",
        ollama_base_url="http://localhost:11434",
    )
    assert "Réponse factice (Ollama mock)" in res["answer"]
    assert len(res["ranking"]) >= 1

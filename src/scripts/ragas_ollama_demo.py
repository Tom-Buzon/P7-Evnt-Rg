
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any, List

import ollama
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------- Minimal LangChain-like result objects ----------
@dataclass
class _FakeGeneration:
    text: str


class _FakeLLMResult:
    """
    Mimics langchain_core.outputs.LLMResult just enough for ragas:
      .generations -> List[List[_FakeGeneration]]
    """

    def __init__(self, texts: List[str]):
        self.generations = [[_FakeGeneration(t)] for t in texts]

    # NEW: make the object awaitable
    def __await__(self):
        async def _coro():
            return self
        return _coro().__await__()


# ---------- Ollama wrapper shaped like a LangChain LLM ----------
class OllamaJudge:
    """
    Tiny adapter with the interface Ragas expects.

    Implements:
      - set_run_config
      - generate(prompts, **kwargs) -> _FakeLLMResult
      - agenerate(prompts, **kwargs) -> _FakeLLMResult
      - generate_text / agenerate_text helpers

    Accepts **kwargs (callbacks, temperature, etc.) and ignores them
    so ragas can pass whatever it wants.
    """

    def __init__(self, model: str = "mistral"):
        self.model = model
        self._run_config = None

    # ---- compatibility with ragas ----
    def set_run_config(self, run_config: Any) -> None:
        self._run_config = run_config  # ragas calls this; we don't use it

    # ---- internal ----
    @staticmethod
    def _coerce_prompt(prompt: Any) -> str:
        # Ragas sometimes passes tuples like ("prompt_str", "<text>")
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, tuple) and len(prompt) == 2 and isinstance(prompt[1], str):
            return prompt[1]
        try:
            return str(prompt)
        except Exception:
            return repr(prompt)

    def _chat_once(self, prompt: Any) -> str:
        text = self._coerce_prompt(prompt)
        resp = ollama.chat(model=self.model, messages=[{"role": "user", "content": text}])
        return resp.get("message", {}).get("content", "")

    # ---- text helpers (some codepaths may use these) ----
    def generate_text(self, prompt: Any, **kwargs) -> str:
        return self._chat_once(prompt)

    async def agenerate_text(self, prompt: Any, **kwargs) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._chat_once, prompt)

    # ---- LangChain-like batch APIs that ragas actually calls ----
    def generate(self, prompts: List[Any], **kwargs) -> _FakeLLMResult:
        texts = [self._chat_once(p) for p in prompts]
        return _FakeLLMResult(texts)

    async def agenerate(self, prompts: List[Any], **kwargs) -> _FakeLLMResult:
        loop = asyncio.get_running_loop()
        texts = await asyncio.gather(*[loop.run_in_executor(None, self._chat_once, p) for p in prompts])
        return _FakeLLMResult(list(texts))


def build_demo_dataset() -> Dataset:
    """Minimal dataset for context_* metrics."""
    rows = [
        {
            "user_input": "Quels concerts à Toulouse en juin 2024 ?",
            "retrieved_contexts": [
                "Titre: Concert à Toulouse le 30 juin 2024 au Rex (rock) | Ville: Toulouse | Début: 2024-06-30T14:30:00Z | URL: https://exemple.org/rex",
                "Titre: Festival Musiques du Monde | Ville: Toulouse | Début: 2024-06-15T20:00:00Z | URL: https://exemple.org/fmm",
                "Titre: Fête de la Musique à Paris | Ville: Paris | Début: 2024-06-21T10:00:00Z | URL: https://exemple.org/fdm-paris",
            ],
            "reference": "Concert au Rex (30 juin) et Festival Musiques du Monde (15 juin) à Toulouse.",
        },
        {
            "user_input": "Expositions à Montpellier en septembre 2024 ?",
            "retrieved_contexts": [
                "Titre: Expo A | Ville: Montpellier | Début: 2024-09-05T09:00:00Z | URL: https://exemple.org/expo-a",
                "Titre: Expo B | Ville: Montpellier | Début: 2024-09-01T00:00:00Z | URL: https://exemple.org/expo-b",
            ],
            "reference": "Expo A (5–20 sept.) et Expo B (tout septembre) à Montpellier.",
        },
        {
            "user_input": "Que faire à Paris cet été ?",
            "retrieved_contexts": [
                "Titre: Paris Plages 2024 | Ville: Paris | Début: 2024-07-01T00:00:00Z | URL: https://exemple.org/paris-plages",
                "Titre: Fête de la Musique à Paris | Ville: Paris | Début: 2024-06-21T10:00:00Z | URL: https://exemple.org/fdm-paris",
                "Titre: Cinéma en plein air | Ville: Paris | Début: 2024-07-10T21:30:00Z | URL: https://exemple.org/cinema-plein-air",
            ],
            "reference": "Paris Plages, Fête de la Musique, Cinéma en plein air (été).",
        },
    ]
    return Dataset.from_list(rows)


def main() -> int:
    parser = argparse.ArgumentParser("Ragas + Ollama demo")
    parser.add_argument("--ollama-model", default="mistral", help="Ollama model (mistral, llama3.2, ...)")
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embeddings model",
    )
    args = parser.parse_args()

    ds = build_demo_dataset()
    print("🔎 Exemple envoyé à Ragas:")
    print(ds[0])

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=args.embed_model)

    # Judge
    judge = OllamaJudge(model=args.ollama_model)

    # Metrics (these require an LLM in ragas==0.1.19)
    metrics = [context_precision, context_recall]

    print("⚙️  Lancement de l'évaluation (context_precision, context_recall) ...")
    res = evaluate(dataset=ds, metrics=metrics, llm=judge, embeddings=embeddings)

    df = res.to_pandas()
    print("\n✅ RÉSULTATS")
    print(df)
    print("\nMOYENNES:")
    print(df.mean(numeric_only=True))

    out = "tests/ragas_report_demo.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n📝 Rapport sauvegardé: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

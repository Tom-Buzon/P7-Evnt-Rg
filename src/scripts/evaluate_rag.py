# src/scripts/evaluate_rag.py
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

from datasets import Dataset, Features, Sequence, Value
from ragas import evaluate
from ragas.metrics import context_precision, context_recall

# Use the community import (works with langchain 0.3.x)
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
os.environ.setdefault("OPENAI_API_KEY", "sk-fake-please-ignore")


# In ragas==0.1.19, context_* metrics still expect a 'reference' (string).
RAGAS_REQUIRED = {"question", "contexts", "ground_truths", "reference"}


def load_and_sanitize(path: pathlib.Path) -> List[Dict[str, Any]]:
    """
    Load JSONL and normalize keys/types for ragas==0.1.19:
      - Accepts either 'contexts' or 'retrieved_contexts' (renamed to 'contexts')
      - Accepts either 'ground_truths' or 'ground_truth'      (renamed to 'ground_truths')
      - Ensures 'reference' is a string; if missing, it concatenates ground_truths
    """
    items: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

            # Accept legacy keys and remap
            if "contexts" not in obj and "retrieved_contexts" in obj:
                obj["contexts"] = obj.pop("retrieved_contexts")
            if "ground_truths" not in obj and "ground_truth" in obj:
                obj["ground_truths"] = obj.pop("ground_truth")

            # Pull and coerce types
            q = str(obj.get("question", ""))
            ctx = obj.get("contexts") or []
            gts = obj.get("ground_truths") or []

            if not isinstance(ctx, list):
                ctx = [ctx]
            ctx = [str(x) for x in ctx]

            if not isinstance(gts, list):
                gts = [gts]
            gts = [str(x) for x in gts]

            # In 0.1.19, 'reference' is required by context_* validators
            reference = obj.get("reference")
            if not isinstance(reference, str):
                reference = " | ".join(gts) if gts else ""

            cleaned = {
                "question": q,
                "contexts": ctx,
                "ground_truths": gts,
                "reference": reference,
            }
            items.append(cleaned)

    return items


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ragas evaluation (retrieval-only metrics) for ragas==0.1.19"
    )
    parser.add_argument(
        "--data",
        default="tests/dataset.jsonl",
        help="Path to JSONL dataset (one JSON object per line).",
    )
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model name (used by context_* metrics).",
    )
    parser.add_argument(
        "--out-csv",
        default="tests/ragas_report.csv",
        help="Where to write the CSV report.",
    )
    args = parser.parse_args(argv)

    ds_path = pathlib.Path(args.data)
    if not ds_path.exists():
        print(f"❌ File not found: {ds_path}")
        return 1

    print(f"📥 Loading {ds_path} ...")
    items = load_and_sanitize(ds_path)

    # Quick schema check to fail fast if something is missing
    for idx, row in enumerate(items, 1):
        missing = RAGAS_REQUIRED - set(row.keys())
        if missing:
            print(f"❌ Row {idx} is missing keys: {sorted(missing)}")
            return 1

    # Enforce explicit schema for ragas==0.1.19
    features = Features(
        {
            "question": Value("string"),
            "contexts": Sequence(Value("string")),
            "ground_truths": Sequence(Value("string")),
            "reference": Value("string"),
        }
    )
    ds = Dataset.from_list(items, features=features)

    # Show a small preview (helps debug any future schema regressions)
    print("🔎 Preview of first 2 items (after cleaning):")
    for i in range(min(2, len(ds))):
        print(json.dumps({k: ds[i][k] for k in ds.column_names}, ensure_ascii=False))

    print("🧪 Final columns sent to Ragas:", ds.column_names)
    print("🧪 Sample[0]:", {k: ds[0][k] for k in ds.column_names})

    # Embeddings for retrieval metrics
    embeddings = HuggingFaceEmbeddings(model_name=args.embed_model)

    # Retrieval-only metrics (no LLM/judge required)
    metrics = [context_precision, context_recall]

    print("⚙️  Running evaluation (context_precision, context_recall) ...")
    result = evaluate(ds, metrics=metrics, embeddings=embeddings)

    # Report
    df = result.to_pandas()
    print("\n✅ RESULTS")
    print(df)
    print("\nMEANS:")
    print(df.mean(numeric_only=True))

    out_csv = pathlib.Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n📝 Report saved to: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

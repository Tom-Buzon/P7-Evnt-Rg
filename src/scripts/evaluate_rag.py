# Placeholder for Ragas pipeline. In a real run, populate 'dataset.jsonl' with
# questions, answers (ground truth), and contexts.
import json, pathlib
from typing import Iterable

def main():
    ds = pathlib.Path("tests/dataset.jsonl")
    if not ds.exists():
        print("No dataset.jsonl found in tests/. Add some Q/A for evaluation.")
        return
    # Here you would import ragas and compute metrics. Keeping light for the POC package.
    with ds.open("r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    print(f"Loaded {len(items)} Q/A items for evaluation.")

if __name__ == "__main__":
    main()

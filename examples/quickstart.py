"""Quickstart: retrieval with query / document prompts (Sentence Transformers).
    python examples/quickstart.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load import load  # noqa: E402

QUERIES = ["what is the capital of france", "how do vaccines work"]
DOCUMENTS = [
    "Paris is the capital and largest city of France.",
    "Vaccines train the immune system to recognise a pathogen without causing the disease.",
    "The mitochondria is the powerhouse of the cell.",
]


def run(model=None):
    model = model or load()
    q = model.encode_query(QUERIES)          # adds the "query: " prompt
    d = model.encode_document(DOCUMENTS)     # adds the "passage: " prompt
    return {"queries": q, "documents": d, "scores": model.similarity(q, d)}


if __name__ == "__main__":
    out = run()
    print("embedding shape:", out["queries"].shape)
    for qi, query in enumerate(QUERIES):
        best = int(out["scores"][qi].argmax())
        print(f"{query!r} -> {DOCUMENTS[best]!r} (cosine {float(out['scores'][qi][best]):.4f})")

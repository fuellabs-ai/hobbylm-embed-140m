"""Plain-text similarity (no prompt): use this for STS, clustering and classification features.
    python examples/similarity.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load import load  # noqa: E402

SENTENCES = [
    "A man is playing a guitar.",
    "Someone is playing an instrument.",
    "The stock market fell sharply today.",
]


def run(model=None):
    model = model or load()
    e = model.encode(SENTENCES)              # no prompt
    return {"sentences": e, "scores": model.similarity(e, e)}


if __name__ == "__main__":
    out = run()
    s = out["scores"]
    for i in range(len(SENTENCES)):
        for j in range(i + 1, len(SENTENCES)):
            print(f"{float(s[i][j]):.4f}  {SENTENCES[i]!r} vs {SENTENCES[j]!r}")

"""Semantic search over a small list of passages.
    python examples/semantic_search.py "how do plants make food" """
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load import load  # noqa: E402

PASSAGES = [
    "Photosynthesis lets plants turn sunlight, water and carbon dioxide into glucose and oxygen.",
    "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
    "A balanced diet includes fruit, vegetables, whole grains and protein.",
    "Python is a popular programming language for data science.",
    "The Great Wall of China stretches for thousands of kilometres.",
    "Regular exercise improves heart health and mood.",
]
DEFAULT_QUERY = "how do plants make food"


def run(query=DEFAULT_QUERY, model=None, top_k=3):
    model = model or load()
    q = model.encode_query([query])
    p = model.encode_document(PASSAGES)
    scores = model.similarity(q, p)[0]
    top = scores.argsort(descending=True)[:top_k]
    return {"query": q, "passages": p, "ranking": [(PASSAGES[i], float(scores[i])) for i in top]}


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    print("query:", query)
    for rank, (text, score) in enumerate(run(query)["ranking"], 1):
        print(f"{rank}. {score:.4f}  {text}")

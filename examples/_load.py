"""Shared loader for the examples: pinned released revision, float32, CPU by default."""
import torch
from sentence_transformers import SentenceTransformer

REPO = "harims95/hobbylm-embed-140m"
# Same weights as the revision scored on MTEB (ff263c85); this revision adds the "document" prompt used by encode_document.
# Default input length 512 tokens (set model.max_seq_length = 2048 for long documents).
REVISION = "ea708cee107818fa0c1871eebd26d82a1e3fb1aa"


def load(device="cpu"):
    model = SentenceTransformer(REPO, revision=REVISION, trust_remote_code=True, device=device)
    # The router picks experts per token; keep float32 so the selection matches the reference.
    assert next(model.parameters()).dtype == torch.float32
    return model

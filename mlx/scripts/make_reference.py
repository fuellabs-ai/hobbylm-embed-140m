"""Step 2: reference outputs from the original PyTorch model via Sentence Transformers (CPU, float32).

Saves:
  ref/ref_vectors.npy   (6, 512) float32, exactly what SentenceTransformer.encode returns
  ref/ref_ids.json      the exact input ids fed to the model for each sentence (prompt included, EOT included)
  ref/ref_hidden.npz    per-sentence hidden states: after the embedding, after each of the 12 blocks, after final norm
Usage: .venv/bin/python -I scripts/make_reference.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_sentences import SENTENCES  # noqa: E402

REF = ROOT / "ref"

model = SentenceTransformer(str(REF), trust_remote_code=True, device="cpu")
model.eval()
print("modules:", [type(m).__name__ for m in model])
print("max_seq_length:", model.max_seq_length)
prompts = model.prompts

# 1) vectors exactly as Sentence Transformers produces them, one call per prompt group
vecs = np.zeros((len(SENTENCES), 512), dtype=np.float32)
for pname in ("query", "passage"):
    idx = [i for i, s in enumerate(SENTENCES) if s["prompt"] == pname]
    out = model.encode([SENTENCES[i]["text"] for i in idx], prompt_name=pname, convert_to_numpy=True, batch_size=8)
    vecs[idx] = out

# 2) exact ids: the same prompt + text through ST's own tokenize (same truncation as encode)
ids_out = []
for s in SENTENCES:
    feats = model.tokenize([prompts[s["prompt"]] + s["text"]])
    ids = feats["input_ids"][0][feats["attention_mask"][0].bool()].tolist()
    ids_out.append({"name": s["name"], "prompt": s["prompt"], "ids": ids, "n_tokens": len(ids)})

# 3) hidden states per sentence (batch 1, no padding) through the underlying HF model
hf = model[0].auto_model
hidden = {}
for i, s in enumerate(ids_out):
    caps = []
    hooks = [hf.embed.register_forward_hook(lambda m, a, o: caps.append(o.detach().clone()))]
    for blk in hf.blocks:
        hooks.append(blk.register_forward_hook(lambda m, a, o: caps.append(o.detach().clone())))
    hooks.append(hf.final_norm.register_forward_hook(lambda m, a, o: caps.append(o.detach().clone())))
    with torch.no_grad():
        ids_t = torch.tensor([s["ids"]])
        out = hf(input_ids=ids_t, attention_mask=torch.ones_like(ids_t))
    for h in hooks:
        h.remove()
    assert len(caps) == 14, len(caps)
    for j, c in enumerate(caps):
        name = "embed" if j == 0 else ("final_norm" if j == 13 else f"block{j - 1}")
        hidden[f"s{i}_{name}"] = c[0].float().numpy()
    hidden[f"s{i}_pooler"] = out.pooler_output[0].float().numpy()
    # sanity: the HF pooler output must equal what ST returned
    cos = float(np.dot(hidden[f"s{i}_pooler"], vecs[i]) / np.linalg.norm(hidden[f"s{i}_pooler"]) / np.linalg.norm(vecs[i]))
    print(f"{s['name']:<16} tokens={s['n_tokens']:<4} HF pooler vs ST cosine={cos:.8f}")

np.save(REF / "ref_vectors.npy", vecs)
(REF / "ref_ids.json").write_text(json.dumps(ids_out, indent=1))
np.savez(REF / "ref_hidden.npz", **hidden)
print("norms:", np.round(np.linalg.norm(vecs, axis=1), 6).tolist())
print("saved ref/ref_vectors.npy, ref/ref_ids.json, ref/ref_hidden.npz")

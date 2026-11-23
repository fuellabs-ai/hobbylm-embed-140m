"""Step 4: check the MLX port against the PyTorch / Sentence Transformers reference.

1. ids: MLX-side tokenizer must reproduce ref/ref_ids.json exactly.
2. hidden states (float32, batch 1, exact reference ids): after embedding, after block 0 (layer 1), then every block
   and the final norm. Reports max abs diff and cosine per sentence.
3. final vectors for all 6 sentences: length 512, norm, cosine vs ref/ref_vectors.npy, float32 and float16,
   encoded as a padded batch (the realistic path).
4. --time: 100 single-sentence encodes (cycling the 6 test sentences, batch 1) after 10 warm-up encodes,
   for float32 and float16. Run this only on an otherwise idle Mac.

Usage: .venv/bin/python -I scripts/check_mlx.py [--time]
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hobbylm_mlx import Encoder  # noqa: E402
from test_sentences import SENTENCES  # noqa: E402

REF = ROOT / "ref"
ref_vecs = np.load(REF / "ref_vectors.npy")
ref_ids = json.loads((REF / "ref_ids.json").read_text())
ref_hidden = np.load(REF / "ref_hidden.npz")


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print(f"mlx {mx.__version__}, default device {mx.default_device()}")
enc32 = Encoder(REF, dtype=mx.float32)

# 1. ids
print("\n== 1. token ids (MLX-side tokenizer vs reference)")
ids_ok = True
for s, r in zip(SENTENCES, ref_ids):
    mine = enc32.tokenize([s["text"]], s["prompt"])[0]
    same = mine == r["ids"]
    ids_ok &= same
    print(f"  {s['name']:<11} ref_tokens={r['n_tokens']:<4} mlx_tokens={len(mine):<4} identical={same}")
print("  ALL IDS IDENTICAL" if ids_ok else "  IDS DIFFER")

# 2. hidden states
print("\n== 2. hidden states, float32 (max abs diff / cosine vs PyTorch)")
names = ["embed"] + [f"block{i}" for i in range(12)] + ["final_norm"]
worst = {n: (0.0, 1.0) for n in names}
for i, r in enumerate(ref_ids):
    ids = mx.array([r["ids"]], dtype=mx.int32)
    _, hidden = enc32.model(ids, mx.ones_like(ids), return_hidden=True)
    mx.eval(hidden)
    for n, h in zip(names, hidden):
        a, b = np.array(h[0]), ref_hidden[f"s{i}_{n}"]
        d, c = float(np.abs(a - b).max()), cos(a, b)
        worst[n] = (max(worst[n][0], d), min(worst[n][1], c))
for n in names:
    label = {"embed": "after embedding", "block0": "after layer 1 (block 0, dense)"}.get(n, n)
    print(f"  {label:<32} worst max|diff|={worst[n][0]:.3e}  worst cosine={worst[n][1]:.8f}")

# 3. final vectors
results = {}
for label, dtype in (("float32", mx.float32), ("float16", mx.float16)):
    enc = enc32 if dtype == mx.float32 else Encoder(REF, dtype=mx.float16)
    vecs = np.zeros_like(ref_vecs)
    for pname in ("query", "passage"):
        idx = [i for i, s in enumerate(SENTENCES) if s["prompt"] == pname]
        vecs[idx] = enc.encode([SENTENCES[i]["text"] for i in idx], prompt_name=pname)
    print(f"\n== 3. final vectors, {label} (padded batch per prompt group)")
    print(f"  {'sentence':<11} {'prompt':<8} {'tokens':>6} {'dim':>4} {'norm':>10} {'cosine vs ST':>14}")
    cs = []
    for i, s in enumerate(SENTENCES):
        c = cos(vecs[i], ref_vecs[i])
        cs.append(c)
        print(f"  {s['name']:<11} {s['prompt']:<8} {ref_ids[i]['n_tokens']:>6} {vecs[i].shape[0]:>4} "
              f"{np.linalg.norm(vecs[i]):>10.6f} {c:>14.8f}")
    results[label] = min(cs)
    print(f"  min cosine = {min(cs):.8f}")

print("\n== summary")
print(f"  ids identical: {ids_ok}")
print(f"  float32 min cosine: {results['float32']:.8f}  target >= 0.999: {'PASS' if results['float32'] >= 0.999 else 'FAIL'}")
print(f"  float16 min cosine: {results['float16']:.8f}  (reported separately)")

# 4. timing
if "--time" in sys.argv:
    print("\n== 4. timing: 100 encodes, batch 1, cycling the 6 test sentences, after 10 warm-up encodes")
    for label, dtype in (("float32", mx.float32), ("float16", mx.float16)):
        enc = Encoder(REF, dtype=dtype)
        work = [(s["text"], s["prompt"]) for s in SENTENCES]
        for j in range(10):
            t, p = work[j % 6]
            enc.encode([t], prompt_name=p)
        t0 = time.perf_counter()
        for j in range(100):
            t, p = work[j % 6]
            enc.encode([t], prompt_name=p)
        secs = time.perf_counter() - t0
        print(f"  {label}: {secs:.3f} s for 100 encodes = {100 / secs:.1f} sentences/s")

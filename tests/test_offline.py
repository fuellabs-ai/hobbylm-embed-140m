"""Offline checks, no download. Run with `python -m unittest discover tests`.

- the model code matches the released hashes;
- the pooling contract on a tiny random model;
- the examples reproduce the stored reference vectors (cosine >= 0.9999). This needs the released weights in the local
  Hugging Face cache (run any example once while online); it is skipped, not failed, if they are missing.
"""
import hashlib
import json
import os
import sys
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))
from hobbylm_hf import HobbyLMConfig, HobbyLMModel  # noqa: E402

EOT = 50256


def tiny():
    torch.manual_seed(0)
    cfg = HobbyLMConfig(vocab_size=50304, hidden_size=64, num_hidden_layers=3, num_dense_layers=1, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, dense_intermediate_size=96, expert_intermediate_size=16,
                        num_experts=8, num_experts_per_tok=2)
    m = HobbyLMModel(cfg).eval()
    for mod in m.modules():                      # expert weights are plain Parameters (not Linear layers): give them values
        if isinstance(getattr(mod, "w13", None), torch.nn.Parameter):
            torch.nn.init.normal_(mod.w13, std=0.02)
            torch.nn.init.normal_(mod.w2, std=0.02)
    return m


class ModelCode(unittest.TestCase):
    def test_files_match_released_hashes(self):
        with open(os.path.join(ROOT, "hobbylm_hf", "SHA256SUMS.txt"), encoding="ascii") as f:
            lines = f.read().splitlines()
        for line in filter(None, lines):
            digest, name = line.split()
            with open(os.path.join(ROOT, "hobbylm_hf", name.lstrip("*")), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), digest, name)

    def test_pooling_excludes_final_eot(self):
        m = tiny()
        ids = torch.tensor([[11, 22, 33, 44, EOT], [55, 66, EOT, EOT, EOT]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=mask)
            # reference: run each row alone and mean-pool the text tokens only (final EOT excluded)
            for r, n in ((0, 5), (1, 3)):
                h = m(input_ids=ids[r:r + 1, :n]).last_hidden_state[0]
                want = F.normalize(h[: n - 1].mean(0), dim=-1)
                self.assertGreater(float(F.cosine_similarity(out.pooler_output[r], want, dim=0)), 0.99999)
                # plain mask-mean of last_hidden_state gives the same embedding
                lh = out.last_hidden_state[r, :n]
                self.assertGreater(float(F.cosine_similarity(F.normalize(lh.mean(0), dim=-1), want, dim=0)), 0.99999)

    def test_no_eot_row_is_left_unchanged(self):
        m = tiny()
        ids = torch.tensor([[11, 22, 33]])
        with torch.no_grad():
            out = m(input_ids=ids)
        want = F.normalize(out.last_hidden_state[0].mean(0), dim=-1)
        self.assertGreater(float(F.cosine_similarity(out.pooler_output[0], want, dim=0)), 0.99999)


class Examples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from _load import load
            cls.model = load()
        except Exception as e:  # weights not in the local cache
            raise unittest.SkipTest(f"released weights not cached locally ({type(e).__name__}); run an example online once")
        with open(os.path.join(ROOT, "tests", "reference_vectors.json"), encoding="utf-8") as f:
            cls.ref = json.load(f)

    def check(self, name, got):
        want = np.asarray(self.ref[name], dtype=np.float64)
        got = np.asarray(got, dtype=np.float64)
        self.assertEqual(got.shape, want.shape, name)
        cos = (got * want).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1))
        self.assertGreaterEqual(float(cos.min()), 0.9999, f"{name}: min cosine {cos.min():.6f}")

    def test_quickstart(self):
        import quickstart
        out = quickstart.run(self.model)
        self.check("quickstart.queries", out["queries"])
        self.check("quickstart.documents", out["documents"])

    def test_semantic_search(self):
        import semantic_search
        out = semantic_search.run(model=self.model)
        self.check("semantic_search.query", out["query"])
        self.check("semantic_search.passages", out["passages"])

    def test_similarity(self):
        import similarity
        self.check("similarity.sentences", similarity.run(self.model)["sentences"])


if __name__ == "__main__":
    unittest.main()

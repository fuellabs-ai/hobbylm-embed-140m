"""HobbyLM-Embed (harims95/hobbylm-embed-140m) in MLX.

Mirrors ref/modeling_hobbylm.py: 12 pre-norm blocks, bidirectional GQA (8 q heads, 2 kv heads, head dim 64) with
QK RMSNorm and rotate-half RoPE, block 0 dense SwiGLU, blocks 1-11 sparse MoE (32 SwiGLU experts, top-8, fp32
sigmoid router, selection bias used for selection only, weights not renormalised), final RMSNorm, mean pooling over
text tokens (padding and final EOT excluded) in float32, then L2 normalisation.

Weights are read straight from model.safetensors through `map_weight_name`. Tokenisation uses the repo's own
tokenizer.json via the `tokenizers` library, so the ids are identical to Sentence Transformers.
"""
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer

PROMPTS = {"query": "query: ", "passage": "passage: ", "document": "passage: "}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


def rope(x, cos, sin):
    # x (B, H, S, D); cos/sin (B, 1, S, D/2); rotate-half, as in the reference
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nq, self.nkv, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.qkv = nn.Linear(c["hidden_size"], (self.nq + 2 * self.nkv) * self.hd, bias=False)
        self.proj = nn.Linear(self.nq * self.hd, c["hidden_size"], bias=False)
        self.q_norm = RMSNorm(self.hd, c["rms_norm_eps"])
        self.k_norm = RMSNorm(self.hd, c["rms_norm_eps"])

    def __call__(self, x, cos, sin, add_mask):
        B, S, _ = x.shape
        qkv = self.qkv(x)
        q = qkv[..., : self.nq * self.hd]
        k = qkv[..., self.nq * self.hd: (self.nq + self.nkv) * self.hd]
        v = qkv[..., (self.nq + self.nkv) * self.hd:]
        q = self.q_norm(q.reshape(B, S, self.nq, self.hd)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, S, self.nkv, self.hd)).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.nkv, self.hd).transpose(0, 2, 1, 3)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        # MLX SDPA handles GQA natively: query heads [0..3] share kv head 0, [4..7] kv head 1 (= repeat_interleave)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(self.hd), mask=add_mask)
        return self.proj(o.transpose(0, 2, 1, 3).reshape(B, S, self.nq * self.hd))


class DenseFFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.inter = c["dense_intermediate_size"]
        self.w13 = nn.Linear(c["hidden_size"], 2 * self.inter, bias=False)
        self.w2 = nn.Linear(self.inter, c["hidden_size"], bias=False)

    def __call__(self, x):
        h = self.w13(x)
        return self.w2(nn.silu(h[..., : self.inter]) * h[..., self.inter:])


class Experts(nn.Module):
    def __init__(self, c):
        super().__init__()
        # stored as (E, in, out): used as x @ w, NOT nn.Linear orientation
        self.w13 = mx.zeros((c["num_experts"], c["hidden_size"], 2 * c["expert_intermediate_size"]))
        self.w2 = mx.zeros((c["num_experts"], c["expert_intermediate_size"], c["hidden_size"]))

    def mm(self, name, x, idx):
        return mx.gather_mm(x, getattr(self, name), rhs_indices=idx)


class QuantizedExperts(nn.Module):
    """Experts with weight-only affine quantization. Weights are transposed to (E, out, in) and quantized along
    `in`, then applied with gather_qmm(transpose=True), which computes the same x @ W as the float version."""

    def __init__(self, experts, group_size, bits, mode="affine"):
        super().__init__()
        self.group_size, self.bits, self.mode = group_size, bits, mode
        for name in ("w13", "w2"):
            r = mx.quantize(getattr(experts, name).swapaxes(-1, -2), group_size=group_size, bits=bits, mode=mode)
            setattr(self, name, r[0])
            setattr(self, name + "_scales", r[1])
            if len(r) > 2:                       # affine has biases; mxfp4 / nvfp4 do not
                setattr(self, name + "_biases", r[2])

    def mm(self, name, x, idx):
        b = getattr(self, name + "_biases", None)
        return mx.gather_qmm(x, getattr(self, name), getattr(self, name + "_scales"), b, rhs_indices=idx,
                             transpose=True, group_size=self.group_size, bits=self.bits, mode=self.mode)


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.top_k = c["num_experts_per_tok"]
        self.inter = c["expert_intermediate_size"]
        self.gate = nn.Linear(c["hidden_size"], c["num_experts"], bias=False)
        self.expert_bias = mx.zeros((c["num_experts"],))
        self.experts = Experts(c)

    def __call__(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        # fp32 router, as in the reference
        scores = mx.sigmoid(x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T)
        biased = scores + self.expert_bias.astype(mx.float32)
        topi = mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[..., : self.top_k]   # selection uses the bias
        topv = mx.take_along_axis(scores, topi, axis=-1)                                  # weights do not
        xe = x[:, None, None, :]                                                           # (N, 1, 1, D)
        h = self.experts.mm("w13", xe, topi)                                               # (N, k, 1, 2I)
        h = nn.silu(h[..., : self.inter]) * h[..., self.inter:]
        y = self.experts.mm("w2", h, topi)                                                 # (N, k, 1, D)
        y = (y.squeeze(-2) * topv[..., None].astype(y.dtype)).sum(axis=1)
        return y.reshape(shape)


class Block(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.attn_norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.attn = Attention(c)
        self.ffn_norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.ffn = DenseFFN(c) if i < c["num_dense_layers"] else MoE(c)

    def __call__(self, x, cos, sin, add_mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, add_mask)
        return x + self.ffn(self.ffn_norm(x))


class HobbyLMModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.blocks = [Block(c, i) for i in range(c["num_hidden_layers"])]
        self.final_norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])

    def __call__(self, input_ids, attention_mask, return_hidden=False):
        c = self.c
        mask = attention_mask.astype(mx.bool_)
        B, S = input_ids.shape
        pos = mx.maximum(mx.cumsum(mask.astype(mx.int32), axis=-1) - 1, 0)
        inv = 1.0 / (c["rope_theta"] ** (mx.arange(0, c["head_dim"], 2).astype(mx.float32) / c["head_dim"]))
        freqs = pos.astype(mx.float32)[..., None] * inv
        x = self.embed(input_ids)
        cos, sin = mx.cos(freqs)[:, None].astype(x.dtype), mx.sin(freqs)[:, None].astype(x.dtype)
        add_mask = mx.where(mask, 0.0, -mx.inf).astype(x.dtype)[:, None, None, :]
        hidden = [x] if return_hidden else None
        for blk in self.blocks:
            x = blk(x, cos, sin, add_mask)
            if return_hidden:
                hidden.append(x)
        h = self.final_norm(x)
        # pooling: mean over text tokens, excluding padding and the final EOT (fallback: the EOT itself)
        ar = mx.arange(S)
        last = mx.argmax(mask.astype(mx.int32) * ar, axis=-1)
        is_eot = mx.take_along_axis(input_ids, last[:, None], axis=1)[:, 0] == c["eot_token_id"]
        keep = mask & ~((ar[None, :] == last[:, None]) & is_eot[:, None])
        cnt = keep.sum(-1, keepdims=True)
        keep = mx.where(cnt == 0, mask, keep)
        hf = h.astype(mx.float32)
        kf = keep.astype(mx.float32)[..., None]
        mean = (hf * kf).sum(1) / kf.sum(1)
        emb = mean / mx.maximum(mx.linalg.norm(mean, axis=-1, keepdims=True), 1e-12)
        if return_hidden:
            hidden.append(h)
            return emb, hidden
        return emb


# ---------------------------------------------------------------- weights

def map_weight_name(name):
    """safetensors name -> MLX parameter path. The MLX modules use the reference names, so this is a checked
    identity map; it exists to fail loudly on any name the port does not know."""
    import re
    patterns = [
        r"embed\.weight", r"final_norm\.weight",
        r"blocks\.\d+\.(attn_norm|ffn_norm)\.weight",
        r"blocks\.\d+\.attn\.(qkv|proj)\.weight", r"blocks\.\d+\.attn\.(q_norm|k_norm)\.weight",
        r"blocks\.\d+\.ffn\.(w13|w2)\.weight",
        r"blocks\.\d+\.ffn\.gate\.weight", r"blocks\.\d+\.ffn\.expert_bias", r"blocks\.\d+\.ffn\.experts\.(w13|w2)",
    ]
    if not any(re.fullmatch(p, name) for p in patterns):
        raise KeyError(f"unmapped weight: {name}")
    return name


QUANT_PARTS = ("embed", "attn", "dense", "experts")


def _part(path):
    if path == "embed":
        return "embed"
    if ".attn." in path:
        return "attn"
    if path.endswith("ffn.w13") or path.endswith("ffn.w2"):
        return "dense"
    return None


def quantize_model(model, bits, group_size=64, parts=None):
    """Weight-only quantization. `parts` optionally maps embed / attn / dense / experts to
    {"bits": b, "group_size": g, "mode": "affine" | "mxfp4" | "nvfp4"} (bits None = leave that part in float), and
    may hold "keep_blocks_8bit": [block indices] whose attention and FFN use 8-bit affine g64 instead.
    Anything not listed uses `bits`/`group_size`, affine. The router (`ffn.gate`, 32x512) and `expert_bias` always stay
    fp32: they decide which experts run, and they are 0.01% of the parameters. Norm weights stay in the float dtype."""
    parts = dict(parts or {})
    keep8 = set(parts.pop("keep_blocks_8bit", []))
    spec = {p: {"bits": bits, "group_size": group_size, "mode": "affine"} for p in QUANT_PARTS}
    for p, v in parts.items():
        spec[p] = {"mode": "affine", **v}
    eight = {"bits": 8, "group_size": 64, "mode": "affine"}

    def blk_of(path):
        return int(path.split(".")[1]) if path.startswith("blocks.") else None

    def pred(path, m):
        p = _part(path)
        if p is None or not isinstance(m, (nn.Linear, nn.Embedding)):
            return False
        s = eight if blk_of(path) in keep8 else spec[p]
        return dict(s) if s["bits"] else False

    nn.quantize(model, class_predicate=pred)
    for i, blk in enumerate(model.blocks):
        if isinstance(blk.ffn, MoE):
            s = eight if i in keep8 else spec["experts"]
            if s["bits"]:
                blk.ffn.experts = QuantizedExperts(blk.ffn.experts, s["group_size"], s["bits"], s["mode"])
    spec["keep_blocks_8bit"] = sorted(keep8)
    model.quant = {"bits": bits, "group_size": group_size, "parts": spec}
    return model


def load_model(repo_dir, dtype=mx.float32, bits=None, group_size=64, parts=None):
    """Load the original float32 safetensors, cast to `dtype` (router stays fp32), optionally quantize to `bits`."""
    repo_dir = Path(repo_dir)
    c = json.loads((repo_dir / "config.json").read_text())
    model = HobbyLMModel(c)
    raw = mx.load(str(repo_dir / "model.safetensors"))
    weights = []
    for k, v in raw.items():
        mk = map_weight_name(k)
        # router and its bias stay fp32 (the reference computes the router in fp32)
        keep_fp32 = mk.endswith("ffn.gate.weight") or mk.endswith("expert_bias")
        weights.append((mk, v if keep_fp32 else v.astype(dtype)))
    model.load_weights(weights, strict=True)    # strict: every parameter must be present with the right shape
    if bits:
        quantize_model(model, bits, group_size, parts)
    model.eval()
    mx.eval(model.parameters())
    return model


def save_quantized(model, out_dir):
    """Write a quantized model as model.safetensors + quant.json; reuse config/tokenizer from the original repo."""
    from mlx.utils import tree_flatten
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (out_dir / "quant.json").write_text(json.dumps(model.quant))
    return out_dir / "model.safetensors"


def save_float(model, out_dir, dtype=mx.float16):
    """Write a float16 (or other float) copy: model.safetensors + quant.json. Router and expert_bias stay fp32."""
    from mlx.utils import tree_flatten
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w = {k: (v if (k.endswith("ffn.gate.weight") or k.endswith("expert_bias")) else v.astype(dtype))
         for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(str(out_dir / "model.safetensors"), w)
    (out_dir / "quant.json").write_text(json.dumps({"bits": None, "dtype": str(dtype).split(".")[-1]}))
    return out_dir / "model.safetensors"


def load_quantized(repo_dir, quant_dir):
    """Build the structure from config (no original weights loaded), then load a saved quantized or float16 file."""
    repo_dir, quant_dir = Path(repo_dir), Path(quant_dir)
    c = json.loads((repo_dir / "config.json").read_text())
    q = json.loads((quant_dir / "quant.json").read_text())
    model = HobbyLMModel(c)
    if q.get("bits"):
        quantize_model(model, q["bits"], q["group_size"], q.get("parts"))
    model.load_weights(str(quant_dir / "model.safetensors"), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------- tokenizer + encode

class Encoder:
    def __init__(self, repo_dir, dtype=mx.float32, max_seq_length=None, bits=None, group_size=64, parts=None,
                 quant_dir=None):
        repo_dir = Path(repo_dir)
        if quant_dir:
            self.model = load_quantized(repo_dir, quant_dir)
        else:
            self.model = load_model(repo_dir, dtype, bits=bits, group_size=group_size, parts=parts)
        self.tok = Tokenizer.from_file(str(repo_dir / "tokenizer.json"))
        if max_seq_length is None:
            max_seq_length = json.loads((repo_dir / "sentence_bert_config.json").read_text())["max_seq_length"]
        self.max_seq_length = max_seq_length
        self.tok.enable_truncation(max_length=max_seq_length)   # truncation reserves room for the appended EOT
        self.pad_id = self.model.c["pad_token_id"]

    def tokenize(self, texts, prompt_name=None):
        prefix = PROMPTS[prompt_name] if prompt_name else ""
        return [e.ids for e in self.tok.encode_batch([prefix + t for t in texts])]

    def encode_ids(self, id_lists):
        S = max(len(x) for x in id_lists)
        ids = np.full((len(id_lists), S), self.pad_id, dtype=np.int32)
        am = np.zeros((len(id_lists), S), dtype=np.int32)
        for i, x in enumerate(id_lists):
            ids[i, : len(x)] = x
            am[i, : len(x)] = 1
        out = self.model(mx.array(ids), mx.array(am))
        mx.eval(out)
        return np.array(out)

    def encode(self, texts, prompt_name=None, batch_size=32):
        id_lists = self.tokenize(texts, prompt_name)
        return np.concatenate([self.encode_ids(id_lists[i: i + batch_size])
                               for i in range(0, len(id_lists), batch_size)])

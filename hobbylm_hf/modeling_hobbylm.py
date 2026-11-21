"""HobbyLM embedding model: bidirectional encoder, layer 0 dense SwiGLU, layers 1-11 sparse mixture-of-experts
(32 SwiGLU experts, top-8, sigmoid router with a per-expert selection bias), grouped-query attention with QK-norm and RoPE.

Embedding: the tokenizer appends EOT (50256); the model mean-pools the final-normed hidden states over the text tokens
(padding and the final EOT excluded; the EOT is attended to) and L2-normalises (`pooler_output`).

`last_hidden_state` holds the per-token states, except that the final EOT position is replaced by the mean of the text
tokens. Plain mean pooling over `attention_mask` (as in Sentence Transformers) therefore gives exactly the same embedding.
"""
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .configuration_hobbylm import HobbyLMConfig


def _rms(x, w, eps):
    return F.rms_norm(x, (x.size(-1),), eps=eps) * w


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return _rms(x, self.weight, self.eps)


def _rope(x, cos, sin):
    # x (B, H, S, D); cos/sin (B, 1, S, D/2); rotate-half
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nq, self.nkv, self.hd, self.eps = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rms_norm_eps
        self.qkv = nn.Linear(c.hidden_size, (self.nq + 2 * self.nkv) * self.hd, bias=False)
        self.proj = nn.Linear(self.nq * self.hd, c.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hd, c.rms_norm_eps)
        self.k_norm = RMSNorm(self.hd, c.rms_norm_eps)

    def forward(self, x, cos, sin, key_mask):
        B, S, _ = x.shape
        q, k, v = self.qkv(x).split([self.nq * self.hd, self.nkv * self.hd, self.nkv * self.hd], dim=-1)
        q = self.q_norm(q.view(B, S, self.nq, self.hd)).transpose(1, 2)
        k = self.k_norm(k.view(B, S, self.nkv, self.hd)).transpose(1, 2)
        v = v.view(B, S, self.nkv, self.hd).transpose(1, 2)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        rep = self.nq // self.nkv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :], is_causal=False)
        return self.proj(o.transpose(1, 2).reshape(B, S, self.nq * self.hd))


class DenseFFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w13 = nn.Linear(c.hidden_size, 2 * c.dense_intermediate_size, bias=False)
        self.w2 = nn.Linear(c.dense_intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Experts(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w13 = nn.Parameter(torch.empty(c.num_experts, c.hidden_size, 2 * c.expert_intermediate_size))
        self.w2 = nn.Parameter(torch.empty(c.num_experts, c.expert_intermediate_size, c.hidden_size))


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_experts, self.top_k = c.num_experts, c.num_experts_per_tok
        self.gate = nn.Linear(c.hidden_size, c.num_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(c.num_experts))
        self.experts = Experts(c)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        scores = torch.sigmoid(F.linear(x.float(), self.gate.weight.float()))          # fp32 router
        topi = torch.topk(scores + self.expert_bias.float(), self.top_k, dim=-1).indices  # bias: selection only
        topv = torch.gather(scores, -1, topi)                                          # weights: unbiased, not renormalised
        flat_e = topi.reshape(-1)
        flat_tok = torch.arange(x.shape[0], device=x.device).repeat_interleave(self.top_k)
        flat_v = topv.reshape(-1)
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            sel = (flat_e == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            toks = flat_tok[sel]
            gate, up = (x[toks] @ self.experts.w13[e]).chunk(2, dim=-1)
            y = (F.silu(gate) * up) @ self.experts.w2[e]
            out.index_add_(0, toks, y.to(x.dtype) * flat_v[sel].unsqueeze(-1).to(x.dtype))
        return out.reshape(shape)


class Block(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.attn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.attn = Attention(c)
        self.ffn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.ffn = DenseFFN(c) if i < c.num_dense_layers else MoE(c)

    def forward(self, x, cos, sin, key_mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, key_mask)
        return x + self.ffn(self.ffn_norm(x))


class HobbyLMPreTrainedModel(PreTrainedModel):
    config_class = HobbyLMConfig
    base_model_prefix = "hobbylm"
    _no_split_modules = ["Block"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, Experts):
            nn.init.normal_(module.w13, std=0.02)
            nn.init.normal_(module.w2, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)


class HobbyLMModel(HobbyLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.num_hidden_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def forward(self, input_ids, attention_mask=None, **kwargs):
        c = self.config
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask = attention_mask.bool()
        B, S = input_ids.shape
        pos = (mask.long().cumsum(-1) - 1).clamp(min=0)                               # 0, 1, ... from the first text token
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, device=input_ids.device).float() / c.head_dim))
        freqs = pos.float()[..., None] * inv                                           # (B, S, D/2)
        cos, sin = freqs.cos()[:, None], freqs.sin()[:, None]
        x = self.embed(input_ids)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
        h = self.final_norm(x)
        # pooling: mean over text tokens, excluding padding and the final EOT (if no text token is left, the EOT itself)
        ar = torch.arange(S, device=h.device)
        last = (mask.long() * ar).argmax(-1)                                           # last real position
        is_eot = input_ids.gather(1, last[:, None]).squeeze(1) == c.eot_token_id
        keep = mask & ~((ar[None, :] == last[:, None]) & is_eot[:, None])
        cnt = keep.sum(-1, keepdim=True)
        keep = torch.where(cnt == 0, mask, keep)
        hf = h.float()
        mean = (hf * keep.unsqueeze(-1)).sum(1) / keep.sum(-1, keepdim=True)
        out = h.clone()
        rows = torch.arange(B, device=h.device)[is_eot]
        out[rows, last[is_eot]] = mean[is_eot].to(h.dtype)                             # mean-pool over the mask == embedding
        return BaseModelOutputWithPooling(last_hidden_state=out, pooler_output=F.normalize(mean, p=2, dim=-1))

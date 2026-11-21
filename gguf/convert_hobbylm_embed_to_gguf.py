#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert harims95/hobbylm-embed-140m (HF, Sentence Transformers) to GGUF for llama.cpp, using the `bailingmoe2` graph.

Why bailingmoe2: it has the same layer structure as HobbyLM-Embed (pre-RMSNorm, fused GQA QKV, per-head RMS QK-norm before
NEOX/rotate-half RoPE, leading dense SwiGLU layer, sigmoid router whose bias is used for selection only). Differences are
handled here, without C++ changes:
  - no shared expert in our model -> an all-zero shared expert (width 32) is written; it adds exactly 0;
  - tied embeddings -> `output.weight` is written as a copy of `token_embd.weight` (unused for embeddings);
  - non-causal attention and MEAN pooling are set in the GGUF metadata;
  - GPT-2 BPE tokenizer, EOT (50256) appended (`add_eos_token`), no BOS.
Known gap (see README): llama.cpp MEAN pooling averages over ALL tokens including the appended EOT; the model was trained to
average the text tokens only. Exact embeddings need `--pooling none` + client-side mean over tokens 0..n-2 + L2 norm.
Also done by the HF tokenizer but NOT by llama.cpp: stripping leading/trailing whitespace, truncating to 511 tokens + EOT.

    python convert_hobbylm_embed_to_gguf.py --out hobbylm-embed-140m-f32.gguf [--revision ff263c85...] [--model-dir DIR]
Requires: numpy, safetensors, huggingface_hub, and gguf-py from the same llama.cpp release (pip install ./llama.cpp/gguf-py).
"""
import argparse
import json
import os

import numpy as np
from safetensors.numpy import load_file

import gguf

REPO = 'harims95/hobbylm-embed-140m'
REV = 'ff263c8593574810700bd3aeba67be7a325738e9'
ARCH = gguf.MODEL_ARCH.BAILINGMOE2
SHEXP_WIDTH = 32


def fetch(model_dir, revision):
    if model_dir:
        return model_dir
    from huggingface_hub import snapshot_download
    return snapshot_download(REPO, revision=revision, allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json'])


def tname(t, bid=None, suffix='weight'):
    n = gguf.TENSOR_NAMES[t]
    n = n.format(bid=bid) if bid is not None else n
    return f'{n}.{suffix}' if suffix else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--revision', default=REV)
    ap.add_argument('--model-dir', default=None, help='local folder with config.json, model.safetensors, tokenizer.json')
    a = ap.parse_args()
    d = fetch(a.model_dir, a.revision)
    cfg = json.load(open(os.path.join(d, 'config.json')))
    sd = load_file(os.path.join(d, 'model.safetensors'))
    tok = json.load(open(os.path.join(d, 'tokenizer.json'), encoding='utf-8'))
    E, H, L = cfg['hidden_size'], cfg['head_dim'], cfg['num_hidden_layers']
    nq, nkv, ne, k = cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['num_experts'], cfg['num_experts_per_tok']
    fd, fe, nd, V = cfg['dense_intermediate_size'], cfg['expert_intermediate_size'], cfg['num_dense_layers'], cfg['vocab_size']
    assert nq * H == E, 'bailingmoe2 needs n_head * head_dim == n_embd'

    w = gguf.GGUFWriter(a.out, gguf.MODEL_ARCH_NAMES[ARCH])
    w.add_name('hobbylm-embed-140m')
    w.add_license('apache-2.0')
    w.add_context_length(cfg.get('max_position_embeddings', 1024))
    w.add_embedding_length(E)
    w.add_block_count(L)
    w.add_feed_forward_length(fd)
    w.add_head_count(nq)
    w.add_head_count_kv(nkv)
    w.add_key_length(H)
    w.add_value_length(H)
    w.add_layer_norm_rms_eps(cfg['rms_norm_eps'])
    w.add_rope_freq_base(cfg['rope_theta'])
    w.add_rope_dimension_count(H)
    w.add_expert_count(ne)
    w.add_expert_used_count(k)
    w.add_expert_feed_forward_length(fe)
    w.add_expert_shared_count(1)
    w.add_expert_shared_feed_forward_length(SHEXP_WIDTH)
    w.add_leading_dense_block_count(nd)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_expert_weights_norm(False)
    w.add_expert_weights_scale(1.0)
    w.add_causal_attention(False)
    w.add_pooling_type(gguf.PoolingType.MEAN)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    # ---- tokenizer: GPT-2 BPE (50,257 tokens) padded to the model vocab (50,304); EOT appended, no BOS
    vocab = tok['model']['vocab']
    inv = {i: t for t, i in vocab.items()}
    eot = cfg.get('eot_token_id', 50256)
    tokens, types = [], []
    for i in range(V):
        if i in inv:
            tokens.append(inv[i]); types.append(gguf.TokenType.CONTROL if i == eot else gguf.TokenType.NORMAL)
        elif i == 50257:
            tokens.append('<|mask|>'); types.append(gguf.TokenType.CONTROL)
        else:
            tokens.append(f'<|unused_{i}|>'); types.append(gguf.TokenType.UNUSED)
    merges = [m if isinstance(m, str) else ' '.join(m) for m in tok['model']['merges']]
    w.add_tokenizer_model('gpt2')
    w.add_tokenizer_pre('gpt-2')
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_eos_token_id(eot)
    w.add_pad_token_id(eot)
    w.add_add_bos_token(False)
    w.add_add_eos_token(True)

    # ---- tensors (numpy row-major; gguf-py reverses to ggml order)
    T = gguf.MODEL_TENSOR
    f32 = lambda x: np.ascontiguousarray(x, dtype=np.float32)
    w.add_tensor(tname(T.TOKEN_EMBD), f32(sd['embed.weight']))
    w.add_tensor(tname(T.OUTPUT), f32(sd['embed.weight']))                       # tied head, copied
    w.add_tensor(tname(T.OUTPUT_NORM), f32(sd['final_norm.weight']))
    for i in range(L):
        p = f'blocks.{i}.'
        w.add_tensor(tname(T.ATTN_NORM, i), f32(sd[p + 'attn_norm.weight']))
        w.add_tensor(tname(T.ATTN_QKV, i), f32(sd[p + 'attn.qkv.weight']))          # (q | k | v, n_embd)
        w.add_tensor(tname(T.ATTN_OUT, i), f32(sd[p + 'attn.proj.weight']))
        w.add_tensor(tname(T.ATTN_Q_NORM, i), f32(sd[p + 'attn.q_norm.weight']))
        w.add_tensor(tname(T.ATTN_K_NORM, i), f32(sd[p + 'attn.k_norm.weight']))
        w.add_tensor(tname(T.FFN_NORM, i), f32(sd[p + 'ffn_norm.weight']))
        if i < nd:
            w13 = sd[p + 'ffn.w13.weight']                                          # (2*fd, E): gate rows, then up rows
            w.add_tensor(tname(T.FFN_GATE, i), f32(w13[:fd]))
            w.add_tensor(tname(T.FFN_UP, i), f32(w13[fd:]))
            w.add_tensor(tname(T.FFN_DOWN, i), f32(sd[p + 'ffn.w2.weight']))
        else:
            w.add_tensor(tname(T.FFN_GATE_INP, i), f32(sd[p + 'ffn.gate.weight']))   # (n_expert, E)
            w.add_tensor(tname(T.FFN_EXP_PROBS_B, i, 'bias'), f32(sd[p + 'ffn.expert_bias']))
            w13 = sd[p + 'ffn.experts.w13']                                         # (n_expert, E, 2*fe): gate | up
            w2 = sd[p + 'ffn.experts.w2']                                           # (n_expert, fe, E)
            w.add_tensor(tname(T.FFN_GATE_EXP, i), f32(np.transpose(w13[:, :, :fe], (0, 2, 1))))   # (n_expert, fe, E)
            w.add_tensor(tname(T.FFN_UP_EXP, i), f32(np.transpose(w13[:, :, fe:], (0, 2, 1))))
            w.add_tensor(tname(T.FFN_DOWN_EXP, i), f32(np.transpose(w2, (0, 2, 1))))             # (n_expert, E, fe)
            z = np.zeros
            w.add_tensor(tname(T.FFN_GATE_SHEXP, i), z((SHEXP_WIDTH, E), np.float32))   # all-zero shared expert
            w.add_tensor(tname(T.FFN_UP_SHEXP, i), z((SHEXP_WIDTH, E), np.float32))
            w.add_tensor(tname(T.FFN_DOWN_SHEXP, i), z((E, SHEXP_WIDTH), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print('wrote', a.out, os.path.getsize(a.out), 'bytes')


if __name__ == '__main__':
    main()

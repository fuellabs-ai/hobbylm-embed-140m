#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exact hobbylm-embed-140m embeddings from llama.cpp (Python standard library only).

llama.cpp's built-in `--pooling mean` averages over ALL tokens, including the end-of-text token (EOT) that the tokenizer
appends. HobbyLM-Embed was trained to average the text tokens only (the EOT is attended to but not pooled). This script runs
`llama-embedding --pooling none`, then averages each prompt's token vectors except the last one (the EOT) and L2-normalises.

    python3 hobbylm_llamacpp_embed.py --bin ./llama-embedding --model hobbylm-embed-140m-f16.gguf --mode query texts.txt
texts.txt: one text per line. --mode: none (similarity / classification / clustering), query or passage (retrieval).
Output: one JSON object per line: {"text": ..., "embedding": [512 floats]}.
Texts are stripped (as the reference tokenizer does). Keep texts under 512 tokens (llama.cpp does not truncate).
"""
import argparse
import json
import math
import re
import subprocess
import sys

SEP = '<#HOBBYLM_SEP#>'
PREFIX = {'none': '', 'query': 'query: ', 'passage': 'passage: '}


def embed(binary, model, prompts, threads=None, ctx=32768):
    cmd = [binary, '-m', model, '-p', SEP.join(prompts), '--embd-separator', SEP, '--pooling', 'none',
           '--embd-normalize', '-1', '--embd-output-format', 'array', '--verbose-prompt', '-c', str(ctx)]
    if threads:
        cmd += ['-t', str(threads)]
    r = subprocess.run(cmd, capture_output=True)
    err = r.stderr.decode('utf-8', errors='replace')                  # token pieces in the log may be partial UTF-8
    if r.returncode != 0:
        sys.exit(f'llama-embedding failed (rc {r.returncode}):\n' + err[-2000:])
    out = r.stdout.decode('utf-8', errors='replace')                  # --verbose-prompt token lines precede the JSON
    counts = [int(n) for n in re.findall(r'number of tokens in prompt = (\d+)', err + out)]
    rows = json.loads(out[out.rindex('[['):])
    if len(counts) != len(prompts) or sum(counts) != len(rows):
        sys.exit(f'unexpected output: {len(prompts)} prompts, {len(counts)} token counts, {len(rows)} rows')
    out, i = [], 0
    for n in counts:
        h = rows[i:i + n]; i += n
        use = h[:-1] if n > 1 else h                      # exclude the final EOT (an empty text keeps only the EOT)
        v = [sum(col) / len(use) for col in zip(*use)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True, help='path to llama-embedding')
    ap.add_argument('--model', required=True, help='hobbylm-embed-140m-*.gguf')
    ap.add_argument('--mode', choices=list(PREFIX), default='none')
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('input', help='text file, one text per line')
    a = ap.parse_args()
    texts = [t.strip() for t in open(a.input, encoding='utf-8').read().splitlines() if t.strip()]
    vecs, counts = embed(a.bin, a.model, [PREFIX[a.mode] + t for t in texts], a.threads)
    for t, v, n in zip(texts, vecs, counts):
        if n > 512:
            print(f'warning: {n} tokens (> 512) for: {t[:60]}', file=sys.stderr)
        print(json.dumps({'text': t, 'embedding': [round(x, 7) for x in v]}, ensure_ascii=False))


if __name__ == '__main__':
    main()

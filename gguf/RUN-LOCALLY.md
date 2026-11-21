# Run hobbylm-embed-140m locally with llama.cpp (GGUF)

Every command below was run on 21 Nov 2026: steps 1 and 3 on Ubuntu 24.04 under WSL2 (CPU only, 16 threads), step 2 on the
Windows host (Python 3.12, Git Bash). The output shown is the real printed output. Steps that were tried and did not pass
are not included.

## 1. Build llama.cpp (release v1.0.0, CPU)

```bash
git clone --depth 1 --branch v1.0.0 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j 14 --target llama-embedding llama-quantize
```

Output (last lines):

```
[100%] Built target llama-common
[100%] Built target llama-embedding
[100%] Built target llama-quantize
```

## 2. Get the GGUF and the helper

```bash
pip install -U huggingface_hub
hf download harims95/hobbylm-embed-140m-gguf hobbylm-embed-140m-f16.gguf hobbylm-embed-140m-q8_0.gguf hobbylm_llamacpp_embed.py --local-dir .
sha256sum hobbylm-embed-140m-f16.gguf hobbylm-embed-140m-q8_0.gguf hobbylm_llamacpp_embed.py
```

Output:

```
7ecb311d06ac5f20e560dbf3ca2f92cd5a364ab7326cee7381f8abad90f77339 *hobbylm-embed-140m-f16.gguf
e192e27ce562a2e19e8980afcb1729b1a53ec1eee284be6ac3039cfea923a957 *hobbylm-embed-140m-q8_0.gguf
d8dd27990663dbe9a60e12f64e989f2bf70c4cff4cc7c00a6760930838454414 *hobbylm_llamacpp_embed.py
```

If the first download fails, retry. Step 3 uses the F16 file; for Q8_0, pass `--model hobbylm-embed-140m-q8_0.gguf`.

## 3. Embed texts

The helper needs only Python 3 (standard library). It runs `llama-embedding --pooling none` and pools the way the model
was trained: the mean of the text tokens, excluding the final end-of-text token, then L2 normalisation.

```bash
printf "what is the capital of france\n" > queries.txt
printf "Paris is the capital and largest city of France.\nThe mitochondria is the powerhouse of the cell.\n" > passages.txt
B=./llama.cpp/build/bin/llama-embedding
python3 hobbylm_llamacpp_embed.py --bin $B --model hobbylm-embed-140m-f16.gguf --mode query queries.txt > q.jsonl
python3 hobbylm_llamacpp_embed.py --bin $B --model hobbylm-embed-140m-f16.gguf --mode passage passages.txt > p.jsonl
cut -c1-120 q.jsonl
python3 -c "
import json
q=[json.loads(l) for l in open('q.jsonl')]; p=[json.loads(l) for l in open('p.jsonl')]
for d in p: print(round(sum(a*b for a,b in zip(q[0]['embedding'], d['embedding'])),4), d['text'])
"
```

Output:

```
{"text": "what is the capital of france", "embedding": [0.0119629, -0.0288813, -0.0404069, -0.0176782, 0.0519803, 0.0387
0.7309 Paris is the capital and largest city of France.
0.378 The mitochondria is the powerhouse of the cell.
```

- `--mode query` / `--mode passage` for retrieval; `--mode none` for similarity, classification and clustering.
- One text per line; texts are stripped. The GGUF files are tested at 512 tokens; their context limit is 1,024 tokens. For long documents (up to 2,048 tokens) use Sentence Transformers with `model.max_seq_length = 2048`. llama.cpp does not truncate, so shorten longer texts yourself.

## Measured agreement with the PyTorch model

100 varied texts (arguments, STS sentences, banking queries, edge cases), each without a prompt and with
`query: ` / `passage: ` (300 inputs), against sentence-transformers with `harims95/hobbylm-embed-140m` at revision
`ff263c85`:

| GGUF (size) | Mean cosine | Min cosine |
|---|---|---|
| F16 (335 MB) | 0.999974 | 0.998881 |
| Q8_0 (179 MB) | 0.999435 | 0.995988 |

Ollama (`num_ctx 512`): works, but its pooling includes the final end-of-text token, so vectors are close to but not
identical with the reference model (task scores within ±0.4 on our checks). Don't mix Ollama vectors with exact vectors
in one index.

LM Studio and llama.cpp's own `--pooling mean`: not supported yet: their mean pooling includes the final end-of-text
token, which this model excludes.

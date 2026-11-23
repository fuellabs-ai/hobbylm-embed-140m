# HobbyLM-Embed-140M on Mac (MLX)

An MLX port of [harims95/hobbylm-embed-140m](https://huggingface.co/harims95/hobbylm-embed-140m) for Apple Silicon. It gives the same 512-d, L2-normalised vectors as Sentence Transformers: in float32 the lowest cosine is 1.00000000 on our test inputs.

Weights (float16): [harims95/hobbylm-embed-140m-mlx](https://huggingface.co/harims95/hobbylm-embed-140m-mlx)

## Use

```bash
pip install mlx tokenizers numpy huggingface_hub
```

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("harims95/hobbylm-embed-140m-mlx")
sys.path.insert(0, path)
from hobbylm_mlx import Encoder

enc = Encoder(path, quant_dir=path)                  # float16 weights
q = enc.encode(["how do vaccines work?"], prompt_name="query")       # adds "query: "
p = enc.encode(["Vaccines train the immune system to recognise a pathogen without causing the disease.",
                "The Eiffel Tower was completed in 1889."], prompt_name="passage")   # adds "passage: "
print(q.shape)        # (1, 512)
print(q @ p.T)        # vectors are L2-normalised, so this is cosine similarity
```

- **Prompts:** as with the main model, queries need `prompt_name="query"` and documents need `prompt_name="passage"` for search and RAG. Use no prompt for similarity, clustering and classification.
- **Input length:** 512 tokens by default, read from the model's `sentence_bert_config.json`. For long documents, pass `max_seq_length=2048`: `Encoder(path, quant_dir=path, max_seq_length=2048)`.
- **Output:** NumPy arrays, one 512-d vector per text.

## Tested

Tested on Apple M4 (macOS 15.5, mlx 0.32.4), Python 3.15.2. The reference is Sentence Transformers 6.3.0 with torch 2.15.0 and transformers 5.24.0 on CPU, float32, model revision `ea708cee`.

| Check (6 inputs: 3 queries, 3 passages, from 5 to 512 tokens) | Result |
| --- | --- |
| Token ids | Identical to Sentence Transformers |
| Hidden states, float32, after the embedding and after every block | Cosine 1.00000000 at every step |
| Final vectors, float32 | Lowest cosine 1.00000000 vs Sentence Transformers |
| Final vectors, float16 | Lowest cosine 0.99965 vs Sentence Transformers |
| Output | 512-d, norm 1.000000 |

Full output: [results/check_mlx_output.txt](results/check_mlx_output.txt).

## Verify it yourself

Run these from this folder. They download the original model into `ref/`, build the PyTorch reference, and compare the MLX port with it layer by layer.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-check.txt     # MLX plus torch and sentence-transformers for the comparison
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('harims95/hobbylm-embed-140m', local_dir='ref')"
.venv/bin/python -I scripts/make_reference.py       # PyTorch reference: vectors, exact ids, per-layer outputs -> ref/
.venv/bin/python -I scripts/check_mlx.py            # ids, layer by layer, final vectors in float32 and float16
```

`requirements.txt` has only what is needed to run the model (no torch).

## What does not match exactly, and why

- **float16, "what is the capital of france": cosine 0.99965.** For one token, the router's top-8 choice flips between two nearly tied experts because of float16 rounding. Encoded on its own, this query gives 0.999999. In float32 the routing is identical and the cosine is 1.0. The router always runs in float32, as in the reference.
- **Small float32 differences inside the MoE blocks (largest 2.7e-04).** The MLX version adds up the expert outputs in a different order from the reference loop. The final vectors are unchanged (cosine 1.00000000).

## How it maps to the reference

`hobbylm_mlx.py` follows [`hobbylm_hf/modeling_hobbylm.py`](../hobbylm_hf/modeling_hobbylm.py) module by module, with the same parameter names. `map_weight_name` fails on any unknown weight, and loading is strict.

- **Attention:** bidirectional GQA (8 query, 2 key-value heads), QK RMSNorm, rotate-half RoPE with positions counted from the first real token, padding keys masked.
- **MoE:** sigmoid router in float32. `expert_bias` is used for selection only, and the weights are not renormalised. Experts are stored as `x @ W` and computed with `mx.gather_mm`.
- **Pooling:** float32 mean over the text tokens, excluding padding and the final end-of-text token, then L2 normalisation.
- **Tokenizer and prompts:** the model's own `tokenizer.json`, with `query: ` and `passage: ` added as text, as in `config_sentence_transformers.json`.

## Licence

Apache-2.0 ([LICENSE](../LICENSE)).

## Team

Hariharan M. S., Prabhurajhan R.

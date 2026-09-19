# FlyGPT: a trained MaleCNS circuit

The default dataset is MaleCNS. Explore is an anatomy inspector. The dedicated FlyGPT tab contains the point cloud, fly, live recurrent diagram and a larger bottom chat area. Every reply runs the checkpoint locally or in the hosted Python function; no external language model supplies answers. NDJSON events carry actual 512-cell states and token text as inference runs. The default visible-step option pauses 50 ms before each real token step, making computation observable without replaying a completed response. Turn it off for unpaced inference. The tokens/s counter divides emitted tokens by elapsed inference time including prompt processing and pacing; it is delivery throughput, not an isolated kernel benchmark.

FlyGPT draws 5,000 structural points per anatomy view at pixel ratio 1. All 512 model cells have a separate activity overlay on a fixed absolute 0–1 scale. Streaming updates affect that overlay rather than recoloring every source position. The full structural overview remains in Explore. The fly does not move from language output: no language-to-motor decoder has been trained.

## What was trained

512 cells selected from MaleCNS v1.0 by incident synapse count within central-brain intrinsic, descending and ascending classes. All 28,452 directed edges in the induced subgraph are retained. Cell IDs, class labels, selection method, input/output indices and data checksums are in `models/malecns-chat/manifest.json`.

128 cells receive token embeddings. The remaining 384 cells supply the readout. These sets do not overlap. There is no embedding-to-logit skip path and no off-graph recurrent connection. Each token performs two graph propagation steps with learned signed edge weights, tanh activation, cell bias and learned per-cell state retention. The graph specifies pathways; these cell dynamics, signs, weights and text interfaces are engineered. This is not a biological spiking model and not the full 166,700-cell nervous system.

The allocated model has 788,480 parameters, including dormant masked recurrent entries. 28,452 recurrent entries are active. The mask stays fixed during optimization. No pretrained language model supplies representations or answers.

MLX 0.32.2 trains the weights on Apple Silicon. The exported runtime uses **NumPy and tokenizers only**. Moving inference to another Python host does not require MLX. A browser implementation could read these same arrays and apply the documented recurrence; browser-native inference has not yet been implemented.

## Data and training

The external source is HuggingFaceTB/smoltalk's `everyday-conversations` subset, pinned to the revision and SHA256 files in the manifest. The original [Everyday Conversations dataset card](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k) declares Apache 2.0 and describes synthetic conversations. Do not describe this corpus as human conversations.

Source conversations are split before tokenization: 2,028 train, 232 validation, 119 source test. A 1,024-token byte-level BPE tokenizer is trained on the training split only. Initial training ran 1,000 steps at learning rate .002, then 4,000 weight-warm-start steps at .001 (optimizer/RNG reset). Best validation checkpoint was selected. The initial random loss was 6.928; best base validation loss was 2.717. The stored base training log covers the second phase.

The active `malecns-lm-512-v3` checkpoint adds 4,000 assistant next-token training steps on 5,697 unique external conversation pairs. The original 7,744 pairs contained 2,019 copies of the same greeting answer. Exact training pairs are deduplicated; no hand-written replacement answers or prompt-routing rules are used. Checkpoint selection uses validation conversation loss. Export and evaluation explicitly load `real/dialogue.safetensors`. The old authored Q&A files have been removed.

## Verification and limitations

`models/malecns-chat/evaluation.json` records:

- Held-out external text loss 2.969, versus 7.256 with recurrent edges disabled.
- Exactly 28,452 nonzero learned recurrent edges; zero off-graph edges and zero input/output overlap.
- CPU versus MLX maximum logit error below 0.000003 on the parity sequence.
- “Hi” generates “Hello! How can I help you today?”; disabling connections produces repeated punctuation.

Some questions receive incorrect, unrelated or malformed answers. Never use it as an authoritative source. The chat UI labels it as a small experimental circuit. Same-checkpoint ablation shows dependence on wiring, not superiority of biological topology; real/shuffled/ordinary-network training comparisons remain a separate experiment.

## Reproduce

With uv installed and the full MaleCNS assets prepared:

```sh
uv pip install --python .venv-physics/bin/python -r scripts/language/requirements.txt
.venv-physics/bin/python scripts/language/prepare.py
.venv-physics/bin/python scripts/language/train.py --steps 1000
.venv-physics/bin/python scripts/language/train.py --steps 4000 --resume --lr .001
.venv-physics/bin/python scripts/language/train-dialogue.py --steps 4000
.venv-physics/bin/python scripts/language/export.py
.venv-physics/bin/python scripts/language/evaluate.py
```

`prepare.py` resolves and records the current source revision; the included checkpoint manifest pins the revision actually used. To reproduce an existing revision strictly, use that manifest revision in preparation instead of resolving HEAD. Training is stochastic but seeded; backend/version changes may alter results. The checkpoint in Git permits inference without retraining or downloading the text dataset. Full MaleCNS assets are still needed for the anatomical view.

The local endpoint is `POST /api/chat/generate` with `{message, history, ablated}`. It allows one bounded inference process, validates input length/roles, accepts same-origin loopback calls only, and uses a fixed script. Chat history lives in browser component state and is sent to the local process; the server does not write conversations to disk.

The diagnostic activity array retains a legacy 0.15-second token index scale. It is not wall-clock or biological time and is not played by the chat UI. Values are signed hidden activations, not measured spikes.

## Prompt conditioning and computation view

`prompt-evaluation.json` records 64 seeded external test pairs, untouched by checkpoint selection. Mean response loss is 2.519 with the correct prompt and 2.792 with a mismatched prompt; 41 greedy replies are distinct. Responses remain frequently irrelevant or malformed. This is evidence of prompt dependence, not reliable comprehension. The new assistant objective worsens general token loss relative to v2 (2.736 to 2.969); it is not an across-the-board improvement.

The FlyGPT tab embeds an unrolled recurrent view, updated from streamed inference events. It shows the previous state and two updates for the first 8 input and first 16 readout cells in checkpoint order. These are the same 24 cells in all three columns, not separate layers. All 512 cells execute. All 82 source edges between this selected subset are shown; omitted cells still contribute. Color and opacity use actual signed state and edge contributions. Embeddings are injected at both updates; retention and bias terms also contribute. Output lines are learned readout weights, not source synapses. Next-token probabilities are raw full-vocabulary softmax before special-token generation masks. Values are captured during inference, not regenerated by an animation.

The live view follows processed prompt and generated tokens. There is no post-response automatic replay. Hover values expose source IDs, weights, contributions and input embeddings. Unit checks compare displayed snapshots with independent inference steps, and browser checks cover open, select and close behavior.

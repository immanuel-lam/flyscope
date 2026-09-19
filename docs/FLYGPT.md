# FlyGPT: a trained MaleCNS circuit

The default dataset is MaleCNS. The right side contains FlyGPT and a Neuron inspector tab. Chat runs locally through the trained checkpoint in `models/malecns-chat`; it does not call an external LLM. Every response returns the circuit states used before each generated token, indexed by real cell ID. The point cloud remains visible. Activity plays briefly, then the viewer returns to idle structural exploration. The body does not move from chat: no language-to-motor decoder has been trained.

## What was trained

512 cells selected from MaleCNS v1.0 by incident synapse count within central-brain intrinsic, descending and ascending classes. All 28,452 directed edges in the induced subgraph are retained. Cell IDs, class labels, selection method, input/output indices and data checksums are in `models/malecns-chat/manifest.json`.

128 cells receive token embeddings. The remaining 384 cells supply the readout. These sets do not overlap. There is no embedding-to-logit skip path and no off-graph recurrent connection. Each token performs two graph propagation steps with learned signed edge weights, tanh activation, cell bias and learned per-cell state retention. The graph specifies pathways; these cell dynamics, signs, weights and text interfaces are engineered. This is not a biological spiking model and not the full 166,700-cell nervous system.

The allocated model has 788,480 parameters, including dormant masked recurrent entries. 28,452 recurrent entries are active. The mask stays fixed during optimization. No pretrained language model supplies representations or answers.

MLX 0.32.2 trains the weights on Apple Silicon. The exported runtime uses **NumPy and tokenizers only**. Moving inference to another Python host does not require MLX. A browser implementation could read these same arrays and apply the documented recurrence; browser-native inference has not yet been implemented.

## Data and training

The external source is HuggingFaceTB/smoltalk's `everyday-conversations` subset, pinned to the revision and SHA256 files in the manifest. The original [Everyday Conversations dataset card](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k) declares Apache 2.0 and describes synthetic conversations. Do not describe this corpus as human conversations.

Source conversations are split before tokenization: 2,028 train, 232 validation, 119 source test. A 1,024-token byte-level BPE tokenizer is trained on the training split only. Initial training ran 1,000 steps at learning rate .002, then 4,000 weight-warm-start steps at .001 (optimizer/RNG reset). Best validation checkpoint was selected. The initial random loss was 6.928; best base validation loss was 2.717. The stored base training log covers the second phase.

Assistant-token fine-tuning uses `scripts/language/fly-dialogues.json`: 17 authored answer templates, with three train phrasings, one validation phrasing and one held-out test phrasing each. 80% of batches use these fly dialogues; 20% use external everyday dialogues. 2,000 optimizer steps were run; the best validation checkpoint was exported. These are small supervised examples, not code that selects responses at inference. Exported inference samples token logits; it never loads the authored answer file.

## Verification and limitations

`models/malecns-chat/evaluation.json` records:

- Held-out external text loss 3.069, versus 6.973 with recurrent edges disabled.
- Exactly 28,452 nonzero learned recurrent edges; zero off-graph edges and zero input/output overlap.
- CPU versus MLX maximum logit error below 0.000003 on the parity sequence.
- 11/17 exact matches on held-out fly phrasings. This is an authored-domain test sharing answer templates, not a general intelligence or broad conversation benchmark.
- “Hi” generates “Hello! How can I help you today?”; disabling connections produces repeated punctuation.

Some questions receive incorrect, unrelated or malformed answers. Never use it as an authoritative source. The chat UI labels it as a small experimental circuit. Same-checkpoint ablation shows dependence on wiring, not superiority of biological topology; real/shuffled/ordinary-network training comparisons remain a separate experiment.

## Reproduce

With uv installed and the full MaleCNS assets prepared:

```sh
uv pip install --python .venv-physics/bin/python -r scripts/language/requirements.txt
.venv-physics/bin/python scripts/language/prepare.py
.venv-physics/bin/python scripts/language/train.py --steps 1000
.venv-physics/bin/python scripts/language/train.py --steps 4000 --resume --lr .001
.venv-physics/bin/python scripts/language/finetune.py --steps 2000
.venv-physics/bin/python scripts/language/export.py
.venv-physics/bin/python scripts/language/evaluate.py
```

`prepare.py` resolves and records the current source revision; the included checkpoint manifest pins the revision actually used. To reproduce an existing revision strictly, use that manifest revision in preparation instead of resolving HEAD. Training is stochastic but seeded; backend/version changes may alter results. The checkpoint in Git permits inference without retraining or downloading the text dataset. Full MaleCNS assets are still needed for the anatomical view.

The local endpoint is `POST /api/chat/generate` with `{message, history, ablated}`. It allows one bounded inference process, validates input length/roles, accepts same-origin loopback calls only, and uses a fixed script. Chat history lives in browser component state and is sent to the local process; the server does not write conversations to disk.

A response's activity clock advances 0.15 seconds per generated token for inspection; this is an animation time scale, not biological time. Values are signed hidden activations. Glow uses absolute magnitude divided by each cell's peak; traces retain their actual sign. No spike events are inferred.

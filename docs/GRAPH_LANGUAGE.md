# experimental source-edge attention language model

This candidate is not deployed. The active FlyGPT checkpoint remains v3 until a replacement shows useful replies, source-path dependence and acceptable CPU performance. No external model supplies answers.

The candidate has 3,803,392 trainable parameters: 512 source cells each carry 256 engineered features, with four graph-attention blocks and eight attention heads. A local feed-forward operation transforms each cell's features. These are machine-learning operations, not measured biological cell dynamics.

128 token slots map to distinct input cells. A source-edge matching assigns a different readout cell to each slot. The remaining 256 cells provide intermediate paths. The mapping uses a fixed seed and source synapse counts, not language-test outcomes. Each block permits communication only on a real directed edge that also satisfies causal slot order. This retains 14,948 inter-cell edges from the 28,452-edge induced source graph. Diagonal attention represents engineered local retention. Future token slots cannot affect earlier predictions. Token input and readout cells are disjoint; there is no input-to-logit skip connection. The token embedding is also used as the learned vocabulary readout matrix.

Training uses the pinned first shard of smol-smoltalk and the existing training-only 2,048-token tokenizer. Conversation hash partitions are unchanged. This preparation preserves system instructions as text and writes separate arrays under `data/graph-language`, so earlier experiments remain reproducible. Data includes synthetic conversations. Model weights and full arrays remain ignored while this is a candidate.

## cpu execution and checks

`scripts/graph-language/runtime.py` uses NumPy and tokenizers only. Each next-token prediction recomputes the source graph over the last 128 tokens. Short inputs are left-padded. CPU traces contain actual per-block cell feature vectors; no UI adapter has yet been added for them. They must not be presented as v3's scalar recurrent states.

```sh
.venv-physics/bin/python scripts/graph-language/prepare.py
.venv-physics/bin/python scripts/graph-language/train.py --steps 40000 --run run-1
VECLIB_MAXIMUM_THREADS=1 .venv-physics/bin/python scripts/graph-language/export.py --run run-1
VECLIB_MAXIMUM_THREADS=1 .venv-physics/bin/python tests/graph_cpu_checks.py
.venv-physics/bin/python tests/graph_language_checks.py
VECLIB_MAXIMUM_THREADS=1 .venv-physics/bin/python scripts/graph-language/probe.py --run run-1
```

The exporter snapshots weights separately, records their SHA256 and compares logits and final cell features with MLX, both with source edges active and cut. It explicitly disables reduced-precision MLX matrix operations during parity checks. Training uses MLX's default precision. [MLX documents this distinction](https://ml-explore.github.io/mlx/build/html/usage/precision.html). Initial default-precision verification differed by up to 0.062 in logits; full-precision verification on the development snapshot matched within 0.00001. Runtime tests also check temporal causality, graph-cut prompt independence and actual layer traces.

An early development snapshot generated about 20.5 tokens/s on this Mac's CPU while training continued. Its replies were repetitive and irrelevant. This is neither a final speed benchmark nor successful conversation quality. The 40,000-step base run is still in progress. Final held-out evaluation and possible assistant fine-tuning remain required; neither lower training loss nor these structural tests is sufficient to promote it.

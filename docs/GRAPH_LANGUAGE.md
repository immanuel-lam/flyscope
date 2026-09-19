# experimental source-edge attention language model

This candidate is not deployed. The active FlyGPT checkpoint remains v3 until a replacement shows useful replies, source-path dependence and acceptable CPU performance. No external model supplies answers.

The candidate has 3,803,392 trainable parameters: 512 source cells each carry 256 engineered features, with four graph-attention blocks and eight attention heads. A local feed-forward operation transforms each cell's features. These are machine-learning operations, not measured biological cell dynamics.

128 token slots map to distinct input cells. A source-edge matching assigns a different readout cell to each slot. The remaining 256 cells provide intermediate paths. The corrected mapping selects high-outdegree input cells, prefers high-indegree readouts, and distributes intermediate cells over token slots. It uses source connectivity only, not language-test outcomes. Each block permits communication only on a real directed edge that also satisfies causal slot order. This retains 13,788 inter-cell edges from the 28,452-edge induced source graph. Diagonal attention represents engineered local retention. Future token slots cannot affect earlier predictions. Token input and readout cells are disjoint; there is no input-to-logit skip connection. The token embedding is also used as the learned vocabulary readout matrix.

Training uses the pinned first shard of smol-smoltalk and the existing training-only 2,048-token tokenizer. Conversation hash partitions are unchanged. This preparation preserves system instructions as text and writes separate arrays under `data/graph-language`, so earlier experiments remain reproducible. Data includes synthetic conversations. Model weights and full arrays remain ignored while this is a candidate.

## cpu execution and checks

`scripts/graph-language/runtime.py` uses NumPy and tokenizers only. Each next-token prediction recomputes the source graph over the last 128 tokens. Short inputs are left-padded. CPU traces contain actual per-block cell feature vectors; no UI adapter has yet been added for them. They must not be presented as v3's scalar recurrent states.

```sh
.venv-physics/bin/python scripts/graph-language/prepare.py
.venv-physics/bin/python scripts/graph-language/train.py --steps 40000 --run run-2
VECLIB_MAXIMUM_THREADS=1 .venv-physics/bin/python scripts/graph-language/export.py --run run-2
VECLIB_MAXIMUM_THREADS=1 .venv-physics/bin/python tests/graph_cpu_checks.py
.venv-physics/bin/python tests/graph_language_checks.py
VECLIB_MAXIMUM_THREADS=1 .venv-physics/bin/python scripts/graph-language/probe.py --run run-2
```

The exporter snapshots weights separately, records their SHA256 and compares logits and final cell features with MLX, both with source edges active and cut. It explicitly disables reduced-precision MLX matrix operations during parity checks. Training uses MLX's default precision. [MLX documents this distinction](https://ml-explore.github.io/mlx/build/html/usage/precision.html). Initial default-precision verification differed by up to 0.062 in logits; full-precision verification on the development snapshot matched within 0.00001. Runtime tests also check temporal causality, graph-cut prompt independence and actual layer traces.

An early development snapshot generated about 20.5 tokens/s on this Mac's CPU while training continued. Its replies were repetitive and irrelevant. This is neither a final speed benchmark nor successful conversation quality. The initial run was stopped after a source-reachability audit found that some input positions could never reach its final readout. The corrected run-2 is in progress, initialized from that run’s retained weights. It has complete 128-slot reachability for generation within four blocks. Auxiliary training heads have different reachable context subsets; 15 heads have their entire causal prefix available. Causality holds for every head, but complete context is only guaranteed at the final generation head. Final held-out evaluation and possible assistant fine-tuning remain required; neither lower training loss nor these structural tests is sufficient to promote it.


## reply training

`prepare-dialogue.py` verifies every token against the existing immutable arrays, then indexes 252,324 training replies, 2,547 validation replies and 2,668 test replies. `train-dialogue.py` samples assistant targets only and pads short contexts on the left exactly as the CPU runtime does. Windows never cross source conversation boundaries, and target tokens cannot enter their own input. The same source hash partitions are retained. A two-update pilot passed; this is a pipeline check, not model-quality evidence. Source indexing and window tests are separate from the topology and numerical checks.

## evaluation

`evaluate.py` selects one reply from each of 64 seeded source conversations, independently of model output. It uses the final CPU readout, not an average over auxiliary heads. The report retains every context, source reply and generated reply, along with correct-prompt, mismatched-prompt and source-cut losses. It also reports first-eight-token losses and repeated token four-grams. Source replies are supplied only to teacher-forced loss computation; free generation receives the prompt alone. Prediction and generation default to a 64-token limit, which is reported explicitly. These held-out source conversations are a reused development split across experiments, not an independent final benchmark.

```sh
VECLIB_MAXIMUM_THREADS=1 .venv-physics/bin/python scripts/graph-language/evaluate.py --run run-2 --examples 64 --tokens 64
.venv-physics/bin/python tests/graph_evaluation_checks.py
```

A two-example, four-token validation run on the pilot checkpoint verifies the pipeline only. Its poor outputs were retained. It does not measure the quality of the still-running corrected training experiment.

A preliminary corrected-run snapshot was evaluated on 16 validation conversations through the CPU runtime. Correct/mismatched/source-cut response losses were 3.341/3.634/7.802. First-eight-token losses were 2.924 with the matching prompt and 3.614 with a mismatched prompt. However, repeated four-gram fraction was 0.334 and raw replies were malformed and irrelevant. This supports prompt and pathway dependence, not useful chat. The snapshot remains undeployed; full training continues. All source contexts, replies, generated tokens and checkpoint identity are retained in `performance/graph-language-corrected-early-validation.json`.

The corrected base run subsequently completed all 40,000 updates in 1,952 seconds, with best validation loss 3.474. Its simple CPU replies remained unusable; see `performance/graph-language-base-probes.json`. Logits matched full-precision MLX within 0.000018; maximum absolute feature difference was 0.000245, with absolute/relative tolerance checks passing. Five CPU behavior/inspection tests passed. The base checkpoint is not deployed. Full 64-conversation CPU evaluation and 10,000 updates of reply-focused fine-tuning have started. This fine-tuning includes short padded contexts that the base random-window objective did not cover.

Reply-focused training completed 10,000 updates in 520 seconds, with best assistant validation loss 3.077. It still failed all six simple development probes: replies were irrelevant or repetitive. It is rejected for deployment; raw outputs and training history are in `performance/graph-language-dialogue-probes.json`. The completed base model's full 64-conversation report is in `performance/graph-language-base-evaluation.json`; dialogue evaluation is still running. Base and dialogue CPU exports now live in separate `portable` and `portable-dialogue` directories. Use `--checkpoint dialogue` with export, probe and evaluation commands to select the latter. A conventional attention control is the next diagnostic: it must remain separate from FlyGPT and must not supply its answers.

## conventional attention diagnostic

`train.py --architecture control --run control-1 --steps 40000` trains ordinary causal attention over 128 token positions. It uses the same tokenizer, arrays, random-window seed, width, block count, optimizer and update count as the graph run. It has 3,705,088 parameters, about 2.6% fewer than the graph model because it has 128 position embeddings rather than 512 cell embeddings. It starts from random weights; graph run-2 warm-started from the interrupted first mapping. This is a pipeline diagnostic, not an exactly matched topology experiment or evidence of biological superiority.

The control has no source-cell IDs or biological wiring. `export-control.py` makes an explicitly labelled CPU export and verifies numerical agreement. It cannot produce a fly-cell inspection, and it is never wired into the website. With attention cut, the control still receives its current-token embedding at the readout, so its ablation must be evaluated per token. The evaluator does not reuse the graph model's prompt-independent source-cut shortcut for this control. A pilot checks training, assistant fine-tuning, CPU parity, causality and the evaluation path before the full diagnostic run.

## computation inspection

The CPU runtime's `inspect(prefix)` records the same forward pass used for prediction. It selects the last eight input cells and last sixteen readout cells, including the actual generation readout. Five columns contain a chosen feature before and after four graph blocks. Per-edge values decompose the attention update after projection into that feature. Separate fields retain actual local feed-forward updates and self-retention contributions. Tests independently compare the sum of edge contributions with the full attention matrix computation and compare displayed logits with ordinary inference.

The anatomical overlay data is RMS over all final features for each of the 512 source cells. It is an engineered continuous model value, not a spike rate. A selected feature cannot explain a whole vocabulary logit; vocabulary readout uses all 256 features of the final readout cell. Edge contributions describe a particular attention update, not a causal attribution of the entire model. Unselected cells still contribute. This interface is not yet connected to the website and must not be passed off as the existing three-column scalar recurrent trace.

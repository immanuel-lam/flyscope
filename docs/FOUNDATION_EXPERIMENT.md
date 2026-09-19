# pretrained foundation adaptation experiment

This experiment is not deployed. Both the roughly four-million-parameter graph model and ordinary attention control failed simple conversation probes after training. A larger pretrained foundation is now being tested to separate basic language learning from adaptation to source wiring.

The pinned source is [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. Its model card declares Apache-2.0. The safetensors file contains 134,515,008 parameters. Exact file, source-cell and wiring fingerprints are in `scripts/foundation/source.json`. The earlier 3.8-million-parameter candidate remains unchanged. TinyStories metadata was also inspected, but its downloaded model card did not state a licence and it was not selected.

## computation contract

Graph mode uses the same 512 MaleCNS source cells, causal mask and disjoint 128 input/readout pairs from the corrected attention experiment. Each cell now carries 576 engineered features. Token embeddings enter only input cells. An explicit initial identity-feature transfer follows each verified input-to-readout source edge. This is a declared engineered synaptic operation, not an embedding-to-logit bypass: cutting inter-cell paths disables the transfer as well as attention.

Thirty subsequent blocks use the pretrained local feature transforms, but token-to-token attention is replaced by source-cell attention under the fixed directed source mask and causal slot order. Padding keys are excluded. Rotary positions use token-slot indices shared by their assigned cells. Local self operations are retained. The vocabulary readout receives only the final source readout cell. No ordinary model is run to supply or correct graph-mode outputs.

The foundation supplies learned language parameters; the connectome supplies communication constraints. This does not mean that a biological fruit fly knows language, that vector features are measured compartments, or that these operations reproduce real neural dynamics. The initial relay and local transforms must remain visible in model documentation and computation inspection.

## implementation and verification

`scripts/foundation/runtime.py` runs on NumPy CPU and tokenizers only. It reads numeric safetensors data without pickle or remote model code. The ordinary-reference mode exists only for diagnostic comparison. `model.py` is the MLX training counterpart and uses the installed MLX-LM Llama modules for the pinned foundation. Dependencies are isolated in `.venv-foundation`; existing physics and language environments are unchanged.

Checks compare CPU ordinary-reference logits with the original library implementation and compare source-mode logits/states with the trainable counterpart. They test causal ordering, an explicit source-paired relay, and loss of prompt information when inter-cell paths are cut. A separate attention-only cut preserves the relay and verifies that it alone cannot process earlier prompt content.

```sh
VECLIB_MAXIMUM_THREADS=4 .venv-foundation/bin/python tests/foundation_checks.py
VECLIB_MAXIMUM_THREADS=4 .venv-foundation/bin/python scripts/foundation/probe.py --tokens 16
```

The first eight-token CPU probe began a correct cat definition in ordinary mode, while source mode became incoherent. Source mode measured about 3.6 tokens/s in that short run. This is an untrained adaptation, not a useful replacement or a deployment performance guarantee. Adaptation training, broader raw-output evaluation, weight packaging, live-UI integration and deployment checks remain open. Ordinary-reference success cannot be counted as success of the source-wired model.

## adaptation training

`prepare.py` re-tokenizes the pinned source conversations with the foundation tokenizer, preserves source system messages, and keeps the existing conversation-level hash partitions. The resulting training partition has 112,742 conversations and 101,630,951 tokens. Assistant targets include the end marker. Explicit validity masks distinguish padding from valid end tokens. Validation and test partitions are held out from this adaptation only: the foundation's model card lists this same instruction corpus, so these are not globally unseen benchmarks.

`train.py` updates only the 26,542,080 attention-projection parameters. Embeddings, local MLPs, source wiring and the initial relay remain fixed. It saves full weights for CPU verification, with a validation-selected checkpoint and the run settings. This is not yet a deployment package.

The first pilot produced non-finite gradients despite finite forward outputs. Structurally unreachable cells remain exactly zero in this bias-free network; repeatedly differentiating their zero-state normalization caused overflow. The trainable implementation now sets these provably unreachable states to constant zero at each block. This preserves forward outputs and source edges while excluding nonexistent derivatives. CPU parity, causality, relay, ablation and finite-gradient checks pass, including both full and padded contexts.

The corrected five-step pilot reduced its fixed 16-example validation loss from 5.5665 to 5.4683 and used about 2.29 GB peak MLX memory. This establishes training feasibility, not conversational ability. Its complete report is `docs/performance/foundation-training-pilot.json`.

```sh
.venv-physics/bin/python scripts/foundation/prepare.py
.venv-foundation/bin/python scripts/foundation/train.py --run adapt-1 --steps 2000
```

The first 2,000-step run finished in 157.37 seconds with best auxiliary-readout validation loss 3.8803. Its saved CPU checkpoint passes the numerical and structural checks, but its raw replies remain repetitive and often incorrect. It is rejected for deployment.

The next experiment scores only the final readout cell, matching the cell used for generation. Earlier auxiliary readouts have incomplete source-path access to some preceding slots, so their average objective differs from the actual generation objective. The new pilot uses batch four and 64 validation targets, with about 5.54 GB peak MLX memory. Its initial loss of 2.4308 is **not comparable** to the prior all-readout average of 3.8803. A five-step pilot had finite gradients but did not improve validation loss; it verifies execution only. A longer run is needed to test usefulness.

```sh
.venv-foundation/bin/python scripts/foundation/train.py --run adapt-readout-1 --initial-run adapt-1 --target last --batch 4 --steps 5000 --seed 197 --validation-examples 64
VECLIB_MAXIMUM_THREADS=4 FOUNDATION_CHECKPOINT=data/foundation/adapt-1 .venv-foundation/bin/python tests/foundation_checks.py
VECLIB_MAXIMUM_THREADS=4 .venv-foundation/bin/python scripts/foundation/evaluate.py --run adapt-1 --tokens 48
```

`--initial-run` creates a separate artifact and records the initial weight hash; it never overwrites the earlier run. Seven checks now include reply-target alignment and conversation-boundary preservation. Generated evaluation reports retain every prompt, token ID, raw response, CPU timing and weight hash. No candidate is promoted based on loss alone.

## smaller CPU weight storage

`quantize.py` exports two-dimensional weight matrices as symmetric signed int8 values with one float32 scale per row; one-dimensional normalization weights remain float32. The CPU runtime verifies each tensor archive's checksum and shape, then reconstructs float32 weights once at load. This reduces artifact size, not resident inference memory, and does not use GPU inference. Source wiring, cell IDs and the source mask are included in the export. The source checkpoint is preserved.

On the rejected `adapt-1` checkpoint, stored tensor bytes fell from 538,090,247 to 135,562,940. A fixed 32-target comparison had 31 matching top tokens and mean reference-to-quantized KL divergence 0.00355. Teacher-forced loss was 3.0212 before and 3.0008 after quantization; this small change is not evidence of better language quality. The source-path cut still produces zero logits. See `docs/performance/foundation-int8-storage.json` for every sample and artifact hash.

Two storage checks verify every tensor's reconstruction error against half its row quantization step and reject corrupted checksums or invalid archive paths. Seven existing CPU, gradient, source and sampling checks also pass after the loader change. These local results do not prove Vercel bundle acceptance or performance, and no quantized candidate is deployed.

```sh
VECLIB_MAXIMUM_THREADS=4 .venv-foundation/bin/python scripts/foundation/quantize.py --run adapt-1
VECLIB_MAXIMUM_THREADS=4 .venv-foundation/bin/python tests/foundation_storage_checks.py
VECLIB_MAXIMUM_THREADS=4 .venv-foundation/bin/python scripts/foundation/check-storage.py --run adapt-1
```

## compact computation inspection

`runtime.next(..., inspect=True)` captures a compact record during the actual CPU calculation. It includes RMS activity for all 512 source cells and feature zero for eight input and sixteen readout cells across all 30 blocks. Each displayed directed connection carries its actual attention-weighted value after the output projection. Contributions from other cells, self operations and local MLP updates are reported separately. The initial source-paired relay is explicit. These are continuous engineered model states, not firing measurements.

The trace includes raw full-vocabulary top-token probabilities before generation masking. It cannot be requested for the ordinary-reference model. Inspection leaves logits and final states unchanged. Tests reconstruct every displayed update from its terms, independently recompute the largest first-block displayed edge from the model weights, and distinguish a complete intercell cut from an attention-only cut that preserves the relay.

```sh
VECLIB_MAXIMUM_THREADS=4 .venv-foundation/bin/python scripts/foundation/inspect-computation.py --run adapt-1 --prompt 'What is a cat?'
```

The sample raw-text probe produced a 177,583-byte JSON record in about 0.410 seconds of CPU computation while training was running. This is a local diagnostic measurement, not a hosted speed guarantee. Full feature-vector debug snapshots are optional and separate. The compact trace is not yet connected to the website; the public FlyGPT still uses its existing scalar-model trace. Any replacement requires reply-quality, stream and browser checks before promotion.

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

# Trained delayed-cue memory

This is a separate non-language experiment on the same 512-cell MaleCNS subgraph. It learns to retain a left/right cue after input is removed. It does not improve FlyGPT automatically and is not biological conditioning.

Each trial contains three cue steps followed by zero-input delay steps. The cue is a two-component vector, mapped to the 128 input cells by a learned encoder. The 384 output cells are disjoint from the inputs. Signed recurrent weights exist only on the source graph's 28,452 directed edges. Learned local retention, bias and a two-class linear readout complete the model. All of these dynamics and mappings are engineering assumptions. One step is an abstract model update, with no biological time calibration.

Training uses seed 73, random balanced-in-expectation labels, cue strength 0.7–1.3 and delay 4–12 steps. It performs 601 optimizer updates on cross-entropy loss. The evaluation uses seed 79, balanced labels, delays 16/24/48 and strengths centred at 0.4/1.6 with 10% random variation. The full evaluation reports all six conditions, not a selected subset.

## Reproduce and inspect

```sh
VECLIB_MAXIMUM_THREADS=1 .venv-physics/bin/python scripts/memory/train.py
VECLIB_MAXIMUM_THREADS=1 .venv-physics/bin/python scripts/memory/evaluate.py
```

`models/malecns-cue-memory/runtime.npz` contains portable CPU weights. Its manifest lists the source cell IDs, tensor indices, graph fingerprint, seed and training history. `weights.safetensors` is the MLX training checkpoint; `evaluation.json` stores the held-out measurements. `scripts/memory/runtime.py` uses only NumPy and performs actual recurrent updates; it has no label lookup or answer table. State is kept inside `CueMemory`, and `reset()` removes it.

All 768 held-out trials were classified correctly. The same evaluation with reset state, graph ablation, or untrained weights reached exactly 50%. CPU and MLX logits agreed within 0.0000153. These controls support learned cue retention through the circuit. They do not establish general memory, semantic learning, reasoning, or biological fidelity. The test has only two cue categories; held-out strengths and delays are not unseen concepts.

## Physical-fly integration

Enable **Trained cue memory** in the local physics panel. The default five-second experiment presents three 100 ms cue updates, followed by 16 zero-input updates. Starting at 1.8 seconds, the trained logit difference supplies a bounded steering stimulus to the descending population of the separate whole-CNS locomotion model. The cue itself is never supplied to that locomotion model. Before response time, its forward stimulus is zero. This 100 ms update interval is an engineering choice, not biological timing.

The **Activity overlay** selector displays either the trained 512-cell memory states or the whole-CNS locomotion rates. They share source cell IDs and a playback clock, but are two distinct computational models. Their values are never merged into a single claimed biological state. The memory panel shows the actual cue, zero-input delay, logits, phase and applied steering. Replay and rewind read the recorded states.

```sh
.venv-physics/bin/python scripts/physics/run.py --memory --duration 5 --output data/physics/memory-left.json
.venv-physics/bin/python scripts/physics/run.py --memory --cue-side right --duration 5 --output data/physics/memory-right.json
```

Generate `memory-reset-left.json` and `memory-reset-right.json` with `--reset-memory`, and `memory-silenced.json` with `--silenced`. Then run `VECLIB_MAXIMUM_THREADS=1 .venv-physics/bin/python tests/memory_physics_checks.py`.

The verified left/right trials had identical body motion before 1.8 seconds. Final lateral positions were +10.049 mm and -5.984 mm. Resetting state after the cue made both body trajectories identical; it does not stop movement, because the trained model has a cue-independent default response. Silencing both models left only 0.126 mm settling displacement and zero recorded states. The checker independently reconstructs each memory step and compares all recorded values to CPU inference. These are controlled engineering experiments, not a claim that the reconstructed fly remembers natural experiences.

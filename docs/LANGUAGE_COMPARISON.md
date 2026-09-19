# Language wiring comparison

This experiment uses the original scalar-state, 1,024-token model and conversation corpus. It does not evaluate the new four-channel chat candidate. The complete measurements are in [the result file](performance/language-topology-comparison.json).

| wiring | active parameters | mean held-out loss | sample standard deviation |
| --- | ---: | ---: | ---: |
| real malecns subgraph | 554,788 | 2.9166 | 0.0306 |
| row-shuffled source mask | 554,788 | 2.8321 | 0.0121 |
| dense recurrent, 391 cells | 555,071 | 5.4306 | 0.3230 |

Lower token cross-entropy is better. The shuffled model performed better than the real graph in all three seeds. This test therefore provides no evidence that real MaleCNS topology helps this language task. The real model's graph-ablation effect establishes dependence on connections, not an advantage of their biological topology.

The dense baseline performed poorly under the shared training settings. Its learning rate and other settings were not tuned separately. This is a weak dense baseline, so these numbers do not establish that sparse models are generally better. The test is small and uses three seeds, a limited corpus and one training budget. It cannot establish biological language ability or general assistant quality.

## Protocol

All models receive the same token batches for each seed (7, 17, 29), 1,000 AdamW updates, batch size 16, sequence length 48, learning rate 0.002 and gradient clipping at 1. Each sees 768,000 training tokens. The final checkpoint is evaluated on the same 64 held-out length-128 sequences. There is no held-out checkpoint selection. Reported losses use the same tokenizer and corpus.

The real and shuffled models have 512 cells and disjoint 128-cell input / 384-cell output pools. Shuffling preserves each destination's incoming edge count; it does not preserve every network statistic. The dense model has 391 cells, 128 input cells and 263 disjoint output cells, giving an active parameter count within 0.1% of the other models. It has more recurrent edges and fewer output units, which is a real architectural difference needed to match the budget.

Run order rotates between seeds. A different training job ran concurrently, so no wall-clock speed comparison is claimed. All nine rows, including the poor dense results, are retained.

```sh
VECLIB_MAXIMUM_THREADS=1 .venv-physics/bin/python scripts/language/compare-topology.py
```

Checkpoints are written under ignored `data/language/topology-comparison/`; the experiment never replaces the active chat checkpoint. The published report includes input-data and script checksums.

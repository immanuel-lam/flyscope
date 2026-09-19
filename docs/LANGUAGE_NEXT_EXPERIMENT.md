# Next language experiment: memory and broader data

The v3 model produces prompt-dependent states but often irrelevant text. The next candidate is an experiment, not a promised intelligent assistant. The deployed checkpoint stays unchanged until the candidate has evidence of useful improvement.

- Keep the same 512 source IDs, 28,452-edge mask and disjoint 128/384 input/readout populations.
- Add four state channels per source cell, with local input/state-dependent retention gates. These channels and gates are engineered memory, not identified biological compartments. Inter-cell messages still require a source edge; no token embedding-to-output bypass is allowed.
- Train a 2,048-token tokenizer on the training split only.
- Use the complete first training shard of HuggingFaceTB/smol-smoltalk at revision f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc, with exact-conversation deduplication and deterministic hash-based train/validation/test partitions. No authored fly Q&A and no hand-selected target answers.
- Select checkpoints on validation loss. Keep test conversations out of training and selection. Inspect unedited replies and quantify correct versus mismatched prompt response loss, ablation, repetition and CPU throughput. Generic loss improvements alone do not establish conversation quality.
- Implement a portable CPU runtime and verify logits and all state channels against MLX before considering export. GPU use is training-only. Do not label multi-channel activations as biological spikes.

A four-channel model has more memory and weights than v3. Measure its cost instead of assuming it is fast or capable. Keep provenance for every training run. If it still produces irrelevant replies, report that failure and keep working; do not add canned fallbacks.


## Current run audit

Base training completed 10,000 updates on 20 September 2026. The CPU export matched MLX logits within 0.00000382. The initial held-out report had correct-prompt loss 3.190 versus mismatched-prompt 3.231 and graph-ablated 8.045. All 64 replies differed, but repetition and irrelevant content remained. Variety is not intelligence. The four-channel model has 5,253,120 allocated and 4,318,352 active parameters. It is not the deployed checkpoint.

A source audit found 17,598 original conversations containing a system message. The current preparation keeps only user and assistant messages, so some rewrite/task instructions were dropped. This is a training-data limitation, particularly for context-dependent rewriting examples. The running dialogue stage uses that same prepared corpus; do not silently describe it as preserving all source instructions. A corrected data experiment must use separate files so it cannot mutate an active memory-mapped training array.


The 4,000-step dialogue stage finished with validation loss 2.843. On the untouched general test prompts, correct/mismatched/ablated losses were 3.049/3.111/8.262, and repeated token four-gram fraction was 0.233. The candidate was rejected for deployment: simple questions still elicited irrelevant, repetitive replies. Unedited test outputs are retained in `performance/language-memory-candidate-evaluation.json`.

A follow-up uses complete short contexts, including source system instructions, and full replies from the same pinned data. Eligibility is based only on context/reply length, not model performance. It retains the same conversation hash split, preserves the original tokenizer, and writes separate arrays. No authored answers are added. The first 16 reply tokens receive twice the training weight to put more pressure on prompt-conditioned answer starts. It will be evaluated against both the original broad prompts and held-out short contexts; a favorable short-context score alone will not justify a general intelligence claim.

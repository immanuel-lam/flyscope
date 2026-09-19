# MaleCNS experimental test subject

Requested 19 September 2026. Finish and verify each feature before advancing.

1. Restore full structural point-cloud exploration when idle; overlay actual simulated rates during playback; explicit paused activity inspection. Implemented; unit/build/browser verification passed.
2. Train a small local MLX chat/language model through real MaleCNS directed pathways. Save checkpoint, tokenizer, IDs, data provenance, held-out evaluation and graph-ablation results. Integrate chat and actual neural state display. Implemented and verified: exported CPU runtime, 11/17 held-out authored phrasings, graph-ablation loss increase, MLX/NumPy parity, default MaleCNS and FlyGPT side panel. See FLYGPT.md.
3. Vision and navigation: display actual model input, test target seeking and visual ablation.
4. Odour tracking: defined field, neural encoding, source-finding test and ablation.
5. Obstacle avoidance: physical collisions, perception and held-out-layout tests.
6. Learning and memory: trained adaptation, retained state and held-out cue tests.
7. Language experiment comparisons: real vs shuffled wiring and a matched ordinary recurrent baseline; report actual results.

Commit every verified step without attribution trailers. Push step 1 only; keep all subsequent work in local commits unless the user asks to push. Do not label the network as biologically validated. Current physics activity is continuous rate dynamics, not spike events.

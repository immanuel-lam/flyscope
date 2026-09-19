# MaleCNS experimental test subject

Requested 19 September 2026. Finish and verify each feature before advancing.

1. Restore full structural point-cloud exploration when idle; overlay actual simulated rates during playback; explicit paused activity inspection. Implemented; unit/build/browser verification passed.
2. Train a small local MLX chat/language model through real MaleCNS directed pathways. Save checkpoint, tokenizer, IDs, data provenance, held-out evaluation and graph-ablation results. Integrate chat and actual neural state display. Implemented and verified: exported CPU runtime, general next-token checkpoint without authored Q&A fine-tuning, graph-ablation loss increase, MLX/NumPy parity, default MaleCNS and FlyGPT side panel. See FLYGPT.md.
3. Vision and navigation: implemented actual bilateral fisheye camera input, pixel-driven target seeking, shared-clock image playback and source-target geometry. Verified left/right target approach against disabled and silenced controls. See PHYSICAL_FLY.md and performance/vision-validation.json.
4. Odour tracking: defined field, neural encoding, source-finding test and ablation.
5. Obstacle avoidance: physical collisions, perception and held-out-layout tests.
6. Learning and memory: trained adaptation, retained state and held-out cue tests.
7. Language experiment comparisons: real vs shuffled wiring and a matched ordinary recurrent baseline; report actual results.

Commit every verified step without attribution trailers. The user subsequently authorized pushing the current verified changes on 19 September 2026. The user has since authorized pushing all completed, verified work when the full goal is done. Do not label the network as biologically validated. Current physics activity is continuous rate dynamics, not spike events.

Chat now uses a pending “Thinking…” indicator and leaves anatomy idle; no post-response state replay. The heading is “Explore the fruit fly nervous system.”

Language follow-up: v3 uses deduplicated external assistant-token training. Prompt-conditioning tests show lower loss with the matching input, but generated semantics remain poor. Computation inspection shows measured recurrent states and source edges with no automatic replay. Authored Q&A files removed.

Follow-up: dedicated live FlyGPT workspace, 5k structural rendering budget with all 512 model activity cells, larger chat, end-to-end tokens/s, and read-only weight inspection command. Intelligence remains unproven; no canned response fallback. Backflip physics experiment added to requested follow-up work.

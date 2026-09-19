# MaleCNS experimental test subject

Requested 19 September 2026. Finish and verify each feature before advancing.

1. Restore full structural point-cloud exploration when idle; overlay actual simulated rates during playback; explicit paused activity inspection. Implemented; unit/build/browser verification passed.
2. Train a small local MLX chat/language model through real MaleCNS directed pathways. Save checkpoint, tokenizer, IDs, data provenance, held-out evaluation and graph-ablation results. Integrate chat and actual neural state display. Implemented and verified: exported CPU runtime, general next-token checkpoint without authored Q&A fine-tuning, graph-ablation loss increase, MLX/NumPy parity, default MaleCNS and FlyGPT side panel. See FLYGPT.md.
3. Vision and navigation: implemented actual bilateral fisheye camera input, pixel-driven target seeking, shared-clock image playback and source-target geometry. Verified left/right target approach against disabled and silenced controls. See PHYSICAL_FLY.md and performance/vision-validation.json.
4. Odour tracking: implemented bilateral antenna sampling of a defined field, engineered neural stimulus, shared-clock observations and source marker. Two source positions improve final distance versus disabled steering; silenced control has zero rates. This demonstrates approach, not reliable arrival or plume tracking. See performance/odour-validation.json.
5. Obstacle avoidance: implemented colliding terrain, head-geometry range rays, neural steering, telemetry and replay. Five layouts show contact reduction, including two held-out layouts without contact. Offset retains contact; held-out right detours with worse forward progress. Efficient general navigation remains unproven. See performance/obstacle-validation.json.
6. Learning and memory: trained delayed-cue circuit, 768 held-out delay/strength trials, reset/untrained/graph-ablation controls, CPU parity and physical delayed steering verified. Separate memory and locomotion overlays share the replay clock. Binary cue retention only, not general semantic memory. See CUE_MEMORY.md.
7. Language experiment comparisons: real vs shuffled wiring and a matched ordinary recurrent baseline; report actual results.

Commit every verified step without attribution trailers. The user subsequently authorized pushing the current verified changes on 19 September 2026. The user has since authorized pushing all completed, verified work when the full goal is done. Do not label the network as biologically validated. Current physics activity is continuous rate dynamics, not spike events.

Chat now uses a pending “Thinking…” indicator and leaves anatomy idle; no post-response state replay. The heading is “Explore the fruit fly nervous system.”

Language follow-up: v3 uses deduplicated external assistant-token training. Prompt-conditioning tests show lower loss with the matching input, but generated semantics remain poor. Computation inspection shows measured recurrent states and source edges with no automatic replay. Authored Q&A files removed.

Follow-up: dedicated live FlyGPT workspace, 5k structural rendering budget with all 512 model activity cells, larger chat, end-to-end tokens/s, and read-only weight inspection command. Intelligence remains unproven; no canned response fallback. Assisted backflip implemented and verified with actual rotation, landing and neural-silencing checks. It uses explicitly labelled external force/torque, not a learned biological skill.

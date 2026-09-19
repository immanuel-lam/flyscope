# Flyscope agent entry point

This repository is a React/TypeScript/Three.js fruit-fly workbench. When asked to build a model, controller, environment, game, task, or simulation for the fly, use the existing integration contracts. Do not create a disconnected demo or insert model logic into the renderer.

## Read first

1. `docs/AGENT_INTEGRATION.md` — integration decision tree and exact build/test workflow.
2. `src/motor/types.ts` — authoritative controller, observation, command, pose, and track types.
3. `docs/DATA_FORMAT.md` — neuron/activity format and provenance.
4. For an external physics simulator, read `docs/EXTERNAL_SIMULATORS.md` before writing an adapter.

## Implementation rules

- In-process controllers belong at `src/controllers/<id>/controller.ts` and export `controller: MotorController`. Generate one with `npm run new:controller -- <id>`. Vite discovers it automatically.
- Keep rendering in `src/Scene.tsx` and the reduced body rig in `src/motor/rig.ts`. Controllers return motor commands, never mutate Three.js objects or React state.
- Call `create()` for fresh state on each run. Use simulation time and fixed `dt`; do not use wall-clock time, unseeded randomness, hidden shared state, network calls, or asynchronous steps in deterministic controllers.
- Read neuron activity by stable string ID. Missing activity is `undefined`, not a measured zero. Use `unsupported()` to reject datasets with the wrong ID, version, units or required channels. Do not invent biological motor mappings.
- Motor commands are forward speed in mm/s, yaw rate in rad/s, and wing spread from 0 to 1. Pose angles are radians relative to the procedural rig's neutral pose; all 20 channels are mandatory. The viewer uses +Y up, +Z forward, +X right.
- External simulators export a validated `Dataset.motor` track. Keep simulator execution and credentials outside the browser. Use the explicit CSV conversion adapter when appropriate; do not relabel reduced-rig replay as a full MuJoCo rig.
- Preserve geometry, activity and motor provenance independently. Our servo/kinematic simulator is not a contact, muscle or flight-physics solver. Synthetic activity is not real firing data. Root registration and morphology within the body remain illustrative.
- Keep legacy datasets without motor tracks working. Bad files must fail before replacing the active dataset. Avoid whole-connectome main-thread imports.
- Do not add arbitrary script execution from imported files. Controller modules are trusted local source code reviewed with the project.
- Retain the user's truth/evidence requirements. Communicate in clear, simple technical English; use standard English in code and files. Never claim external physics or biological validation based on our tests.

## Required completion checks

Run `npm test`, `npm run build`, and `npm run test:browser` for changes to controllers, playback, rig or integration. Run `python3 -m unittest discover -s tests -p '*_test.py'` if changing the CSV adapter. Add task-specific behavioral tests, not just shape checks. Verify replay/rewind, missing-channel behavior, dataset switches and at least one browser-visible result. Report what actually ran and any unavailable external backend.

## Usual deliverable

A discoverable controller or a validated importable run, its declared mappings/units/provenance, focused tests, and an updated integration note. Preserve existing user work. Do not merge, publish or deploy as part of a local integration request.

## Full data and upcoming model work

- Read `docs/LARGE_DATASETS.md` before working with full MaleCNS data. Keep graph chunks and geometry detail out of the React render state unless selected. Render the full point overview at rest; apply explicit interaction LOD and cache budgets.
- For joint neural/motor output, use `simulateExperiment()` and the controller's optional activity declaration. Never show arbitrary synthetic signals as a real cell recording.
- The user's next intended experiments are in `docs/EXPERIMENT_ROADMAP.md`: an actually trained connectome-constrained text model, then a panel showing its actual visual input. A trained 512-cell chat checkpoint now exists; see docs/FLYGPT.md. Vision, odour tracking, obstacle avoidance and delayed cue memory now have local physical experiments and control tests. See docs/PHYSICAL_FLY.md and docs/CUE_MEMORY.md; improving general chat quality remains open.

## Functional physics backend

Read `docs/PHYSICAL_FLY.md` for real NeuroMechFly/MuJoCo runs. The local job API executes the full MaleCNS rate network and contact physics in Python, then replays full body transforms. Its sensory encoding, dynamics and population decoder are engineering assumptions. Keep those labels. Modify the external loop for physical environments; do not add motion in the renderer. Run `npm run test:physics` after generating the three documented control recordings when this backend changes. Preserve the language and vision roadmap.

## FlyGPT runtime

Read `docs/FLYGPT.md` before changing the language model. Keep source cell IDs and fixed directed mask, disjoint text input/output cells, and the no-external-LLM contract. MLX is training-only; inference must remain portable. Tests must include graph ablation and CPU/export parity. The goal and publication boundary are in `docs/GOAL_PROGRESS.md`: the user authorized pushing all completed verified work when the goal is finished. Commit verified steps without attribution trailers, then verify the GitHub push and public deployment.

## Sensory and memory experiments

The authoritative external loop is `scripts/physics/run.py`. Vision uses actual rendered eye frames; odour samples the actual antennal positions; obstacle rays start at the head geometry because this locomotion model has no independent head body. Add colliding terrain to `world.ground_geoms` before adding the fly so FlyGym creates explicit contact pairs. Do not rely on geometry collision flags alone.

Preserve separate states for the trained cue-memory circuit and the whole-CNS locomotion circuit. Their source IDs overlap, but they are two distinct models; `PhysicsRun.memory.activity` must not be merged with `PhysicsRun.activity`. The activity selector changes the inspected model on the same body clock. The assisted backflip uses explicit external force/torque and must retain that label.

Reproduce control recordings and run feature checks listed in docs/PHYSICAL_FLY.md and docs/CUE_MEMORY.md. Report failed layouts and tradeoffs, not only successful cases. Never claim an obstacle controller is universally collision-free from a few fixed-seed tests, or that binary cue recall proves language reasoning.

# Build something that plugs into the fly

Give a coding agent this repository and a task. Repository-aware agents should read the root `AGENTS.md` automatically. For other agents, explicitly say “Read AGENTS.md and docs/AGENT_INTEGRATION.md first.” These files are local project instructions, not global agent memory.

## Choose the integration point

| What you are building | Where it plugs in | Deliverable |
| --- | --- | --- |
| A deterministic behavior or toy environment | `MotorController` | New discovered controller module |
| A brain/model that emits neuron values | `Dataset.activity` | JSON recording with stable neuron IDs and explicit units |
| A decoder from neural activity to locomotion | `MotorController` plus `unsupported()` gate | Documented ID mapping and controller tests |
| Python/MuJoCo/FlyGym/FlyBody body simulation | `Dataset.motor` | Converted pose recording and source provenance |
| Closed-loop physics with real contact sensors | External simulator owns the loop | Export its results; the browser does not supply contact observations |
| A game with high-level commands | Controller if deterministic/offline; otherwise an external run | Explicit observation encoding, motor readout and recording |

The browser runs synchronous offline controllers and can launch bounded local Python physics jobs. See [PHYSICAL_FLY.md](PHYSICAL_FLY.md) for the coupled MaleCNS/NeuroMechFly loop. Jobs compute before playback; there is no live WebSocket control.

## Fast path: add a controller

```sh
npm run new:controller -- my-experiment
```

This creates `src/controllers/my-experiment/controller.ts` and `tests/my-experiment.test.ts`. It refuses existing paths, reserved IDs and unsafe names. Vite finds the module without a registry edit. In the viewer, select it under **Motor lab → Motor controller**, then press Play. The template is neutral until implemented.

The essential module contract:

```ts
import type { MotorController } from '../../motor/types.ts';

export const controller: MotorController = {
  id: 'my-experiment',
  label: 'My experiment',
  description: 'A scripted test; no claim of biological control.',
  unsupported: () => null,
  create: () => ({
    step: ({ time }) => ({
      forward: time < 2 ? 1 : 0,
      turn: time >= 2 && time < 4 ? 0.6 : 0,
      wing: 0,
    }),
  }),
};
```

`create(context)` gets the exact dataset and the workbench's `{forward, turn, wing}` parameters. The built-in sliders are shown for the manual controller; custom controllers can supply their own internal configuration or add a deliberate UI component. Each new run calls `create()` again. State may live inside that closure. Never use module-level mutable state.

`step(observation)` gets:

- `time`: seconds at the start of this step.
- `dt`: seconds for this step (normally 1/120; the final step may be shorter).
- `pose`: a detached copy of the previous reduced-rig motor pose.
- `activity(id)`: the most recent neural sample at this time, or `undefined`.

No ground forces, contact sensors, visual observations or odors are provided. Implement and label those outside the viewer if your task needs them. A controller can calculate geometric toy-environment observations from pose and a declared world, but they are not physical sensor measurements.

Return `{forward, turn, wing}`. The engine rejects non-finite outputs, clamps forward to ±3 mm/s, turn to ±2 rad/s and wing spread to [0,1], smooths commands, generates an alternating-tripod target gait, then integrates damped joint servos. Root movement is planar kinematics, independent of foot-ground forces. Wing spread is a pose command, not aerodynamic flight.

Local simulations run at 120 Hz for up to 60 seconds. Their poses are recorded before playback. Playback speed does not alter simulation physics. Changing manual parameters starts a new run at time zero. Scrubbing interpolates recorded motor poses. This is why rewind is reproducible.

## Wiring real neuron activity

Use `src/controllers/neural-readout/controller.ts` as a structural example, not as a biological decoder. Its six `demo-*` IDs are synthetic and must not be replaced by arbitrary real IDs.

Before enabling a biological dataset:

1. Pin dataset ID, version, specimen and activity unit.
2. List exact input neuron IDs and define missing-value behavior.
3. Cite or label the origin of the readout mapping. Learned mappings need training/evaluation provenance.
4. Normalize values explicitly; Hz, mV and arbitrary units are not interchangeable.
5. Add an `unsupported()` check for all assumptions and tests that wrong versions/missing channels are rejected.
6. Keep synthetic circuit dynamics, neural readout, motor control and physical body dynamics as separate components.

The demo's neural controller changes locomotion if those six channels change. It does not run a recurrent connectome simulation or train a language model.

## Export/import a complete run

```sh
npm run example:motor
npm run validate:dataset -- examples/neural-motor-run.json
```

Import the generated JSON through **Import dataset**. It opens in **Imported motor replay** mode with the same neural activity and motor time axis. **Export motor run** saves the current motor run alongside its dataset; **Export current dataset** saves only the originally loaded dataset, which may already contain a motor track.

For code use:

```ts
import { simulateMotor } from './src/motor/engine.ts';
import { parseDataset } from './src/data.ts';
// dataset must already be parsed; controller must satisfy MotorController.
const motor = simulateMotor(dataset, controller, {forward: 1, turn: 0, wing: 0}, 10);
const output = parseDataset({...dataset, motor});
```

Generated runs include model name, source, dataset ID/version, units and an optional `generator` block with controller ID, command parameters and fixed step. Keep the controller source revision with experiment records as well. Imported motor tracks must refer to the same dataset ID/version. This identifies the run context; it does not prove biological correspondence or anatomical registration.

## Tests an agent must add

Test a neutral condition, a positive response, missing/invalid input, deterministic repeatability and the task's actual success criterion. Examples: changing a mapped neuron changes steering; silence stops the neural controller; a turning command changes heading; a model cannot run with the wrong dataset version.

Then run:

```sh
npm test
npm run build
npm run test:browser
```

The browser suite checks pose changes, time seeking, export/import replay, dataset switching and unsupported neural control. Use browser inspection to verify the fly actually moves; telemetry alone is insufficient for rig changes.

## Example requests to give an agent

> Read AGENTS.md. Build a controller called odor-seek with a declared synthetic odor field. Use the previous root pose to sample the field, turn toward increasing concentration, and stop near the source. Do not label its samples as biological olfactory activity. Register it through the existing controller convention and test that it reaches the target more often than a neutral controller.

> Read AGENTS.md. Add a decoder for my model output using this dataset version and these exact neuron IDs. Define the normalization and missing-data behavior. Keep the renderer unchanged. Export a ten-second synchronized neural/motor run and test that channel ablation changes the behavior.

> Read AGENTS.md and docs/EXTERNAL_SIMULATORS.md. Adapt my pinned FlyGym run to the procedural rig. Export pose CSV, document each reduced-joint mapping and axis/unit conversion, validate the JSON, and verify replay. Do not claim full-rig physical accuracy in this viewer.

## File map

- `src/data.ts`: structure/activity schema and integrated validation.
- `src/motor/types.ts`: stable TypeScript contracts and joint inventory.
- `src/motor/engine.ts`: fixed-step servo/kinematic engine and deterministic pose sampling.
- `src/motor/validation.ts`: runtime motor import checks.
- `src/motor/rig.ts`: articulated procedural legs.
- `src/motor/MotorPanel.tsx`: controller selection, manual inputs and telemetry.
- `src/controllers/registry.ts`: source-module discovery.
- `src/Scene.tsx`: pose application, wing rig and camera following.
- `scripts/new-controller.mjs`: agent scaffolding.
- `scripts/convert-motor-csv.py`: explicit external pose adapter.

## Coupled activity + motor output

A controller can declare `activity: {kind, unit}` and emit `neural: Record<string, number>` alongside its motor command. `simulateExperiment()` records those exact values and the resulting motor poses on one clock. All emitted IDs must exist in the dataset, values must be finite, and the channel set must remain constant through the run. The returned `{motor, activity}` can be merged into the dataset and validated/exported. `simulateMotor()` alone records only motor output.

The walking-circuit module is a concrete example: 24 synthetic oscillator rates determine its forward command. Its input assumptions forbid using the readout on real MaleCNS cells. The UI shows these controller signals instead of the unrelated pre-generated demo signal. This illustrates integration, not a validated spiking connectome model.

For full connectomes, use [LARGE_DATASETS.md](LARGE_DATASETS.md), not the small JSON-import path. The language-model and vision requirements are preserved in [EXPERIMENT_ROADMAP.md](EXPERIMENT_ROADMAP.md).


## Extend the physical experiments

Use `scripts/physics/run.py` for simulator-owned sensing, neural updates and body dynamics. The existing modules are `vision.py` (actual eye pixels), `odour.py` (defined field at antennal positions), `obstacles.py` (actual MuJoCo rays and terrain contact pairs), and `backflip.py` (explicitly external assistance). A separate trained circuit in `scripts/memory/runtime.py` supplies delayed-cue decisions. See PHYSICAL_FLY.md and CUE_MEMORY.md for exact commands and control recordings.

Add typed optional observations to `src/physics/types.ts`, record their actual simulator timestamps, and display the previous observation on the common playback clock in `PhysicsPanel.tsx`. Render only geometry/transforms delivered by the simulator. Keep old recordings without the new field valid. The local endpoint has a fixed argument allowlist in `server/physics-plugin.ts`; extend that validation rather than accepting arbitrary executable names or flags.

Each feature needs a disabled-sensor or disabled-controller comparison, a neural-silencing control where applicable, and a browser check for playback and rewind. Keep actual learned-model values separate when two models reuse source neuron IDs. Record assumptions and negative results. Publishing remains controlled by the user's current request, not by the existence of a deploy script.

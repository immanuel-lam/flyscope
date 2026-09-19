# Functional physical fly

The MaleCNS view now runs a local, coupled neural/body simulation. Choose **Run physics**, wait for computation, and press **Play** or scrub the time slider. This is computed replay, not a real-time interactive simulator. Two simulated seconds currently take about 14.5 wall-clock seconds on this Mac.

The body is NeuroMechFly/FlyGym 2.1.0 with MuJoCo 3.9.0. Joint position actuators, gravity, collisions, foot adhesion and the published hybrid walking controller produce physical motion. The browser renders the simulator's compiled meshes and every body's measured world transform. It does not use the old procedural gait for these runs. Wings are passive; this implementation does not fly.

## Neural connection and its limits

The model evaluates a state for all 166,700 classified MaleCNS v1.0 neurons using the downloaded weighted directed graph. It is an **engineered rate network**, not a validated biological fly nervous system. Normalized state values are not spikes or measured firing rates.

For each 10 ms neural step:

```
r += (dt / 0.05) * (tanh(max(0, 0.9 * W @ r + input)) - r)
```

`W[post, pre]` contains synapse counts normalized by each postsynaptic cell's total absolute retained input. Source consensus neurotransmitters assign assumed signs: acetylcholine +1; GABA and glutamate -1; unknown and other transmitters 0. This ignores receptor dependence and neuromodulation. The original graph has 25,582,938 edges; 24,469,412 have a nonzero sign in this model. No evidence supports treating these assumed dynamics as identified physiology.

The stimulus is injected into descending neurons with L/R soma annotations (656/648 cells). Mean rates from thoracic T1–T3 motor cells (251/249) are multiplied by 24 and clipped to [0, 1.3], then supplied to the hybrid controller's two descending signals. Motor populations are not mapped to individual muscles. The hybrid CPG/reflex controller supplies the detailed gait. This population decoder is an engineering assumption, not a discovered biological walking circuit.

Every 10 ms the backend counts feet with measured ground-contact force by side. Counts inject up to 0.08 into root-side-labelled VNC sensory populations (3,186/3,170 cells). This is also an assumed sensory encoding. The hybrid controller independently retains its own physical reflex inputs when neural feedback is disabled.

The browser records 756 source cells: every decoder cell plus 256 distributed input/sensory/intermediate cells. Unrecorded cells have **missing display data**, not zero activity. Neural placement inside the body's head is illustrative, not anatomical registration. The whole CNS includes the VNC and should not be interpreted as fitting inside the head.

## Reproduce

```sh
npm run setup:physics
# Requires the catalog and graph from npm run prepare:malecns.
npm run simulate:physics -- --duration 2 --output data/physics/driven.json
npm run simulate:physics -- --duration 2 --silenced --output data/physics/silenced.json
npm run simulate:physics -- --duration 2 --no-feedback --output data/physics/no-feedback.json
npm run test:physics
```

The setup uses a separate Python 3.12 environment. Source NT data comes from the [MaleCNS download page](https://male-cns.janelia.org/download/), with its CC BY 4.0 attribution. Physics and controller sources: [NeuroMechFly](https://neuromechfly.org/) and [FlyGym repository](https://github.com/NeLy-EPFL/flygym). Dependency versions are pinned in `scripts/physics/requirements.txt`.

The fixed-step physics loop is 0.1 ms, neural updates are 10 ms, and exported poses are approximately 30 Hz. Frames preserve MuJoCo wxyz quaternions and source coordinates X forward, Y left, Z up, in millimetres. The viewer maps source coordinates to [-Y, Z, X]. Mesh-local transformations are preserved. A `mj_forward` call initializes derived poses after resetting the simulation, before recording frame zero.

The signed sparse matrix is cached at `data/physics/signed-normalized-v1.npz`. Remove that generated cache if the dataset or sign/normalization policy changes. Keep full dataset versions pinned; this model is specific to the prepared v1.0 catalog ordering.

## Evidence (19 September 2026)

Identical two-second runs, seed 0, stimulus 1, bias 0:

| Condition | Horizontal thorax displacement |
| --- | ---: |
| Neural output + contact feedback | 11.147 mm |
| Neural model silenced | 0.126 mm |
| Neural output, contact input to network disabled | 10.922 mm |

Silenced displacement is passive settling, not walking. All neural output is exactly zero in that control. Graph ablation also makes the decoder output zero. Tests verify deterministic neural states, finite body transforms, unit quaternions, actual contacts and changed neural states/trajectory when sensory input is removed. Final active thorax height is 1.129 mm. These checks establish software coupling and physical movement; they do not validate biological behavior or demonstrate learned control.

See `docs/performance/physics-validation.json` for measured results. Browser tests verify synchronized playback, rewind, rendering and dataset switching. Training a language model and showing actual visual observations remain separate, unfinished work.

## Agent integration

- Neural dynamics/encoding/readout: `scripts/physics/brain.py`.
- Body construction, physics loop, observation extraction and recording: `scripts/physics/run.py`.
- Local bounded job API: `server/physics-plugin.ts` (`POST /api/physics/run`, `GET /status`, `GET /result`, `POST /cancel`). Only same-origin loopback requests; fixed executable, validated bounded parameters, one run at a time, 180-second limit.
- Browser controls: `src/physics/PhysicsPanel.tsx`.
- Versioned full-body recording contract: `src/physics/types.ts`; actual mesh replay: `src/physics/rig.ts`.
- `PhysicsRun` export preserves the full physical model, transforms, sampled neural signals, parameters and provenance. It is a separate format from `Dataset.motor`; the small Dataset importer does not accept it yet. **Load latest run** restores the local backend's most recent result.
- `physicsTelemetry()` supplies only root telemetry for existing UI components. Its zero joint slots must never be used as a physical joint export or to render the physical fly.

When adding an environment or model, keep the closed loop in Python. Extend the observation/decoder deliberately and test ablations. Do not insert a second independent motion path in Three.js. New datasets require a new explicit annotation, encoding and decoder policy; matching cell names is insufficient.

## Activity display

Physical replay retains every recorded neuron in a separate glow layer in both views. The original point cloud remains visible. When playback stops, the activity overlay is hidden and the viewer returns to structural exploration. Check **Show activity while paused** to inspect a recorded time explicitly. Glow strength is the current nonnegative rate divided by that cell’s maximum recorded rate; this relative scaling exposes low-rate motor cells without changing exported values or the trace. It is not a shared amplitude comparison across cells. Steady rates stay steady, and zero rates produce no glow. No spike times or periodic flashes are invented. The threshold slider uses the shared absolute range during activity display; it does not hide the idle explorer. The initial inspector selection chooses the recorded cell with the largest rate variation after the initial 20% of the run.

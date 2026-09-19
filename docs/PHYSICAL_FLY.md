# Functional physical fly

The MaleCNS view now runs a local, coupled neural/body simulation. Choose **Run physics**, wait for computation, and press **Play** or scrub the time slider. This is computed replay, not a real-time interactive simulator. An earlier isolated two-second run took about 14.5 wall-clock seconds on this Mac. The latest parallel validation runs, concurrent with language training and browser checks, took about 49 seconds; those timings are not a matched performance comparison.

The body is NeuroMechFly/FlyGym 2.1.0 with MuJoCo 3.9.0. Joint position actuators, gravity, collisions, foot adhesion and the published hybrid walking controller produce physical motion. The browser renders the simulator's compiled meshes and each compiled body's measured world transform. Fixed segments fused into a parent, such as the head, retain their mesh on that parent and are omitted as independent body rows. New exports include compiled body IDs and source-file checksums; missing body IDs must never index another body's transform. It does not use the old procedural gait for these runs. Wings are passive; this implementation does not fly.

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

## Eye-camera navigation

Enable **Eye-camera navigation** before running physics. The backend adds the actual left/right FlyGym eye cameras and a red visual target. Every 100 ms it renders two 450 × 512 fisheye RGB frames. A defined red-pixel mask measures each eye's red fraction; the normalized bilateral difference supplies a bounded descending-neuron stimulus bias. The full MaleCNS rate network then determines the motor population drive. No target coordinates or bearing are supplied to the controller. The target coordinates are used only for world construction and distance metrics.

The browser shows 225 × 256 display thumbnails on the shared physics clock, retaining the original resolution for control. Scrubbing and rewind select the preceding sampled observation. The target sphere is reproduced in the physical view at its recorded source position. It is visual-only and does not collide.

This is engineered color-target seeking, not identified retinal neuron mapping or biological vision. The rate network, sensory encoder and motor decoder retain their earlier assumptions. It has been tested on two target positions at a fixed seed, not arbitrary environments.

Two-second target-distance results (mm): left 1.310 with visual steering versus 1.993 disabled; right 3.056 versus 6.112 disabled. Neural silencing leaves only 0.126 mm passive settling and zero recorded rates. See `docs/performance/vision-validation.json`. Checks verify camera images, observation clocks, changed neural trajectories, steering ablation and browser rewind.

```sh
.venv-physics/bin/python scripts/physics/run.py --duration 2 --vision --output data/physics/vision-left.json
.venv-physics/bin/python scripts/physics/run.py --duration 2 --vision --no-vision-control --output data/physics/vision-disabled.json
.venv-physics/bin/python scripts/physics/run.py --duration 2 --vision --target-y -4 --output data/physics/vision-right.json
.venv-physics/bin/python scripts/physics/run.py --duration 2 --vision --target-y -4 --no-vision-control --output data/physics/vision-right-disabled.json
.venv-physics/bin/python scripts/physics/run.py --duration 2 --vision --silenced --output data/physics/vision-silenced.json
.venv-physics/bin/python tests/vision_checks.py
```

## Odour tracking

Enable **Odour tracking** in the local physics panel and choose the source side. The simulator samples `exp(-distance² / (2 × 5²))` at the actual left and right funiculus positions in millimetres. A bounded bilateral contrast supplies the same engineered descending-neuron stimulus pathway used for locomotion. Motor output still comes from the full rate network. The source is a scalar field, not a simulated turbulent plume, chemical receptor model, or identified olfactory circuit.

The panel displays both antenna concentrations and applied steering at the shared playback time; rewind restores the original observations. A green ground ring marks the source for the viewer only. It adds no physical force or observation to the controller. Vision and odour, if both enabled, share the source position; their steering stimuli are summed and bounded.

Reproduce the controls with `.venv-physics/bin/python scripts/physics/run.py --odour --duration 2 --output data/physics/odour-left.json`. Add `--target-y -4` for the right source, `--no-odour-control` to retain sensing without steering, or `--silenced` to zero neural activity. Save the corresponding `odour-right`, `odour-disabled`, `odour-right-disabled`, and `odour-silenced` JSON files before running `.venv-physics/bin/python tests/odour_checks.py`.

For the two fixed-seed, two-second tests, left-source final distance was 1.654 mm versus 1.993 mm without odour steering; right-source distance was 5.047 mm versus 6.110 mm. Neural silencing left 0.126 mm of settling displacement and zero recorded rates. These runs show a modest control effect through the rate model; they do not establish reliable source arrival, general plume tracking, or biological olfaction. See `performance/odour-validation.json`.

## Physical obstacle avoidance

Enable **Physical obstacles** to add a colliding cylinder. **Range-based avoidance** controls whether sensor readings influence the neural stimulus; disabling it leaves the same physical world in place. The simulator uses nine artificial horizontal MuJoCo rays from the head geometry, filtered to obstacle geometry, with an 8 mm range. A bounded turn stimulus enters the descending population and is held for 0.5 seconds after the rays clear. This is engineered perception and control, not biological vision. Terrain contact feedback also includes obstacle contacts.

FlyGym uses explicit contact pairs: the obstacles are included in its terrain geometry collection before the fly is compiled. The head body is attached to the thorax in this locomotion model, so rays use the actual head geometry position, not a missing head body ID. Rendering uses recorded obstacle dimensions and poses; it adds no avoidance motion.

Run `.venv-physics/bin/python scripts/physics/run.py --obstacles --layout training --duration 3 --output data/physics/obstacle-training.json`. Layouts are `training`, `offset`, `wide`, `heldout-left`, and `heldout-right`. For each layout, save a matching `obstacle-<layout>-disabled.json` using `--no-avoidance`. Also save `obstacle-silenced.json` using `--silenced`. Run `.venv-physics/bin/python tests/obstacle_checks.py` to verify measured contact reduction, continued movement, the real ray intersections, and the silenced control.

Results are in `performance/obstacle-validation.json`. Contact time fell in all five fixed-seed layouts. Four runs had no obstacle contact; `offset` retained 0.459 seconds of contact versus 1.377 seconds disabled. Both held-out layouts had no contact, but the right layout made only 6.759 mm forward progress versus 17.026 mm disabled: avoidance traded speed for clearance. The tests therefore verify contact reduction and movement, not efficient route completion or universal collision-free navigation. The three development layouts informed implementation; the two held-out layouts were run after the clearance rule was fixed.

## Assisted backflip

Enable **Assisted backflip** to run an explicitly external-force stunt. Once mean neural motor output exceeds 0.1 after 0.6 seconds, a controller applies bounded lift and torque to the thorax for 0.9 seconds. It tracks an 8 mm lift arc and a full backward rotation, then removes all applied forces. MuJoCo integrates the motion and landing; no body poses or joint positions are teleported. Joint locomotion control continues. This is not a learned backflip, biological muscle mapping, flight model, or evidence that the connectome specifies this behavior.

The UI records applied force and torque at each body sample and labels the assist state. The force/torque values use the simulator's model units. Neural activity remains the actual whole-CNS rate output; it gates the external controller but does not generate its trajectory.

```sh
.venv-physics/bin/python scripts/physics/run.py --backflip --duration 3 --output data/physics/backflip.json
.venv-physics/bin/python scripts/physics/run.py --backflip --silenced --duration 3 --output data/physics/backflip-silenced.json
.venv-physics/bin/python tests/backflip_checks.py
```

The final verified fixed-seed run completed a backward rotation, reached 9.024 mm maximum thorax height, and landed at 1.128 mm with upright cosine 0.999 and eight contact points. The test measures actual recorded body quaternions, not the commanded angle. All external force and torque were zero after the assist ended. Neural silencing prevented activation entirely. Earlier tuning produced an incomplete rotation; only the final measured checkpoint is claimed successful. See `performance/backflip-validation.json`.

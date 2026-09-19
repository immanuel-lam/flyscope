# Flyscope

A local React + TypeScript + Three.js workbench for fruit fly neuron structure and activity. Includes two real MaleCNS skeletons and a separate synthetic demonstration circuit. No API keys required. The optional physical fly uses a local Python backend.

## Run

```sh
npm install
npm run dev
```

Open the local URL printed by Vite. `npm run build` creates a static site in `dist/`; `npm run preview` serves it locally.

## Features

- Linked neuron and procedural fly views, orbit/zoom, camera reset and neuron selection.
- Region filtering, neuron search, activity threshold, optional directed-edge geometry.
- Playback, scrub, speed control, per-neuron trace and explicit activity provenance.
- Local JSON/SWC imports with validation and JSON export. Files remain in the browser.
- Bundled MaleCNS v1.0 DNge104 pair; original units and attribution retained.
- Responsive controls; reduced-motion-friendly initial paused state; accessible form controls and list-based neuron selection.

The synthetic fixture contains 720 neurons and generated signals. It is not a reconstructed brain or a biological dynamics model. The fly is a procedural illustration, and brain placement within its head is not registered to an anatomical template. The motor lab supports manual locomotion and a synthetic neural readout. This is a reduced servo/kinematic model, not ground-contact or flight physics.

## Motor lab and agent integrations

Select **Motor lab → Manual walking** or **Demo neural readout**, then press Play. Walking, turning and wing spread drive articulated joints and a moving root. The camera follows the body. Playback, pause and rewind use a deterministic fixed-step run. Manual parameter changes reset the run. Real anatomy can be shown with manual motion, but no biological motor mapping is inferred.

**Export motor run** saves structure, activity and motor poses in one JSON file. Import it to replay the run. External simulators can supply poses through an explicit CSV adapter. The MaleCNS physical mode runs a local NeuroMechFly backend; see below.

Agents should start with [AGENTS.md](AGENTS.md) and [the integration guide](docs/AGENT_INTEGRATION.md). The project instructions explain automatic controller discovery, neuron/channel mapping, provenance and tests.

```sh
npm run new:controller -- my-experiment
npm run example:motor
npm run validate:dataset -- examples/neural-motor-run.json
```

See [external simulator adapters](docs/EXTERNAL_SIMULATORS.md) for FlyGym/FlyBody boundaries and a runnable pose conversion example. The generated sample is synthetic; it is not an external physics result.

## Reuse in another project

`src/data.ts` defines the versioned input contract and validation. `src/Scene.tsx` renders a dataset independent of its source. `src/App.tsx` handles imports, selection and playback. Structure and activity use identical string neuron IDs. Export the synthetic fixture for a complete example; see `docs/DATA_FORMAT.md` for a minimal file.

Load a small dataset subset, not a whole synapse dump. Current limits are 30 MB per imported file, 20,000 neurons, 100,000 directed edges, and 500,000 skeleton segments. Those are safety caps, not performance guarantees. JSON is parsed on the main thread. Only the first 80 search matches appear in the list; refine the query or click a neuron in 3D to select others. Coordinates are normalized for rendering only. Unknown SWC coordinate units remain explicitly unspecified.

Connection lines describe source-to-target records but do not display directional arrowheads. Filtered-out neurons are dimmed, not removed. Connection lines are shown only with no region/activity filter. Activity colors use the full dataset value range, not a biological firing threshold. The right-hand active count means values above 50% of that range.

## Verify

```sh
npm test
npx playwright install chromium
npm run test:browser
npm run build
python3 -m unittest discover -s tests -p '*_test.py'
```

Data tests validate source SWCs, ID integrity, temporal sampling and malformed data rejection. The browser tests cover WebGL mounting, playback, search, real skeleton loading, imports, mobile layout, visible motor movement, exact rewind and exported motor replay. Motor tests cover determinism, neutral commands, neural ablation, unsupported dataset rejection and pose validation.

## Research and attribution

See `docs/RESEARCH.md` for the MaleCNS/FlyWire distinction, model options, primary sources and a proposed connectome-constrained language experiment. See `public/DATA_SOURCES.md` for the CC BY 4.0 attribution and exact source URLs of the bundled real neurons.

Future work includes live model streaming, full-brain tiling, a registered body rig and model training. This first version is the viewer foundation.

## Full MaleCNS model

Choose **MaleCNS · full catalog** or open `/?dataset=malecns`. The prepared local dataset contains 166,700 classified neurons and their source positions, plus 25,582,938 directed weighted connections. Detailed skeletons and selected outgoing edges load on demand; the overview does not draw every neurite or edge at once. See [large dataset preparation and limits](docs/LARGE_DATASETS.md) and [measured performance](docs/PERFORMANCE.md).

For walking with visible controller activity, choose the synthetic dataset and **Walking neural circuit**. This records the actual synthetic rates used by that toy controller. Direct manual commands still do not simulate real neurons.

[Language model and visual-input roadmap](docs/EXPERIMENT_ROADMAP.md) preserves the requested next stages; neither a trained language checkpoint nor a fly-eye input renderer is implemented yet.

## Functional physical fly

In the full MaleCNS view, choose **Run physics**, then **Play**. NeuroMechFly and MuJoCo simulate forces and contacts; an experimental whole-connectome rate model supplies population motor drive and receives contact feedback. The viewer replays actual simulator meshes/body transforms and sampled neural signals. A two-second run takes about 14.5 seconds locally. Use the silence and feedback controls to compare behavior.

Setup, measured controls, assumptions, full-run export and agent extension points: [PHYSICAL_FLY.md](docs/PHYSICAL_FLY.md). This is a functioning coupled engineering simulation, not a biologically validated neural walking model. Language training and visual input are still unfinished.

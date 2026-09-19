# Flyscope dataset v1

Imports accept JSON or a single SWC file. JSON source IDs must be strings. SWC alone supplies structure; it cannot supply connections between cells or electrical activity. Export current dataset in the UI to get a complete runnable fixture.

```json
{
  "schemaVersion": 1,
  "id": "my-circuit",
  "name": "Two-neuron example",
  "source": "My simulator; synthetic fixture",
  "version": "1.0",
  "geometry": "synthetic",
  "coordinateSpace": "example-local",
  "units": "um",
  "neurons": [
    {"id": "a", "label": "Neuron A", "region": "Example", "position": [-1, 0, 0], "skeleton": [-1, 0, 0, 0, 1, 0]},
    {"id": "b", "label": "Neuron B", "region": "Example", "position": [1, 0, 0], "skeleton": [1, 0, 0, 0, 1, 0]}
  ],
  "edges": [{"source": "a", "target": "b", "weight": 3}],
  "activity": {
    "kind": "simulation",
    "unit": "mV",
    "times": [0, 0.1, 0.2],
    "values": {"a": [-70, -50, -65], "b": [-70, -65, -50]}
  }
}
```

`geometry` is `synthetic` or `reconstruction`. Each skeleton is an optional flat array of line segment endpoints `[x1,y1,z1,x2,y2,z2,...]`, in the dataset coordinate space and units. `position` is the selection point; a SWC import uses the first root node, which is not guaranteed to be a soma. Edges are directed and weights are non-negative source-defined values; weights are retained but not used as synaptic dynamics or linewidth.

`activity` is optional. `kind` must be `synthetic`, `simulation` or `recording`; times are strictly increasing non-negative seconds. Each recorded neuron has one finite value per timestamp. Missing neurons have no activity. Before the first sample there is no activity; between samples the preceding value is held. Frames describe values such as binned firing rates or membrane potential, not a sparse event stream. Convert spike trains to explicit bins first, and label the units.

Color normalization uses `[min(0, observed minimum), observed maximum]`. A constant range is expanded by one unit. Missing activity uses the structural region color. Threshold and active-count percentages refer to this display range. They do not indicate biological activation.

Preserve dataset, specimen, version and coordinate space when adapting source data. Never join MaleCNS and FlyWire IDs directly. Merge all desired morphology into a common registered frame outside the viewer before import. Credentials must stay outside exported datasets.


## Optional motor track

`motor` adds synchronized body replay without changing the neuron schema. Existing schemaVersion 1 datasets remain valid. It is optional and independently declares motor provenance.

```json
{
  "schemaVersion": 1,
  "rig": "flyscope-procedural-v1",
  "kind": "simulation",
  "source": "My pose exporter; declared mapping",
  "model": "my-backend@pinned-version",
  "datasetId": "my-circuit",
  "datasetVersion": "1.0",
  "units": {"position": "mm", "angle": "rad", "time": "s"},
  "times": [0],
  "poses": [{
    "position": [0, 0, 0],
    "heading": 0,
    "joints": {
      "LF.sweep": 0, "LF.lift": 0, "LF.knee": 0,
      "LM.sweep": 0, "LM.lift": 0, "LM.knee": 0,
      "LH.sweep": 0, "LH.lift": 0, "LH.knee": 0,
      "RF.sweep": 0, "RF.lift": 0, "RF.knee": 0,
      "RM.sweep": 0, "RM.lift": 0, "RM.knee": 0,
      "RH.sweep": 0, "RH.lift": 0, "RH.knee": 0,
      "wing.L": 0, "wing.R": 0
    }
  }]
}
```

Place that object in the containing dataset's `motor` field, not at the file root. Identity and version must match the containing dataset. `times` and `poses` have equal lengths, with at most 18,001 strictly increasing non-negative timestamps. Positions are limited to ±10,000 mm, and all 20 joint offsets must be within ±π radians. A pose-only dataset still needs a structure section; the small external-structure fixture is a runnable example.

Body-world coordinates are separate from neuron coordinates: +Y up, +Z forward, +X right. Heading is unwrapped yaw in radians. Joint offsets are relative to the procedural neutral pose, not universal anatomical angles. The brain is attached to the moving root with an illustrative transform. See [EXTERNAL_SIMULATORS.md](EXTERNAL_SIMULATORS.md) for axis signs and explicit retargeting.

The shared viewer time spans the longer of the activity and motor tracks. Motor poses interpolate linearly between timestamps, stay neutral before the first frame, and hold the final pose afterwards. Neural values retain previous-sample holding. An exported local motor simulation is limited to 60 seconds; imports can represent longer time spans within the frame/file limits.

Local simulations also export `generator: {controllerId, parameters: {forward, turn, wing}, stepSeconds}`. This optional metadata preserves run inputs; the replay always uses recorded poses, not an automatic controller re-run.

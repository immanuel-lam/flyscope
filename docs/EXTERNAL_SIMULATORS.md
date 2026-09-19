# External physics simulator adapters

Checked 19 September 2026. The local motor lab integrates damped joint servos with a procedural gait and planar root kinematics. It has no contact, muscle, gravity or aerodynamic solver. Its browser rig is a reduced visual model, not a validated physical digital twin.

For actual body dynamics, the relevant primary projects are:

- [NeuroMechFly/FlyGym](https://neuromechfly.org/): embodied simulation with joint actuation and sensory feedback. Its [model composition tutorial](https://neuromechfly.org/tutorials/1a_basic_model_composition/) describes position and torque actuation, axis order and neutral poses.
- [FlyBody](https://github.com/TuragaLab/flybody): MuJoCo fruit-fly body models and locomotion tasks developed by Google DeepMind and HHMI Janelia.

A local NeuroMechFly 2.1.0 / MuJoCo 3.9.0 backend now runs coupled MaleCNS experiments. See [PHYSICAL_FLY.md](PHYSICAL_FLY.md) for execution, assumptions, actual full-body geometry and measured controls. The CSV bridge below remains available for other backends. APIs, DOFs and joint conventions vary.

## Pose contract

A dataset may have a `motor` member. See `src/motor/types.ts` and `src/motor/validation.ts` for the authoritative runtime contract:

- `schemaVersion: 1`, `rig: "flyscope-procedural-v1"`.
- `kind: "simulation" | "recording"`, `source`, `model`.
- `datasetId` and `datasetVersion` must match the containing dataset.
- `units: {position: "mm", angle: "rad", time: "s"}`.
- Strictly increasing non-negative `times`, at most 18,001 frames.
- One pose per time: `{position: [x,y,z], heading, joints}`.
- Position is in a right-handed body-world frame: +Y up, +Z forward at zero heading, +X right. Positive heading rotates +Z toward +X; heading must be unwrapped before export.
- All 20 joints are required: `LF`, `LM`, `LH`, `RF`, `RM`, `RH`, each with `.sweep`, `.lift`, `.knee`; plus `wing.L`, `wing.R`. L/R mean anatomical left/right; F/M/H mean front/middle/hind.
- Joint values are additive radians from this renderer's neutral geometry, within ±π. Source absolute joint angles are not directly interchangeable with these offsets.

Before the first motor frame the rig is neutral. Between frames the viewer interpolates root position, unwrapped heading and joint offsets. After the last frame it holds the final pose. Neural activity uses its own timestamp series and previous-sample holding; both clocks are in seconds from a common run origin.

## Reduced joint convention

The procedural rig's leg sweep rotates around local Y, multiplied by side (-1 left, +1 right). Lift rotates around local Z multiplied by side. Knee rotates around local Z multiplied by negative side. Wing spread rotates the wing pivot around local Z with side sign. This rig lacks many DOFs and uses illustrative segment dimensions; source angles require an explicit retargeting policy, not a name match.

For a fully faithful physical replay, add a separate versioned rig with its actual body transforms, joint axes, meshes and limits. Do not overload `flyscope-procedural-v1` or silently discard source DOFs.

## Working CSV bridge

Export a pose CSV from your simulator and prepare a mapping JSON. Each destination channel has either `{ "column": "name", "scale": 1, "offset": 0 }` or `{ "constant": 0 }`. Constants are explicit omissions, and must be documented. The mapping must contain `time`, `x`, `y`, `z`, `heading` and all 20 joints.

- For metre positions, use scale 1000.
- For degrees, use scale 0.017453292519943295.
- If the source is Z-up, first identify its forward/right directions, then explicitly permute and sign the axes. No universal FlyGym/FlyBody axis mapping is assumed here.
- For quaternion orientations, compute an unwrapped heading in the target frame in the source exporter. This adapter does not guess quaternion ordering or project rotations.
- Downsample high-rate poses to a suitable rendering rate without aliasing. Preserve the raw simulation separately.

```sh
python3 scripts/convert-motor-csv.py \
  --dataset examples/external-structure.json \
  --csv examples/external-poses.csv \
  --mapping examples/external-mapping.json \
  --source "Synthetic adapter fixture; not a MuJoCo run" \
  --model "adapter-test-v1" \
  --output examples/my-converted-run.json
npm run validate:dataset -- examples/my-converted-run.json
```

The bundled fixture demonstrates the converter, not physics. Import the result in the viewer. The converter refuses to overwrite an existing output. It validates pose values; the subsequent TypeScript validation checks the complete containing dataset with the same parser used by the browser.

## Closed-loop ownership

Run the actual loop in the backend: environment observations → neural model → motor decoder → simulator step → measured simulator observations. Record the neural values and resulting poses on a shared clock. Keep model/decoder versions, time step, random seed, joint mapping, source dataset identity, licensing and parameters in your experiment records. The viewer replays output; it does not send physical commands or infer motor-neuron correspondence from neuron labels.

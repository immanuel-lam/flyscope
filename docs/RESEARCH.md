# Fruit fly connectomes and a language-learning experiment

Research checked 19 September 2026. The implementation in this repository is a visualization workbench, not a trained language model or a validated whole-brain simulator.

## Which Google release?

Google's 3 September 2026 announcement describes the MaleCNS project with HHMI Janelia and collaborators: more than 166,000 neurons and 125 million synaptic connections, including the brain and ventral nerve cord. This differs from the earlier female FlyWire brain. The project page dates MaleCNS v1.0 to 8 June 2026; the announcement and data release are different events.

- [Google Research announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
- [MaleCNS project and release history](https://male-cns.janelia.org/)
- [MaleCNS download documentation](https://male-cns.janelia.org/download/)
- [FlyWire dataset/version and access FAQ](https://codex.flywire.ai/faq)

## Data paths

The canonical MaleCNS page documents neuron annotations, neurotransmitter predictions, a weighted connection graph, synaptic locations, and skeletons. It recommends neuPrint for programmatic queries and navis for morphology. Its raw public SWC objects can also be downloaded without an account. Bulk data sizes on the page include 1.1 GB for connection weights and 12.7 GB for synapse points; whole-dataset rendering is a separate engineering task.

This workbench bundles two unchanged SWCs explicitly shown in the project's documentation: DNge104_R (body ID 12781) and DNge104_L (556329), from MaleCNS v1.0. Coordinates use 8 nm voxels, not micrometres. The viewer fits their joint bounding box into a display space and preserves their relative positions. It does not change the exported coordinates. The files contain reconstructed centerlines, not measured electrical activity, and no connectivity between the pair is claimed. See DATA_SOURCES.md for attribution and exact URLs.

FlyWire Codex supports multiple datasets, but public bulk downloads in its own portal currently focus on FAFB and BANC. Use the project-specific source for MaleCNS. Snapshot IDs, cell identities, units, coordinate frames and biological specimens must remain explicit. String IDs avoid JavaScript precision loss for large FlyWire root IDs.

## What the scan does and does not provide

A structural connectome is not a recording of a working brain. Synapse counts alone do not supply all synaptic strengths, delays, receptor dynamics, cell parameters, plasticity or body physics. A simulation must make and document additional assumptions. Neuron morphology is useful for spatial visualization but does not itself prescribe neural dynamics.

Shiu and colleagues provide a leaky integrate-and-fire model with activation and silencing experiments. Their code exports spike times and rates indexed by FlyWire IDs. That offers an external source of activity for an adapter. It is not a language model, and its identifiers must not be attached directly to MaleCNS body IDs.

- [Model code and usage](https://github.com/philshiu/Drosophila_brain_model)
- [Published paper: A Drosophila computational brain model reveals sensorimotor processing](https://doi.org/10.1038/s41586-024-07763-9)
- [Connectome-constrained networks predict neural activity across the fly visual system](https://www.nature.com/articles/s41586-024-07939-3)

## Proposed language experiment (engineering hypothesis)

A possible experiment is a token encoder feeding a recurrent circuit whose connectivity mask comes from a selected fly graph, followed by a trainable token readout. Options include rate-based units for easier differentiation or spiking units with a documented surrogate-gradient method. Biological neurons and artificial units are different abstractions; record the mapping explicitly.

Start with a controlled character prediction or sequence memory task. Compare against a parameter-matched small recurrent network, a shuffled-degree-preserving connectome, and a simple non-neural baseline. Separate the influence of fixed topology from learned weights, encoder capacity and decoder capacity. Record held-out loss, accuracy where appropriate, sample quality, seed variability, memory and runtime. A graph-derived model can learn a task without establishing that the biological fly circuit could perform it. No evidence reviewed here establishes useful language ability from this connectome.

## Visualization architecture

- Structure: versioned dataset, source, coordinate space, units, stable string neuron IDs, skeleton segments, directed edges.
- Activity: separate provenance (`synthetic`, `simulation`, `recording`), unit, time axis in seconds and values keyed by the same neuron IDs. Values are held from the preceding sample, with no interpolation or invented data for missing neurons.
- Body: an independent procedural Three.js model. Both views share neuron selection and activity, but anatomical registration is illustrative. The motor lab now integrates damped joint servos with a procedural gait and planar locomotion. Manual commands or a deliberately synthetic neural readout produce recorded poses. External pose recordings can be converted through the CSV adapter. No physical contacts, muscles or flight forces are simulated locally.

Next extensions: dataset-specific preprocessing to spatial tiles or ROI subsets; per-neuron lazy skeleton fetches; binary typed arrays; worker parsing; sparse live activity frames over a documented transport; registered body/brain transforms; GLTF body rigs; and a reproducible model runtime. None of those are claimed as implemented by this prototype.

## Motor integration update

The implemented reduced-rig contract and agent workflow are in [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md). [EXTERNAL_SIMULATORS.md](EXTERNAL_SIMULATORS.md) links primary NeuroMechFly/FlyBody sources and defines the external pose-recording path. Real physics backend execution and biologically validated motor decoding remain unimplemented.

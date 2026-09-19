# Next experiments: a language model and the fly's visual input

User direction, 19 September 2026: use real fly wiring; train an actual language model; show neural activity linked to behavior; later show the image the fly receives. Do not substitute a pretrained assistant answering beside a decorative fly.

## Current working foundation

- Real MaleCNS catalog: 166,700 classified cells, all with source positions.
- Real weighted graph: 25,582,938 connections between those classified cells, 124,177,617 summed synapses.
- Selected-cell source skeleton and connection loading, with explicit display limits.
- Coupled synthetic walking demonstration: `src/controllers/walking-circuit/controller.ts`; `src/motor/experiment.ts` records the emitted activity on the motor clock.
- MaleCNS now has an experimental whole-connectome rate network coupled to actual NeuroMechFly/MuJoCo contact physics. See PHYSICAL_FLY.md for tested neural silencing and feedback controls. Its population motor decoder and sensory encoding are engineering assumptions, not validated biology. This model is untrained.

## Stage 1: actually train a text predictor

Default proposal pending user preference: a small local character/byte-level proof of concept on this Mac. A larger GPU job is an alternative, not authorized spending. No language training or trained checkpoint exists yet.

The model must perform next-token prediction through a recurrent circuit constrained by a documented subgraph of the real MaleCNS connections. Start with a tractable selected subgraph, not a dense 166,700-by-166,700 weight matrix. Define:

1. An exact cell-ID list and source-version manifest; preserve mapping between tensor indices and real body IDs.
2. A token encoder that injects signals into specified circuit units.
3. Sparse or indexed recurrent propagation using the graph's directed mask, with explicitly documented learned weights, signs, nonlinearities and time dynamics. Synapse count is not automatically a physiological weight.
4. A readout from circuit state to vocabulary logits. Avoid a powerful independent language model that can bypass the circuit.
5. A licensed/public-domain text dataset with immutable train/validation/test splits.
6. Training code, seed, checkpoint, tokenizer, configuration and a reproducible generation command.
7. Exported activity values indexed by source neuron ID and token/step timestamp so the viewer shows the actual trained circuit state.

Success is measured held-out next-token loss, plus qualitative generation and baseline comparisons. Compare against shuffled connectivity and a parameter-matched ordinary small recurrent model. Test that circuit ablation changes predictions. Report failures. A trained small predictor is not a general assistant-level LLM or evidence of biological language capacity.

Motor behavior may later use a separate documented decoder of the trained state. Do not invent a natural language-to-leg mapping or imply language training validates the walking system.

## Stage 2: see what the fly sees

Add a body-attached camera and a visible input panel on the same run clock. Show the input frame actually delivered to the model, including its resolution, channels, timestamp and transformations. A scene camera image is a synthetic visual observation, not a biologically accurate compound eye.

Useful progressive views:

- Raw left/right camera frames from the simulated environment.
- The exact resized/normalized image tensor supplied to the model.
- An optional explicitly approximate facet/ommatidial sampling view.
- Neural responses from the visual model, joined to neuron IDs through a declared mapping.

For a biological retina/optic-lobe model, use a pinned external model and its documented visual sampling geometry. Do not infer pixel-to-neuron identities from spatial proximity alone. When an external simulator owns rendering, record its actual frames and poses rather than recreating potentially different images in the viewer.

## Instructions to the next agent

Read AGENTS.md, docs/AGENT_INTEGRATION.md and docs/LARGE_DATASETS.md. Complete Stage 1 as an actual training-and-evaluation task, with a checkpoint and viewer-compatible output; do not present only a plan or precomputed animation. Verify the available local compute/runtime before choosing the implementation. Stage 2 should display the real input to that pipeline, not decorative unrelated images. Both stages are unimplemented at the time of this note.

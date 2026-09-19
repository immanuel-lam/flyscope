# Full MaleCNS loader

## What is loaded

The full catalog is derived from `body-annotations-male-cns-v1.0-minconf-0.5.feather`. The inclusion rule is all rows with a non-null `superclass`: 166,700 classified neurons. This includes incompletely traced cells; the status is shown in the inspector. The original file has 211,577 rows, including unclassified segments and glia. We do not call every raw segment a neuron.

All 166,700 cells have real source positions. Where supplied, the viewer uses `somaLocation`, then `tosomaLocation`. Otherwise the preparation script fetches the first vertex of the source skeleton with a 20-byte HTTP range request. These positions are explicitly labeled `skeleton-anchor`, not soma. The source precomputed skeleton uses nanometres; the adapter divides by 8 to match the catalog's 8 nm voxel units. No random or invented positions are inserted.

The original weighted connection table is downloaded locally in full. Browser assets retain every edge with both endpoints in the classified catalog: 25,582,938 directed records, with weights summing to 124,177,617 synapses. The original table contains 151,856,684 segment-to-segment records, many involving unclassified fragments. A weighted edge record is not one synapse. Detailed synaptic point coordinates and EM images are separate source products and are not downloaded.

## Load levels and budgets

- Whole-CNS overview: all source positions in one Three.js point draw in the brain view at rest. During interaction it samples 20,000 points, then restores all points; the small fly overlay stays capped at 20,000 points. This is not every neurite rendered at once.
- Search: the full catalog, with 80 visible results and exact string body IDs.
- Selected morphology: source precomputed binary skeleton, parsed in a Web Worker. All edges for normal-sized cells; uniformly sampled original segments if the cell exceeds 150,000 edges. The inspector reports shown/total segment counts. This sampling is a display limit, not a topology-preserving simplification.
- Selected connectivity: one of 128 binary shards, parsed in the worker; all outgoing matches counted, the strongest 2,000 eligible connections sent for display. All retained graph records remain on disk.
- CPU cache: up to 64 MB skeleton bytes in the worker, plus at most four graph shards. Largest single source skeleton: 32 MB.
- Local server cache: 512 MB on disk, four concurrent source downloads, up to 32 queued requests. Source URL is fixed; only numeric IDs are accepted.
- The fly anatomy and brain placement remain illustrative. There is no recorded or inferred electrical activity in the MaleCNS catalog.

The whole catalog is fetched and parsed in a worker; no giant JSON file is parsed on the React main thread. GPU buffers are reused across selection changes. Only selected detail geometry is replaced. Core geometry is disposed when datasets or views change. Source graphs are never expanded into 25 million JavaScript edge objects.

## Prepare from source

```sh
python3 -m venv .venv
.venv/bin/pip install pyarrow numpy aiohttp
mkdir -p data/malecns
curl -fL --retry 3 -C - https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather -o data/malecns/annotations.feather
curl -fL --retry 3 -C - https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather -o data/malecns/connections.feather
npm run prepare:malecns
npm run dev -- --port 5174
```

Anchor downloads resume from `data/malecns/anchors.jsonl`. Successful anchors and source 404s are cached; transient failures are retried on the next run. Preparation initially writes an overview using available positions, then replaces it atomically when anchor fetching completes. Regenerating graph shards must finish before treating the dataset as ready. Never mix catalog rows from one version with graph indices from another.

Raw data, prepared public assets, virtual environment and caches are excluded from Git. A production build copies the prepared public assets into `dist`; expect roughly 310 MB of graph assets plus the catalog. Use `npm run preview` for the local preview gateway. A static-only host cannot service the skeleton cache API: deployment needs an equivalent fixed-source endpoint or pre-fetched selected skeletons.

The full dataset is selected through **Dataset → MaleCNS · full catalog**. The original two-neuron example is explicitly named **2-neuron sample**. Bulk JSON export is disabled because that format would omit the sharded graph. Motor exports from the full catalog contain the selected neuron and motor track, not an implicit whole-brain copy.

## Extend this loader

See `src/large/loader.worker.ts`, `server/malecns-plugin.ts` and `scripts/prepare-malecns.py`. Do not reuse the 20,000-neuron local JSON import route for the full connectome. Keep source identity, position kind, graph inclusion and display budgets visible. A new connectome needs its own versioned preparation adapter and matching catalog/graph ordering.

Performance evidence and limitations are in `PERFORMANCE.md`. Benchmark output lives in `docs/performance/`.

## Source and attribution

[Canonical MaleCNS downloads](https://male-cns.janelia.org/download/). Creators: FlyEM (HHMI Janelia), University of Cambridge, MRC LMB and Google Research. Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The browser assets are derived by filtering classified cells, selecting source positions and indexing connections. Geometry display is transformed for viewing; source coordinates are preserved in the catalog and detail parser.

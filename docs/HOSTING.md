# Vercel deployment

The raw dataset and prepared `public/malecns` directory are excluded from Git. A plain Vite deployment therefore returns 404 for `outgoing-counts.bin`, catalog and graph shards.

`vercel.json` now runs `scripts/fetch-hosted-data.mjs` before the build. It downloads the versioned public GitHub release archive pinned in `hosting-data.json`, verifies SHA256, and extracts all 128 shards, catalog, graph metadata and counts. Failed download or checksum verification fails the build; it does not substitute a synthetic brain. The archive is derived CC BY 4.0 source data; attribution and transformations are in the release notes and LARGE_DATASETS.md.

A fixed-source external rewrite routes numeric skeleton IDs to the canonical source bucket. It is not an arbitrary URL proxy. Browser detail budgets still apply.

`api/chat/status.py` and `api/chat/generate.py` run the same committed NumPy checkpoint as local inference. Root requirements contain only inference dependencies, not MLX or MuJoCo. The function validates input and caps output at 40 tokens; NDJSON sends cell states and text as computed. It does not persist chat history. Exclusion rules keep the anatomical assets and development environments out of function bundles.

Physics remains local. The hosted status endpoint declares it unavailable and the UI disables the run button. A separately hosted physics service is needed for remote MuJoCo experiments; the browser must not imply that a static host can execute the local backend.

After deployment, verify `/malecns/outgoing-counts.bin` (666,800 bytes), `/malecns/catalog.json`, `/malecns/graph.json`, a connection shard, a selected source skeleton, and `/api/chat/status`. Send a streamed chat request and verify a `state` event precedes `result`. A local handler test does not by itself verify Vercel deployment.

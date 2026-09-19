# Full MaleCNS performance

Measured 19 September 2026 on this Mac, using headless Chromium with SwiftShader software rendering, 1440 × 1000, and the local Vite development server. These are local development measurements, not a claim about physical-GPU performance, remote hosting, every device, or all neurons' complete skeletons rendered together.

## Results

| Measurement | Result |
| --- | ---: |
| Classified cells in catalog | 166,700 |
| Cells with a real source position | 166,700 |
| Directed graph records between classified cells | 25,582,938 |
| Sum of retained connection weights | 124,177,617 synapses |
| Catalog + count-array transfer | 12.5 MiB |
| Graph assets on disk | 306,995,256 bytes in 128 chunks |
| Select full dataset → catalog visible | 1,132 ms |
| Catalog fetch/parse/build in worker | 94 ms |
| Search update in the full catalog | 48 ms |
| Orbit with 20,000-point interaction LOD, median / p95 RAF interval | 16.7 / 16.7 ms |
| Idle median / p95 RAF interval | 16.7 / 16.8 ms |
| Approximate browser-reported used JS heap | 188 MB |
| Brain view at rest | 166,700 points; four draw calls with selected skeleton |
| Small body overlay | 20,000 sampled overview points; selected skeleton retained |
| Selected reference skeleton 12781 | 15,942 segments, 311 KiB source binary |

The full catalog remains loaded during interaction. Point sampling affects rendering only. After camera motion stops, the brain view restores all points. The body overlay remains capped at 20,000 points. Selected morphology and connections have separate explicit budgets.

## Before/after

The first implementation redrew both complete point clouds continuously, even with no input or activity. Software-rendered RAF intervals were around 166.6 ms (roughly six frames per second). This was not acceptable for interaction.

The revised renderer redraws only on camera, selection, detail, display-layer or motor/activity changes. It samples an evenly distributed set of 20,000 points during orbit, then restores the whole brain point set. It caps the small body overview at 20,000 points. In the recorded scripted orbit, the scheduling interval was about 16.7 ms. Idle scheduling is also 16.7 ms but idle rendering is stopped: do not interpret that as 60 new full-brain renders per second.

The initial benchmark's `loadMs` included an intentional two-second settling wait; do not compare that value directly with the corrected final load timing. The frame-timing baseline is retained in `docs/performance/baseline.json`; the current measured output is `docs/performance/latest.json`.

## Remaining costs and limits

- Startup included a roughly 1.97-second main-thread long task during initial software-renderer/shader setup. Other observed tasks reached 461 ms around data/view setup. Background parsing does not eliminate structured cloning, React setup or GPU-buffer construction on the main thread.
- JS heap is Chromium's approximate report; it excludes some native/GPU allocations and is not a peak resident-memory measurement. GC timing affects the result.
- Real hardware GPU behavior has not been benchmarked. Dense selected morphologies and many visible connection lines can still cost more than the reference test.
- Source skeleton latency depends on network and cache. The benchmark uses a cached local skeleton. The first fetch goes through the fixed-source gateway and can be slower.
- Every neuron is searchable and represented in the overview. Every original neurite and every graph edge is not drawn simultaneously. Those requests require different spatial tiling and aggregation strategies.
- The fixed-step motor model is an illustrative simulator. These graphics measurements say nothing about whole-connectome neural simulation or language-model training throughput.

## Reproduce

Prepare the assets using `docs/LARGE_DATASETS.md`, run the dev server on port 5174, then:

```sh
npm run benchmark:large
npm run test:browser
```

The benchmark selects the full catalog, loads neuron 12781, samples frame scheduling, captures draw/geometry counters, measures a scripted orbit and a search, and writes JSON. `tests/large.spec.ts` also checks counts, real detail loading, connectivity, stopped idle rendering and dataset-switch isolation.

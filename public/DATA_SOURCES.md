# Bundled morphology attribution

Dataset: MaleCNS v1.0.
Creators: FlyEM (HHMI Janelia), University of Cambridge Department of Zoology, MRC Laboratory of Molecular Biology, and Google Research.
Project: https://male-cns.janelia.org/
Download documentation: https://male-cns.janelia.org/download/
License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
Retrieved: 2026-09-19.

- 12781.swc — DNge104_R: https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/12781.swc
- 556329.swc — DNge104_L: https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/556329.swc

The source files are unmodified. SWC coordinates are in Male CNS EM space, in 8 nm units. Rendering recentres and uniformly scales the combined skeletons to fit the viewport. Brain-to-body placement is illustrative, not anatomically registered.

The synthetic dataset and procedural fly are original demonstration fixtures, not data from these projects. No recorded or biologically validated activity is bundled.

## Full catalog and graph additions

The full local source tables are stored under data/malecns (not committed). Derived browser assets are under public/malecns. See docs/LARGE_DATASETS.md for the exact classification filter, position derivation and connection inclusion rule. All 166,700 classified cells have source coordinates. Raw skeleton geometry is fetched through a fixed-source local gateway. These derived data retain the MaleCNS CC BY 4.0 attribution above. No measured electrical activity is supplied by these files.

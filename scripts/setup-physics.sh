#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
UV="$(command -v uv || true)"
if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi
if [ -z "$UV" ]; then echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ first.' >&2; exit 1; fi
"$UV" venv --python 3.12 .venv-physics
"$UV" pip install --python .venv-physics/bin/python -r scripts/physics/requirements.txt
mkdir -p data/malecns
if [ ! -f data/malecns/neurotransmitters.feather ]; then
 curl --fail --location --output data/malecns/neurotransmitters.feather.tmp https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather
 mv data/malecns/neurotransmitters.feather.tmp data/malecns/neurotransmitters.feather
fi
echo 'Physics installed. Run npm run prepare:malecns if the full dataset is not yet prepared.'

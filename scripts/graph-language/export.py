"""Export an isolated candidate and verify CPU logits and all cell features."""
import argparse
import hashlib
import json
import shutil
import os
# Verify the mathematical float32 model, without backend-specific reduced precision.
os.environ['MLX_ENABLE_TF32'] = '0'
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from model import GraphLanguageModel, ROOT
from runtime import GraphRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='run-1')
    parser.add_argument('--checkpoint', choices=['base', 'dialogue'], default='base')
    args = parser.parse_args()
    run = ROOT / 'data/graph-language' / args.run
    out = run / 'portable'
    out.mkdir(exist_ok=True)
    training_file = 'training.json' if args.checkpoint == 'base' else 'dialogue-training.json'
    weights_file = 'weights.safetensors' if args.checkpoint == 'base' else 'dialogue.safetensors'
    training = json.loads((run / training_file).read_text())
    config = training['arguments']
    # Snapshot first: a concurrent trainer can replace its best checkpoint.
    shutil.copyfile(run / weights_file, out / 'checkpoint.safetensors')
    wiring = dict(np.load(run / 'wiring.npz'))
    source_graph = ROOT / 'data/language/graph.npz'
    source_manifest = ROOT / 'data/language/manifest.json'
    expected_source = np.load(source_graph)['counts'] > 0
    np.testing.assert_array_equal(wiring['sourceMask'], expected_source)
    off_diagonal = ~np.eye(512, dtype=bool)
    if np.any(wiring['mask'] & off_diagonal & ~expected_source):
        raise ValueError('Export contains an inter-cell edge absent from the source graph')
    model = GraphLanguageModel(width=config['width'], layers=config['layers'],
                              wiring_data=tuple(wiring[k] for k in ['inputs','outputs','slots','mask','sourceMask']))
    model.load_weights(str(out / 'checkpoint.safetensors'))
    mx.eval(model.parameters())
    weights = {key: np.array(value) for key, value in tree_flatten(model.parameters())}
    weights['attention_mask'] = wiring['mask']
    np.savez_compressed(out / 'runtime.npz', **weights)
    source = json.loads(source_manifest.read_text())
    tokenizer = ROOT / 'data/language-memory/tokenizer.json'
    manifest = dict(modelId='malecns-graph-attention-candidate', neurons=source['neurons'],
                    datasetId=source['datasetId'], datasetVersion=source['datasetVersion'],
                    inputIndices=wiring['inputs'].tolist(), outputIndices=wiring['outputs'].tolist(),
                    slots=wiring['slots'].tolist(), heads=8, layers=config['layers'], width=config['width'],
                    parameters=training['parameters'], training=training, checkpoint=args.checkpoint,
                    checkpointSha256=hashlib.sha256((out / 'checkpoint.safetensors').read_bytes()).hexdigest(),
                    weightsSha256=hashlib.sha256((out / 'runtime.npz').read_bytes()).hexdigest(),
                    tokenizerSha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
                    sourceGraphSha256=hashlib.sha256(source_graph.read_bytes()).hexdigest(),
                    sourceCellManifestSha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
                    wiringSha256=hashlib.sha256((run / 'wiring.npz').read_bytes()).hexdigest(),
                    sourceIntercellEdges=int(np.count_nonzero(wiring['mask'] & off_diagonal)),
                    source=json.loads((ROOT / 'data/graph-language/source.json').read_text()),
                    parityPrecision='Full float32, MLX_ENABLE_TF32=0; training used MLX default precision',
                    assumptions=training['assumptions'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copyfile(tokenizer, out / 'tokenizer.json')
    cpu = GraphRuntime(out)
    errors = []
    for ablated in [False, True]:
        tokens = np.random.default_rng(157).integers(5, 2048, 128, dtype=np.int32)
        tokens[:64] = 0
        expected, expected_states = model(mx.array(tokens[None]), ablated, True)
        mx.eval(expected, expected_states)
        actual, states, _ = cpu.forward(tokens, ablated)
        logit_error = float(np.max(np.abs(np.array(expected)[0] - actual)))
        state_error = float(np.max(np.abs(np.array(expected_states)[0] - states)))
        np.testing.assert_allclose(actual, np.array(expected)[0], atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(states, np.array(expected_states)[0], atol=2e-4, rtol=2e-4)
        errors.append(dict(ablated=ablated, maxLogitError=logit_error, maxStateError=state_error))
    (out / 'parity.json').write_text(json.dumps(errors, indent=2))
    print(json.dumps(dict(directory=str(out), parity=errors)))


if __name__ == '__main__':
    main()

"""Export a conventional attention diagnostic, never a FlyGPT checkpoint."""
import os
os.environ['MLX_ENABLE_TF32'] = '0'
import argparse
import hashlib
import json
import shutil
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from model import AttentionControl, ROOT
from runtime import GraphRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='control-1')
    parser.add_argument('--checkpoint', choices=['base', 'dialogue'], default='base')
    args = parser.parse_args()
    run = ROOT / 'data/graph-language' / args.run
    training = json.loads((run / ('training.json' if args.checkpoint == 'base' else 'dialogue-training.json')).read_text())
    if training['architecture'] != 'control':
        raise ValueError('This exporter is only for the non-biological control')
    config = training['arguments']
    out = run / ('portable' if args.checkpoint == 'base' else 'portable-dialogue')
    out.mkdir(exist_ok=True)
    shutil.copyfile(run / ('weights.safetensors' if args.checkpoint == 'base' else 'dialogue.safetensors'), out / 'checkpoint.safetensors')
    model = AttentionControl(width=config['width'], layers=config['layers'])
    model.load_weights(str(out / 'checkpoint.safetensors'))
    mx.eval(model.parameters())
    weights = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    weights['attention_mask'] = np.tril(np.ones((128, 128), bool))
    np.savez_compressed(out / 'runtime.npz', **weights)
    manifest = {'modelId': 'ordinary-attention-diagnostic', 'architecture': 'ordinary-attention-control',
                'inputIndices': list(range(128)), 'outputIndices': list(range(128)),
                'heads': 8, 'layers': config['layers'], 'width': config['width'],
                'parameters': training['parameters'], 'training': training,
                'checkpointSha256': hashlib.sha256((out / 'checkpoint.safetensors').read_bytes()).hexdigest(),
                'limitations': 'No fly source cells or pathways. Diagnostic only. Never deployed or used to supply FlyGPT answers.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copyfile(ROOT / 'data/language-memory/tokenizer.json', out / 'tokenizer.json')
    cpu = GraphRuntime(out)
    tokens = np.random.default_rng(157).integers(5, 2048, 128, dtype=np.int32)
    tokens[:64] = 0
    expected, states = model(mx.array(tokens[None]), return_states=True)
    mx.eval(expected, states)
    actual, cpu_states, _ = cpu.forward(tokens)
    np.testing.assert_allclose(actual, np.array(expected)[0], atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(cpu_states, np.array(states)[0], atol=2e-4, rtol=2e-4)
    report = {'maxLogitError': float(np.max(np.abs(actual - np.array(expected)[0]))),
              'maxStateError': float(np.max(np.abs(cpu_states - np.array(states)[0])))}
    (out / 'parity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()

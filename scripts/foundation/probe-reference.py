"""Evaluate a training reference offline; never a source-wired FlyGPT reply path."""
import argparse
import hashlib
import json
import time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_flatten
from runtime import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    directory = ROOT / 'data/foundation' / args.run
    model, tokenizer = load(str(directory))
    cases = json.loads((ROOT / 'docs/performance/foundation-early-evaluation.json').read_text())['replies']
    replies = []
    for case in cases:
        text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
        text += f'<|im_start|>user\n{case["prompt"]}<|im_end|>\n<|im_start|>assistant\n'
        start = time.perf_counter()
        answer = generate(model, tokenizer, prompt=text, max_tokens=48, sampler=make_sampler(temp=0), verbose=False)
        row = {'prompt': case['prompt'], 'text': answer, 'seconds': time.perf_counter() - start}
        replies.append(row)
        print(json.dumps(row), flush=True)
    with (directory / 'model.safetensors').open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    report = {'run': args.run, 'mode': 'offline-training-reference-only', 'engine': 'MLX',
              'parameters': sum(v.size for _, v in tree_flatten(model.parameters())),
              'weightSha256': digest, 'maxNewTokens': 48, 'temperature': 0,
              'peakMlxMemoryBytes': mx.get_peak_memory(), 'replies': replies,
              'limitation': 'Ordinary pretrained reference, without source wiring. This is not the FlyGPT inference model and its replies do not satisfy the source-wired goal. No reference model is connected to the chat UI.'}
    (directory / 'reference-probes.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

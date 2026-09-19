"""Retain ordinary-reference and source-wired replies without selecting successes."""
import argparse
import json
from pathlib import Path
from runtime import FoundationRuntime, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, default=16)
    args = parser.parse_args()
    model = FoundationRuntime()
    replies = []
    for prompt in ['hi', 'who are you?', 'what is a cat?', 'what is a tree?',
                   'what is two plus three?', 'I feel sad today.']:
        row = {'prompt': prompt}
        for graph, label in [(False, 'ordinaryReference'), (True, 'sourceGraph')]:
            row[label] = model.generate(prompt, max_tokens=args.tokens, graph=graph)
        replies.append(row)
        print(json.dumps(row), flush=True)
    report = {'source': json.loads((Path(__file__).parent / 'source.json').read_text()),
              'maxTokens': args.tokens, 'replies': replies,
              'limitations': 'Frozen pretrained weights; graph adaptation has not been trained. Ordinary reference is diagnostic only and is not wired into FlyGPT. Short development probes are not a quality benchmark. Timings are CPU inference with full graph recomputation and exclude weight loading.'}
    (ROOT / 'data/foundation/initial-probes.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

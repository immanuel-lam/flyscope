"""Keep raw CPU replies and timing for a candidate, without selecting good outputs."""
import argparse
import json
from pathlib import Path
from runtime import GraphRuntime

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='run-1')
    args = parser.parse_args()
    directory = ROOT / 'data/graph-language' / args.run / 'portable'
    model = GraphRuntime(directory)
    prompts = ['hi', 'who are you?', 'what is a cat?', 'what is a tree?',
               'what is two plus three?', 'I feel sad today.']
    replies = []
    for prompt in prompts:
        result = {'prompt': prompt, **model.generate(prompt, max_tokens=48)}
        replies.append(result)
        print(json.dumps(result), flush=True)
    report = {'checkpointSha256': model.config['checkpointSha256'],
              'trainingHistoryAtExport': model.config['training']['history'][-1],
              'parameters': model.config['parameters'], 'replies': replies,
              'limitations': 'Qualitative development probes, not held-out quality measurement. Timing includes full CPU graph recomputation per generated token, while MLX training may be running.'}
    (directory / 'probes.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

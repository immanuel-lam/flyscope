"""Export one actual CPU computation for UI-independent source-cell inspection."""
import argparse
import hashlib
import json
import time
from runtime import FoundationRuntime, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--prompt', default='What is a cat?')
    args = parser.parse_args()
    directory = ROOT / 'data/foundation' / args.run
    model = FoundationRuntime(directory)
    tokens = model.tokenizer.encode(args.prompt).ids
    start = time.perf_counter()
    _, _, details = model.next(tokens, inspect=True)
    elapsed = time.perf_counter() - start
    source_file = directory / ('weights.json' if (directory / 'weights.json').exists() else 'model.safetensors')
    result = {'prompt': args.prompt, 'tokenIds': tokens, 'inputFormat': 'Raw text probe, without chat template',
              'run': args.run, 'artifactSha256': hashlib.sha256(source_file.read_bytes()).hexdigest(),
              'engine': 'NumPy CPU', 'computeSeconds': elapsed, 'inspection': details['inspection']}
    encoded = json.dumps(result, allow_nan=False)
    (directory / 'inspection.json').write_text(encoded)
    print(json.dumps({'computeSeconds': elapsed, 'jsonBytes': len(encoded.encode()),
                      'cells': len(details['inspection']['activityIds']),
                      'blocks': len(details['inspection']['blocks']),
                      'output': str(directory / 'inspection.json')}))


if __name__ == '__main__':
    main()

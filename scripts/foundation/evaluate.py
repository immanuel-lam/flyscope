"""Record every CPU development reply and source-path cuts for an adapted run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from runtime import FoundationRuntime, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--tokens', type=int, default=48)
    args = parser.parse_args()
    directory = ROOT / 'data/foundation' / args.run
    model = FoundationRuntime(directory)
    prompts = ['hi', 'who are you?', 'what is a cat?', 'what is a tree?',
               'what is two plus three?', 'I feel sad today.',
               'Why does ice melt in a warm room?',
               'Name one thing a bicycle and a car have in common.',
               'My name is Mira. What is my name?',
               'Is a stone alive? Explain in one short sentence.']
    replies = []
    for prompt in prompts:
        row = {'prompt': prompt, **model.generate(prompt, max_tokens=args.tokens)}
        replies.append(row)
        print(json.dumps(row), flush=True)
    prefixes = [model.tokenizer.encode(s).ids for s in ['What is a cat?', 'What is a tree?']]
    full = [model.next(p)[0] for p in prefixes]
    cut = [model.next(p, ablated=True)[0] for p in prefixes]
    attention_cut = [model.next(p, attention_cut=True)[0] for p in prefixes]
    report = {'run': args.run, 'maxTokens': args.tokens, 'replies': replies,
              'weightSha256': hashlib.sha256((directory / 'model.safetensors').read_bytes()).hexdigest(),
              'training': json.loads((directory / 'training.json').read_text()),
              'ablation': {'fullPromptMaxLogitDifference': float(np.max(np.abs(full[0] - full[1]))),
                           'cutPromptMaxLogitDifference': float(np.max(np.abs(cut[0] - cut[1]))),
                           'cutMaxAbsoluteLogit': float(np.max(np.abs(cut))),
                           'attentionCutPromptMaxLogitDifference': float(np.max(np.abs(attention_cut[0] - attention_cut[1])))},
              'limitations': 'Development probes, not an unseen benchmark. All replies are raw greedy CPU generation through source wiring, without external answer generation. Training validation is held out from adaptation only. Reply usefulness requires inspection; prompt dependence alone is insufficient.'}
    (directory / 'evaluation.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

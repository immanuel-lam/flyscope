"""Evaluate the actual CPU generation head; keep every selected source reply."""
import argparse
import json
from pathlib import Path
import numpy as np
from runtime import GraphRuntime

ROOT = Path(__file__).resolve().parents[2]


def examples(tokens, spans, count, seed=173):
    rng = np.random.default_rng(seed)
    conversations = np.unique(spans[:, 0])
    selected = rng.choice(conversations, min(count, len(conversations)), replace=False)
    result = []
    for conversation in selected:
        index = int(rng.choice(np.flatnonzero(spans[:, 0] == conversation)))
        start, reply, end = map(int, spans[index])
        result.append({'spanIndex': index, 'prefix': tokens[max(start, reply - 128):reply].tolist(),
                       'target': tokens[reply:end].tolist()})
    return result


def token_loss(logits, token):
    values = logits.astype(np.float64)
    peak = values.max()
    return float(peak + np.log(np.exp(values - peak).sum()) - values[token])


def response_loss(model, prefix, target):
    losses = []
    context = list(prefix)
    for token in target:
        logits, _ = model.next(context)
        losses.append(token_loss(logits, token))
        context.append(token)
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='run-2')
    parser.add_argument('--examples', type=int, default=64)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--checkpoint', choices=['base', 'dialogue'], default='base')
    args = parser.parse_args()
    if args.examples < 2 or args.tokens < 1:
        raise ValueError('Use at least two examples and one token')
    directory = ROOT / 'data/graph-language' / args.run / ('portable' if args.checkpoint == 'base' else 'portable-dialogue')
    model = GraphRuntime(directory)
    tokens = np.load(ROOT / 'data/graph-language' / f'{args.split}.npy', mmap_mode='r')
    spans = np.load(ROOT / 'data/graph-language' / f'{args.split}-reply-spans.npy')
    rows = examples(tokens, spans, args.examples)
    # Source-cut logits are prompt-independent; this is separately tested.
    ablated_logits, _ = model.next(rows[0]['prefix'], ablated=True)
    outputs = []
    for index, row in enumerate(rows):
        target = row['target'][:args.tokens]
        correct = response_loss(model, row['prefix'], target)
        mismatched = response_loss(model, rows[(index + 1) % len(rows)]['prefix'], target)
        cut = [token_loss(ablated_logits, token) for token in target]
        generation = model.generate_tokens(row['prefix'], max_tokens=args.tokens)
        generated = generation['tokenIds']
        grams = [tuple(generated[i:i + 4]) for i in range(max(0, len(generated) - 3))]
        output = {'spanIndex': row['spanIndex'],
                  'context': model.tokenizer.decode(row['prefix'], skip_special_tokens=False),
                  'sourceReply': model.tokenizer.decode(row['target']),
                  'sourceReplyTokens': len(row['target']), 'evaluatedReplyTokens': len(target),
                  'correctPromptLoss': float(np.mean(correct)),
                  'mismatchedPromptLoss': float(np.mean(mismatched)),
                  'ablatedLoss': float(np.mean(cut)),
                  'firstEightCorrectLoss': float(np.mean(correct[:8])),
                  'firstEightMismatchedLoss': float(np.mean(mismatched[:8])),
                  'repeatedFourGramFraction': 1 - len(set(grams)) / len(grams) if grams else 0,
                  'generation': generation}
        outputs.append(output)
        print(json.dumps({'example': index + 1, 'reply': generation['text'],
                          'correctLoss': output['correctPromptLoss']}), flush=True)
    metrics = {key: float(np.mean([row[key] for row in outputs])) for key in
               ['correctPromptLoss', 'mismatchedPromptLoss', 'ablatedLoss',
                'firstEightCorrectLoss', 'firstEightMismatchedLoss', 'repeatedFourGramFraction']}
    report = {'checkpointSha256': model.config['checkpointSha256'],
              'parameters': model.config['parameters'], 'arguments': vars(args), 'seed': 173,
              'examples': len(rows), 'metrics': metrics, 'outputs': outputs,
              'limitations': 'Untrained source conversations, selected without model scores. Reused development split across architecture experiments, not an independent final benchmark. Response losses use at most the declared token limit and the actual final CPU readout. Full source replies retained for inspection. Generation receives context only, never source answer tokens. Timings may overlap MLX training.'}
    (directory / f'{args.split}-evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

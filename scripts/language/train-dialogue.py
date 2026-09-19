"""Train assistant next-token loss on external dialogues; no authored answers.

Deduplicate identical (prompt, reply) pairs in training only. Select by held-out
conversation loss, never by the interactive examples used to inspect the model.
"""
import argparse, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tokenizers import Tokenizer
from model import CircuitLM, DATA


def examples(split, tokenizer, length=192):
    rows = json.loads((DATA / f'{split}-conversations.json').read_text())
    pairs = []
    seen = set()
    for row in rows:
        messages = row['messages']
        for left, right in zip(messages, messages[1:]):
            if left['role'] != 'user' or right['role'] != 'assistant':
                continue
            key = (left['content'], right['content'])
            if split == 'train' and key in seen:
                continue
            seen.add(key)
            prefix = tokenizer.encode(f'<user> {key[0]} <end>\n<assistant>').ids[-128:]
            reply = tokenizer.encode(' ' + key[1] + ' <end>').ids
            ids = (prefix + reply)[:length + 1]
            x = np.zeros(length, dtype=np.int32)
            y = x.copy()
            mask = np.zeros(length, dtype=np.float32)
            x[:len(ids)-1], y[:len(ids)-1] = ids[:-1], ids[1:]
            mask[len(prefix)-1:len(ids)-1] = 1
            pairs.append((x, y, mask))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=4000)
    args = parser.parse_args()
    mx.random.seed(29)
    rng = np.random.default_rng(29)
    model = CircuitLM()
    model.load_weights(str(DATA / 'real/weights.safetensors'))
    tokenizer = Tokenizer.from_file(str(DATA / 'tokenizer.json'))
    train = examples('train', tokenizer)
    validation = examples('validation', tokenizer)
    # Fixed validation sample selected before training, independent of test data.
    indices = np.random.default_rng(31).choice(len(validation), 128, replace=False)
    def batch(rows, indices):
        return [mx.array(np.stack([rows[int(i)][j] for i in indices])) for j in range(3)]
    def loss(model, x, y, mask):
        values = nn.losses.cross_entropy(model(x), y, reduction='none')
        return mx.sum(values * mask) / mx.maximum(mask.sum(), 1)
    def evaluate():
        return float(mx.mean(mx.stack([loss(model, *batch(validation, chunk)) for chunk in np.array_split(indices, 8)])))
    grad = nn.value_and_grad(model, loss)
    optimizer = optim.AdamW(learning_rate=.0005, weight_decay=.001)
    best = evaluate()
    initial = best
    output = DATA / 'real/dialogue.safetensors'
    model.save_weights(str(output))
    history = []
    started = time.perf_counter()
    print(json.dumps({'initial': best, 'trainingPairs': len(train)}), flush=True)
    for step in range(1, args.steps + 1):
        value, gradients = grad(model, *batch(train, rng.integers(0, len(train), 16)))
        gradients, _ = optim.clip_grad_norm(gradients, 1)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, value)
        if step % 200 == 0 or step == args.steps:
            val = evaluate()
            row = {'step': step, 'trainingLoss': float(value), 'validationLoss': val, 'seconds': time.perf_counter()-started}
            history.append(row)
            print(json.dumps(row), flush=True)
            if val < best:
                best = val
                model.save_weights(str(output))
            (DATA / 'real/dialogue-training.json').write_text(json.dumps({'seed':29, 'trainingPairs':len(train), 'validationPairs':len(validation), 'validationSample':128, 'initialValidationLoss':initial, 'bestValidationLoss':best, 'objective':'assistant next-token cross entropy on external conversation pairs; exact training pair deduplication; no authored Q&A', 'history':history}, indent=2))

if __name__ == '__main__':
    main()

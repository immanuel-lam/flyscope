"""Re-tokenize pinned source conversations; retain adaptation split boundaries."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/foundation/corpus'


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'data/language-memory/source/data/train-00000-of-00004.parquet'
    tokenizer_file = ROOT / 'data/foundation/smollm2-135m/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    parts = {s: [] for s in ['train', 'validation', 'test']}
    spans = {s: [] for s in parts}
    offsets = {s: 0 for s in parts}
    counts = {s: 0 for s in parts}
    seen = set()
    tail = tokenizer.encode('<|im_end|>\n').ids
    for row in pq.read_table(source).to_pylist():
        retained = [m for m in row['messages'] if m['role'] in ['user', 'assistant']]
        digest = hashlib.sha256(json.dumps(retained, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        bucket = int(digest[:8], 16) % 100
        split = 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'
        messages = [m for m in row['messages'] if m['role'] in ['system', 'user', 'assistant']]
        if not messages or messages[0]['role'] != 'system':
            messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}] + messages
        tokens = []
        base = offsets[split]
        for message in messages:
            header = f'<|im_start|>{message["role"]}\n'
            content = message['content'].strip()
            prefix = tokenizer.encode(header).ids
            body = tokenizer.encode(content).ids
            complete = prefix + body + tail
            if complete != tokenizer.encode(header + content + '<|im_end|>\n').ids:
                raise ValueError('Message token boundaries differ from full chat encoding')
            if message['role'] == 'assistant':
                start = base + len(tokens) + len(prefix)
                spans[split].append([base, start, start + len(body) + 1])
            tokens.extend(complete)
        parts[split].append(np.asarray(tokens, np.int32))
        offsets[split] += len(tokens)
        counts[split] += 1
    report = {'sourceSha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'tokenizerSha256': hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
              'sourceRevision': 'f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc',
              'format': 'SmolLM chat markers. Source system messages retained; generic assistant system header inserted only when absent. Content boundary whitespace stripped; source answers otherwise unchanged.',
              'splitMeaning': 'Same deduplicated conversation hash partitions as prior experiments. Validation/test are held out from this adaptation only. The pretrained instruction foundation may already have seen this corpus, so these are not globally unseen language benchmarks.',
              'splits': {}}
    for split in parts:
        tokens = np.concatenate(parts[split])
        np.save(OUT / f'{split}.npy', tokens)
        np.save(OUT / f'{split}-reply-spans.npy', np.asarray(spans[split], np.int64))
        report['splits'][split] = {'conversations': counts[split], 'replies': len(spans[split]), 'tokens': len(tokens)}
        print(split, report['splits'][split], flush=True)
    (OUT / 'source.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

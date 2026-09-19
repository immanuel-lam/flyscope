"""Index actual source replies; verify them against the immutable base arrays."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/graph-language'


def main():
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/language-memory/tokenizer.json'))
    source = ROOT / 'data/language-memory/source/data/train-00000-of-00004.parquet'
    arrays = {s: np.load(DATA / f'{s}.npy', mmap_mode='r') for s in ['train', 'validation', 'test']}
    spans = {s: [] for s in arrays}
    offsets = {s: 0 for s in arrays}
    seen = set()
    assistant = tokenizer.token_to_id('<assistant>')
    end = tokenizer.token_to_id('<end>')
    for row in pq.read_table(source).to_pylist():
        retained = [m for m in row['messages'] if m['role'] in ['user', 'assistant']]
        digest = hashlib.sha256(json.dumps(retained, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        bucket = int(digest[:8], 16) % 100
        split = 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'
        text = ''
        for message in row['messages']:
            if message['role'] == 'system':
                text += 'Instructions: ' + message['content'] + '\n'
            elif message['role'] in ['user', 'assistant']:
                text += f'<{message["role"]}> ' + message['content'] + ' <end>\n'
        tokens = np.array(tokenizer.encode(text).ids, np.int32)
        offset = offsets[split]
        np.testing.assert_array_equal(tokens, arrays[split][offset:offset + len(tokens)])
        for start in np.flatnonzero(tokens == assistant):
            endings = np.flatnonzero(tokens[start + 1:] == end)
            if not len(endings):
                raise ValueError('Assistant turn has no end token')
            stop = int(start + 1 + endings[0] + 1)
            spans[split].append([offset, offset + int(start) + 1, offset + stop])
        offsets[split] += len(tokens)
    report = {'sourceSha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'splitRule': 'Same non-system conversation hash as base training; no changes to base arrays.',
              'objective': 'Assistant next-token loss only; conversation-bounded, left-padded windows matching CPU inference.',
              'splits': {}}
    for split in arrays:
        assert offsets[split] == len(arrays[split])
        filename = DATA / f'{split}-reply-spans.npy'
        np.save(filename, np.asarray(spans[split], np.int64))
        report['splits'][split] = {'replies': len(spans[split]), 'tokensVerified': offsets[split],
                                   'spansSha256': hashlib.sha256(filename.read_bytes()).hexdigest()}
    (DATA / 'dialogue-source.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

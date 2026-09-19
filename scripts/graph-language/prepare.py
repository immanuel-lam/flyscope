"""Pinned source corpus with source instructions retained and old splits preserved."""
from pathlib import Path
import json,hashlib
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'data/language-memory';OUT=ROOT/'data/graph-language'

def main():
    OUT.mkdir(exist_ok=True);tok=Tokenizer.from_file(str(OLD/'tokenizer.json'));groups={k:[] for k in ['train','validation','test']};seen=set()
    source=OLD/'source/data/train-00000-of-00004.parquet'
    for row in pq.read_table(source).to_pylist():
        retained=[m for m in row['messages'] if m['role'] in ['user','assistant']]
        digest=hashlib.sha256(json.dumps(retained,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if digest in seen:continue
        seen.add(digest);bucket=int(digest[:8],16)%100;split='test' if bucket==0 else 'validation' if bucket==1 else 'train'
        text=''
        for m in row['messages']:
            if m['role']=='system':text+='Instructions: '+m['content']+'\n'
            elif m['role'] in ['user','assistant']:text+=f'<{m["role"]}> '+m['content']+' <end>\n'
        groups[split].append(text)
    counts={}
    for split,rows in groups.items():
        parts=[np.array(tok.encode(text).ids,np.int32) for text in rows];tokens=np.concatenate(parts);np.save(OUT/f'{split}.npy',tokens);counts[split]={'conversations':len(rows),'tokens':len(tokens)};print(split,counts[split],flush=True)
    manifest=json.loads((OLD/'source.json').read_text());manifest.update(systemInstructions='Retained as Instructions: text. Hash split uses non-system turns to preserve the original split.',splits=counts,tokenizerSha256=hashlib.sha256((OLD/'tokenizer.json').read_bytes()).hexdigest())
    (OUT/'source.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()

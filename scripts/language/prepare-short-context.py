"""Complete short-context supervision, preserving source system instructions.
No authored answers and no filtering by evaluation performance.
"""
import json,hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'data/language-memory'

def main():
    source=DATA/'source/data/train-00000-of-00004.parquet';tok=Tokenizer.from_file(str(DATA/'tokenizer.json'));out=DATA/'short-context';out.mkdir(exist_ok=True)
    groups={k:[] for k in ['train','validation','test']};seen=set()
    for row in pq.read_table(source).to_pylist():
        turns=[m for m in row['messages'] if m['role'] in ['user','assistant']]
        digest=hashlib.sha256(json.dumps(turns,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if digest in seen:continue
        seen.add(digest);bucket=int(digest[:8],16)%100;split='test' if bucket==0 else 'validation' if bucket==1 else 'train'
        system='\n'.join(m['content'] for m in row['messages'] if m['role']=='system')
        prefix=f'Instructions: {system}\n' if system else ''
        for m in turns:
            if m['role']=='assistant':
                context=tok.encode(prefix+'<assistant>').ids;reply=tok.encode(' '+m['content']+' <end>').ids
                if len(context)<=128 and 8<=len(reply)<=96:
                    ids=context+reply;x=np.zeros(224,np.int32);y=x.copy();mask=np.zeros(224,np.float32)
                    x[:len(ids)-1]=ids[:-1];y[:len(ids)-1]=ids[1:];mask[len(context)-1:len(ids)-1]=1
                    groups[split].append((x,y,mask,digest))
            prefix+=f'<{m["role"]}> {m["content"]} <end>\n'
    for split,rows in groups.items():
        np.savez_compressed(out/f'{split}.npz',x=np.stack([r[0] for r in rows]),y=np.stack([r[1] for r in rows]),mask=np.stack([r[2] for r in rows]))
    report={'sourceSha256':hashlib.sha256(source.read_bytes()).hexdigest(),'tokenizerSha256':hashlib.sha256((DATA/'tokenizer.json').read_bytes()).hexdigest(),'criteria':'Complete serialized context including source system instructions <=128 tokens; full source reply 8..96 tokens. Same conversation hash splits as prior experiment. No evaluation-based filtering or authored answers.','samples':{k:len(v) for k,v in groups.items()},'conversationHashes':{k:sorted(set(r[3] for r in v)) for k,v in groups.items()}}
    for a,b in [('train','test'),('train','validation'),('validation','test')]:assert not set(report['conversationHashes'][a])&set(report['conversationHashes'][b])
    (out/'source.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='conversationHashes'},indent=2))
if __name__=='__main__':main()

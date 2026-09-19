"""Prepare a broader, pinned external dialogue corpus without authored examples."""
from pathlib import Path
import hashlib,json
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/language-memory';OUT.mkdir(exist_ok=True)
REPO='HuggingFaceTB/smol-smoltalk';REV='f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc'

def main():
    name='data/train-00000-of-00004.parquet'
    path=hf_hub_download(REPO,name,repo_type='dataset',revision=REV,local_dir=OUT/'source')
    rows=pq.read_table(path).to_pylist();groups={'train':[],'validation':[],'test':[]};seen=set()
    for row in rows:
        messages=[m for m in row['messages'] if m['role'] in ['user','assistant']]
        if not messages:continue
        text=json.dumps(messages,sort_keys=True,ensure_ascii=False);digest=hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        bucket=int(digest[:8],16)%100
        split='test' if bucket==0 else 'validation' if bucket==1 else 'train'
        groups[split].append({'messages':messages,'source':row['source']})
    def serialize(row):return ''.join(f'<{m["role"]}> '+m['content']+' <end>\n' for m in row['messages'])
    tok=Tokenizer(models.BPE(unk_token='<unk>'));tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tok.decoder=decoders.ByteLevel()
    tok.train_from_iterator((serialize(r) for r in groups['train']),trainers.BpeTrainer(vocab_size=2048,special_tokens=['<pad>','<unk>','<user>','<assistant>','<end>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(OUT/'tokenizer.json'))
    for split,group in groups.items():
        tokens=[];offsets=[0]
        for row in group:
            ids=tok.encode(serialize(row)).ids;tokens.extend(ids);offsets.append(len(tokens))
        np.save(OUT/f'{split}.npy',np.array(tokens,dtype=np.int32));np.save(OUT/f'{split}-offsets.npy',np.array(offsets,dtype=np.int64))
        (OUT/f'{split}-conversations.json').write_text(json.dumps(group))
        print(split,len(group),len(tokens),flush=True)
    manifest={'repo':REPO,'revision':REV,'file':name,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'license':'Apache-2.0','synthetic':True,'selection':'Entire first training shard; exact conversation deduplication. SHA256 hash buckets 0=test,1=validation,2..99=train. Original source test shard is not used.','splits':{k:len(v) for k,v in groups.items()},'vocab':2048}
    (OUT/'source.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()

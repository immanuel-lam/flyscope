"""Prepare a pinned dialogue corpus and a real directed MaleCNS subgraph."""
from pathlib import Path
import json,hashlib
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders
from huggingface_hub import HfApi,hf_hub_download
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/language';OUT.mkdir(parents=True,exist_ok=True)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    repo='HuggingFaceTB/smoltalk';rev=HfApi().dataset_info(repo).sha
    paths={s:hf_hub_download(repo,f'data/everyday-conversations/{s}-00000-of-00001.parquet',repo_type='dataset',revision=rev,local_dir=OUT/'source') for s in ['train','test']}
    rows={s:pq.read_table(p).to_pylist() for s,p in paths.items()}
    # Split by source conversation before tokenization; tokenizer sees train only.
    train=[];valid=[]
    for row in rows['train']:
        key=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
        (valid if int(key[:8],16)%10==0 else train).append(row)
    groups={'train':train,'validation':valid,'test':rows['test']}
    def serialize(row):return ''.join(f'<{m["role"]}> '+m['content']+' <end>\n' for m in row['messages'])
    tokenizer=Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    tokenizer.train_from_iterator((serialize(r) for r in train),trainers.BpeTrainer(vocab_size=1024,special_tokens=['<pad>','<unk>','<user>','<assistant>','<end>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer.save(str(OUT/'tokenizer.json'))
    for name,group in groups.items():
        sequences=[tokenizer.encode(serialize(r)).ids for r in group]
        np.save(OUT/f'{name}.npy',np.concatenate(sequences).astype(np.int32))
        (OUT/f'{name}-conversations.json').write_text(json.dumps(group))
    catalog=json.loads((ROOT/'public/malecns/catalog.json').read_text());n=len(catalog['rows'])
    eligible=np.array([r[2] in ['cb_intrinsic','descending_neuron','ascending_neuron'] for r in catalog['rows']])
    score=np.zeros(n)
    for i in range(128):
        edges=np.fromfile(ROOT/f'public/malecns/connections/{i}.bin',dtype='<u4').reshape(-1,3)
        edges=edges[eligible[edges[:,0]] & eligible[edges[:,1]]]
        score+=np.bincount(edges[:,0],weights=edges[:,2],minlength=n)+np.bincount(edges[:,1],weights=edges[:,2],minlength=n)
    selected=np.sort(np.argsort(-score,kind='stable')[:512]);lookup=np.full(n,-1,dtype=np.int32);lookup[selected]=np.arange(len(selected))
    counts=np.zeros((512,512),dtype=np.float32)
    for i in range(128):
        edges=np.fromfile(ROOT/f'public/malecns/connections/{i}.bin',dtype='<u4').reshape(-1,3)
        pre,post=lookup[edges[:,0]],lookup[edges[:,1]];ok=(pre>=0)&(post>=0)
        np.add.at(counts,(post[ok],pre[ok]),edges[ok,2])
    # Input and output units are disjoint. No embedding-to-logit skip path.
    inputs=np.argsort(-counts.sum(axis=0),kind='stable')[:128]
    outputs=np.setdiff1d(np.arange(512),inputs)
    assert np.count_nonzero(counts[np.ix_(outputs,inputs)])>100
    np.savez(OUT/'graph.npz',counts=counts,inputs=inputs,outputs=outputs)
    manifest={'datasetId':catalog['id'],'datasetVersion':catalog['version'],'neurons':[catalog['rows'][int(i)][:3] for i in selected], 'selection':'512 cells with largest incident synapse sum within cb_intrinsic/descending_neuron/ascending_neuron classes; ties stable by catalog order', 'edges':int(np.count_nonzero(counts)),'inputIndices':inputs.tolist(),'outputIndices':outputs.tolist(),'source':{'repo':repo,'revision':rev,'subset':'everyday-conversations','license':'Apache-2.0 according to original Everyday Conversations card','url':'https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k','files':{s:{'sha256':sha(p),'file':Path(p).name} for s,p in paths.items()}},'splits':{s:len(r) for s,r in groups.items()},'tokenizerSha256':sha(OUT/'tokenizer.json'),'graphSha256':sha(OUT/'graph.npz'),'seed':7}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print({k:manifest[k] for k in ['edges','splits','graphSha256']})
if __name__=='__main__':main()

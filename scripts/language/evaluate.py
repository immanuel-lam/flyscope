"""Held-out loss, portable-runtime parity and causal graph-ablation evaluation."""
import json,hashlib
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from model import CircuitLM,DATA,ROOT
from runtime import ChatCircuit

def main():
    report={};test=np.load(DATA/'test.npy');rng=np.random.default_rng(91)
    blocks=np.stack([test[i:i+65] for i in rng.integers(0,len(test)-65,128)])
    x=mx.array(blocks[:,:-1]);y=mx.array(blocks[:,1:]);model=CircuitLM();weights=DATA/'real/chat.safetensors';model.load_weights(str(weights if weights.exists() else DATA/'real/weights.safetensors'))
    for name,ablated in [('normal',False),('connectionsDisabled',True)]:
        value=float(nn.losses.cross_entropy(model(x,ablated),y,reduction='mean'));report[name]={'heldOutTokenLoss':value,'perplexity':float(np.exp(value))}
    assert report['normal']['heldOutTokenLoss']<report['connectionsDisabled']['heldOutTokenLoss']
    runtime=ChatCircuit();source=np.load(DATA/'graph.npz');mask=source['counts']>0
    assert np.all(runtime.weights['recurrent'][~mask]==0)
    assert not set(runtime.inputs)&set(runtime.outputs)
    assert np.isfinite(runtime.weights['recurrent']).all()
    report['topology']={'neurons':runtime.n,'sourceEdges':int(mask.sum()),'nonzeroLearnedEdges':int(np.count_nonzero(runtime.weights['recurrent'])),'offGraphEdges':0,'inputOutputOverlap':0}
    h=mx.zeros((1,512));cpu=np.zeros(512,dtype=np.float32);error=0
    for token in [2,89,144,71,4,3,891,312]:
        logits,h=model.step(mx.array([token]),h);cpu_logits,cpu=runtime.step(token,cpu);mx.eval(logits,h);error=max(error,float(np.max(np.abs(np.array(logits)[0]-cpu_logits))))
    assert error<1e-4;report['portableRuntimeMaxLogitError']=error
    groups=json.loads((ROOT/'scripts/language/fly-dialogues.json').read_text());outputs=[]
    for row in groups:
        prompt=row['questions'][4];r=runtime.generate(prompt,max_tokens=64);outputs.append({'prompt':prompt,'expected':row['answer'],'generated':r['text'],'exactMatch':r['text']==row['answer']})
    report['heldOutFlyPhrasings']={'count':len(outputs),'exactMatches':sum(r['exactMatch'] for r in outputs),'outputs':outputs}
    normal=runtime.generate('Hi');ablated=runtime.generate('Hi',ablated=True);assert normal['text']!=ablated['text'];assert normal['text'];assert len(normal['activity']['values'])==512
    report['greeting']={'normal':normal['text'],'connectionsDisabled':ablated['text']}
    report['limitations']=['Small authored-domain test with shared answer templates; not a general chat benchmark.','External text test includes greetings; no claim of biological language ability.','512 selected cells, not the full MaleCNS network; artificial dynamics and learned weights.']
    report['weightsSha256']=hashlib.sha256((ROOT/'models/malecns-chat/runtime.npz').read_bytes()).hexdigest()
    (ROOT/'models/malecns-chat/evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

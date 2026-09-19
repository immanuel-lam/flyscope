"""Fixed-budget real/shuffled/approximately parameter-matched dense comparison.

Uses the original 1024-token scalar-state model and data, not the new chat candidate.
"""
import json,time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from model import CircuitLM,DATA,ROOT

def make_model(variant):
    m=CircuitLM(variant='shuffled' if variant=='shuffled' else 'real')
    if variant=='dense':
        n=391;m._n=n;m._mask=mx.ones((n,n));m._inputs=mx.arange(128);m._outputs=mx.arange(128,n)
        m.recurrent=mx.random.normal((n,n))*(.7/np.sqrt(n));m.bias=mx.zeros(n);m.retention=mx.zeros(n);m.readout=nn.Linear(n-128,1024)
    return m

def main():
    train=np.load(DATA/'train.npy');test=np.load(DATA/'test.npy');out=DATA/'topology-comparison';out.mkdir(exist_ok=True)
    starts=np.random.default_rng(83).integers(0,len(test)-129,64)
    def batch(data,starts,length):
        a=np.stack([data[i:i+length+1] for i in starts]);return mx.array(a[:,:-1]),mx.array(a[:,1:])
    def loss(m,x,y):return nn.losses.cross_entropy(m(x),y,reduction='mean')
    rows=[]
    for seed in [7,17,29]:
        variants=['real','shuffled','dense']
        if seed==17:variants=variants[1:]+variants[:1]
        if seed==29:variants=variants[2:]+variants[:2]
        for variant in variants:
            mx.random.seed(seed);rng=np.random.default_rng(seed);model=make_model(variant);optimizer=optim.AdamW(learning_rate=.002,weight_decay=.01);grad=nn.value_and_grad(model,loss)
            for step in range(1000):
                x,y=batch(train,rng.integers(0,len(train)-49,16),48);value,grads=grad(model,x,y);grads,_=optim.clip_grad_norm(grads,1);optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state,value)
            losses=[float(loss(model,*batch(test,chunk,128))) for chunk in np.array_split(starts,8)]
            allocated=sum(v.size for _,v in tree_flatten(model.parameters()));edges=int(mx.sum(model._mask).item());active=allocated-model.recurrent.size+edges
            row={'seed':seed,'variant':variant,'steps':1000,'tokensSeen':768000,'allocatedParameters':allocated,'activeParameters':active,'recurrentEdges':edges,'heldOutLoss':float(np.mean(losses))};rows.append(row)
            model.save_weights(str(out/f'{variant}-{seed}.safetensors'));print(json.dumps(row),flush=True)
            report={'protocol':'Original scalar-state model, 1024-token corpus, 1000 fixed AdamW updates, identical token batches per seed, final checkpoint, identical 64 held-out length-128 sequences. No held-out model selection. Dense has 391 cells to match active parameter count within 0.1%.','results':rows,'limitations':'Small corpus and three seeds. Incoming-degree-preserving row shuffling does not preserve all graph statistics. This does not benchmark the newer four-channel candidate or establish biological language capacity. Concurrent training makes wall-clock comparisons invalid; none are claimed.'}
            (out/'results.json').write_text(json.dumps(report,indent=2))
    (ROOT/'docs/performance/language-topology-comparison.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()

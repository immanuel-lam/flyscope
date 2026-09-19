"""Learn delayed left/right cue recall through the MaleCNS directed subgraph."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
ROOT=Path(__file__).resolve().parents[2]

class CueCircuit(nn.Module):
    def __init__(self):
        super().__init__();graph=np.load(ROOT/'data/language/graph.npz')
        mask=(graph['counts']>0).astype(np.float32)
        self._mask=mx.array(mask);self._inputs=mx.array(graph['inputs']);self._outputs=mx.array(graph['outputs'])
        self.encoder=mx.random.normal((2,len(graph['inputs'])))*.3
        self.recurrent=mx.random.normal(mask.shape)*mx.array(.5/np.sqrt(np.maximum(mask.sum(1,keepdims=True),1)))
        self.retention=mx.full((len(mask),),2.)
        self.bias=mx.zeros((len(mask),));self.readout=nn.Linear(len(graph['outputs']),2)
    def step(self,cue,state):
        external=mx.zeros(state.shape).at[:,self._inputs].add(cue@self.encoder)
        gate=mx.sigmoid(self.retention)
        state=gate*state+(1-gate)*mx.tanh(state@(self.recurrent*self._mask).T+external+self.bias)
        return self.readout(state[:,self._outputs]),state
    def __call__(self,cues):
        state=mx.zeros((cues.shape[0],512))
        for i in range(cues.shape[1]):logits,state=self.step(cues[:,i],state)
        return logits

def batch(rng,delay,n=64,amplitudes=(.7,1.3)):
    labels=rng.integers(0,2,n);values=np.zeros((n,delay+3,2),np.float32)
    strength=rng.uniform(*amplitudes,n)
    for i in range(n):values[i,:3,labels[i]]=strength[i]
    # Blank delay is exactly zero: no cue, label, or trial ID remains in the input.
    return mx.array(values),mx.array(labels)

def main():
    mx.random.seed(73);rng=np.random.default_rng(73);model=CueCircuit();optimizer=optim.Adam(learning_rate=.003)
    def loss(m,x,y):return nn.losses.cross_entropy(m(x),y,reduction='mean')
    grad=nn.value_and_grad(model,loss);started=time.perf_counter();history=[]
    for step in range(601):
        x,y=batch(rng,int(rng.integers(4,13)));value,grads=grad(model,x,y);grads,_=optim.clip_grad_norm(grads,1);optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state,value)
        if step%100==0:row={'step':step,'loss':float(value),'seconds':time.perf_counter()-started};history.append(row);print(json.dumps(row),flush=True)
    out=ROOT/'models/malecns-cue-memory';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'runtime.npz',encoder=np.array(model.encoder),recurrent=np.array(model.recurrent*model._mask),retention=np.array(model.retention),bias=np.array(model.bias),readout=np.array(model.readout.weight),readout_bias=np.array(model.readout.bias))
    graph=np.load(ROOT/'data/language/graph.npz');manifest=json.loads((ROOT/'data/language/manifest.json').read_text())
    manifest={k:manifest[k] for k in ['datasetId','datasetVersion','neurons','selection','edges','graphSha256','inputIndices','outputIndices'] if k in manifest}
    manifest.update(modelId='malecns-cue-memory-v1',seed=73,task='Three cue steps, then zero-input delay, then left/right classification',trainingDelays=[4,12],trainingAmplitude=[.7,1.3],history=history,assumptions='Engineered input encoding, learned signed recurrence on fixed source edges, local retention and trained output decoder; not biological conditioning.')
    manifest['weightsSha256']=hashlib.sha256((out/'runtime.npz').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    model.save_weights(str(out/'weights.safetensors'))
if __name__=='__main__':main()

"""Unseen delay/strength controls and export parity for the cue-memory experiment."""
import json
import numpy as np
import mlx.core as mx
from train import CueCircuit,ROOT
from runtime import CueMemory

def main():
    directory=ROOT/'models/malecns-cue-memory';cpu=CueMemory(directory)
    model=CueCircuit();model.load_weights(str(directory/'weights.safetensors'));mx.eval(model.parameters())
    mx.random.seed(73);untrained=CueCircuit();mx.eval(untrained.parameters())
    rng=np.random.default_rng(79);rows=[];max_error=0.
    for delay in [16,24,48]:
        for amplitude in [.4,1.6]:
            hits={k:0 for k in ['trained','reset','ablated','untrained']};n=128
            for trial in range(n):
                label=trial%2;cue=np.zeros(2,np.float32);cue[label]=amplitude*rng.uniform(.9,1.1)
                sequences=[cue]*3+[np.zeros(2,np.float32)]*delay
                for condition in hits:
                    cpu.reset();state=mx.zeros((1,512))
                    for i,x in enumerate(sequences):
                        if condition=='reset' and i==3:cpu.reset()
                        if condition=='untrained':logits,state=untrained.step(mx.array(x[None,:]),state)
                        else:logits=cpu.step(x,ablated=condition=='ablated')
                    predicted=int(mx.argmax(logits)) if condition=='untrained' else int(np.argmax(logits))
                    hits[condition]+=predicted==label
            rows.append({'delaySteps':delay,'cueAmplitude':amplitude,'trials':n,'accuracy':{k:v/n for k,v in hits.items()}})
    cpu.reset();state=mx.zeros((1,512))
    for i in range(30):
        cue=np.array([.9,0],np.float32) if i<3 else np.zeros(2,np.float32)
        a=cpu.step(cue);b,state=model.step(mx.array(cue[None,:]),state);mx.eval(b,state)
        max_error=max(max_error,float(np.max(np.abs(a-np.array(b)[0]))))
        np.testing.assert_allclose(cpu.state,np.array(state)[0],atol=1e-5)
    assert max_error<1e-4
    assert all(r['accuracy']['trained']>=.95 for r in rows)
    assert all(r['accuracy']['reset']==.5 and r['accuracy']['ablated']==.5 for r in rows)
    report={'conditions':rows,'cpuMlxMaxLogitError':max_error,'limitations':'Artificial binary cue recall with engineered dynamics; longer delays and strengths held out, not unseen semantic concepts or biological learning. No claim that this improves language ability.'}
    (directory/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

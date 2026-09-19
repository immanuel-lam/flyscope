from pathlib import Path
import json,hashlib,shutil
import numpy as np
import mlx.core as mx
from model import CircuitLM,DATA,ROOT
from runtime import ChatCircuit

def main():
    model=CircuitLM();checkpoint=DATA/'real/dialogue.safetensors';model.load_weights(str(checkpoint));mx.eval(model.parameters())
    out=ROOT/'models/malecns-chat';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'runtime.npz',embedding=np.array(model.embedding.weight),recurrent=np.array(model.recurrent*model._mask),bias=np.array(model.bias),retention=np.array(model.retention),readout=np.array(model.readout.weight),readout_bias=np.array(model.readout.bias))
    shutil.copy(DATA/'tokenizer.json',out/'tokenizer.json');manifest=json.loads((DATA/'manifest.json').read_text());training=json.loads((DATA/'real/training.json').read_text())
    finetuning=json.loads((DATA/'real/dialogue-training.json').read_text())
    manifest.update({'finetuning':finetuning,'modelId':'malecns-lm-512-v3','architecture':'512 continuous recurrent cell states, 2 graph propagation steps/token, learned per-cell retention; 128 input cells and 384 disjoint readout cells; only observed directed edges permit inter-cell messages','dynamics':'Artificial tanh states and learned signed weights; source counts define topology, not fixed physiological weights. State retention is a model assumption. No biological spiking claim.','runtime':'NumPy + tokenizers; MLX required only for training/export','training':training,'weightsSha256':hashlib.sha256((out/'runtime.npz').read_bytes()).hexdigest()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));runtime=ChatCircuit(out)
    # Compare token-by-token CPU and MLX state/logits using an unseen sequence of token IDs.
    h=mx.zeros((1,512));state=np.zeros(512,dtype=np.float32);errors=[]
    for token in [2,71,283,4,3,421,91,104]:
        logits,h=model.step(mx.array([token]),h);mx.eval(logits,h);cpu,state=runtime.step(token,state);errors.append(float(np.max(np.abs(cpu-np.array(logits)[0]))));assert np.allclose(np.array(h)[0],state,atol=2e-5)
    assert max(errors)<1e-4,errors
    print(json.dumps({'runtimeParityMaxLogitError':max(errors),'model':str(out)}))
if __name__=='__main__':main()

"""Export an experimental candidate separately from the deployed checkpoint."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
from memory_model import MemoryCircuit,ROOT
from memory_runtime import MemoryRuntime

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='run-1');p.add_argument('--checkpoint',choices=['base','dialogue','short'],default='base');a=p.parse_args()
    data=ROOT/'data/language-memory';run=data/a.run;out=run/'portable';out.mkdir(exist_ok=True)
    checkpoint={'base':'weights.safetensors','dialogue':'dialogue.safetensors','short':'short.safetensors'}[a.checkpoint]
    training_file={'base':'training.json','dialogue':'dialogue-training.json','short':'short-training.json'}[a.checkpoint]
    model=MemoryCircuit();model.load_weights(str(run/checkpoint));mx.eval(model.parameters())
    weights={k:np.array(getattr(model,k)) for k in ['bias','retention','input_gate','state_gate']}
    weights.update(embedding=np.array(model.embedding.weight),recurrent=np.array(model.recurrent*model._mask),readout=np.array(model.readout.weight),readout_bias=np.array(model.readout.bias))
    np.savez_compressed(out/'runtime.npz',**weights)
    manifest=json.loads((ROOT/'data/language/manifest.json').read_text());manifest.update(modelId='malecns-memory-512x4-candidate',channels=4,architecture='Four state channels per source cell; two masked recurrent updates per token; local input/state-dependent retention gates; disjoint input/readout cells',dynamics='Engineered continuous memory channels, not biological compartments or spikes',source=json.loads((data/'source.json').read_text()),training=json.loads((run/training_file).read_text()),checkpoint=checkpoint,tokenizerSha256=hashlib.sha256((data/'tokenizer.json').read_bytes()).hexdigest(),weightsSha256=hashlib.sha256((out/'runtime.npz').read_bytes()).hexdigest())
    manifest['trainingLineage']=[json.loads((run/f).read_text()) for f in ['training.json','dialogue-training.json','short-training.json'] if (run/f).exists() and (f==training_file or f=='training.json' or a.checkpoint=='short')]
    if a.checkpoint=='short':manifest['fineTuningSource']={k:v for k,v in json.loads((data/'short-context/source.json').read_text()).items() if k!='conversationHashes'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copy(data/'tokenizer.json',out/'tokenizer.json')
    runtime=MemoryRuntime(out);h=mx.zeros((1,4,512));cpu=np.zeros((4,512),dtype=np.float32);error=0
    for t in [2,75,614,1832,4,3,1256,91]:
        logits,h=model.step(mx.array([t]),h);actual,cpu=runtime.step(t,cpu);mx.eval(logits,h)
        error=max(error,float(np.max(np.abs(np.array(logits)[0]-actual))))
        assert np.allclose(np.array(h)[0],cpu,atol=3e-5)
    assert error<1e-4,error
    print(json.dumps({'directory':str(out),'maxLogitError':error}))
if __name__=='__main__':main()

"""Export an experimental candidate separately from the deployed checkpoint."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
from memory_model import MemoryCircuit,ROOT
from memory_runtime import MemoryRuntime

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='run-1');p.add_argument('--checkpoint',choices=['base','dialogue','short','context'],default='base');a=p.parse_args()
    data=ROOT/'data/language-memory';run=data/a.run;out=run/'portable';out.mkdir(exist_ok=True)
    checkpoint={'base':'weights.safetensors','dialogue':'dialogue.safetensors','short':'short.safetensors','context':'context.safetensors'}[a.checkpoint]
    training_file={'base':'training.json','dialogue':'dialogue-training.json','short':'short-training.json','context':'context-training.json'}[a.checkpoint]
    if a.checkpoint=='context':
        from context_model import ContextCircuit
        model=ContextCircuit()
    else:model=MemoryCircuit()
    model.load_weights(str(run/checkpoint));mx.eval(model.parameters())
    weights={k:np.array(getattr(model,k)) for k in ['bias','retention','input_gate','state_gate']}
    weights.update(embedding=np.array(model.embedding.weight),recurrent=np.array(model.recurrent*model._mask),readout=np.array(model.readout.weight),readout_bias=np.array(model.readout.bias))
    if a.checkpoint=='context':weights['context_gain']=np.array(model.context_gain)
    np.savez_compressed(out/'runtime.npz',**weights)
    manifest=json.loads((ROOT/'data/language/manifest.json').read_text());manifest.update(modelId='malecns-memory-512x4-candidate',channels=4,architecture='Four state channels per source cell; two masked recurrent updates per token; local input/state-dependent retention gates; disjoint input/readout cells',dynamics='Engineered continuous memory channels, not biological compartments or spikes',source=json.loads((data/'source.json').read_text()),training=json.loads((run/training_file).read_text()),checkpoint=checkpoint,tokenizerSha256=hashlib.sha256((data/'tokenizer.json').read_bytes()).hexdigest(),weightsSha256=hashlib.sha256((out/'runtime.npz').read_bytes()).hexdigest())
    manifest['trainingLineage']=[json.loads((run/f).read_text()) for f in ['training.json','dialogue-training.json','short-training.json'] if (run/f).exists() and (f==training_file or f=='training.json' or a.checkpoint=='short')]
    if a.checkpoint=='short':manifest['fineTuningSource']={k:v for k,v in json.loads((data/'short-context/source.json').read_text()).items() if k!='conversationHashes'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copy(data/'tokenizer.json',out/'tokenizer.json')
    if a.checkpoint=='context':
        manifest['modelId']='malecns-prompt-context-512x4-candidate'
        manifest['promptConditioning']='Mean prompt embeddings are held as an external drive into the 128 input cells; no drive into output cells or logits. Engineered text interface, not biological memory.'
        manifest['trainingLineage']=[json.loads((run/f).read_text()) for f in ['training.json','dialogue-training.json','context-training.json']]
        manifest['fineTuningSource']={k:v for k,v in json.loads((data/'short-context/source.json').read_text()).items() if k!='conversationHashes'}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    runtime=MemoryRuntime(out);h=mx.zeros((1,4,512));cpu=np.zeros((4,512),dtype=np.float32);error=0
    context=runtime.prompt_context([2,75,614,4,3]);mx_context=None
    if context is not None:
        mx_context=model.prompt_context(mx.array([[2,75,614,4,3]]),mx.array([[0,0,0,0,1]]))
        np.testing.assert_allclose(context,np.array(mx_context)[0],atol=1e-6)
    for t in [2,75,614,1832,4,3,1256,91]:
        logits,h=model.step(mx.array([t]),h,**({'context':mx_context} if mx_context is not None else {}));actual,cpu=runtime.step(t,cpu,context=context);mx.eval(logits,h)
        error=max(error,float(np.max(np.abs(np.array(logits)[0]-actual))))
        assert np.allclose(np.array(h)[0],cpu,atol=3e-5)
    assert error<1e-4,error
    print(json.dumps({'directory':str(out),'maxLogitError':error}))
if __name__=='__main__':main()

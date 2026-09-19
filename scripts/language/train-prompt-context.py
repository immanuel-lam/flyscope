"""Fine-tune on complete short source contexts; no answer rules or retrieval."""
import json,time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from memory_model import ROOT
from context_model import ContextCircuit
DATA=ROOT/'data/language-memory'

def main():
    mx.random.seed(89);rng=np.random.default_rng(89);model=ContextCircuit();run=DATA/'run-1';model.load_weights(str(run/'dialogue.safetensors'),strict=False)
    data={k:dict(np.load(DATA/f'short-context/{k}.npz')) for k in ['train','validation']}
    def batch(split,indices):return tuple(mx.array(data[split][k][indices]) for k in ['x','y','mask'])
    def loss(m,x,y,mask):
        # Emphasize the start of each answer, where the prompt must determine content.
        ordinal=mx.cumsum(mask,axis=1);weight=mask*mx.where(ordinal<=16,2.,1.)
        return mx.sum(nn.losses.cross_entropy(m(x,mask),y,reduction='none')*weight)/mx.maximum(weight.sum(),1)
    ids=np.random.default_rng(97).integers(0,len(data['validation']['x']),64)
    def evaluate():return float(mx.mean(mx.stack([loss(model,*batch('validation',chunk)) for chunk in np.array_split(ids,8)])))
    initial=evaluate();best=initial;model.save_weights(str(run/'context.safetensors'));history=[];optimizer=optim.AdamW(learning_rate=.0005,weight_decay=.001);grad=nn.value_and_grad(model,loss);start=time.perf_counter()
    print(json.dumps({'initialValidationLoss':initial,'samples':len(data['train']['x'])}),flush=True)
    for step in range(1,2001):
        value,grads=grad(model,*batch('train',rng.integers(0,len(data['train']['x']),16)));grads,_=optim.clip_grad_norm(grads,1);optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state,value)
        if step%200==0 or step==1:
            val=evaluate();row={'step':step,'loss':float(value),'validationLoss':val,'seconds':time.perf_counter()-start};history.append(row);print(json.dumps(row),flush=True)
            if val<best:best=val;model.save_weights(str(run/'context.safetensors'))
            (run/'context-training.json').write_text(json.dumps({'seed':89,'initialValidationLoss':initial,'bestValidationLoss':best,'history':history,'objective':'Persistent mean prompt embedding into source input cells; complete short contexts, first 16 reply tokens weighted 2x. No authored answers. Validation-selected checkpoint.','source':str(DATA/'short-context/source.json')},indent=2))
if __name__=='__main__':main()

"""Assistant next-token training with actual preceding context, without authored replies."""
import argparse,json,time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from memory_model import MemoryCircuit,ROOT
DATA=ROOT/'data/language-memory'

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=4000);p.add_argument('--run',default='run-1');a=p.parse_args()
    mx.random.seed(53);rng=np.random.default_rng(53);model=MemoryCircuit();out=DATA/a.run;model.load_weights(str(out/'weights.safetensors'))
    arrays={k:np.load(DATA/f'{k}.npy',mmap_mode='r') for k in ['train','validation']}
    boundaries={k:np.load(DATA/f'{k}-offsets.npy') for k in arrays}
    starts={k:np.flatnonzero(v==3) for k,v in arrays.items()}
    def batch(split,indices):
        data=arrays[split];offsets=boundaries[split];xs=[];ys=[];ms=[]
        for i in indices:
            assistant=int(starts[split][i]);conversation_start=int(offsets[np.searchsorted(offsets,assistant,side='right')-1]);begin=max(conversation_start,assistant-192)
            tail=data[assistant+1:assistant+193];ends=np.flatnonzero(tail==4);count=int(ends[0])+1 if len(ends) else len(tail)
            ids=data[begin:assistant+1+count]
            x=np.zeros(384,dtype=np.int32);y=x.copy();mask=np.zeros(384,dtype=np.float32)
            x[:len(ids)-1]=ids[:-1];y[:len(ids)-1]=ids[1:];mask[assistant-begin:len(ids)-1]=1
            xs.append(x);ys.append(y);ms.append(mask)
        return mx.array(np.stack(xs)),mx.array(np.stack(ys)),mx.array(np.stack(ms))
    def loss(m,x,y,mask):return mx.sum(nn.losses.cross_entropy(m(x),y,reduction='none')*mask)/mx.maximum(mask.sum(),1)
    val_indices=np.random.default_rng(59).integers(0,len(starts['validation']),64)
    def evaluate():return float(mx.mean(mx.stack([loss(model,*batch('validation',c)) for c in np.array_split(val_indices,8)])))
    initial=evaluate();best=initial;model.save_weights(str(out/'dialogue.safetensors'));history=[];started=time.perf_counter();optimizer=optim.AdamW(learning_rate=.0003,weight_decay=.001);grad=nn.value_and_grad(model,loss)
    print(json.dumps({'initialAssistantLoss':initial}),flush=True)
    for step in range(1,a.steps+1):
        value,grads=grad(model,*batch('train',rng.integers(0,len(starts['train']),8)));grads,_=optim.clip_grad_norm(grads,1);optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state,value)
        if step==1 or step%200==0 or step==a.steps:
            val=evaluate();row={'step':step,'loss':float(value),'validationLoss':val,'seconds':time.perf_counter()-started};history.append(row);print(json.dumps(row),flush=True)
            if val<best:best=val;model.save_weights(str(out/'dialogue.safetensors'))
            (out/'dialogue-training.json').write_text(json.dumps({'initialLoss':initial,'bestValidationLoss':best,'seed':53,'objective':'assistant next-token loss with up to 192 preceding context tokens and 192 reply tokens; source conversations only','history':history},indent=2))
if __name__=='__main__':main()

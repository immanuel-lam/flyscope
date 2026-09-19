"""Broader next-token training; model choice uses validation, never test replies."""
import argparse,json,time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from memory_model import MemoryCircuit,ROOT
DATA=ROOT/'data/language-memory'

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch',type=int,default=12);p.add_argument('--length',type=int,default=192);p.add_argument('--resume',action='store_true');p.add_argument('--out',default='run-1');a=p.parse_args()
    mx.random.seed(43);rng=np.random.default_rng(43);model=MemoryCircuit();out=DATA/a.out;out.mkdir(exist_ok=True)
    if a.resume:model.load_weights(str(out/'weights.safetensors'))
    train=np.load(DATA/'train.npy',mmap_mode='r');valid=np.load(DATA/'validation.npy',mmap_mode='r')
    optimizer=optim.AdamW(learning_rate=.001,weight_decay=.01)
    def loss(m,x,y):return nn.losses.cross_entropy(m(x),y,reduction='mean')
    grad=nn.value_and_grad(model,loss)
    starts=np.random.default_rng(47).integers(0,len(valid)-a.length-1,32)
    def batch(data,starts):
        b=np.stack([data[i:i+a.length+1] for i in starts]);return mx.array(b[:,:-1]),mx.array(b[:,1:])
    def evaluate():return float(mx.mean(mx.stack([loss(model,*batch(valid,c)) for c in np.array_split(starts,8)])))
    best=evaluate();initial=best;model.save_weights(str(out/'weights.safetensors'));history=[];start=time.perf_counter()
    print(json.dumps({'initialValidationLoss':initial,'parameters':sum(v.size for _,v in tree_flatten(model.parameters()))}),flush=True)
    for step in range(1,a.steps+1):
        x,y=batch(train,rng.integers(0,len(train)-a.length-1,a.batch));value,grads=grad(model,x,y);grads,_=optim.clip_grad_norm(grads,1);optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state,value)
        if step==1 or step%200==0 or step==a.steps:
            val=evaluate();row={'step':step,'loss':float(value),'validationLoss':val,'seconds':time.perf_counter()-start};history.append(row);print(json.dumps(row),flush=True)
            if val<best:best=val;model.save_weights(str(out/'weights.safetensors'))
            (out/'training.json').write_text(json.dumps({'seed':43,'initialValidationLoss':initial,'bestValidationLoss':best,'parameters':sum(v.size for _,v in tree_flatten(model.parameters())),'arguments':vars(a),'history':history},indent=2))
if __name__=='__main__':main()

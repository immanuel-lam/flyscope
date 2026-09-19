import argparse,json,time,hashlib
from functools import partial
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from model import CircuitLM,DATA

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=2000);p.add_argument('--variant',choices=['real','shuffled','dense'],default='real');p.add_argument('--resume',action='store_true');p.add_argument('--lr',type=float,default=.002);p.add_argument('--batch',type=int,default=16);p.add_argument('--length',type=int,default=48);a=p.parse_args()
    mx.random.seed(7);rng=np.random.default_rng(7);model=CircuitLM(variant=a.variant);out=DATA/a.variant;out.mkdir(exist_ok=True)
    if a.resume:model.load_weights(str(out/'weights.safetensors'))
    train=np.load(DATA/'train.npy');valid=np.load(DATA/'validation.npy')
    optimizer=optim.AdamW(learning_rate=a.lr,weight_decay=.01)
    def loss(model,x,y):return nn.losses.cross_entropy(model(x),y,reduction='mean')
    loss_grad=nn.value_and_grad(model,loss)
    state=[model.state,optimizer.state,mx.random.state]
    def update(x,y):
        value,grads=loss_grad(model,x,y);grads,norm=optim.clip_grad_norm(grads,1.0);optimizer.update(model,grads);return value
    def batch(data):
        starts=rng.integers(0,len(data)-a.length-1,size=a.batch);b=np.stack([data[s:s+a.length+1] for s in starts]);return mx.array(b[:,:-1]),mx.array(b[:,1:])
    def evaluate():
        values=[]
        for start in np.linspace(0,len(valid)-a.length-1,16,dtype=int):
            x=mx.array(valid[start:start+a.length][None,:]);y=mx.array(valid[start+1:start+a.length+1][None,:]);values.append(loss(model,x,y))
        return float(mx.mean(mx.stack(values)))
    mx.eval(model.parameters());start=time.perf_counter();initial=evaluate();best=initial;history=[]
    print(json.dumps({'initialValidationLoss':initial,'parameters':sum(v.size for _,v in tree_flatten(model.parameters()))}),flush=True)
    for step in range(1,a.steps+1):
        x,y=batch(train);value=update(x,y);mx.eval(state,value)
        if step%100==0 or step==1 or step==a.steps:
            val=evaluate();row={'step':step,'loss':float(value),'validationLoss':val,'seconds':time.perf_counter()-start};history.append(row);print(json.dumps(row),flush=True)
            if val<best:best=val;model.save_weights(str(out/'weights.safetensors'))
            (out/'training.json').write_text(json.dumps({'variant':a.variant,'initialValidationLoss':initial,'bestValidationLoss':best,'history':history,'arguments':vars(a)},indent=2))
    print('Training complete',flush=True)
if __name__=='__main__':main()

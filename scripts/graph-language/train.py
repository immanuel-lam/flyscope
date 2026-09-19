"""Train the source-masked graph language model on pinned next-token data."""
import argparse,json,time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from model import GraphLanguageModel,ROOT,wiring
DATA=ROOT/'data/graph-language'

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch',type=int,default=8);p.add_argument('--width',type=int,default=256);p.add_argument('--layers',type=int,default=4);p.add_argument('--run',default='run-1');a=p.parse_args()
    mx.random.seed(127);rng=np.random.default_rng(127);model=GraphLanguageModel(width=a.width,layers=a.layers);optimizer=optim.AdamW(learning_rate=.0003,weight_decay=.01)
    train=np.load(DATA/'train.npy',mmap_mode='r');valid=np.load(DATA/'validation.npy',mmap_mode='r');out=DATA/a.run;out.mkdir(exist_ok=True)
    def batch(data,starts):
        x=np.stack([data[s:s+129] for s in starts]);return mx.array(x[:,:-1]),mx.array(x[:,1:])
    def loss(m,x,y):return nn.losses.cross_entropy(m(x),y,reduction='mean')
    grad=nn.value_and_grad(model,loss);valid_starts=np.random.default_rng(131).integers(0,len(valid)-129,32)
    def evaluate():return float(mx.mean(mx.stack([loss(model,*batch(valid,chunk)) for chunk in np.array_split(valid_starts,4)])))
    initial=evaluate();best=initial;history=[];started=time.perf_counter();parameters=sum(v.size for _,v in tree_flatten(model.parameters()));model.save_weights(str(out/'weights.safetensors'))
    inputs,outputs,slots,mask,source=wiring();np.savez_compressed(out/'wiring.npz',inputs=inputs,outputs=outputs,slots=slots,mask=mask,sourceMask=source)
    print(json.dumps({'initialValidationLoss':initial,'parameters':parameters,'sourceEdgesUsed':int(np.count_nonzero(mask&source&~np.eye(512,dtype=bool)))}),flush=True)
    for step in range(1,a.steps+1):
        optimizer.learning_rate=.0003*min(step/200,1)
        value,grads=grad(model,*batch(train,rng.integers(0,len(train)-129,a.batch)));grads,_=optim.clip_grad_norm(grads,1);optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state,value)
        if step==1 or step%200==0 or step==a.steps:
            val=evaluate();row={'step':step,'loss':float(value),'validationLoss':val,'seconds':time.perf_counter()-started};history.append(row);print(json.dumps(row),flush=True)
            if val<best:best=val;model.save_weights(str(out/'weights.safetensors'))
            (out/'training.json').write_text(json.dumps({'arguments':vars(a),'seed':127,'parameters':parameters,'initialValidationLoss':initial,'bestValidationLoss':best,'history':history,'assumptions':'128 token slots mapped to distinct input/readout source cells. Causal source-edge attention, local self-retention and local feed-forward operations. Engineered feature vectors, not biological compartments. No external LM or logits bypass.'},indent=2))
if __name__=='__main__':main()

"""Causal language prediction using graph-masked attention between source cells.

Each token slot has a distinct input cell and a directly connected readout cell.
Other cells act as intermediates. Feature vectors/local MLPs are engineered.
"""
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2]

def wiring():
    counts=np.load(ROOT/'data/language/graph.npz')['counts'];source=counts>0
    inputs=np.sort(np.random.default_rng(0).choice(512,128,replace=False));remaining=np.setdiff1d(np.arange(512),inputs)
    cost=(~source[np.ix_(remaining,inputs)].T).astype(float);r,c=linear_sum_assignment(cost)
    if cost[r,c].sum()!=0:raise ValueError('Missing source paths for token slots')
    outputs=remaining[c];slots=np.zeros(512,np.int32);slots[inputs]=np.arange(128);slots[outputs]=np.arange(128)
    intermediate=np.setdiff1d(remaining,outputs)
    for cell in intermediate:slots[cell]=int(np.argmax(counts[cell,inputs]))
    mask=source&(slots[:,None]>=slots[None,:])
    # Diagonal attention is local state retention, not an added inter-cell edge.
    mask|=np.eye(512,dtype=bool)
    return inputs,outputs,slots,mask,source

class GraphBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.norm1=nn.RMSNorm(width);self.norm2=nn.RMSNorm(width)
        self.qkv=nn.Linear(width,3*width,bias=False);self.projection=nn.Linear(width,width,bias=False)
        self.up=nn.Linear(width,4*width,bias=False);self.down=nn.Linear(4*width,width,bias=False);self._heads=heads
    def __call__(self,x,mask):
        b,n,d=x.shape;q,k,v=mx.split(self.qkv(self.norm1(x)),3,axis=-1)
        q,k,v=[a.reshape(b,n,self._heads,d//self._heads).transpose(0,2,1,3) for a in [q,k,v]]
        value=mx.fast.scaled_dot_product_attention(q,k,v,scale=(d//self._heads)**-.5,mask=mask)
        x=x+self.projection(value.transpose(0,2,1,3).reshape(b,n,d))
        hidden=self.up(self.norm2(x));activated=.5*hidden*(1+mx.tanh((2/np.pi)**.5*(hidden+.044715*hidden**3)))
        return x+self.down(activated)

class GraphLanguageModel(nn.Module):
    def __init__(self,width=256,layers=4,heads=8):
        super().__init__();inputs,outputs,slots,mask,source=wiring()
        self._injection=mx.array(np.eye(512,dtype=np.float32)[:,inputs]);self._inputs=mx.array(inputs);self._outputs=mx.array(outputs);self._mask=mx.array(mask);self._width=width
        self.embedding=nn.Embedding(2048,width);self.cell_embedding=nn.Embedding(512,width)
        self.blocks=[GraphBlock(width,heads) for _ in range(layers)];self.norm=nn.RMSNorm(width)
    def __call__(self,tokens,ablated=False,return_states=False):
        if tokens.shape[1]!=128:raise ValueError('Expected exactly 128 token slots')
        x=mx.broadcast_to(self.cell_embedding(mx.arange(512))[None,:,:],(tokens.shape[0],512,self._width))
        drive=self.embedding(tokens)*mx.expand_dims(tokens!=0,-1)
        x=x+mx.matmul(self._injection,drive)
        mask=mx.eye(512,dtype=mx.bool_) if ablated else self._mask
        for block in self.blocks:x=block(x,mask)
        logits=self.norm(x[:,self._outputs,:])@self.embedding.weight.T
        return (logits,x) if return_states else logits

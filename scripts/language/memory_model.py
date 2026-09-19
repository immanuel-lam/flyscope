"""Four trainable state channels per source cell, with source-masked recurrence.
Channels and local retention gates are engineering assumptions, not biological compartments.
"""
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
ROOT=Path(__file__).resolve().parents[2]

class MemoryCircuit(nn.Module):
    def __init__(self,vocab=2048,channels=4):
        super().__init__()
        graph=np.load(ROOT/'data/language/graph.npz')
        mask=(graph['counts']>0).astype(np.float32)
        self._mask=mx.array(mask)[None,:,:]
        self._inputs=mx.array(graph['inputs']);self._outputs=mx.array(graph['outputs'])
        self._channels=channels;self._cells=len(mask)
        self.embedding=nn.Embedding(vocab,len(graph['inputs'])*channels)
        self.recurrent=mx.random.normal((channels,*mask.shape))*mx.array(.5/np.sqrt(np.maximum(mask.sum(1,keepdims=True),1)))[None,:,:]
        self.bias=mx.zeros((channels,self._cells))
        fractions=mx.linspace(.6,.95,channels)[:,None]
        self.retention=mx.broadcast_to(mx.log(fractions/(1-fractions)),(channels,self._cells))
        self.input_gate=mx.zeros((channels,self._cells))
        self.state_gate=mx.zeros((channels,self._cells))
        self.readout=nn.Linear(len(graph['outputs'])*channels,vocab)
    def step(self,token,state,ablated=False):
        external=mx.zeros(state.shape)
        external=external.at[:,:,self._inputs].add(self.embedding(token).reshape(token.shape[0],self._channels,-1))
        weight=self.recurrent*self._mask*(0 if ablated else 1)
        for _ in range(2):
            messages=mx.matmul(state.transpose(1,0,2),weight.transpose(0,2,1)).transpose(1,0,2)
            gate=mx.sigmoid(self.retention+self.input_gate*external+self.state_gate*state)
            state=gate*state+(1-gate)*mx.tanh(messages+external+self.bias)
        return self.readout(state[:,:,self._outputs].reshape(token.shape[0],-1)),state
    def __call__(self,tokens,ablated=False):
        state=mx.zeros((tokens.shape[0],self._channels,self._cells));out=[]
        for i in range(tokens.shape[1]):
            logits,state=self.step(tokens[:,i],state,ablated);out.append(logits)
        return mx.stack(out,axis=1)

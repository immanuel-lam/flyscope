"""Tiny trainable recurrent circuit. Every off-diagonal recurrent edge is source-masked."""
from pathlib import Path
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'data/language'
class CircuitLM(nn.Module):
    def __init__(self,vocab=1024,variant='real'):
        super().__init__();g=np.load(DATA/'graph.npz');counts=g['counts'];mask=(counts>0).astype(np.float32)
        if variant=='shuffled':
            rng=np.random.default_rng(71);mask=np.stack([rng.permutation(row) for row in mask])
        elif variant=='dense': mask=np.ones_like(mask)
        self._mask=mx.array(mask);self._inputs=mx.array(g['inputs']);self._outputs=mx.array(g['outputs']);self._n=counts.shape[0]
        self.embedding=nn.Embedding(vocab,len(g['inputs']))
        self.recurrent=mx.random.normal(mask.shape)*mx.array(0.7/np.sqrt(np.maximum(mask.sum(1,keepdims=True),1)))
        self.bias=mx.zeros((self._n,));self.retention=mx.zeros((self._n,))
        self.readout=nn.Linear(len(g['outputs']),vocab)
    def step(self,token,h,ablated=False):
        external=mx.zeros((token.shape[0],self._n));external=external.at[:,self._inputs].add(self.embedding(token))
        w=self.recurrent*self._mask*(0 if ablated else 1)
        gate=mx.sigmoid(self.retention)
        for _ in range(2):h=gate*h+(1-gate)*mx.tanh(h@w.T+external+self.bias)
        return self.readout(h[:,self._outputs]),h
    def __call__(self,tokens,ablated=False):
        h=mx.zeros((tokens.shape[0],self._n));out=[]
        for t in range(tokens.shape[1]):
            logits,h=self.step(tokens[:,t],h,ablated);out.append(logits)
        return mx.stack(out,axis=1)

"""Persistent pooled prompt drive into source input cells; no output bypass."""
import mlx.core as mx
from memory_model import MemoryCircuit

class ContextCircuit(MemoryCircuit):
    def __init__(self):
        super().__init__();self.context_gain=mx.ones((self._channels,128))
    def prompt_context(self,tokens,reply_mask):
        prefix=(mx.cumsum(reply_mask,axis=1)==0).astype(mx.float32)
        encoded=self.embedding(tokens)
        pooled=mx.sum(encoded*prefix[:,:,None],axis=1)/mx.maximum(prefix.sum(axis=1,keepdims=True),1)
        return pooled.reshape(tokens.shape[0],self._channels,-1)
    def step(self,token,state,ablated=False,context=None):
        drive=self.embedding(token).reshape(token.shape[0],self._channels,-1)
        if context is not None:drive=drive+self.context_gain*context
        external=mx.zeros(state.shape).at[:,:,self._inputs].add(drive)
        weight=self.recurrent*self._mask*(0 if ablated else 1)
        for _ in range(2):
            messages=mx.matmul(state.transpose(1,0,2),weight.transpose(0,2,1)).transpose(1,0,2)
            gate=mx.sigmoid(self.retention+self.input_gate*external+self.state_gate*state)
            state=gate*state+(1-gate)*mx.tanh(messages+external+self.bias)
        return self.readout(state[:,:,self._outputs].reshape(token.shape[0],-1)),state
    def __call__(self,tokens,reply_mask):
        context=self.prompt_context(tokens,reply_mask);state=mx.zeros((tokens.shape[0],self._channels,self._cells));out=[]
        for i in range(tokens.shape[1]):
            logits,state=self.step(tokens[:,i],state,context=context);out.append(logits)
        return mx.stack(out,axis=1)

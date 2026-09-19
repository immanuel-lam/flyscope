"""NumPy-only inference for the multi-channel source-masked circuit."""
from pathlib import Path
import json,time
import numpy as np
from tokenizers import Tokenizer

class MemoryRuntime:
    def __init__(self,directory):
        directory=Path(directory)
        self.weights=dict(np.load(directory/'runtime.npz'))
        self.config=json.loads((directory/'manifest.json').read_text())
        self.tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'))
        self.inputs=np.array(self.config['inputIndices']);self.outputs=np.array(self.config['outputIndices'])
        self.channels=self.weights['bias'].shape[0];self.cells=self.weights['bias'].shape[1]
    def step(self,token,state,ablated=False):
        w=self.weights
        external=np.zeros_like(state);external[:,self.inputs]=w['embedding'][token].reshape(self.channels,-1)
        for _ in range(2):
            message=np.zeros_like(state) if ablated else np.matmul(w['recurrent'],state[:,:,None])[:,:,0]
            gate=1/(1+np.exp(-np.clip(w['retention']+w['input_gate']*external+w['state_gate']*state,-80,80)))
            state=gate*state+(1-gate)*np.tanh(message+external+w['bias'])
        return w['readout']@state[:,self.outputs].reshape(-1)+w['readout_bias'],state
    def generate(self,message,history=None,max_tokens=64,ablated=False):
        started=time.perf_counter()
        prompt=''.join(f'<{m["role"]}> {m["content"]} <end>\n' for m in (history or [])[-8:])+f'<user> {message} <end>\n<assistant>'
        prefix=self.tokenizer.encode(prompt).ids[-256:]
        state=np.zeros((self.channels,self.cells),dtype=np.float32)
        for token in prefix:logits,state=self.step(token,state,ablated)
        forbidden=[self.tokenizer.token_to_id(t) for t in ['<pad>','<unk>','<user>','<assistant>']]
        end=self.tokenizer.token_to_id('<end>');tokens=[]
        for _ in range(max_tokens):
            logits=logits.copy();logits[forbidden]=-1e9;token=int(np.argmax(logits))
            if token==end:break
            tokens.append(token);logits,state=self.step(token,state,ablated)
        elapsed=time.perf_counter()-started
        return {'text':self.tokenizer.decode(tokens).strip(),'tokenIds':tokens,'elapsedSeconds':elapsed,'tokensPerSecond':len(tokens)/max(elapsed,1e-9),'channels':self.channels,'cells':self.cells,'engine':'NumPy CPU','ablated':ablated}

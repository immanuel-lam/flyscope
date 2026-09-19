"""Portable CPU inference: NumPy + tokenizer. No MLX or pretrained LLM at runtime."""
from pathlib import Path
import json,time
import numpy as np
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[2];DEFAULT=ROOT/'models/malecns-chat'
class ChatCircuit:
    def __init__(self,directory=DEFAULT):
        directory=Path(directory);self.config=json.loads((directory/'manifest.json').read_text());self.weights=dict(np.load(directory/'runtime.npz'));self.tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'))
        self.inputs=np.array(self.config['inputIndices']);self.outputs=np.array(self.config['outputIndices']);self.n=len(self.config['neurons'])
        self.gate=1/(1+np.exp(-self.weights['retention']));self.end=self.tokenizer.token_to_id('<end>')
    def step(self,token,state,ablated=False):
        external=np.zeros(self.n,dtype=np.float32);external[self.inputs]=self.weights['embedding'][token]
        for _ in range(2):
            message=np.zeros_like(state) if ablated else self.weights['recurrent']@state
            state=self.gate*state+(1-self.gate)*np.tanh(message+external+self.weights['bias'])
        logits=self.weights['readout']@state[self.outputs]+self.weights['readout_bias']
        return logits,state
    def generate(self,message,history=None,max_tokens=40,temperature=0,seed=7,ablated=False):
        started=time.perf_counter();history=history or []
        prompt=''.join(f'<{turn["role"]}> {turn["content"]} <end>\n' for turn in history[-4:] if turn['role'] in ['user','assistant'])+f'<user> {message} <end>\n<assistant>'
        prompt_ids=self.tokenizer.encode(prompt).ids[-128:];h=np.zeros(self.n,dtype=np.float32)
        for token in prompt_ids:logits,h=self.step(token,h,ablated)
        rng=np.random.default_rng(seed);tokens=[];states=[]
        forbidden=[self.tokenizer.token_to_id(t) for t in ['<pad>','<unk>','<user>','<assistant>']]
        for _ in range(max_tokens):
            logits=logits.copy();logits[forbidden]=-1e9
            if temperature>0:
                p=np.exp((logits-logits.max())/temperature);p/=p.sum();token=int(rng.choice(len(p),p=p))
            else:token=int(np.argmax(logits))
            if token==self.end:break
            tokens.append(token);states.append(h.copy());logits,h=self.step(token,h,ablated)
        text=self.tokenizer.decode(tokens).strip()
        matrix=np.array(states,dtype=np.float32) if states else np.zeros((1,self.n),dtype=np.float32)
        return {'text':text,'tokens':[self.tokenizer.decode([t]) for t in tokens],'tokenIds':tokens,'elapsedSeconds':time.perf_counter()-started,'modelId':self.config['modelId'],'datasetId':self.config['datasetId'],'datasetVersion':self.config['datasetVersion'],'neurons':self.n,'edges':self.config['edges'],'ablated':ablated,'activity':{'kind':'simulation','unit':'hidden activation (signed a.u.)','times':[i*.15 for i in range(len(matrix))],'values':{row[0]:matrix[:,i].round(6).tolist() for i,row in enumerate(self.config['neurons'])}},'note':'Trained artificial dynamics on a real MaleCNS subgraph. Continuous hidden states, not biological spike measurements.'}
if __name__=='__main__':
    import sys
    request=json.load(sys.stdin);model=ChatCircuit();print(json.dumps(model.generate(request['message'],request.get('history'),max_tokens=request.get('maxTokens',40),ablated=request.get('ablated',False)),allow_nan=False))

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
    def step(self,token,state,ablated=False,trace=None):
        snapshots=[state.copy()] if trace is not None else None
        external=np.zeros(self.n,dtype=np.float32);external[self.inputs]=self.weights['embedding'][token]
        for _ in range(2):
            message=np.zeros_like(state) if ablated else self.weights['recurrent']@state
            state=self.gate*state+(1-self.gate)*np.tanh(message+external+self.weights['bias'])
            if snapshots is not None: snapshots.append(state.copy())
        logits=self.weights['readout']@state[self.outputs]+self.weights['readout_bias']
        if trace is not None:
            z=logits-logits.max();prob=np.exp(z);prob/=prob.sum();top=np.argsort(prob)[-5:][::-1]
            cells=self.view_cells
            trace.append({'token':self.tokenizer.decode([int(token)],skip_special_tokens=False),'tokenId':int(token),'states':[v[cells].round(6).tolist() for v in snapshots],'input':external[cells].round(6).tolist(),'predictions':[{'token':self.tokenizer.decode([int(t)],skip_special_tokens=False),'probability':float(prob[t]),'weights':[float(self.weights['readout'][t,np.where(self.outputs==c)[0][0]]) if c in self.outputs else 0.0 for c in cells]} for t in top]})
        return logits,state
    def generate(self,message,history=None,max_tokens=40,temperature=0,seed=7,ablated=False,inspect=False):
        trace=[] if inspect else None
        self.view_cells=np.concatenate([self.inputs[:8],self.outputs[:16]])
        started=time.perf_counter();history=history or []
        prompt=''.join(f'<{turn["role"]}> {turn["content"]} <end>\n' for turn in history[-4:] if turn['role'] in ['user','assistant'])+f'<user> {message} <end>\n<assistant>'
        prompt_ids=self.tokenizer.encode(prompt).ids[-128:];h=np.zeros(self.n,dtype=np.float32)
        for token in prompt_ids:logits,h=self.step(token,h,ablated,trace)
        rng=np.random.default_rng(seed);tokens=[];states=[]
        forbidden=[self.tokenizer.token_to_id(t) for t in ['<pad>','<unk>','<user>','<assistant>']]
        for _ in range(max_tokens):
            logits=logits.copy();logits[forbidden]=-1e9
            if temperature>0:
                p=np.exp((logits-logits.max())/temperature);p/=p.sum();token=int(rng.choice(len(p),p=p))
            else:token=int(np.argmax(logits))
            if token==self.end:break
            tokens.append(token);states.append(h.copy());logits,h=self.step(token,h,ablated,trace)
        text=self.tokenizer.decode(tokens).strip()
        matrix=np.array(states,dtype=np.float32) if states else np.zeros((1,self.n),dtype=np.float32)
        view=None
        if inspect:
            cells=self.view_cells
            edges=[{'from':i,'to':j,'weight':float(self.weights['recurrent'][dst,src])*(0 if ablated else 1)} for i,src in enumerate(cells) for j,dst in enumerate(cells) if self.weights['recurrent'][dst,src]!=0]
            view={'cells':[{'id':self.config['neurons'][int(c)][0],'input':bool(c in self.inputs),'retention':float(self.gate[c])} for c in cells],'edges':edges,'steps':trace,'promptSteps':len(prompt_ids),'selection':'First 8 input cells and first 16 readout cells in checkpoint order; induced connections only. All 512 cells are computed.'}
        return {'inspection':view,'text':text,'tokens':[self.tokenizer.decode([t]) for t in tokens],'tokenIds':tokens,'elapsedSeconds':time.perf_counter()-started,'modelId':self.config['modelId'],'datasetId':self.config['datasetId'],'datasetVersion':self.config['datasetVersion'],'neurons':self.n,'edges':self.config['edges'],'ablated':ablated,'activity':{'kind':'simulation','unit':'hidden activation (signed a.u.)','times':[i*.15 for i in range(len(matrix))],'values':{row[0]:matrix[:,i].round(6).tolist() for i,row in enumerate(self.config['neurons'])}},'note':'Trained artificial dynamics on a real MaleCNS subgraph. Continuous hidden states, not biological spike measurements.'}
if __name__=='__main__':
    import sys
    request=json.load(sys.stdin);model=ChatCircuit();print(json.dumps(model.generate(request['message'],request.get('history'),max_tokens=request.get('maxTokens',40),ablated=request.get('ablated',False),inspect=True),allow_nan=False))

"""All held-out complete short contexts, with raw free generation and controls."""
import json
from pathlib import Path
import numpy as np
from memory_runtime import MemoryRuntime
ROOT=Path(__file__).resolve().parents[2]

def main():
    directory=ROOT/'data/language-memory/run-1/portable';m=MemoryRuntime(directory);data=np.load(ROOT/'data/language-memory/short-context/test.npz');examples=[]
    for x,y,mask in zip(data['x'],data['y'],data['mask']):
        k=int(np.flatnonzero(mask)[0]);n=int(mask.sum());examples.append((x[:k+1].tolist(),y[k:k+n].tolist()))
    def prefix_state(prefix,ablated=False):
        state=np.zeros((m.channels,m.cells),np.float32)
        for t in prefix:logits,state=m.step(t,state,ablated)
        return logits,state
    def loss(prefix,target,ablated=False):
        logits,state=prefix_state(prefix,ablated);total=0.
        for t in target:
            z=logits.astype(np.float64);total+=float(z.max()+np.log(np.exp(z-z.max()).sum())-z[t]);logits,state=m.step(t,state,ablated)
        return total/len(target)
    outputs=[];correct=[];wrong=[];ablated=[]
    for i,(prefix,target) in enumerate(examples):
        logits,state=prefix_state(prefix);tokens=[]
        for _ in range(96):
            selected=logits.copy();selected[[0,1,2,3]]=-1e9;t=int(np.argmax(selected))
            if t==4:break
            tokens.append(t);logits,state=m.step(t,state)
        correct.append(loss(prefix,target));wrong.append(loss(examples[(i+1)%len(examples)][0],target));ablated.append(loss(prefix,target,True))
        grams=[tuple(tokens[j:j+4]) for j in range(max(0,len(tokens)-3))]
        outputs.append({'context':m.tokenizer.decode(prefix,skip_special_tokens=False),'sourceReply':m.tokenizer.decode(target),'generatedReply':m.tokenizer.decode(tokens),'tokenIds':tokens,'repeatedFourGramFraction':1-len(set(grams))/len(grams) if grams else 0})
    report={'checkpoint':m.config['checkpoint'],'examples':len(examples),'meanCorrectPromptLoss':float(np.mean(correct)),'meanMismatchedPromptLoss':float(np.mean(wrong)),'meanAblatedLoss':float(np.mean(ablated)),'meanRepeatedFourGramFraction':float(np.mean([x['repeatedFourGramFraction'] for x in outputs])),'outputs':outputs,'limitations':'Length-filtered held-out contexts, not a general intelligence benchmark. Raw generation retained. No source replies used at inference.'}
    (directory/'short-evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='outputs'},indent=2))
if __name__=='__main__':main()

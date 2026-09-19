"""Candidate audit: CPU throughput, repetition, prompt dependence and ablation.
Test results are reports, not a prompt-selection or response-replacement mechanism.
"""
import argparse,json
import numpy as np
from memory_model import ROOT
from memory_runtime import MemoryRuntime

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='run-1');a=p.parse_args();directory=ROOT/'data/language-memory'/a.run/'portable';m=MemoryRuntime(directory)
    rows=json.loads((ROOT/'data/language-memory/test-conversations.json').read_text());pairs=[]
    for row in rows:
        for left,right in zip(row['messages'],row['messages'][1:]):
            if left['role']=='user' and right['role']=='assistant':pairs.append((left['content'],right['content']))
    rng=np.random.default_rng(61);sample=[pairs[i] for i in rng.choice(len(pairs),64,replace=False)]
    def loss(prompt,reply,ablated=False):
        prefix=m.tokenizer.encode(f'<user> {prompt} <end>\n<assistant>').ids[-256:];target=m.tokenizer.encode(' '+reply+' <end>').ids[:192];state=np.zeros((m.channels,m.cells),dtype=np.float32)
        context=m.prompt_context(prefix)
        for token in prefix:logits,state=m.step(token,state,ablated,context)
        total=0
        for token in target:
            z=logits.astype(np.float64);total+=float(np.log(np.exp(z-z.max()).sum())+z.max()-z[token]);logits,state=m.step(token,state,ablated,context)
        return total/max(len(target),1)
    outputs=[];correct=[];wrong=[];ablated=[]
    for i,(prompt,reply) in enumerate(sample):
        generated=m.generate(prompt);ids=generated['tokenIds'];grams=[tuple(ids[j:j+4]) for j in range(max(0,len(ids)-3))]
        outputs.append({'prompt':prompt,**generated,'repeated4GramFraction':1-len(set(grams))/max(1,len(grams)) if grams else 0})
        correct.append(loss(prompt,reply));wrong.append(loss(sample[(i+1)%len(sample)][0],reply));ablated.append(loss(prompt,reply,True))
    report={'modelId':m.config['modelId'],'testPairs':len(sample),'meanCorrectPromptLoss':float(np.mean(correct)),'meanMismatchedPromptLoss':float(np.mean(wrong)),'meanAblatedLoss':float(np.mean(ablated)),'medianCpuTokensPerSecond':float(np.median([o['tokensPerSecond'] for o in outputs])),'meanRepeated4GramFraction':float(np.mean([o['repeated4GramFraction'] for o in outputs])),'uniqueReplies':len(set(o['text'] for o in outputs)),'outputs':outputs,'limitations':'Short greedy generations, one local CPU environment; loss and variety do not prove relevance, reasoning or biological language ability. Read unedited outputs.'}
    (directory/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='outputs'},indent=2))
if __name__=='__main__':main()

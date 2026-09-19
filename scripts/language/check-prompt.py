"""Report prompt conditioning on untouched external test conversations.
No selected answer strings, exact-match target, or score-based retraining.
"""
import json
import numpy as np
from runtime import ChatCircuit
from model import DATA, ROOT

model = ChatCircuit()
pairs = []
for row in json.loads((DATA/'test-conversations.json').read_text()):
    for left, right in zip(row['messages'], row['messages'][1:]):
        if left['role']=='user' and right['role']=='assistant':
            pairs.append((left['content'], right['content']))
rng = np.random.default_rng(173)
chosen = rng.choice(len(pairs), min(64,len(pairs)), replace=False)
sample = [pairs[i] for i in chosen]

def response_loss(prompt, reply):
    prefix = model.tokenizer.encode(f'<user> {prompt} <end>\n<assistant>').ids[-128:]
    target = model.tokenizer.encode(' '+reply+' <end>').ids
    h = np.zeros(model.n, dtype=np.float32)
    for t in prefix:
        logits,h = model.step(t,h)
    loss = 0
    for t in target:
        z = logits.astype(np.float64)
        loss += float(np.log(np.exp(z-z.max()).sum())+z.max()-z[t])
        logits,h = model.step(t,h)
    return loss/len(target)

correct=[];mismatched=[];outputs=[]
for i,(prompt,reply) in enumerate(sample):
    correct.append(response_loss(prompt,reply))
    mismatched.append(response_loss(sample[(i+1)%len(sample)][0],reply))
    outputs.append({'prompt':prompt, 'generated':model.generate(prompt)['text']})
report={'modelId':model.config['modelId'],'testPairs':len(sample), 'meanResponseLossCorrectPrompt':float(np.mean(correct)), 'meanResponseLossMismatchedPrompt':float(np.mean(mismatched)), 'uniqueGreedyReplies':len(set(r['generated'] for r in outputs)), 'outputs':outputs, 'interpretation':'Lower loss with the correct prompt suggests conditioning; diversity alone does not establish relevant or correct answers. No exact answer targets used for checkpoint selection.'}
(ROOT/'models/malecns-chat/prompt-evaluation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='outputs'},indent=2))

"""Supervised assistant-token training on documented fly dialogues, with held-out phrasings."""
import argparse,json,time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tokenizers import Tokenizer
from model import CircuitLM,DATA,ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=2000);p.add_argument('--variant',default='real');a=p.parse_args()
    mx.random.seed(17);rng=np.random.default_rng(17);model=CircuitLM(variant=a.variant);out=DATA/a.variant;model.load_weights(str(out/'weights.safetensors'));tok=Tokenizer.from_file(str(DATA/'tokenizer.json'))
    groups=json.loads((ROOT/'scripts/language/fly-dialogues.json').read_text())
    def encode(question,answer):
        prefix=tok.encode(f'<user> {question} <end>\n<assistant>').ids;reply=tok.encode(' '+answer+' <end>').ids
        ids=(prefix+reply)[:97];x=np.zeros(96,dtype=np.int32);y=x.copy();mask=np.zeros(96,dtype=np.float32);x[:len(ids)-1]=ids[:-1];y[:len(ids)-1]=ids[1:];mask[max(len(prefix)-1,0):len(ids)-1]=1
        return x,y,mask
    train=[encode(q,row['answer']) for row in groups for q in row['questions'][:3]]
    validation=[encode(row['questions'][3],row['answer']) for row in groups]
    everyday=[]
    for row in json.loads((DATA/'train-conversations.json').read_text()):
        messages=row['messages']
        for left,right in zip(messages,messages[1:]):
            if left['role']=='user' and right['role']=='assistant':everyday.append(encode(left['content'],right['content']))
    def loss(model,x,y,mask):
        losses=nn.losses.cross_entropy(model(x),y,reduction='none');return mx.sum(losses*mask)/mx.maximum(mx.sum(mask),1)
    grad=nn.value_and_grad(model,loss);optimizer=optim.AdamW(learning_rate=.0008,weight_decay=.001)
    def makebatch(rows,indices):return [mx.array(np.stack([rows[int(i)][j] for i in indices])) for j in range(3)]
    valid=makebatch(validation,np.arange(len(validation)));start=time.perf_counter();best=float(loss(model,*valid));history=[]
    print(json.dumps({'initialFlyValidationLoss':best}),flush=True)
    for step in range(1,a.steps+1):
        source=everyday if step%5==0 else train
        inputs=makebatch(source,rng.integers(0,len(source),32));value,grads=grad(model,*inputs);grads,norm=optim.clip_grad_norm(grads,1);optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state,value)
        if step%100==0 or step==1 or step==a.steps:
            validation_loss=float(loss(model,*valid));r={'step':step,'loss':float(value),'validationLoss':validation_loss,'seconds':time.perf_counter()-start};history.append(r);print(json.dumps(r),flush=True)
            if validation_loss<best:best=validation_loss;model.save_weights(str(out/'chat.safetensors'))
            (out/'finetuning.json').write_text(json.dumps({'variant':a.variant,'steps':a.steps,'seed':17,'trainQuestions':len(train),'validationQuestions':len(validation),'testQuestions':len(groups),'authoredData':'scripts/language/fly-dialogues.json; first 3 phrasings train, fourth validation, fifth test','mix':'80% authored fly dialogue batches, 20% external everyday dialogue batches','bestValidationLoss':best,'history':history},indent=2))
if __name__=='__main__':main()

"""Read checkpoint arrays and ranked incoming/outgoing weights by real cell ID."""
import argparse,json
import numpy as np
from runtime import ChatCircuit
parser=argparse.ArgumentParser()
parser.add_argument('--cell',help='MaleCNS body ID from manifest.json')
parser.add_argument('--limit',type=int,default=10)
args=parser.parse_args()
model=ChatCircuit()
print(json.dumps({'modelId':model.config['modelId'],'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'nonzero':int(np.count_nonzero(v))} for k,v in model.weights.items()}},indent=2))
if args.cell:
    ids=[row[0] for row in model.config['neurons']]
    if args.cell not in ids:parser.error('Cell is not in the model subgraph')
    i=ids.index(args.cell);w=model.weights['recurrent']
    for name,vector in [('incoming',w[i,:]),('outgoing',w[:,i])]:
        order=np.argsort(np.abs(vector))[::-1]
        print(json.dumps({name:[{'cell':ids[int(j)],'weight':float(vector[j])} for j in order if vector[j]!=0][:max(1,min(args.limit,100))]},indent=2))
    print(json.dumps({'cell':args.cell,'bias':float(model.weights['bias'][i]),'retention':float(model.gate[i]),'receivesTokens':bool(i in model.inputs),'suppliesReadout':bool(i in model.outputs)},indent=2))

"""Build bounded browser assets from the original MaleCNS v1.0 tables.
Run with .venv/bin/python scripts/prepare-malecns.py [--anchors] [--graph].
The neuron inclusion rule is superclass != null, not every raw segment.
"""
import argparse, asyncio, hashlib, json, struct, time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/malecns'; OUT=ROOT/'public/malecns'; OUT.mkdir(parents=True,exist_ok=True)
BASE='https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-precomputed/'
rows=[r for r in feather.read_table(RAW/'annotations.feather').to_pylist() if r['superclass']]
rows.sort(key=lambda r:r['bodyId'])
anchors={}
cache=RAW/'anchors.jsonl'
if cache.exists():
 for line in cache.read_text().splitlines():
  try:
   k,p=json.loads(line);anchors[k]=p
  except ValueError: pass

def catalog():
 data=[]
 for r in rows:
  p=r['somaLocation'] or r['tosomaLocation'] or anchors.get(str(r['bodyId']))
  kind='soma' if r['somaLocation'] else 'to-soma' if r['tosomaLocation'] else 'skeleton-anchor' if p else 'missing'
  data.append([str(r['bodyId']),r['instance'] or r['type'] or str(r['bodyId']),r['superclass'],p,kind,r['status'] or 'Unspecified'])
 manifest={'schemaVersion':1,'id':'male-cns-v1-full','version':'male-cns:v1.0','name':'MaleCNS · full neuron catalog','source':'HHMI Janelia, Cambridge, MRC LMB and Google Research; CC BY 4.0; https://male-cns.janelia.org/download/','units':'8 nm voxels','coordinateSpace':'Male CNS EM','inclusion':'All annotation rows with a non-null superclass; includes incompletely traced cells. Raw unclassified fragments and glia are excluded.','total':len(data),'positioned':sum(x[3] is not None for x in data),'skeletonBase':BASE,'rows':data}
 tmp=OUT/'catalog.tmp';tmp.write_text(json.dumps(manifest,separators=(',',':')));tmp.replace(OUT/'catalog.json')
 print(f'Catalog: {manifest["total"]:,} classified cells, {manifest["positioned"]:,} source positions',flush=True)

async def fetch_anchors():
 import aiohttp
 missing=[str(r['bodyId']) for r in rows if not(r['somaLocation'] or r['tosomaLocation']) and str(r['bodyId']) not in anchors]
 print(f'Fetching {len(missing):,} missing skeleton anchors (20-byte range requests; 24 concurrent)',flush=True)
 queue=asyncio.Queue()
 for id in missing: queue.put_nowait(id)
 done=0;started=time.time()
 timeout=aiohttp.ClientTimeout(total=30)
 with cache.open('a') as log:
  async with aiohttp.ClientSession(timeout=timeout,connector=aiohttp.TCPConnector(limit=24)) as session:
   async def worker():
    nonlocal done
    while not queue.empty():
     id=await queue.get();point=None;definitive=False
     for attempt in range(3):
      try:
       async with session.get(BASE+id,headers={'Range':'bytes=0-19'}) as response:
        if response.status==404: definitive=True;break
        if response.status!=206: raise ValueError(f'HTTP {response.status}')
        data=await response.read();n,e,x,y,z=struct.unpack('<IIfff',data)
        if n>0 and all(np.isfinite([x,y,z])):point=[x/8,y/8,z/8]
        definitive=True;break
      except Exception:
       await asyncio.sleep(.2*(attempt+1))
     if definitive:anchors[id]=point;log.write(json.dumps([id,point])+'\n')
     done+=1
     if done%1000==0:log.flush();print(f'Anchors {done:,}/{len(missing):,} · {time.time()-started:.1f}s',flush=True)
     queue.task_done()
   await asyncio.gather(*(worker() for _ in range(24)))
 catalog()

def graph():
 dest=OUT/'connections';dest.mkdir(exist_ok=True)
 ids=np.array([r['bodyId'] for r in rows],dtype=np.int64)
 handles=[(dest/f'{i}.bin').open('wb') for i in range(128)]
 counts=np.zeros(len(ids),dtype=np.uint32);total=0;weight_sum=0;started=time.time()
 with pa.memory_map(str(RAW/'connections.feather'),'r') as source:
  reader=pa.ipc.open_file(source)
  for i in range(reader.num_record_batches):
   batch=reader.get_batch(i);pre=batch.column(0).to_numpy();post=batch.column(1).to_numpy();weights=batch.column(2).to_numpy()
   a=np.searchsorted(ids,pre);b=np.searchsorted(ids,post)
   keep=(a<len(ids))&(b<len(ids));safea=np.minimum(a,len(ids)-1);safeb=np.minimum(b,len(ids)-1);keep &= (ids[safea]==pre)&(ids[safeb]==post)
   records=np.column_stack((a[keep],b[keep],weights[keep])).astype('<u4')
   if len(records):
    np.add.at(counts,records[:,0],1);total+=len(records);weight_sum+=int(weights[keep].sum())
    shard=records[:,0]%128
    order=np.argsort(shard,kind='stable');records=records[order];shard=shard[order]
    starts=np.r_[0,np.flatnonzero(np.diff(shard))+1,len(shard)]
    for l,h in zip(starts[:-1],starts[1:]):handles[int(shard[l])].write(records[l:h].tobytes())
   if i%300==0:print(f'Graph batch {i}/{reader.num_record_batches} · {total:,} retained edges · {time.time()-started:.1f}s',flush=True)
 for h in handles:h.close()
 counts.tofile(OUT/'outgoing-counts.bin')
 report={'schemaVersion':1,'shards':128,'record':'little-endian uint32 sourceCatalogIndex,targetCatalogIndex,synapseCount','edges':total,'synapseCount':weight_sum,'rawEdges':151856684,'inclusion':'Both endpoint IDs appear in the classified neuron catalog; all retained weights, not a top-k filter.','bytes':sum(p.stat().st_size for p in dest.glob('*.bin'))}
 (OUT/'graph.json').write_text(json.dumps(report,indent=2));print(report,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--anchors',action='store_true');p.add_argument('--graph',action='store_true');args=p.parse_args();catalog()
 if args.graph:graph()
 if args.anchors:asyncio.run(fetch_anchors())

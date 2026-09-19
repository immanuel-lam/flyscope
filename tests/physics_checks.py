"""Run after the documented driven/silenced/no-feedback experiments."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/physics'))
from brain import ConnectomeBrain

def read(name): return json.loads((ROOT/f'data/physics/{name}.json').read_text())
a,b,c=[read(n) for n in ['driven','silenced','no-feedback']]
assert a['metrics']['neurons']==166700
assert a['metrics']['effectiveSignedEdges']==24469412
assert a['metrics']['displacementMm']>5
assert a['metrics']['displacementMm']>5*b['metrics']['displacementMm']
assert 0.5<a['metrics']['finalHeightMm']<2
assert a['metrics']['maxContacts']>0
assert max(max(f['feedback']) for f in a['frames'])>0
assert all(max(f['drive'])==0 for f in b['frames'])
assert all(all(v==0 for v in row) for row in b['activity']['values'].values())
assert abs(a['metrics']['displacementMm']-c['metrics']['displacementMm'])>0.05
assert any(a['activity']['values'][k]!=c['activity']['values'][k] for k in a['activity']['values'])
for run in [a,b,c]:
    assert all(x<y for x,y in zip(run['times'],run['times'][1:]))
    assert len(run['frames'])==len(run['times'])
    if 'compiledBodyIds' in run:
        assert len(run['compiledBodyIds'])==len(run['bodyNames'])
        assert all(i>=0 for i in run['compiledBodyIds'])
    for frame in run['frames']:
        assert len(frame['positions'])==3*len(run['bodyNames'])
        assert len(frame['quaternions'])==4*len(run['bodyNames'])
        assert np.isfinite(frame['positions']).all() and np.isfinite(frame['quaternions']).all()
        assert np.allclose(np.linalg.norm(np.array(frame['quaternions']).reshape(-1,4),axis=1),1,atol=1e-5)
brain=ConnectomeBrain()
for _ in range(20): assert np.array_equal(brain.step(0,0,[0,0]),[0,0])
outputs=[]
for _ in range(2):
    brain.rates[:]=0
    outputs.append(np.stack([brain.step(1,0,[2,3]) for _ in range(100)]))
assert np.array_equal(*outputs), 'Neural model is not deterministic'
assert outputs[0][-1].min()>0.1
brain.rates[:]=0;brain.weights.data[:]=0
for _ in range(100): disconnected=brain.step(1,0,[2,3])
assert np.array_equal(disconnected,[0,0]), 'Decoder bypasses the connectome'
report={'checks':['physical displacement','neural silencing','contact feedback changes neural output and trajectory','finite body transforms','unit quaternions','silent input','deterministic neural dynamics','graph ablation stops descending drive'], 'recordingSha256':{n:hashlib.sha256((ROOT/f'data/physics/{n}.json').read_bytes()).hexdigest() for n in ['driven','silenced','no-feedback']},'recordedSourceChecksums':{n:r.get('sourceChecksums',{}) for n,r in zip(['driven','silenced','no-feedback'],[a,b,c])},'runs':{n:r['metrics'] for n,r in zip(['driven','silenced','no-feedback'],[a,b,c])}}
(ROOT/'docs/performance/physics-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

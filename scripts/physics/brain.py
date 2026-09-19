"""Experimental whole-MaleCNS rate dynamics, not a biological model of walking."""
from pathlib import Path
import json
import numpy as np
import pyarrow.feather as feather
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]

class ConnectomeBrain:
    dt = 0.01
    def __init__(self):
        self.catalog = json.loads((ROOT/'public/malecns/catalog.json').read_text())
        self.rows = self.catalog['rows']
        n = len(self.rows)
        annotations = {str(r['bodyId']): r for r in feather.read_table(ROOT/'data/malecns/annotations.feather', columns=['bodyId','somaSide','superclass','somaNeuromere','rootSide']).to_pylist()}
        classes = np.array([r[2] for r in self.rows])
        sides = np.array([annotations[r[0]]['somaSide'] or '' for r in self.rows])
        neuromeres = np.array([annotations[r[0]]['somaNeuromere'] or '' for r in self.rows])
        self.inputs = [np.flatnonzero((classes=='descending_neuron') & (sides==s)) for s in ['L','R']]
        # Thoracic motor pools: a population-level engineering decoder, NOT muscle identity.
        self.outputs = [np.flatnonzero((classes=='vnc_motor') & (sides==s) & np.isin(neuromeres,['T1','T2','T3'])) for s in ['L','R']]
        if any(len(a)==0 for a in self.outputs):
            raise ValueError('No thoracic motor pool found; verify annotation version.')
        roots = np.array([annotations[r[0]]['rootSide'] or '' for r in self.rows])
        self.sensory = [np.flatnonzero((classes=='vnc_sensory') & (roots==s)) for s in ['L','R']]
        if any(len(a)==0 for a in self.inputs+self.sensory): raise ValueError('Missing bilateral neural populations')
        cache = ROOT/'data/physics/signed-normalized-v1.npz'
        cache.parent.mkdir(parents=True,exist_ok=True)
        if cache.exists(): self.weights = sparse.load_npz(cache)
        else:
            nts = {str(r['body']): r['consensus_nt'] for r in feather.read_table(ROOT/'data/malecns/neurotransmitters.feather',columns=['body','consensus_nt']).to_pylist()}
            # Receptor-independent signs are assumptions; neuromodulators/unknown are omitted.
            sign = np.array([{'acetylcholine':1,'gaba':-1,'glutamate':-1}.get(nts.get(r[0]),0) for r in self.rows],dtype=np.float32)
            records = np.concatenate([np.fromfile(ROOT/f'public/malecns/connections/{i}.bin',dtype='<u4').reshape(-1,3) for i in range(128)])
            src,dst = records[:,0],records[:,1]
            values = records[:,2].astype(np.float32)*sign[src]
            norm = np.bincount(dst,weights=np.abs(values),minlength=n).astype(np.float32)
            values /= np.maximum(norm[dst],1)
            self.weights = sparse.csr_matrix((values,(dst,src)),shape=(n,n));self.weights.eliminate_zeros()
            sparse.save_npz(cache,self.weights,compressed=False)
        self.rates=np.zeros(n,dtype=np.float32)
        # Record all decoder cells plus a deterministic spread of input/sensory/intermediate cells.
        pools=np.concatenate([*self.inputs,*self.sensory,np.flatnonzero(classes=='vnc_intrinsic')])
        self.recorded=np.unique(np.concatenate([*self.outputs,pools[np.linspace(0,len(pools)-1,256,dtype=int)]]))
        self.output_gain=24.0

    def step(self, drive, turn, contacts, silenced=False, feedback=True):
        external=np.zeros_like(self.rates)
        for side in range(2):
            lateral=turn if side==0 else -turn
            external[self.inputs[side]]=max(0,drive+lateral)
            if feedback: external[self.sensory[side]]=0.08*float(np.clip(contacts[side],0,3))/3
        target=np.tanh(np.maximum(0,0.9*(self.weights@self.rates)+external))
        self.rates += (self.dt/0.05)*(target-self.rates)
        if silenced: self.rates[:]=0
        output=np.array([self.rates[indices].mean() for indices in self.outputs])
        return np.clip(self.output_gain*output,0,1.3)

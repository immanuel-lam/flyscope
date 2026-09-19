"""NumPy-only delayed-cue circuit. One state value per source cell."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

class CueMemory:
    def __init__(self,directory=None):
        directory=Path(directory or ROOT/'models/malecns-cue-memory')
        self.config=json.loads((directory/'manifest.json').read_text())
        w=np.load(directory/'runtime.npz')
        for name in w.files:setattr(self,name,w[name])
        self.inputs=np.array(self.config['inputIndices']);self.outputs=np.array(self.config['outputIndices'])
        self.gate=1/(1+np.exp(-self.retention));self.state=np.zeros(512,np.float32)
    def reset(self):self.state.fill(0)
    def step(self,cue,ablated=False):
        cue=np.asarray(cue,dtype=np.float32)
        if cue.shape!=(2,) or not np.isfinite(cue).all():raise ValueError('Expected two finite cue strengths')
        external=np.zeros(512,np.float32);external[self.inputs]=cue@self.encoder
        message=np.zeros(512,np.float32) if ablated else self.recurrent@self.state
        self.state=self.gate*self.state+(1-self.gate)*np.tanh(message+external+self.bias)
        return self.readout@self.state[self.outputs]+self.readout_bias

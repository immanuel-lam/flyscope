"""Report achieved body rotation and landing, independently of controller targets."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
r=json.loads((ROOT/'data/physics/backflip.json').read_text())
q=np.array([f['quaternions'][:4] for f in r['frames']]);angles=np.unwrap(2*np.arctan2(q[:,2],q[:,0]))
z=np.array([f['positions'][2] for f in r['frames']]);upright=1-2*(q[-1,1]**2+q[-1,2]**2)
print(json.dumps({'minimumPitchRadians':float(angles.min()),'finalPitchRadians':float(angles[-1]),'maximumHeightMm':float(z.max()),'finalHeightMm':float(z[-1]),'finalUprightCosine':float(upright),'finalContacts':r['frames'][-1]['contacts'],'assistEnded':not r['backflip']['frames'][-1]['active']},indent=2))

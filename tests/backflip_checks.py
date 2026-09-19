"""Check achieved body rotation/landing and neural-gated assist, not target angles."""
from pathlib import Path
import json,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
class BackflipChecks(unittest.TestCase):
    def test_full_rotation_then_unassisted_ground_contact(self):
        r=json.loads((ROOT/'data/physics/backflip.json').read_text());q=np.array([f['quaternions'][:4] for f in r['frames']]);angles=np.unwrap(2*np.arctan2(q[:,2],q[:,0]))
        upright=1-2*(q[-1,1]**2+q[-1,2]**2);height=np.array([f['positions'][2] for f in r['frames']])
        self.assertLess(angles.min(),-6)
        self.assertAlmostEqual(angles[-1],-2*np.pi,delta=.3)
        self.assertGreater(upright,.95);self.assertGreater(r['frames'][-1]['contacts'],0)
        self.assertTrue(.8<height[-1]<1.5);self.assertGreater(height.max(),5)
        final=r['backflip']['frames'][-1];self.assertFalse(final['active']);self.assertEqual(final['force'],[0,0,0]);self.assertEqual(final['torque'],[0,0,0])
        self.assertTrue(all(not f['active'] for f in r['backflip']['frames'] if f['time']>1.6))
        for f in r['frames']:self.assertTrue(np.isfinite(f['positions']).all())
        silent=json.loads((ROOT/'data/physics/backflip-silenced.json').read_text())
        self.assertTrue(all(not f['active'] and f['force']==[0,0,0] and f['torque']==[0,0,0] for f in silent['backflip']['frames']))
        self.assertLess(silent['metrics']['displacementMm'],.3)
        self.assertEqual(np.max(np.abs(list(silent['activity']['values'].values()))),0)
        (ROOT/'docs/performance/backflip-validation.json').write_text(json.dumps({'achievedMinimumPitchRadians':float(angles.min()),'achievedFinalPitchRadians':float(angles[-1]),'maximumHeightMm':float(height.max()),'finalHeightMm':float(height[-1]),'finalUprightCosine':float(upright),'finalContacts':r['frames'][-1]['contacts'],'silencedAssistActive':False,'limitations':'Fixed-seed externally assisted stunt. Neural output gates the assist; the network did not learn a biological backflip.'},indent=2))
if __name__=='__main__':unittest.main()

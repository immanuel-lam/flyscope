"""Sensor-field invariants independent of physics target-reaching outcomes."""
import sys, unittest, json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/physics'))
from odour import sample,steering

class OdourChecks(unittest.TestCase):
    def test_geometry_and_symmetry(self):
        antennae=np.array([[0,.3,1],[0,-.3,1]])
        left=sample(antennae,[12,4]);right=sample(antennae,[12,-4])
        np.testing.assert_allclose(left,right[::-1])
        self.assertLess(steering(left),0)
        self.assertGreater(steering(right),0)
        self.assertEqual(steering(sample(antennae,[12,0])),0)
        translated=sample(antennae+np.array([5,9,0]),[17,13])
        np.testing.assert_allclose(left,translated)
        self.assertGreater(sample(antennae,[2,0]).mean(),sample(antennae,[12,0]).mean())

    def test_absence_and_bad_inputs(self):
        self.assertEqual(steering([0,0]),0)
        self.assertEqual(steering([1,0]),-.6)
        for bad in [[float('nan'),0],[-1,0],[1]]:
            with self.assertRaises(ValueError):steering(bad)

class RecordedOdourChecks(unittest.TestCase):
    def test_closed_loop_controls(self):
        root=Path(__file__).resolve().parents[1]
        results={}
        for side,baseline in [('left','disabled'),('right','right-disabled')]:
            active=json.loads((root/f'data/physics/odour-{side}.json').read_text())
            disabled=json.loads((root/f'data/physics/odour-{baseline}.json').read_text())
            self.assertLess(active['metrics']['odourFinalDistanceMm'],disabled['metrics']['odourFinalDistanceMm'])
            self.assertNotEqual(active['activity']['values'],disabled['activity']['values'])
            self.assertTrue(all(f['turn']==0 for f in disabled['odour']['frames']))
            frames=active['odour']['frames']
            self.assertEqual(len(frames),201)
            self.assertTrue(all(a['time']<b['time'] for a,b in zip(frames,frames[1:])))
            for frame in frames:
                expected=sample(frame['antennae'],active['odour']['sourcePosition'][:2])
                np.testing.assert_allclose(frame['concentration'],expected,atol=1e-6)
                self.assertAlmostEqual(frame['turn'],steering(expected),places=5)
            results[side]={key:run['metrics']['odourFinalDistanceMm'] for key,run in [('activeMm',active),('disabledMm',disabled)]}
        silent=json.loads((root/'data/physics/odour-silenced.json').read_text())
        self.assertLess(silent['metrics']['displacementMm'],.3)
        self.assertEqual(np.max(np.abs(list(silent['activity']['values'].values()))),0)
        (root/'docs/performance/odour-validation.json').write_text(json.dumps({'conditions':results,'silencedDisplacementMm':silent['metrics']['displacementMm'],'limitations':'Two sources, one fixed seed, static field and engineered DN mapping; not turbulent plume tracking or biological olfaction.'},indent=2))

if __name__=='__main__':unittest.main()

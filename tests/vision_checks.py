"""Validate physical camera-guided runs against disabled and silenced controls."""
from pathlib import Path
import base64,json,unittest,sys
from io import BytesIO
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/physics'))
from vision import observe

class VisionChecks(unittest.TestCase):
    def test_visual_decoder_uses_pixels(self):
        class Simulator:
            def get_raw_vision(self,name):return self.images
        sim=Simulator();sim.images=np.zeros((2,16,16,3),dtype=np.uint8)
        self.assertEqual(observe(sim,'test')[2],0)
        sim.images[0,:8,:,0]=255
        self.assertLess(observe(sim,'test')[2],0)
        sim.images=sim.images[::-1].copy()
        self.assertGreater(observe(sim,'test')[2],0)

    def test_measured_navigation_and_ablation(self):
        results={}
        for side,active_name,disabled_name in [('left','vision-left','vision-disabled'),('right','vision-right','vision-right-disabled')]:
            active=json.loads((ROOT/f'data/physics/{active_name}.json').read_text())
            disabled=json.loads((ROOT/f'data/physics/{disabled_name}.json').read_text())
            self.assertLess(active['metrics']['targetFinalDistanceMm'],disabled['metrics']['targetFinalDistanceMm'])
            self.assertLess(active['metrics']['targetFinalDistanceMm'],active['metrics']['targetStartDistanceMm'])
            self.assertNotEqual(active['activity']['values'],disabled['activity']['values'])
            self.assertTrue(all(f['turn']==0 for f in disabled['vision']['frames']))
            frames=active['vision']['frames']
            self.assertEqual(len(frames),21)
            self.assertTrue(all(a['time']<b['time'] for a,b in zip(frames,frames[1:])))
            for f in frames:
                self.assertTrue(np.isfinite(f['redFraction']).all())
                for data in f['eyes']:
                    image=Image.open(BytesIO(base64.b64decode(data.split(',',1)[1])))
                    self.assertEqual(image.size,(225,256))
            results[side]={'activeFinalDistanceMm':active['metrics']['targetFinalDistanceMm'],'disabledFinalDistanceMm':disabled['metrics']['targetFinalDistanceMm']}
        silent=json.loads((ROOT/'data/physics/vision-silenced.json').read_text())
        self.assertLess(silent['metrics']['displacementMm'],.3)
        self.assertEqual(np.max(np.abs(list(silent['activity']['values'].values()))),0)
        (ROOT/'docs/performance/vision-validation.json').write_text(json.dumps({'conditions':results,'silencedDisplacementMm':silent['metrics']['displacementMm'],'limitations':'Two target positions, fixed seed and red-target decoder; not general visual navigation or biological validation.'},indent=2))

if __name__=='__main__':unittest.main()

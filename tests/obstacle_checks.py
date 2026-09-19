"""Real MuJoCo ray intersection and collision sensor checks."""
from pathlib import Path
import sys,unittest,json
from types import SimpleNamespace
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/physics'))
from obstacles import RangeSensor

class ObstacleChecks(unittest.TestCase):
    def test_actual_rays_and_direction_hold(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="head" pos="0 0 1"><geom name="head_geom" type="sphere" size=".01" group="0"/></body><body name="thorax"/><geom name="avoidance_obstacle_0" type="cylinder" pos="4 0 1" size="1 1" group="5"/><geom name="ignored" type="sphere" pos="1 0 1" size=".1" group="0"/></worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        sensor=RangeSensor(SimpleNamespace(mj_model=model,mj_data=data),mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"head_geom"),2)
        obs=sensor.observe();self.assertAlmostEqual(obs['distances'][4],3.)
        self.assertEqual(obs['distances'][0],8.)
        self.assertGreater(obs['turn'],0)
        self.assertEqual(obs['contacts'],0)
        self.assertEqual(sensor.observe()['turn'],obs['turn'])
        model.geom_pos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"avoidance_obstacle_0")]=[40,0,1];mujoco.mj_forward(model,data)
        clear=sensor.observe();self.assertEqual(clear['distances'],[8.]*9);self.assertGreater(clear['turn'],0)
        for _ in range(49):clear=sensor.observe()
        self.assertEqual(clear['turn'],0)

class PhysicalAvoidanceChecks(unittest.TestCase):
    def test_held_out_layouts_and_controls(self):
        root=Path(__file__).resolve().parents[1];report={}
        for layout in ['training','offset','wide','heldout-left','heldout-right']:
            active=json.loads((root/f'data/physics/obstacle-{layout}.json').read_text())
            disabled=json.loads((root/f'data/physics/obstacle-{layout}-disabled.json').read_text())
            a,d=active['metrics'],disabled['metrics']
            self.assertLess(a['obstacleContactSteps'],d['obstacleContactSteps'])
            self.assertGreater(d['obstacleContactSteps'],0)
            self.assertLessEqual(d['obstacleContactSteps'],30001)
            self.assertGreater(a['displacementMm'],10)
            self.assertGreater(a['forwardProgressMm'],0)
            self.assertNotEqual(active['activity']['values'],disabled['activity']['values'])
            self.assertTrue(all(f['turn']==0 for f in disabled['obstacles']['frames']))
            for f in active['obstacles']['frames']:
                self.assertTrue(np.isfinite(f['distances']).all())
                self.assertTrue(all(0<=v<=8 for v in f['distances']))
                self.assertEqual(len(f['distances']),9)
            report[layout]={'activeForwardMm':a['forwardProgressMm'],'disabledForwardMm':d['forwardProgressMm'],'activeContactSeconds':a['obstacleContactSteps']*.0001,'disabledContactSeconds':d['obstacleContactSteps']*.0001}
        silent=json.loads((root/'data/physics/obstacle-silenced.json').read_text())
        self.assertLess(silent['metrics']['displacementMm'],.3)
        self.assertEqual(np.max(np.abs(list(silent['activity']['values'].values()))),0)
        (root/'docs/performance/obstacle-validation.json').write_text(json.dumps({'conditions':report,'limitations':'One seed, five single-cylinder layouts; heldout-left and heldout-right tested after freezing the clearance-hold controller. Training, offset and wide are development cases. Artificial obstacle-only range sensors, not biological vision. Offset retains contact. Heldout-right avoids contact with less forward progress than disabled control. No guarantee of efficient navigation or cluttered-environment success.'},indent=2))

if __name__=='__main__':unittest.main()

"""Verify exported body transforms against compiled MuJoCo identities."""
from pathlib import Path
import json,unittest
import numpy as np
import mujoco
from flygym.compose import FlatGroundWorld
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly
ROOT=Path(__file__).resolve().parents[1]
class CompiledBodyChecks(unittest.TestCase):
    def test_initial_transforms_and_fixed_head_geometry(self):
        run=json.loads((ROOT/'data/physics/compiled-bodies.json').read_text())
        fly=make_locomotion_fly();world=FlatGroundWorld();world.add_fly(fly,spawn_position=[0,0,.3],spawn_rotation=Rotation3D('quat',[1,0,0,0]),add_ground_contact_sensors=False)
        sim=Simulation(world);sim.reset();mujoco.mj_forward(sim.mj_model,sim.mj_data)
        ids=run['compiledBodyIds'];self.assertTrue(all(i>=0 for i in ids));self.assertEqual(len(ids),len(set(ids)))
        self.assertNotIn('c_head',run['bodyNames'])
        for name,i in zip(run['bodyNames'],ids):self.assertEqual(mujoco.mj_id2name(sim.mj_model,mujoco.mjtObj.mjOBJ_BODY,i),fly.name+'/'+name)
        origin=sim.mj_data.xpos[ids[0]].copy();origin[2]=0
        np.testing.assert_allclose(np.array(run['frames'][0]['positions']).reshape(-1,3),sim.mj_data.xpos[ids]-origin,atol=1e-6)
        np.testing.assert_allclose(np.array(run['frames'][0]['quaternions']).reshape(-1,4),sim.mj_data.xquat[ids],atol=1e-6)
        head=next(g for g in run['geometry'] if g['name']=='c_head');self.assertEqual(run['bodyNames'][head['body']],'c_thorax')
        self.assertTrue(all(0<=g['body']<len(ids) for g in run['geometry']))
        sim.close()
if __name__=='__main__':unittest.main()

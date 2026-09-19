"""Verify graph attention cannot use future tokens or bypass source paths."""
from pathlib import Path
import sys,unittest
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/graph-language'))
from model import wiring,GraphLanguageModel
class GraphLanguageChecks(unittest.TestCase):
    def test_source_edges_and_slot_paths(self):
        inputs,outputs,slots,mask,source=wiring()
        self.assertFalse(set(inputs)&set(outputs));self.assertEqual(len(set(outputs)),128)
        self.assertTrue(np.all(source[outputs,inputs]))
        off_diagonal=~np.eye(512,dtype=bool)
        self.assertFalse(np.any(mask&off_diagonal&~source))
        post,pre=np.where(mask);self.assertTrue(np.all(slots[post]>=slots[pre]))
    def test_causality_and_graph_dependency(self):
        mx.random.seed(113);model=GraphLanguageModel(width=32,layers=2,heads=4)
        a=mx.full((1,128),75);b=a.at[:,65:].add(614-75)
        x=model(a);y=model(b);mx.eval(x,y)
        np.testing.assert_allclose(np.array(x[:,:65]),np.array(y[:,:65]),atol=1e-5)
        self.assertGreater(float(mx.max(mx.abs(x[:,65:]-y[:,65:]))),1e-4)
        x=model(a,ablated=True);y=model(b,ablated=True);mx.eval(x,y)
        np.testing.assert_array_equal(np.array(x),np.array(y))
if __name__=='__main__':unittest.main()

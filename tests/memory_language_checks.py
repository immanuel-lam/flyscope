"""Portable four-channel candidate invariants; does not assert useful language."""
from pathlib import Path
import sys,json,hashlib,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/language'))
from memory_runtime import MemoryRuntime
class CandidateChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=ROOT/'data/language-memory/run-1/portable'
        cls.model=MemoryRuntime(cls.directory)
    def test_source_mask_export_and_disjoint_pools(self):
        m=self.model;graph=np.load(ROOT/'data/language/graph.npz');mask=graph['counts']>0
        self.assertEqual(m.weights['recurrent'].shape,(4,512,512))
        self.assertEqual(np.count_nonzero(m.weights['recurrent'][:,~mask]),0)
        self.assertFalse(set(m.inputs)&set(m.outputs))
        source=json.loads((ROOT/'data/language/manifest.json').read_text())
        self.assertEqual(m.config['neurons'],source['neurons'])
        self.assertEqual(hashlib.sha256((self.directory/'tokenizer.json').read_bytes()).hexdigest(),m.config['tokenizerSha256'])
        self.assertEqual(hashlib.sha256((self.directory/'runtime.npz').read_bytes()).hexdigest(),m.config['weightsSha256'])
    def test_inputs_affect_output_only_through_graph(self):
        m=self.model;zero=np.zeros((4,512),np.float32)
        a,sa=m.step(75,zero);b,sb=m.step(614,zero)
        self.assertGreater(float(np.max(np.abs(a-b))),1e-5)
        self.assertGreater(float(np.max(np.abs(sa-sb))),1e-5)
        a,_=m.step(75,zero,True);b,_=m.step(614,zero,True)
        np.testing.assert_array_equal(a,b)
    def test_text_is_decoded_generated_tokens(self):
        m=self.model;r=m.generate('Describe a tree.',max_tokens=16)
        self.assertEqual(r['text'],m.tokenizer.decode(r['tokenIds']).strip())
        self.assertEqual(r['engine'],'NumPy CPU')
        self.assertLessEqual(len(r['tokenIds']),16)
if __name__=='__main__':unittest.main()

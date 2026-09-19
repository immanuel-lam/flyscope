"""Portable inference checks: decoding and causal use of the source graph."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/language'))
from runtime import ChatCircuit

class CircuitChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=ChatCircuit()

    def test_replies_are_decoded_generated_tokens(self):
        for prompt in ['Hi', 'Describe a forest.', 'The temperature fell overnight.']:
            reply=self.model.generate(prompt)
            self.assertEqual(reply['text'], self.model.tokenizer.decode(reply['tokenIds']).strip())
            self.assertTrue(np.isfinite(list(reply['activity']['values'].values())).all())

    def test_inspection_matches_inference_and_prompt_changes_states(self):
        reply=self.model.generate('Describe a forest.',inspect=True)
        view=reply['inspection']
        indices=[next(i for i,row in enumerate(self.model.config['neurons']) if row[0]==c['id']) for c in view['cells']]
        state=np.zeros(self.model.n,dtype=np.float32)
        for entry in view['steps']:
            np.testing.assert_allclose(entry['states'][0],state[indices],atol=1e-6)
            logits,state=self.model.step(entry['tokenId'],state)
            np.testing.assert_allclose(entry['states'][2],state[indices],atol=1e-6)
            self.assertEqual(entry['predictions'][0]['token'],self.model.tokenizer.decode([int(np.argmax(logits))],skip_special_tokens=False))
        for edge in view['edges']:
            self.assertEqual(edge['weight'],float(self.model.weights['recurrent'][indices[edge['to']],indices[edge['from']]]))
        other=self.model.generate('Hi',inspect=True)['inspection']
        left=np.array(view['steps'][view['promptSteps']-1]['states'][-1])
        right=np.array(other['steps'][other['promptSteps']-1]['states'][-1])
        self.assertGreater(float(np.max(np.abs(left-right))),1e-4)

    def test_input_reaches_readout_only_through_connections(self):
        state=np.zeros(self.model.n,dtype=np.float32)
        a,_=self.model.step(71,state)
        b,_=self.model.step(283,state)
        self.assertGreater(float(np.max(np.abs(a-b))),1e-5)
        a,_=self.model.step(71,state,ablated=True)
        b,_=self.model.step(283,state,ablated=True)
        np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()

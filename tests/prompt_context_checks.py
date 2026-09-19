"""Prompt-conditioning causality checks independent of response quality."""
from pathlib import Path
import sys,unittest
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/language'))
from context_model import ContextCircuit
class PromptContextChecks(unittest.TestCase):
    def test_answer_tokens_do_not_enter_prompt_pool(self):
        mx.random.seed(107);model=ContextCircuit()
        mask=mx.array([[0,0,0,1,1,1,0,0]])
        a=mx.array([[2,75,4,3,100,101,0,0]])
        b=mx.array([[2,75,4,3,1900,1800,75,614]])
        np.testing.assert_array_equal(np.array(model.prompt_context(a,mask)),np.array(model.prompt_context(b,mask)))
        c=mx.array([[2,614,4,3,100,101,0,0]])
        self.assertGreater(float(mx.max(mx.abs(model.prompt_context(a,mask)-model.prompt_context(c,mask)))),1e-5)
    def test_persistent_input_still_requires_source_edges(self):
        mx.random.seed(109);model=ContextCircuit();state=mx.zeros((1,4,512));token=mx.array([3])
        a=mx.zeros((1,4,128));b=mx.ones((1,4,128))
        x,_=model.step(token,state,True,a);y,_=model.step(token,state,True,b)
        np.testing.assert_array_equal(np.array(x),np.array(y))
        x,_=model.step(token,state,False,a);y,_=model.step(token,state,False,b)
        self.assertGreater(float(mx.max(mx.abs(x-y))),1e-5)
if __name__=='__main__':unittest.main()

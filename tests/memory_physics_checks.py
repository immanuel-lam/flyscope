from pathlib import Path
import json,unittest,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/memory'))
from runtime import CueMemory

def load(name):return json.loads((ROOT/f'data/physics/memory-{name}.json').read_text())

class MemoryPhysicsChecks(unittest.TestCase):
    def test_actual_states_and_delayed_motor_response(self):
        left,right=load('left'),load('right')
        for label,run in enumerate([left,right]):
            model=CueMemory();previous=np.zeros(512);j=0
            for i,frame in enumerate(run['memory']['frames']):
                logits=model.step(frame['cue'])
                np.testing.assert_allclose(frame['logits'],logits,atol=2e-5)
                if i>=3:self.assertEqual(frame['cue'],[0,0])
                if i<18:self.assertEqual(frame['turn'],0)
                else:self.assertEqual(int(np.argmax(logits)),label)
            self.assertEqual(len(run['memory']['activity']['values']),512)
            self.assertEqual(run['memory']['activity']['times'],run['times'])
            model.reset();index=0
            for i,t in enumerate(run['times']):
                while index<len(run['memory']['frames']) and run['memory']['frames'][index]['time']<=t+1e-8:
                    model.step(run['memory']['frames'][index]['cue']);index+=1
                values=[run['memory']['activity']['values'][row[0]][i] for row in model.config['neurons']]
                np.testing.assert_allclose(values,model.state,atol=1e-6)
        self.assertGreater(left['frames'][-1]['positions'][1],0)
        self.assertLess(right['frames'][-1]['positions'][1],0)
        for t,a,b in zip(left['times'],left['frames'],right['frames']):
            if t<1.8:self.assertEqual(a,b)

    def test_reset_removes_cue_dependency_and_silencing_stops_output(self):
        left,right=load('reset-left'),load('reset-right')
        self.assertEqual(left['frames'],right['frames'])
        self.assertEqual(left['activity'],right['activity'])
        silent=load('silenced')
        self.assertLess(silent['metrics']['displacementMm'],.3)
        for activity in [silent['activity'],silent['memory']['activity']]:self.assertEqual(np.max(np.abs(list(activity['values'].values()))),0)

if __name__=='__main__':unittest.main()

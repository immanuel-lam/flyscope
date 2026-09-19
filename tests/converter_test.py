import importlib.util
import json
import math
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('converter', Path(__file__).resolve().parents[1] / 'scripts/convert-motor-csv.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)

class ConverterTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.dataset = json.loads((root / 'examples/external-structure.json').read_text())
        self.mapping = json.loads((root / 'examples/external-mapping.json').read_text())
        self.row = {'time':'0', 'x':'0', 'y':'0', 'z':'1', 'heading':'0', 'lf_sweep_degrees':'90'}

    def test_units_and_explicit_neutral_channels(self):
        run = converter.convert(self.dataset, [self.row], self.mapping, 'test source', 'test model', 'simulation')
        self.assertAlmostEqual(run['motor']['poses'][0]['joints']['LF.sweep'], math.pi / 2)
        self.assertEqual(run['motor']['poses'][0]['joints']['RF.sweep'], 0)
        self.assertNotIn('activity', run)

    def test_missing_mapping_rejected(self):
        del self.mapping['RH.knee']
        with self.assertRaises(ValueError):
            converter.convert(self.dataset, [self.row], self.mapping, 'test', 'test', 'simulation')

    def test_bad_time_and_nonfinite_values_rejected(self):
        with self.assertRaises(ValueError):
            converter.convert(self.dataset, [self.row, self.row], self.mapping, 'test', 'test', 'simulation')
        self.row['lf_sweep_degrees'] = 'nan'
        with self.assertRaises(ValueError):
            converter.convert(self.dataset, [self.row], self.mapping, 'test', 'test', 'simulation')

if __name__ == '__main__':
    unittest.main()

"""Convert explicitly mapped external simulator poses to Flyscope's reduced rig.
Standard library only. See docs/EXTERNAL_SIMULATORS.md. Does not run physics.
"""
import argparse
import csv
import json
import math
from pathlib import Path

LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
JOINTS = [f'{leg}.{axis}' for leg in LEGS for axis in ('sweep', 'lift', 'knee')] + ['wing.L', 'wing.R']

def convert(dataset, rows, mapping, source, model, kind):
    if not source.strip() or not model.strip():
        raise ValueError('Source and model are required')
    if set(mapping) != {'time', 'x', 'y', 'z', 'heading', *JOINTS}:
        raise ValueError('Mapping must define time, x, y, z, heading and all 20 joints')
    def value(row, channel):
        spec = mapping[channel]
        if 'constant' in spec:
            if set(spec) != {'constant'}:
                raise ValueError(f'{channel}: constants cannot include other keys')
            result = float(spec['constant'])
        else:
            if set(spec) - {'column', 'scale', 'offset'} or 'column' not in spec:
                raise ValueError(f'{channel}: expected column, optional scale and offset')
            result = float(row[spec['column']]) * float(spec.get('scale', 1)) + float(spec.get('offset', 0))
        if not math.isfinite(result):
            raise ValueError(f'{channel}: value is not finite')
        return result
    times, poses = [], []
    for row in rows:
        t = value(row, 'time')
        if t < 0 or (times and t <= times[-1]):
            raise ValueError('Time must be non-negative and strictly increasing')
        joints = {j: value(row, j) for j in JOINTS}
        if any(abs(v) > math.pi for v in joints.values()):
            raise ValueError('Joint angles must be additive radians within ±pi; check your mapping')
        position = [value(row, c) for c in ('x', 'y', 'z')]
        if any(abs(v) > 10000 for v in position):
            raise ValueError('Position exceeds 10,000 mm viewer limit')
        times.append(t)
        poses.append({'position': position, 'heading': value(row, 'heading'), 'joints': joints})
        if len(times) > 18001:
            raise ValueError('Export at most 18,001 frames; downsample before conversion')
    if not times:
        raise ValueError('CSV has no poses')
    dataset['motor'] = {
        'schemaVersion': 1, 'rig': 'flyscope-procedural-v1', 'kind': kind,
        'source': source, 'model': model, 'datasetId': dataset['id'], 'datasetVersion': dataset['version'],
        'units': {'position': 'mm', 'angle': 'rad', 'time': 's'}, 'times': times, 'poses': poses,
    }
    return dataset

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('dataset', 'csv', 'mapping', 'source', 'model', 'output'):
        p.add_argument(f'--{arg}', required=True)
    p.add_argument('--kind', choices=('simulation', 'recording'), default='simulation')
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        p.error('Output already exists; choose a new path to preserve prior runs')
    with open(args.csv, newline='') as f:
        result = convert(json.loads(Path(args.dataset).read_text()), csv.DictReader(f), json.loads(Path(args.mapping).read_text()), args.source, args.model, args.kind)
    output.write_text(json.dumps(result, allow_nan=False))
    print(f'Wrote {output}. Next run: npm run validate:dataset -- "{output}"')

if __name__ == '__main__':
    main()

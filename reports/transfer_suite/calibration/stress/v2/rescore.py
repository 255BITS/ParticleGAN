import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, '/ml2/hypergan/ParticleGAN-transfer-stress')
from benchmarks.transfer_suite import stress_tasks, vector_tasks
from benchmarks.locked_shared.observation import sustained
from benchmarks.locked_shared.baseline import write_json

source = Path('/tmp/pr36-transfer-stress-v1/results.json')
out = Path(__file__).parent
if (out/'results.json').exists():
    raise FileExistsError('Refuse to overwrite prior rescoring')
original_bytes = source.read_bytes()
original = json.loads(original_bytes)
specs = {spec['name']: spec for spec in stress_tasks.TASKS}
report = {
    'evaluation_protocol': stress_tasks.PROTOCOL,
    'recorded_training_protocol': original['protocol'],
    'source_result': str(source),
    'source_result_sha256': hashlib.sha256(original_bytes).hexdigest(),
    'evaluation_source_sha256': {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                 for module in (stress_tasks, vector_tasks)},
    'rescorer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'development_specs': stress_tasks.TASKS, 'reserved_specs': stress_tasks.RESERVED_TASKS,
    'retrained': False, 'reserved_evaluated': False, 'rows': [],
}
for index, row in enumerate(original['rows']):
    spec = specs[row['task']]
    result = row['result']
    thresholds = spec['thresholds']
    required = {key for key, _, _ in thresholds}
    assert required <= result['live'].keys()
    assert all(required <= point.keys() for point in result['observations'])
    expected = {math.ceil(i * spec['steps'] / 24) for i in range(1, 25)}
    convergence = sustained(result['observations'], thresholds, expected_steps=expected, minimum=5)
    report['rows'].append({
        'task': row['task'], 'family': row['family'], 'tier': row['tier'], 'reference': row['reference'],
        'source_row_index': index, 'live': result['live'], 'ema': result['ema'],
        'old_status': result['status'], 'old_convergence': result['convergence'],
        'status': 'PASS' if vector_tasks.passes(result['live'], thresholds) else 'FAIL',
        'ema_status': 'PASS' if vector_tasks.passes(result['ema'], thresholds) else 'FAIL',
        'convergence': convergence, 'thresholds': thresholds, 'seconds_recorded': result['seconds'],
    })
report['reference_solvable'] = {name: any(row['task']==name and row['convergence']['confirmed_step'] is not None
                                        for row in report['rows']) for name in specs}
report['outcome_changes'] = [row['source_row_index'] for row in report['rows']
    if row['status'] != row['old_status'] or row['convergence'] != row['old_convergence']]
assert source.read_bytes() == original_bytes
write_json(out/'frozen_specs.json', {'evaluation_protocol': stress_tasks.PROTOCOL,
                                   'development': stress_tasks.TASKS, 'reserved': stress_tasks.RESERVED_TASKS,
                                   'evaluation_source_sha256': report['evaluation_source_sha256']})
write_json(out/'results.json', report)
print(json.dumps({'rows':len(report['rows']), 'changes':report['outcome_changes'],
                  'reference_solvable':report['reference_solvable'],
                  'source_sha256':report['source_result_sha256'],
                  'evaluation_source_sha256':report['evaluation_source_sha256']}, indent=2))

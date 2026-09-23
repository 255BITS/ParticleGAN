import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, '/ml2/hypergan/ParticleGAN-transfer-stress')
import torch
from benchmarks.transfer_suite import stress_tasks, vector_tasks
from benchmarks.locked_shared.baseline import write_json

torch.set_num_threads(1)
out = Path(__file__).parent
if (out/'results.json').exists():
    raise FileExistsError('Refuse to overwrite prior attempts')
protocol = vector_tasks.fingerprint()
module = Path(stress_tasks.__file__)
protocol['source_sha256']['benchmarks/transfer_suite/stress_tasks.py'] = hashlib.sha256(module.read_bytes()).hexdigest()
protocol['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
report = {'protocol': protocol, 'stress_protocol': stress_tasks.PROTOCOL,
          'development_specs': stress_tasks.TASKS, 'reserved_specs': stress_tasks.RESERVED_TASKS,
          'reserved_evaluated': False, 'rows': [], 'complete': False}
write_json(out/'frozen_specs.json', {'protocol': protocol, 'development': stress_tasks.TASKS,
                                   'reserved': stress_tasks.RESERVED_TASKS})
write_json(out/'results.json', report)
for spec in stress_tasks.TASKS:
    assert spec['split'] == 'development'
    for schedule in ('cosine', 'constant'):
        print('START', spec['name'], schedule, 'tier='+spec['tier'], 'steps='+str(spec['steps']), flush=True)
        policy = vector_tasks.fixed_policy(schedule)
        result = stress_tasks.run_episode(spec, policy, fixed=True)
        row = {'task': spec['name'], 'family': spec['family'], 'tier': spec['tier'],
               'reference': 'fixed_'+schedule, 'policy': policy, 'result': result}
        report['rows'].append(row)
        write_json(out/'results.json', report)
        print('DONE', spec['name'], schedule, result.get('status'),
              'live='+str({key:result.get('live',{}).get(key) for key,_,_ in spec['thresholds']}),
              'suffix='+str(result.get('convergence',{}).get('passing_suffix')),
              'confirmed='+str(result.get('convergence',{}).get('confirmed_step')),
              'seconds='+str(result.get('seconds')), flush=True)
        if result.get('error'):
            print(result['error'], flush=True)
report['complete'] = True
report['stress_source_unchanged'] = hashlib.sha256(module.read_bytes()).hexdigest() == protocol['source_sha256']['benchmarks/transfer_suite/stress_tasks.py']
report['reference_solvable'] = {
    spec['name']: any(row['task']==spec['name'] and row['result'].get('convergence',{}).get('confirmed_step') is not None
                      for row in report['rows']) for spec in stress_tasks.TASKS}
write_json(out/'results.json', report)
print('COMPLETE reference_solvable='+str(report['reference_solvable']), flush=True)

"""Regrade retained candidate runs and reuse the native-stream controls."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

started = time.perf_counter()
root = Path(__file__).resolve().parent
repo = root.parents[2]
sys.path.insert(0, str(root / 'prepared/repos/cuda'))
from benchmarks.transfer_suite.protocol import test_verdict

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((root / 'prepared/prepared-sources.json').read_text())['cuda']
assert all(sha(root / 'prepared/repos/cuda' / name) == h for name, h in manifest.items())
checks = []
for artifact in sorted((root / 'runs').glob('*/*/result.json')):
    candidate, gate = artifact.parts[-3:-1]
    record = json.loads(artifact.read_text())
    if record['status'] == 'ERROR':
        checks.append(dict(candidate=candidate, gate=gate, setup_error=record.get('error')))
        continue
    code = root / 'candidates' / candidate
    declaration = json.loads((code / 'declaration.json').read_text())
    assert all(sha(code / name) == h for name, h in declaration['code_sha256'].items())
    assert record['worker_sha256'] == declaration['code_sha256']['probe.py']
    assert sha(root / 'prepared/prepared-sources.json') == declaration['prepared_manifest_sha256']
    steps = record['spec']['steps']
    assert record['proof']['adam_calls'] == 2 * steps
    assert sorted(item['calls'] for item in record['proof']['optimizers'].values()) == [steps, steps]
    assert record['particle_update']['updates'] == [steps]
    assert all(row['device'] == row['moment_device'] == 'cuda:0'
               for row in record['particle_update']['trace'])
    assert record['deterministic'] and not record['tf32'] and not record['cpu_random']
    verdict = test_verdict(record['spec'], record['result'])
    assert verdict == record['verdict']
    assert verdict['status'] == record['status']
    control = repo / 'reports/toy100/cpu-recipe-gpu-port/runs/cuda_cpu_init' / (gate + '.json.gz')
    identical_rng = identical_init = None
    if control.exists():
        baseline = json.load(gzip.open(control))
        assert record['config'] == baseline['config']
        assert record['result']['actions'] == baseline['result']['actions']
        identical_rng = record['randomness'] == baseline['randomness']
        identical_init = record['proof']['initial_optimizers'] == baseline['proof']['initial_optimizers']
        assert identical_rng and identical_init
    checks.append(dict(candidate=candidate, gate=gate, status=record['status'],
                       sustained_suffix=verdict['convergence']['passing_suffix'],
                       frozen_steps=steps, actual_adam_calls=record['proof']['adam_calls'],
                       particle_updates=record['particle_update']['updates'],
                       identical_native_random_receipt=identical_rng,
                       identical_cpu_initialization=identical_init))
result = dict(status='PASS', gates=len(checks), prepared_files=len(manifest),
              checks=checks, seconds=time.perf_counter()-started)
(root / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'checks'}), flush=True)
with (repo.parent / 'tests.jsonl').open('a') as stream:
    stream.write(json.dumps(dict(candidate='regression', gate='source_rng_counts_and_frozen_regrade',
                                status='PASS', seconds=result['seconds'],
                                metrics=dict(audited_gates=len(checks)), artifact=str(root / 'audit.json'))) + '\n')

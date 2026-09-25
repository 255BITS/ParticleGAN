"""One GPU worker, immutable declarations, concise gate ledger."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--candidate', required=True)
p.add_argument('--tasks', nargs='+', default=['mode_hold', 'vector_unequal_mass'])
a = p.parse_args()
root = Path(__file__).resolve().parent
repo = root.parents[2]
attempt = repo.parent
supervisor = attempt / 'supervisor.md'
if supervisor.exists():
    steering = supervisor.read_text()
    print(steering.strip(), flush=True)
    if 'STOP' in steering:
        raise SystemExit('Supervisor STOP: batch not launched')
code = root / 'candidates' / a.candidate
declaration = json.loads((code / 'declaration.json').read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert all(sha(code / key) == value for key, value in declaration['code_sha256'].items())
env = os.environ.copy()
env.pop('LD_PRELOAD', None)
env.pop('PYTHONPATH', None)
env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           ATEN_CPU_CAPABILITY='avx2', MKL_ENABLE_INSTRUCTIONS='AVX2',
           ONEDNN_MAX_CPU_ISA='AVX2', DNNL_MAX_CPU_ISA='AVX2',
           CUDA_VISIBLE_DEVICES='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69',
           CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='0')
for task in a.tasks:
    output = root / 'runs' / a.candidate / task
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture = repo / 'reports/toy100/cpu-recipe-gpu-port/initialization-fixtures' / task / 'initial-values.pt'
    assert fixture.exists(), fixture
    command = [sys.executable, '-u', str(code / 'probe.py'),
               '--repo', str(root / 'prepared/repos/cuda'),
               '--config', str(repo / 'configs/toy100/constraints_simple_regularization.json'),
               '--task', task, '--backend', 'cuda', '--initial-state', str(fixture),
               '--output', str(output)]
    launch = dict(candidate=a.candidate, gate=task, command=command,
                  shell_command=shlex.join(command), declaration_sha256=sha(code / 'declaration.json'),
                  fixture_sha256=sha(fixture), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with (root / 'commands.jsonl').open('a') as stream:
        stream.write(json.dumps(launch) + '\n')
    print(json.dumps(dict(event='START', candidate=a.candidate, gate=task,
                          log=str(output.with_suffix('.log')))), flush=True)
    start = time.perf_counter()
    with output.with_suffix('.log').open('w') as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    artifact = output / 'result.json'
    record = json.loads(artifact.read_text()) if artifact.exists() else dict(
        status='ERROR', seconds=time.perf_counter()-start, error=f'exit {completed.returncode}')
    metrics = record.get('result', {}).get('live', {})
    row = dict(candidate=a.candidate, gate=task, status=record['status'],
               seconds=record['seconds'], metrics=metrics, artifact=str(artifact),
               convergence=record.get('verdict', {}).get('convergence'),
               adam_calls=record.get('proof', {}).get('adam_calls'),
               error=record.get('error'))
    with (attempt / 'tests.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps({k:v for k,v in row.items() if k != 'metrics'} | {'metrics': {
        k:v for k,v in metrics.items() if isinstance(v, (int, float))}}), flush=True)

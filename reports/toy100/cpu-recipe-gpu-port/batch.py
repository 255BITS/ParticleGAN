"""Run fixed-seed backend controls three at a time, preserving every result."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--gpu', default='1')
p.add_argument('--profiles', nargs='+', default=['cpu', 'cuda', 'cuda_cpu_random'])
p.add_argument('--tasks', nargs='+', default=[
    'trajectory', 'img_intensity2', 'img_bars4', 'img_blobs4', 'mode_hold', 'vector_unequal_mass'])
a = p.parse_args()
jobs = [(profile, task) for task in a.tasks for profile in a.profiles
        if not (a.root / 'runs' / profile / task / 'result.json').exists()]


def run(job):
    profile, task = job
    output = a.root / 'runs' / profile / task
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(Path(__file__).with_name('probe.py')),
               '--repo', str(a.root / 'repos' / profile), '--config', str(a.config.resolve()),
               '--task', task, '--backend', 'cpu' if profile == 'cpu' else 'cuda',
               '--output', str(output)]
    if profile == 'cuda_cpu_random':
        command.append('--cpu-random')
    if profile == 'cuda_cpu_init':
        command.extend(['--initial-state', str(a.root / 'initialization-fixtures' / task / 'initial-values.pt')])
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    env.pop('PYTHONPATH', None)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               ATEN_CPU_CAPABILITY='avx2', MKL_ENABLE_INSTRUCTIONS='AVX2',
               ONEDNN_MAX_CPU_ISA='AVX2', DNNL_MAX_CPU_ISA='AVX2',
               CUDA_VISIBLE_DEVICES='' if profile == 'cpu' else a.gpu,
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='0')
    log = output.with_suffix('.log')
    with log.open('w') as stream:
        code = subprocess.call(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if not (output / 'result.json').exists():
        raise RuntimeError(f'{profile}/{task}: process {code}, see {log}')
    record = json.loads((output / 'result.json').read_text())
    return dict(profile=profile, task=task, status=record['status'], seconds=record['seconds'],
                live=record.get('result', {}).get('live'), command=command, log=str(log),
                error=record.get('error'))


with ThreadPoolExecutor(max_workers=3) as pool:
    futures = [pool.submit(run, job) for job in jobs]
    for future in as_completed(futures):
        record = future.result()
        with (a.root / 'ledger.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
print(json.dumps(dict(event='COMPLETE', runs=len(jobs))), flush=True)

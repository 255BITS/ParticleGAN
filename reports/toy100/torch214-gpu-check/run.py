"""Isolated 2.14/cu126 comparison on the two outstanding baseline blockers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = Path('/ml2/hypergan/ParticleGAN-epsilon-gan-followup')
SOURCE = Path('/ml2/hypergan/cpu-recipe-port-prepare-check/repos/cuda')
WORKER = REPO / 'reports/toy100/cpu-recipe-gpu-port/probe.py'
CONFIG = REPO / 'configs/toy100/constraints_simple_regularization.json'
env = os.environ.copy()
env.pop('LD_PRELOAD', None)
env.pop('PYTHONPATH', None)
env.update(CUDA_VISIBLE_DEVICES='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           ATEN_CPU_CAPABILITY='avx2', MKL_ENABLE_INSTRUCTIONS='AVX2',
           ONEDNN_MAX_CPU_ISA='AVX2', DNNL_MAX_CPU_ISA='AVX2', PYTHONHASHSEED='0')
os.environ.update({k: env[k] for k in ['CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG']})
import torch
assert torch.__version__ == '2.14.0+cu126' and torch.cuda.is_available()
assert Path(torch.__file__).is_relative_to(sys.prefix)
environment = dict(torch=torch.__version__, torch_file=torch.__file__, python=sys.version,
                   cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                   gpu=torch.cuda.get_device_name(0), gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                   dependencies={p: importlib.metadata.version(p) for p in
                       ['triton', 'torchvision', 'nvidia-cublas-cu12', 'nvidia-cudnn-cu12',
                        'nvidia-cuda-runtime-cu12', 'numpy']},
                   worker_sha256=hashlib.sha256(WORKER.read_bytes()).hexdigest(),
                   config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                   source_manifest_sha256=hashlib.sha256((SOURCE.parents[1]/'prepared-sources.json').read_bytes()).hexdigest())
(ROOT / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
manifest = json.loads((SOURCE.parents[1] / 'prepared-sources.json').read_text())['cuda']
assert all(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == h for name, h in manifest.items())


def run(profile, task):
    output = ROOT / 'runs' / profile / task
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(WORKER), '--repo', str(SOURCE), '--config', str(CONFIG),
               '--task', task, '--backend', 'cuda', '--output', str(output)]
    if profile == 'cpu_init':
        command += ['--initial-state', str(REPO / 'reports/toy100/cpu-recipe-gpu-port/initialization-fixtures' / task / 'initial-values.pt')]
    with output.with_suffix('.log').open('w') as log:
        code = subprocess.call(command, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if not (output / 'result.json').exists():
        raise RuntimeError(f'{profile}/{task} process failed: {code}')
    record = json.loads((output / 'result.json').read_text())
    row = dict(profile=profile, task=task, status=record['status'], seconds=record['seconds'],
               live=record.get('result', {}).get('live'), command=command, artifact=str(output/'result.json'))
    print(json.dumps(row), flush=True)
    return row


with ThreadPoolExecutor(max_workers=3) as pool:
    jobs = [pool.submit(run, profile, task)
            for task in ['mode_hold', 'vector_unequal_mass'] for profile in ['native', 'cpu_init']]
    results = [future.result() for future in as_completed(jobs)]
sys.path.insert(0, str(SOURCE))
from benchmarks.transfer_suite.protocol import test_verdict
for row in results:
    record = json.loads(Path(row['artifact']).read_text())
    assert record['status'] != 'ERROR', record.get('error')
    assert record['torch'] == '2.14.0+cu126'
    assert record['proof']['adam_calls'] == 2 * record['spec']['steps']
    assert all(value['device'] == 'cuda:0' for value in record['proof']['optimizers'].values())
    assert test_verdict(record['spec'], record['result']) == record['verdict']
summary = dict(status='COMPLETE', evidence_audit='PASS', results=results,
               scope='Two blockers, two initialization profiles; no 22-toy score')
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('COMPLETE; all four verdicts and CUDA update counts audited', flush=True)

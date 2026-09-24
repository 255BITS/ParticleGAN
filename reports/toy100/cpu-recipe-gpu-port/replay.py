"""Replay one backend control from the retained source archives."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from prepare import prepare

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--workdir', type=Path, required=True)
p.add_argument('--gpu', default='0')
p.add_argument('--profile', required=True, choices=['cpu', 'cuda', 'cuda_cpu_random', 'cuda_cpu_init'])
p.add_argument('--task', required=True, choices=['trajectory', 'img_intensity2', 'img_bars4',
                                               'img_blobs4', 'mode_hold', 'vector_unequal_mass'])
a = p.parse_args()
bundle = Path(__file__).resolve().parent
root = prepare(a.workdir.resolve())
config = bundle.parents[2] / 'configs/toy100/constraints_simple_regularization.json'
env = os.environ.copy()
env.pop('LD_PRELOAD', None)
env.pop('PYTHONPATH', None)
env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           ATEN_CPU_CAPABILITY='avx2', MKL_ENABLE_INSTRUCTIONS='AVX2',
           ONEDNN_MAX_CPU_ISA='AVX2', DNNL_MAX_CPU_ISA='AVX2', PYTHONHASHSEED='0',
           CUDA_VISIBLE_DEVICES=a.gpu, CUBLAS_WORKSPACE_CONFIG=':4096:8')
if a.profile == 'cuda_cpu_init':
    command = [sys.executable, str(bundle / 'probe.py'), '--repo', str(root / 'repos/cpu'),
               '--config', str(config), '--task', a.task, '--backend', 'cpu', '--init-only',
               '--output', str(root / 'initialization-fixtures' / a.task)]
    subprocess.run(command, env=env | {'CUDA_VISIBLE_DEVICES': ''}, check=True)
command = [sys.executable, str(bundle / 'batch.py'), '--root', str(root), '--config', str(config),
           '--gpu', a.gpu, '--profiles', a.profile, '--tasks', a.task]
subprocess.run(command, env=env, check=True)
result = json.loads((root / 'runs' / a.profile / a.task / 'result.json').read_text())
raise SystemExit(2 if result['status'] == 'ERROR' else 0)

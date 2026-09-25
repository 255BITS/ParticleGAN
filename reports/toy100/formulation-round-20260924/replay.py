"""Replay one retained formulation with verified archived sources and a fresh output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

bundle = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--lane', choices=['critic_formulation', 'particle_geometry', 'game_dynamics', 'followups'], required=True)
p.add_argument('--candidate', required=True)
p.add_argument('--task', choices=['mode_hold', 'vector_unequal_mass'], required=True)
p.add_argument('--workdir', type=Path, required=True)
p.add_argument('--gpu', default='1')
a = p.parse_args()
parent = bundle / a.lane if a.lane == 'followups' else bundle / a.lane / 'candidates'
candidate = (parent / a.candidate).resolve()
if candidate.parent != parent.resolve() or not candidate.is_dir():
    p.error('Choose a retained candidate directory')
declaration = json.loads((candidate / 'declaration.json').read_text())
hashes = declaration.get('code_sha256', declaration.get('file_hashes'))
if hashes is None:
    hashes = {'probe.py': declaration['probe_sha256'], 'config.json': declaration['config_sha256']}
for name, expected in hashes.items():
    assert hashlib.sha256((candidate / name).read_bytes()).hexdigest() == expected, name
sys.path.insert(0, str(bundle.parent / 'cpu-recipe-gpu-port'))
from prepare import prepare
root = prepare(a.workdir.resolve())
manifest_lane = 'critic_formulation' if a.lane == 'followups' else a.lane
expected = json.loads((bundle / manifest_lane / 'prepared-sources.json').read_text())
assert json.loads((root / 'prepared-sources.json').read_text()) == expected
config = candidate / 'config.json'
if not config.exists():
    config = bundle.parents[2] / 'configs/toy100/constraints_simple_regularization.json'
fixture = bundle.parent / 'cpu-recipe-gpu-port/initialization-fixtures' / a.task / 'initial-values.pt'
env = os.environ.copy()
for key in ('PYTHONPATH', 'LD_PRELOAD'):
    env.pop(key, None)
env.update(CUDA_VISIBLE_DEVICES=a.gpu, CUBLAS_WORKSPACE_CONFIG=':4096:8',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           PYTHONHASHSEED='0', ATEN_CPU_CAPABILITY='avx2', MKL_ENABLE_INSTRUCTIONS='AVX2',
           ONEDNN_MAX_CPU_ISA='AVX2', DNNL_MAX_CPU_ISA='AVX2')
command = [sys.executable, str(candidate / 'probe.py'), '--repo', str(root / 'repos/cuda'),
           '--config', str(config), '--task', a.task, '--backend', 'cuda',
           '--initial-state', str(fixture), '--output', str(root / 'result')]
subprocess.run(command, env=env, check=True)
result = json.loads((root / 'result/result.json').read_text())
print(json.dumps({'task': a.task, 'candidate': a.candidate, 'status': result['status']}))

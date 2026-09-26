"""Seed-shifted full training run for vector_unequal_mass (no frozen-code edits).

Same remap shim as seed_capture.py ({0:S,1:S+1,2:S+2}, eval 990/991/402
fixed), then exec the frozen probe source end-to-end on CUDA with the
per-seed CPU fixture. Mechanism/latent/response/checkpoint come from the
exact ka2 copy in /tmp/opencode/rt_ka2_mass.
"""
import hashlib
import os
import pathlib
import subprocess
import sys

SEED = int(sys.argv[1])
FIX = pathlib.Path(sys.argv[2])
OUT = pathlib.Path(sys.argv[3])
PROBE = pathlib.Path('/tmp/opencode/rt_ka2_mass/probe.py')
REPO = pathlib.Path('/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda')
CONFIG = pathlib.Path('/tmp/opencode/rt_ka2_mass/config.json')

source = PROBE.read_text()
preamble = (
    'import torch as _torch\n'
    'import torch\n'
    f'_SEED = {SEED}\n'
    '_REMAP = {0: _SEED, 1: _SEED + 1, 2: _SEED + 2}\n'
    '_orig_manual_seed = _torch.manual_seed\n'
    'def _mapped_manual_seed(s):\n'
    '    return _orig_manual_seed(_REMAP.get(s, s))\n'
    '_torch.manual_seed = _mapped_manual_seed\n'
    'torch.manual_seed = _mapped_manual_seed\n'
    '_OrigGen = _torch.Generator\n'
    'class _MappedGen(_OrigGen):\n'
    '    def manual_seed(self, s):\n'
    '        return super().manual_seed(_REMAP.get(s, s))\n'
    '_torch.Generator = _MappedGen\n'
    'torch.Generator = _MappedGen\n'
)
assert 'import torch\n' in source
source = source.replace('import torch\n', preamble, 1)
tmp = pathlib.Path('/tmp/opencode/rt_ka2_mass') / f'_run_s{SEED}.py'
tmp.write_text(source)
env = dict(os.environ)
env['CUDA_VISIBLE_DEVICES'] = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
env['OMP_NUM_THREADS'] = '1'
cmd = [sys.executable, '-u', str(tmp), '--repo', str(REPO),
       '--task', 'vector_unequal_mass', '--output', str(OUT),
       '--config', str(CONFIG), '--backend', 'cuda',
       '--initial-state', str(FIX)]
r = subprocess.run(cmd, env=env, capture_output=True, text=True)
print(r.stdout[-3000:])
print(r.stderr[-3000:])
print('rc=', r.returncode)

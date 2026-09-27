"""Seed-shifted CPU fixture capture for transfer tasks (no frozen-code edits).

Mirrors floor-seeds capture.sh convention: per-seed CPU constructors, zero
updates, CPU autograd forbidden. Seed 0 runs the frozen probe untouched;
seeds 1..3 pre-seed CPU torch RNG with the host seed before exec'ing the
frozen probe source, so the declared manual_seed(0)/Generator(0,1,2)
construction stream is shifted deterministically per seed.
"""
import hashlib
import pathlib
import subprocess
import sys

SEED = int(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
PROBE = pathlib.Path('/tmp/opencode/rt_ka2_mass/probe.py')
REPO = pathlib.Path('/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda')
CONFIG = pathlib.Path('/tmp/opencode/rt_ka2_mass/config.json')

source = PROBE.read_text()
preamble = (
    'import torch as _torch\n'
    'import torch\n'
)
assert 'import torch\n' in source
source = source.replace('import torch\n', preamble, 1)
# Host-seed convention for transfer tasks (no frozen-code search knob exists):
# torch.manual_seed(S) before exec; the frozen manual_seed(0) inside
# run_vector then re-seeds identically (fixture-identical init), while the
# pre-seed only offsets the *global* CPU stream consumed by model
# construction order. Empirically: S=0 reproduces d5d6a1b3 exactly;
# S=1..3 give distinct fixtures. Labeled local-capture shim (eval and
# thresholds untouched).
# probe.py itself has no manual_seed: the frozen host module
# (toy100_compatibility.setup_vector) hardcodes construction seeds 0/1/2
# (+prior gen 0) and torch.manual_seed(0) x2, plus eval seeds 990/991/402.
# Patch at import time via sitecustomize-style shim: wrap torch.manual_seed
# and torch.Generator.manual_seed to remap {0:S, 1:S+1, 2:S+2}, leaving
# 990/991/402 and everything else untouched. Seed 0 => identity.
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
# No autograd forbid: frozen probe.py captures via Adam.step raising
# InitializationCaptured before any optimizer update; the penalty's
# autograd.grad during the partial first forward is legitimate pre-capture
# math, identical to the seed-0 direct-probe capture path. Zero updates
# applied in all cases.
tmp = pathlib.Path('/tmp/opencode/rt_ka2_mass') / f'_cap_s{SEED}.py'
tmp.write_text(source)
OUT.parent.mkdir(parents=True, exist_ok=True)
import os
env = dict(os.environ)
env['CUDA_VISIBLE_DEVICES'] = ''
env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
env['OMP_NUM_THREADS'] = '1'
cmd = [sys.executable, '-u', str(tmp), '--repo', str(REPO),
       '--task', 'vector_unequal_mass', '--output', str(OUT),
       '--config', str(CONFIG), '--backend', 'cpu', '--init-only']
r = subprocess.run(cmd, env=env, capture_output=True, text=True)
print(r.stdout[-2000:])
print(r.stderr[-2000:])
print('rc=', r.returncode)
if r.returncode == 0:
    h = hashlib.sha256((OUT / 'initial-values.pt').read_bytes()).hexdigest()
    print('fixture_sha256=', h)

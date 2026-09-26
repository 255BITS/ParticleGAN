"""Ortho-init K3P task runner. Usage: ortho_run.py PLAN_NAME JOBS WALL_SECONDS
Plans: init (determinism + inventory), prio0 (8 priority gates, offset 0), full0 (remaining 14 toys + staggered100, offset 0),
       seeds (8 priority gates x offsets 101..707)."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
W = Path('/tmp/k3p-ortho-20260925'); SRC = W / 'repo'; T = SRC / 'reports/toy100'; K = T / 'gap-fill-20260925/sources/k3p'
REPO = W / 'prep/repos/cuda'; PY = '/tmp/k3p-audit-20260925/venv/bin/python'; SHIM = str(T / 'ortho-init-k3p/orthoshim.py')
PLAN, JOBS, DEADLINE = sys.argv[1], int(sys.argv[2]), time.time() + float(sys.argv[3])
VARIANTS = os.environ.get('K3P_ORTHO_VARIANTS', os.environ.get('K3P_ORTHO_VARIANT', 'qr')).split(':')
for _v in VARIANTS: (W / 'out' / _v / 'logs').mkdir(parents=True, exist_ok=True)
TOYS = ['ae_gan_hold', 'cover_leftover', 'img_bars4', 'img_blobs4', 'img_intensity2', 'img_stripes2', 'mid_scale_identity', 'mode_hold',
        'residual_student', 'trajectory', 'two_pole', 'unipolar', 'unused_token_hold', 'vector_two_broad', 'vector_unequal_mass',
        'vector_unequal_width', 'vector_anisotropic', 'vector_overlap', 'vector_spiral']
PRIO = [('toy', 'mode_hold'), ('hold', 'mode_hold'), ('toy', 'vector_unequal_mass'), ('toy', 'img_stripes2'), ('toy', 'img_blobs4'),
        ('native', 'grid100'), ('native', 'rotated100'), ('shift', 'mode_hold')]
REST = [('toy', t) for t in TOYS if ('toy', t) not in PRIO] + [('native', 'staggered100')]
def cmd(kind, t):  # no --initial-state: ortho init replaces the CPU-built fixture
    if kind == 'native':
        return [str(K / 'native100.py'), '--repo', str(REPO), '--task', t, '--candidate', str(K)]
    if kind in ('hold', 'shift'):
        return [str(K / f'{kind}.py'), '--repo', str(REPO), '--task', 'mode_hold', '--config', str(K / 'config.json'), '--backend', 'cuda',
                '--network-floor', '0.01', '--prior-floor', '0.05']
    return [str(K / 'probe.py'), '--repo', str(REPO), '--task', t, '--config', str(K / 'config.json'), '--backend', 'cuda']
tasks = []
if PLAN == 'init1':
    tasks = [('init-s0', 0, k, t, True) for k, t in PRIO + REST if k != 'shift'] + [('init-s101', 101, k, t, True) for k, t in PRIO + REST if k != 'shift']
elif PLAN == 'init':
    for off in (0, 101):
        for kind, t in PRIO + REST:
            if kind == 'shift': continue
            tasks.append((f'init-s{off}', off, kind, t, True))
elif PLAN == 'prio0':
    tasks = [('s0', 0, k, t, False) for k, t in PRIO]
elif PLAN == 'full0':
    tasks = [('s0', 0, k, t, False) for k, t in REST]
elif PLAN == 'early':
    tasks = [('s0', 0, k, t, False) for k, t in (('toy', 'mode_hold'), ('toy', 'vector_unequal_mass'))]
elif PLAN == 'screen':
    tasks = [(f's{o}', o, 'toy', 'mode_hold', False) for o in (0, 101, 202, 303)] + [(f's{o}', o, 'toy', 'vector_unequal_mass', False) for o in (0, 101)]
elif PLAN == 'ringseeds':
    tasks = [(f's{o}', o, 'toy', 'mode_hold', False) for o in (101, 202, 303, 404, 505, 606, 707)]
elif PLAN == 'unequalseeds':
    tasks = [(f's{o}', o, 'toy', 'vector_unequal_mass', False) for o in (101, 202, 303, 404, 505, 606, 707)]
elif PLAN == 'seeds':
    tasks = [(f's{o}', o, k, t, False) for o in (101, 202, 303, 404, 505, 606, 707) for k, t in PRIO]
tasks = [(var,) + x for x in tasks for var in VARIANTS] if PLAN in ('early', 'init1', 'ringseeds', 'unequalseeds', 'screen') else [(VARIANTS[0],) + x for x in tasks]
env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONHASHSEED='0',
           ATEN_CPU_CAPABILITY='avx2', MKL_ENABLE_INSTRUCTIONS='AVX2', ONEDNN_MAX_CPU_ISA='AVX2', DNNL_MAX_CPU_ISA='AVX2', K3P_ORTHO='1')
env.pop('PYTHONPATH', None); env.pop('LD_PRELOAD', None)
assert env.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')
T0 = time.time()
def run(x):
    var, v, off, kind, t, init_only = x
    OUT = W / 'out' / var
    o = OUT / v / f'{kind}-{t}'
    done = o / ('ortho-init.json' if init_only else 'result.json')
    if done.exists(): return
    if time.time() > DEADLINE - 60:
        print(json.dumps(dict(skip=f'{var}/{v}/{kind}-{t}')), flush=True); return
    subprocess.run(['rm', '-rf', str(o)]); o.parent.mkdir(parents=True, exist_ok=True); s = time.time()
    e = dict(env, K3P_ORTHO_VARIANT=var, K3P_SEED_OFFSET=str(off), K3P_INIT_ONLY='1' if init_only else '0')
    with open(OUT / 'logs' / f'{v}-{kind}-{t}.log', 'w') as log:
        rc = subprocess.call([PY, '-u', SHIM] + cmd(kind, t) + ['--output', str(o)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=e)
    print(json.dumps(dict(done=f'{var}/{v}/{kind}-{t}', rc=rc, seconds=round(time.time() - s, 1), t=round(time.time() - T0))), flush=True)
with ThreadPoolExecutor(JOBS) as ex: list(ex.map(run, tasks))
print(json.dumps(dict(event='PLAN_DONE', plan=PLAN, variants=VARIANTS, wall=round(time.time() - T0))), flush=True)

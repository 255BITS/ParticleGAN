"""Frozen paths and static integrity checks for the original CUDA screen lane."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness')
SCREEN = HARNESS / 'screen.py'
PACKAGE = STUDY / 'pkg-CB64-RA4'
CONFIG = STUDY / 'configs/overrides-CB64-RA4.json'
PYTHON = '/tmp/pr38-default-env/bin/python'
GPU0_UUID = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
SCREEN_SHA = 'ee8193adbdf09e93511befae7b6491143c26de88612eddf065cbb92eb2153c3c'
PACKAGE_SHA = 'e34bcb21aaa64caa0601cea5dc1f9b8eaebee9578686ebff39b459676063deb2'
CONFIG_SHA = 'd2b1018854671ded2ffe92b2f30c97a3871cf15911c6c241ceaadd281d94e7e7'
PORTABILITY = ('mode_hold', 'img_intensity2', 'img_blobs4', 'img_stripes2', 'img_bars4',
    'vector_two_broad', 'vector_unequal_mass', 'vector_unequal_width', 'vector_anisotropic',
    'vector_overlap', 'vector_spiral', 'ring_shift', 'stationary')
NATIVE = ('grid100', 'rotated100', 'staggered100')
TASKS = PORTABILITY + NATIVE
OPTIONS = dict(eval_output_noise=True, save_final_state=True, strict_streams=True, diagnostics=True)
ENV = dict(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1',
    MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
UNSET = ('ABSENT', 'ABSENT_START', 'ABSENT_END', 'LRFREE_NATIVE_TEST_STEPS')

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def package_digest(root=PACKAGE):
    package = Path(root) / 'particlegan'
    digest = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        digest.update(str(path.relative_to(package)).encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()

def task_plan(task):
    if task in NATIVE:
        steps = 7000
        observations = [0, 1, 10, 25, 50, 100] + list(range(250, 7001, 250))
        return dict(steps=steps, num_particles=20000, z_dim=2, batch_size=2048, seed=1234,
            observation_steps=observations, terminal_steps=observations[-5:], terminal_samples=20000,
            holdout_samples=100000, holdout_seeds=dict(target=2835, noise=2836, latent=2837))
    if task == 'mode_hold':
        return dict(steps=1200, num_particles=12, z_dim=4, batch_size=128, seed=0,
            observation_steps=list(range(50, 1201, 50)))
    if task.startswith('img_'):
        spec = read(HARNESS / 'tasks/image_task_specs.json')[task]
        return dict(steps=spec['steps'], num_particles=spec['particles'], z_dim=spec['z_dim'],
            batch_size=spec['batch_size'], seed=0, observation_steps=list(range(25, 601, 25)))
    if task.startswith('vector_'):
        spec = read(HARNESS / 'tasks/vector_task_specs.json')[task]['spec']
        steps = spec['steps']
        return dict(steps=steps, num_particles=spec['particles'], z_dim=spec['z_dim'],
            batch_size=spec['batch'], seed=0, observation_steps=[(i * steps + 23) // 24 for i in range(1, 25)])
    steps = 4600 if task == 'ring_shift' else 7500
    return dict(steps=steps, num_particles=20000, z_dim=2, batch_size=2048, seed=0,
        observation_steps=list(range(10, steps + 1, 10)))

def command(task):
    return [PYTHON, '-u', str(ROOT / 'run_screen.py'), '--package-root', str(PACKAGE),
        '--overrides', str(CONFIG), '--task', task, '--output', str(ROOT / 'runs' / task),
        '--device', 'cuda:0', '--candidate-options', str(ROOT / 'candidate-options.json'), '--cand', 'CB64-RA4']

def verify_frozen(check_ready=True):
    frozen = read(ROOT / 'source-freeze.json')
    failures = []
    for name, item in frozen['files'].items():
        try:
            got = sha(item['path'])
        except OSError as error:
            failures.append(f'{name}: {error}')
            continue
        if got != item['sha256']:
            failures.append(f'{name}: {got} != {item["sha256"]}')
    got = package_digest()
    if got != PACKAGE_SHA:
        failures.append(f'candidate package digest: {got} != {PACKAGE_SHA}')
    if check_ready:
        ready = read(ROOT / 'READY.json')
        for rel, wanted in ready['lane_source_sha256'].items():
            if sha(ROOT / rel) != wanted:
                failures.append(f'lane source changed: {rel}')
    if failures:
        raise RuntimeError('frozen source verification failed: ' + '; '.join(failures))
    return dict(status='VALID', file_count=len(frozen['files']), package_sha256=got,
        config_sha256=sha(CONFIG), screen_sha256=sha(SCREEN))

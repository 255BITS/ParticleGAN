#!/usr/bin/env python
"""Read saved CUDA evidence on CPU; never construct a trainer or run a scorer."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import zipfile

sys.dont_write_bytecode = True
# Even optional tensor inspection must remain a read of CPU-mapped storage.
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
OLD = ROOT.parent / 'feature-cells-config-20260929'
PREV = ROOT.parent / 'scaling-portability-20260929/validation'
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness')
PACKAGE_SHA = '13f5bbe4de824e6899cb28ee4dff8f35d74bbaa6243bfe9c18b8df44d173a1ce'
CONFIG_SHA = 'd2b1018854671ded2ffe92b2f30c97a3871cf15911c6c241ceaadd281d94e7e7'
CHECKPOINTS = [0, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
PORTABILITY = ('mode_hold', 'img_intensity2', 'img_blobs4', 'img_stripes2', 'img_bars4',
               'vector_two_broad', 'vector_unequal_mass', 'vector_unequal_width',
               'vector_anisotropic', 'vector_overlap', 'vector_spiral', 'ring_shift', 'stationary')
NATIVE = ('grid100', 'rotated100', 'staggered100')
NATIVE_STEPS = [0, 1, 10, 25, 50, 100] + list(range(250, 7001, 250))
OPTIONS = dict(eval_output_noise=True, strict_streams=True, save_final_state=True,
               diagnostics=True, evaluation_generate='plain', serial_backward_argument=True,
               initialization='batch_feature_zero', image_prior_perturb=False, ring_frozen_control=False)
EXCLUDED = ['birth_death.last.eval_seconds']
REQUIRED_STATE = {'schema', 'recipe', 'optimizer_options', 'penalty_options', 'device', 'dtype',
                  'models', 'requires_grad', 'optimizers', 'initial_lrs', 'completed_steps',
                  'streams', 'cpu_rng', 'cuda_rng', 'controller', 'output_noise', 'lr_settle',
                  'birth_death', 'row_evidence', 'serial_backward'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def package_map(package):
    return {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob('*.py'))}


def package_digest(package):
    h = hashlib.sha256()
    for p in sorted(package.rglob('*.py')):
        h.update(str(p.relative_to(package)).encode() + b'\0' + p.read_bytes() + b'\0')
    return h.hexdigest()


class Checks:
    def __init__(self):
        self.rows = []
        self.outcomes = {'learned': [], 'replays': [], 'screens': []}
        self.activity = []

    def check(self, name, condition, **details):
        self.rows.append(dict(check=name, ok=bool(condition), **details))

    def guarded(self, name, function):
        try:
            return function()
        except Exception as error:
            self.check(name, False, error_type=type(error).__name__, error=str(error))
            return None


def check_hash_map(checks, name, mapping, base=None):
    failures = []
    for path, expected in mapping.items():
        p = Path(path) if base is None else base / path
        try:
            got = sha(p)
            if got != expected:
                failures.append(dict(path=str(p), expected=expected, actual=got))
        except OSError as error:
            failures.append(dict(path=str(p), error=str(error)))
    checks.check(name, not failures, files=len(mapping), failures=failures)


def sources(checks):
    old = read(OLD / 'review/source-freeze.json')
    package = OLD / 'pkg-CB64-RA/particlegan'
    wanted = {str(Path(p).relative_to('pkg-CB64-RA/particlegan')): v
              for p, v in old['candidate_files'].items() if p.startswith('pkg-CB64-RA/particlegan/')}
    checks.check('candidate complete Python file map', package_map(package) == wanted,
                 actual_files=len(package_map(package)), expected_files=len(wanted))
    checks.check('candidate package digest', package_digest(package) == PACKAGE_SHA)
    checks.check('candidate config digest', sha(OLD / 'configs/overrides-CB64-RA.json') == CONFIG_SHA)
    checks.check('candidate original READY receipt', sha(old['ready_receipt']['path']) == old['ready_receipt']['sha256'])
    baseline = read(AUDIT / 'source-baseline.json')
    checks.check('original source baseline matched prior receipts', baseline['all_match'])
    check_hash_map(checks, 'original sources and real data unchanged',
                   {p: record['expected_sha256'] for p, record in baseline['files'].items()})
    frozen = read(ROOT / 'source-freeze.json')
    check_hash_map(checks, 'coordinator local sources unchanged', frozen['local_sources'], ROOT)
    check_hash_map(checks, 'coordinator external sources unchanged', frozen['external_sources'])
    cpu = frozen['original_cpu_archive']
    checks.check('previous CPU manifest identity', sha(cpu['path']) == cpu['sha256'])
    check_hash_map(checks, 'previous CPU archive unchanged', cpu['files'], OLD)
    for lane, record in frozen['lanes'].items():
        checks.check(f'{lane} READY receipt identity', sha(record['path']) == record['sha256'])
    start_path = ROOT / 'execution-started.json'
    if start_path.exists():
        start = read(start_path)
        checks.check('execution uses common frozen sources', start['source_freeze_sha256'] == sha(ROOT / 'source-freeze.json'))
        checks.check('execution serial GPU policy', start['numerical_gpu_parallelism'] == 1 and start['physical_gpu'] == 0)
    return frozen


def task_plan(task):
    if task in NATIVE:
        return dict(steps=7000, num_particles=20000, z_dim=2, batch_size=2048,
                    observation_steps=NATIVE_STEPS, seed=1234)
    if task == 'mode_hold':
        return dict(steps=1200, num_particles=12, z_dim=4, batch_size=128,
                    observation_steps=list(range(50, 1201, 50)), seed=0)
    if task.startswith('img_'):
        spec = read(HARNESS / 'tasks/image_task_specs.json')[task]
        return dict(steps=600, num_particles=spec['particles'], z_dim=spec['z_dim'],
                    batch_size=spec['batch_size'], observation_steps=list(range(25, 601, 25)), seed=0)
    if task.startswith('vector_'):
        spec = read(HARNESS / 'tasks/vector_task_specs.json')[task]['spec']
        steps = spec['steps']
        return dict(steps=steps, num_particles=spec['particles'], z_dim=spec['z_dim'], batch_size=spec['batch'],
                    observation_steps=[math.ceil(i * steps / 24) for i in range(1, 25)], seed=0)
    steps = 4600 if task == 'ring_shift' else 7500
    return dict(steps=steps, num_particles=20000, z_dim=2, batch_size=2048,
                observation_steps=list(range(10, steps + 1, 10)), seed=0)


def npz_headers(path):
    """Read .npy metadata from saved ZIP members without evaluating cloud metrics."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue
            with archive.open(name) as f:
                if f.read(6) != b'\x93NUMPY':
                    raise ValueError(f'{path}:{name} has no NPY magic')
                major, minor = f.read(2)
                n = int.from_bytes(f.read(2 if major == 1 else 4), 'little')
                header = ast.literal_eval(f.read(n).decode('utf-8' if major == 3 else 'latin1'))
                result[name[:-4]] = dict(shape=list(header['shape']), dtype=header['descr'],
                                        fortran_order=header['fortran_order'])
    return result


def cloud(checks, name, path, count):
    headers = npz_headers(path)
    checks.check(name, {'live', 'ema', 'target'}.issubset(headers)
                 and all(headers[k]['shape'] == [count, 2] and 'f' in headers[k]['dtype']
                         for k in ('live', 'ema', 'target')), path=str(path), arrays=headers)


def native(checks, task, out, result):
    fixture = read(HARNESS / 'tasks/native100_fixture.json')
    actual = read(out / 'native-fixture.json')
    matches = {role: {k: actual['initial'][role][k] == v for k, v in values.items()}
               for role, values in fixture['expected_parameters'].items()}
    checks.check(f'{task} canonical CUDA initialization', all(all(v.values()) for v in matches.values())
                 and actual['prior_range'] == fixture['prior_range'] and actual['G_kept'] and actual['prior_kept'],
                 matches=matches, prior_range=actual['prior_range'])
    checks.check(f'{task} native fixture sidecar parity', result['native_fixture'] == actual)
    for kind in ('noisy', 'clean'):
        prefix = f'{task}/{kind}'
        directory = out / f'native-{kind}'
        config, summary = read(directory / 'config.json'), read(directory / 'summary.json')
        checks.check(prefix + ' declared canonical resources',
                     all(config[k] == v for k, v in dict(steps=7000, seed=1234, num_particles=20000,
                     z_dim=2, batch_size=2048, eval_samples=20000, snapshot_samples=4096,
                     threads=1, device='cuda:0').items()) and summary['config'] == config)
        checks.check(prefix + ' complete observation schedule', summary['completed_steps'] == 7000
                     and summary['eval_steps'] == NATIVE_STEPS and summary['snapshot_steps'] == NATIVE_STEPS)
        checks.check(prefix + ' independent accuracy declaration',
                     summary['accuracy_check_steps'] == NATIVE_STEPS[-5:]
                     and summary['accuracy']['check_steps'] == NATIVE_STEPS[-5:]
                     and summary['accuracy']['protocol'] == 'toy100-accuracy-v1'
                     and summary['accuracy']['sample_count'] == 20000
                     and summary['accuracy']['holdout_samples'] == 100000
                     and summary['accuracy']['holdout_seed_offsets'] == dict(target=1601, noise=1602, latent=1603))
        events = lines(directory / 'events.jsonl')
        for model in ('live', 'ema'):
            steps = [r['step'] for r in events if r.get('event') == 'eval' and r.get('model') == model]
            checks.check(prefix + f' {model} event schedule', steps == NATIVE_STEPS, events=len(steps))
        for step in NATIVE_STEPS:
            cloud(checks, prefix + f' snapshot {step}', directory / 'snapshots' / f'step_{step:06d}.npz', 4096)
        for step in NATIVE_STEPS[-5:]:
            cloud(checks, prefix + f' terminal cloud {step}', directory / 'quality_checks' / f'step_{step:06d}.npz', 20000)
        cloud(checks, prefix + ' final cloud', directory / 'final_samples.npz', 20000)
        cloud(checks, prefix + ' holdout cloud', directory / 'holdout_samples.npz', 100000)
        verdict = read(directory / 'verdict.json')
        coverage, accuracy = verdict['coverage'], verdict['accuracy']
        checks.check(prefix + ' unchanged official scorer receipt', verdict['sources'] == fixture['host_source_sha256'])
        checks.check(prefix + ' official evidence valid', coverage['status'] in ('PASS', 'FAIL')
                     and accuracy['status'] in ('PASS', 'FAIL'),
                     coverage_status=coverage['status'], accuracy_status=accuracy['status'],
                     coverage_reason=coverage.get('reason'), accuracy_reason=accuracy.get('reason'))
        terminal = accuracy.get('terminal_checks', [])
        checks.check(prefix + ' official terminal flags', [r['step'] for r in terminal] == NATIVE_STEPS[-5:]
                     and all(type(r['passed']) is bool for r in terminal))
        if accuracy['status'] == 'PASS':
            h = accuracy['holdout_metrics']
            checks.check(prefix + ' official PASS prerequisites', coverage['status'] == 'PASS'
                         and all(r['passed'] for r in terminal) and h['frozen_pass'] and h['accuracy_pass'])
        if kind == 'noisy':
            checks.check(f'{task} primary live noisy status', result['status'] == accuracy['status']
                         and result['native']['coverage_status'] == coverage['status']
                         and result['native']['accuracy_status'] == accuracy['status']
                         and result['native']['terminal_accuracy'] == [r['passed'] for r in terminal])
        else:
            checks.check(f'{task} secondary clean status', result['clean_status'] == accuracy['status'])


def screen(checks, task):
    out = ROOT / 'screens/runs' / task
    result, execution = read(out / 'result.json'), read(out / 'execution-receipt.json')
    plan, header = task_plan(task), result.get('header', {})
    status = result['status']
    checks.outcomes['screens'].append(dict(task=task, status=status, completed_steps=result.get('completed_steps'),
                                         observations=result.get('observations'), warnings=result.get('warnings', []),
                                         error=result.get('error')))
    checks.check(task + ' original wrapper source', execution['original_screen'] == str(HARNESS / 'screen.py'))
    checks.check(task + ' execution source integrity',
                 execution['source_integrity_before']['status'] == 'VALID'
                 and execution['source_integrity_after']['status'] == 'VALID')
    checks.check(task + ' device and candidate receipt', header.get('device') == 'cuda:0'
                 and header.get('cuda_visible_devices') == '0' and header.get('package_sha256') == PACKAGE_SHA
                 and header.get('package_root') == str(OLD / 'pkg-CB64-RA'))
    checks.check(task + ' exact original scorer flags', header.get('options') == OPTIONS,
                 resolved_options=header.get('options'))
    checks.check(task + ' exact frozen overrides', header.get('overrides') == read(OLD / 'configs/overrides-CB64-RA.json'))
    checks.check(task + ' strict streams intact', result.get('stream_deviations') == 0,
                 deviations=result.get('stream_deviations'))
    checks.check(task + ' resource and environment guard',
                 execution['resources']['physical_gpu'] == 0 and execution['resources']['cuda_memory_fraction'] == .2
                 and execution['resources']['numeric_threads'] == 1
                 and all(execution['unset_environment'].values()))
    if status not in ('PASS', 'FAIL'):
        checks.check(task + ' completed quality evidence', False, status=status, error=result.get('error'))
        return
    rows = lines(out / 'metrics.jsonl')
    checks.check(task + ' full update and observation budget', result['completed_steps'] == plan['steps']
                 and result['observations'] == len(plan['observation_steps'])
                 and [r['step'] for r in rows] == plan['observation_steps'])
    checks.check(task + ' frozen recipe resources',
                 all(result['recipe'][k] == plan[k] for k in ('num_particles', 'z_dim', 'batch_size')))
    checks.check(task + ' saved final state', (out / 'final-state.pt').is_file())
    checks.check(task + ' saved execution result identity', execution['result_sha256'] == sha(out / 'result.json'))
    for row in rows:
        counters = row.get('diag', {}).get('birth_death', {}).get('counters', {})
        if counters:
            checks.activity.append(dict(lane='screen', task=task, step=row['step'], counters=counters))
    if task in NATIVE:
        native(checks, task, out, result)


def learned(checks, problem, variant):
    name = f'{problem}/{variant}'
    directory = ROOT / 'learned/training' / problem / variant
    result_path = directory / 'result.json'
    if not result_path.exists() and (directory / 'error.json').exists():
        error = read(directory / 'error.json')
        checks.outcomes['learned'].append(dict(problem=problem, variant=variant, status='ERROR',
                                              error=error.get('error'), phase=error.get('phase'),
                                              completed_steps=error.get('completed_steps')))
        checks.check(name + ' completed learned budget', False, error=error.get('error'))
        return
    result, config = read(result_path), read(directory / 'config.json')
    checks.outcomes['learned'].append(dict(problem=problem, variant=variant, status=result['status'],
                                          steps=result['steps'], training_seconds=result['training_seconds']))
    rows = lines(directory / 'metrics.jsonl')
    previous = read(PREV / 'runs' / problem / 'E22/config.json')
    initial_keys = ('initial_generator_sha256', 'initial_critic_sha256', 'initial_prior_sha256')
    checks.check(name + ' archived and current initial tensor identity',
                 all(config[k] == previous[k] for k in initial_keys),
                 initial={k: config[k] for k in initial_keys})
    checks.check(name + ' current CUDA and seed receipt', config['device'] == 'cuda:0'
                 and result['device'] == 'cuda:0' and config['seed'] == 314159
                 and config['serial_backward'] and config['prior_learnable']
                 and config['previous_gpu_initialization_verified'])
    runtime = config['runtime']
    checks.check(name + ' runtime guard', runtime['device'] == 'cuda:0'
                 and runtime['physical_gpu']['physical_index'] == 0
                 and runtime['physical_gpu']['uuid'] == 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
                 and runtime['visible_devices'] == '0' and runtime['cuda_memory_fraction'] == .2
                 and runtime['cpu_threads'] == 2 and runtime['interop_threads'] == 1
                 and runtime['deterministic_algorithms'] and not runtime['tf32_matmul']
                 and not runtime['tf32_cudnn'] and not runtime['cudnn_benchmark']
                 and runtime['cublas_workspace_config'] == ':4096:8')
    selected = config['package']
    checks.check(name + ' frozen selected source and config',
                 sha(selected['config_path']) == selected['config_sha256']
                 and package_digest(Path(selected['package_root']) / 'particlegan') == selected['package_sha256'])
    wanted_recipe = read(selected['config_path'])
    wanted_recipe.update(z_dim=128, num_particles=1024, batch_size=128)
    checks.check(name + ' only declared recipe dimensions override frozen config',
                 all(config['recipe'][k] == value for k, value in wanted_recipe.items()))
    checks.check(name + ' full learned budget and checkpoints', result['status'] == 'COMPLETE'
                 and config['steps'] == 2000 and result['steps'] == 2000
                 and [r['step'] for r in rows] == CHECKPOINTS)
    times = [r['training_seconds'] for r in rows]
    checks.check(name + ' training timing receipts', times[0] == 0 and all(math.isfinite(t) and t >= 0 for t in times)
                 and times == sorted(times) and times[-1] == result['training_seconds']
                 and result['whole_run_seconds'] >= result['training_seconds']
                 and result['peak_gpu_allocated_bytes'] > 0 and result['peak_gpu_reserved_bytes'] > 0)
    checks.check(name + ' final metric receipt parity', result['final']['step'] == 2000
                 and result['final']['metrics'] == rows[-1]['metrics'])
    for step in CHECKPOINTS:
        path = directory / f'checkpoint-{step:04d}.pt'
        checks.check(name + f' checkpoint {step} file identity', path.is_file()
                     and sha(path) == result['checkpoint_sha256'][path.name])
    for row in rows:
        counters = row.get('diagnostics', {}).get('birth_death', {}).get('counters', {})
        if counters and variant == 'CB64-RA':
            checks.activity.append(dict(lane='learned', task=name, step=row['step'], counters=counters))
    if problem == 'toy':
        metrics = rows[-1]['metrics']
        gate = metrics['precision'] >= .9 and metrics['coverage'] == 25 and metrics['mass_tv'] <= .1
        checks.outcomes['learned'][-1]['independent_toy_quality_gate'] = 'PASS' if gate else 'FAIL'
    else:
        evaluator = read(directory / 'evaluator.json')
        checks.check(name + ' real only evaluator dimensions', evaluator['active_dimensions'] == 39
                     and evaluator['accuracy'] >= .97
                     and evaluator['evaluator_model_sha256'] == sha(PREV / 'evaluator.pt'))
        checks.check(name + ' original raw and active metrics', all(
            all(k in row['metrics'] for k in ('class_mass_tv', 'confident_class_coverage',
                'confident_fraction', 'mean_classifier_confidence', 'pixel_clipping_fraction',
                'raw_embedding', 'active_embedding')) for row in rows))


class TensorReader:
    """Read trusted local checkpoints with all storage mapped to CPU.

    Storage origin tags preserve the replay digest's recorded devices without
    initializing CUDA or moving a tensor to CUDA.
    """
    def __init__(self):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        self.origins = {}

    def load(self, path):
        self.origins = {}
        def remap(storage, location):
            self.origins[storage._cdata] = location
            return storage
        return self.torch.load(path, map_location=remap, weights_only=False)

    def digest(self, value, path=(), semantic=False):
        h = hashlib.sha256()
        def token(x):
            b = x if isinstance(x, bytes) else str(x).encode()
            h.update(str(len(b)).encode() + b':' + b)
        def add(x, prefix):
            if isinstance(x, self.torch.Tensor):
                token('tensor'); token(tuple(x.shape)); token(x.dtype)
                token(self.origins.get(x.untyped_storage()._cdata, str(x.device)))
                token(x.detach().contiguous().reshape(-1).view(self.torch.uint8).numpy().tobytes())
            elif isinstance(x, dict):
                items = [(k, v) for k, v in x.items() if not (semantic and prefix == ('birth_death', 'last') and k == 'eval_seconds')]
                token('dict'); token(len(items))
                for k, v in sorted(items, key=lambda item: (type(item[0]).__name__, repr(item[0]))):
                    add(k, prefix + ('<key>',)); add(v, prefix + (k,))
            elif isinstance(x, (list, tuple)):
                token(type(x).__name__); token(len(x))
                for v in x:
                    add(v, prefix)
            elif isinstance(x, float):
                token('float64'); token(struct.pack('!d', x))
            else:
                token(type(x).__name__); token(repr(x))
        add(value, path)
        return h.hexdigest()

    def raw(self, tensor):
        return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()

    def model(self, values):
        h = hashlib.sha256()
        for key, value in values.items():
            h.update(key.encode()); h.update(value.detach().contiguous().numpy().tobytes())
        return h.hexdigest()

    def rng(self, state):
        values = {'cpu_rng': state['cpu_rng'], 'cuda_rng': state['cuda_rng'], **state['streams']}
        values['birth_death.stream'] = state['birth_death']['stream']
        return all(isinstance(x, self.torch.Tensor) and x.dtype == self.torch.uint8
                   and self.origins.get(x.untyped_storage()._cdata) == 'cpu' for x in values.values())


def inspect_initial_pair(checks, reader, problem):
    fingerprints = []
    for variant in ('E22', 'CB64-RA'):
        directory = ROOT / 'learned/training' / problem / variant
        config = read(directory / 'config.json')
        checkpoint = reader.load(directory / 'checkpoint-0000.pt')
        state = checkpoint['trainer']
        checks.check(f'{problem}/{variant} initial saved state scope', state['device'] == 'cuda:0'
                     and state['completed_steps'] == 0 and checkpoint['data_position'] == 0
                     and checkpoint['receipt_sha256'] == sha(directory / 'config.json')
                     and REQUIRED_STATE.issubset(state) and reader.rng(state))
        actual = dict(initial_generator_sha256=reader.model(state['models']['G']),
                      initial_critic_sha256=reader.model(state['models']['D']),
                      initial_prior_sha256=reader.raw(state['models']['prior']['z']))
        checks.check(f'{problem}/{variant} saved initial tensor receipt parity',
                     all(config[k] == value for k, value in actual.items()), computed=actual)
        fingerprints.append({k: reader.digest(state[k]) for k in ('models', 'streams', 'cpu_rng', 'cuda_rng')})
    checks.check(f'{problem} paired initial model and RNG state parity', fingerprints[0] == fingerprints[1])
    if problem == 'mnist':
        a = read(ROOT / 'learned/training/mnist/E22/evaluator.json')
        b = read(ROOT / 'learned/training/mnist/CB64-RA/evaluator.json')
        keys = ('accuracy', 'active_dimensions', 'active_mask_sha256', 'active_mean_sha256',
                'active_std_sha256', 'raw_reference_sha256', 'active_reference_sha256', 'real_vs_real')
        checks.check('MNIST paired evaluator and real control parity', all(a[k] == b[k] for k in keys))


def replay(checks, reader, problem, variant):
    name = f'replay {problem}/{variant}'
    path = ROOT / 'learned/replay' / problem / variant / 'result.json'
    aggregate = read(ROOT / 'learned' / f'replay-{variant}.json')[problem]
    if aggregate['status'] == 'ERROR':
        checks.outcomes['replays'].append(dict(problem=problem, variant=variant, status='ERROR', error=aggregate.get('error')))
        checks.check(name + ' completed CUDA continuation', False, error=aggregate.get('error'))
        return
    result = read(path)
    checks.outcomes['replays'].append(dict(problem=problem, variant=variant, status=result['status']))
    checks.check(name + ' aggregate parity', result == aggregate)
    checks.check(name + ' continuation scope', result['device'] == 'cuda:0' and result['start_step'] == 1000
                 and result['steps_replayed'] == 10 and result['excluded_observational_fields'] == EXCLUDED
                 and result['checkpoint_sha256'] == sha(result['checkpoint']))
    keys = set(result['semantic_sections_bit_identical'])
    checks.check(name + ' complete semantic sections', REQUIRED_STATE.issubset(keys))
    per_update = result['per_update_comparison']
    checks.check(name + ' complete per update equality record', [r['step'] for r in per_update] == list(range(1001, 1011))
                 and all(set(r['semantic_sections_bit_identical']) == keys for r in per_update))
    branches = result['branches']
    checks.check(name + ' two saved endpoint branches', len(branches) == 2)
    observed = []
    for branch in branches:
        endpoint = Path(branch['endpoint'])
        checks.check(name + f" branch {branch['branch']} endpoint identity", sha(endpoint) == branch['endpoint_sha256'])
        saved = reader.load(endpoint)
        state = saved['trainer']
        full = reader.digest(state)
        semantic = reader.digest(state, semantic=True)
        sections = {k: reader.digest(value, (k,), semantic=True) for k, value in state.items()}
        losses = reader.digest(saved['loss_tensors'])
        checks.check(name + f" branch {branch['branch']} payload fingerprint parity", full == branch['whole_state_sha256']
                     and semantic == branch['semantic_state_sha256'] and sections == branch['semantic_sections']
                     and losses == branch['losses_sha256'])
        checks.check(name + f" branch {branch['branch']} saved CUDA endpoint scope", state['device'] == 'cuda:0'
                     and state['completed_steps'] == 1010 and saved['data_position'] == 2 * 1010 * 128
                     and saved['source_checkpoint_sha256'] == result['checkpoint_sha256'] and reader.rng(state)
                     and len(saved['loss_tensors']) == 10)
        observed.append(dict(semantic=semantic, sections=sections, losses=losses))
    actual_same = observed[0] == observed[1]
    recorded_same = result['semantic_state_bit_identical'] and result['losses_bit_identical']
    checks.check(name + ' endpoint equality matches independent payload read', actual_same == recorded_same)
    required = (actual_same and all(result['semantic_sections_bit_identical'].values())
                and result['restoration_semantic_bit_identical']
                and all(r['losses_bit_identical'] and r['semantic_state_bit_identical']
                        and all(r['semantic_sections_bit_identical'].values()) for r in per_update))
    checks.check(name + ' correctness label agrees with complete evidence',
                 result['status'] == ('PASS' if required else 'FAIL'))


def execution(checks):
    jobs, completed = read(ROOT / 'jobs.json'), read(ROOT / 'execution-results.json')
    expected = [f'learned-{problem}-{variant}' for problem in ('toy', 'mnist') for variant in ('E22', 'CB64-RA')]
    expected += ['replay-E22', 'replay-CB64-RA'] + [f'screen-{task}' for task in PORTABILITY + NATIVE]
    checks.check('exactly the declared serial jobs', [r['name'] for r in jobs] == expected
                 and [r['name'] for r in completed] == expected, jobs=len(jobs), completed=len(completed))
    events = lines(ROOT / 'run.log')
    starts = [r['name'] for r in events if r['event'] == 'job_start']
    ends = [r['name'] for r in events if r['event'] == 'job_complete']
    active = None
    serial = True
    for row in events:
        if row['event'] == 'job_start':
            serial &= active is None
            active = row['name']
        if row['event'] == 'job_complete':
            serial &= active == row['name']
            active = None
    checks.check('one recorded attempt per job with no numerical overlap',
                 starts == expected and ends == expected and serial and active is None)
    for row in completed:
        p = Path(row['result'])
        checks.check(row['name'] + ' coordinator result identity',
                     row['result_sha256'] == (sha(p) if p.exists() else None), returncode=row['returncode'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources-only', action='store_true')
    parser.add_argument('--output', type=Path, default=AUDIT / 'CHECKS.json')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(AUDIT):
        parser.error('audit output must remain within the assigned audit directory')
    checks = Checks()
    checks.guarded('source receipt audit', lambda: sources(checks))
    if not args.sources_only:
        checks.guarded('coordinator execution audit', lambda: execution(checks))
        for problem in ('toy', 'mnist'):
            for variant in ('E22', 'CB64-RA'):
                checks.guarded(f'{problem}/{variant} learned artifact audit',
                               lambda p=problem, v=variant: learned(checks, p, v))
        reader = TensorReader()
        for problem in ('toy', 'mnist'):
            checks.guarded(problem + ' initial checkpoint parity audit',
                           lambda p=problem: inspect_initial_pair(checks, reader, p))
            for variant in ('E22', 'CB64-RA'):
                checks.guarded(f'replay {problem}/{variant} tensor audit',
                               lambda p=problem, v=variant: replay(checks, reader, p, v))
        checks.check('checkpoint inspection never initialized CUDA', not reader.torch.cuda.is_initialized())
        for task in PORTABILITY + NATIVE:
            checks.guarded(task + ' original screen artifact audit', lambda t=task: screen(checks, t))
    failed = [r for r in checks.rows if not r['ok']]
    screens = checks.outcomes['screens']
    complete = len(screens) == 16 and all(r['status'] in ('PASS', 'FAIL') for r in screens)
    canonical = ('PENDING' if args.sources_only else
                 'ERROR' if not complete or failed else
                 'PASS' if all(r['status'] == 'PASS' for r in screens) else 'FAIL')
    activity = dict(ordinary_evaluations=max([int(r['counters'].get('cell_evals', 0)) for r in checks.activity] or [0]),
                    ordinary_moves=max([int(r['counters'].get('ordinary_moves', 0)) for r in checks.activity] or [0]))
    report = dict(audited_at_utc=datetime.now(timezone.utc).isoformat(),
                  mode='sources_only' if args.sources_only else 'final_saved_artifacts',
                  artifact_validity='VALID' if not failed else 'INVALID',
                  canonical_gpu_acceptance=canonical, checks=len(checks.rows), passed=len(checks.rows) - len(failed),
                  failed=failed, outcomes=checks.outcomes,
                  outcome_counts={name: dict(Counter(row['status'] for row in rows)) for name, rows in checks.outcomes.items()},
                  feature_cell_activity=activity, ordinary_reaction_exercised=bool(activity['ordinary_evaluations'] and activity['ordinary_moves']),
                  scope='Read-only saved artifacts; no trainer construction, replay updates, metric recomputation or GPU workload',
                  all_checks=checks.rows)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('artifact_validity', 'canonical_gpu_acceptance', 'checks', 'passed',
                                             'outcome_counts', 'feature_cell_activity')}))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())

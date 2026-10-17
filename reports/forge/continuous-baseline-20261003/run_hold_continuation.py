"""Two preregistered C6 checkpoint continuations; no sweep or automatic retry.

The maintained parent owns admission, leases and immutable source export. A
fresh child imports only the original 8021 scientific namespace, restores the
complete 1,200-update public fixture and performs exactly 150 additional updates.
The original ordinary PASS/study INCOMPLETE remains an immutable input.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
SELF = 'reports/forge/continuous-baseline-20261003/run_hold_continuation.py'
ORIGINAL_ROOT = Path('/ml2/hypergan/ParticleGAN-forge-family-winner-round1')
ORIGINAL_COMMIT = '8021a1c50c4aff90ddea5010d368cffdc857b2f6'
ARCHIVE = Path('/ml2/hypergan/forge-policy-family-round4-20261002')
CASE = 'api-vector-two-broad'
SEED, EVAL_SEED, SAMPLES = 24002, 34002, 4096
START, END, BOUNDARIES = 1200, 1350, (1250, 1300, 1350)
CAP, GRACE, TOTAL_CAP = 180., 60., 480.
PREVIOUS_OUTPUT = Path('/ml2/hypergan/forge-continuous-leaderboard-20261003/hold-continuation')
PREVIOUS_SOURCE = {
    'commit': 'e97cae6d897369354588c29b65b02893ab066484',
    'execution_digest': '29069307aaaefe61b6ab5c2c9b5b26a8b536323570db4054f2abea8f64dd7b29',
    'files_sha256': {SELF: '93342f094a40d1e36df7161b0d250cef90ff7a9ffb965e0041c8365f13e85128'},
}
PREVIOUS_STUDIES = {
    'atlas': {
        'study_sha256': 'e551fe443879ab2a487de8e372c54295206bef3fdb04dc7b7ed0e096956bb54d',
        'request_sha256': 'ecf016dea5ac00bdb6beab89d149cacc53b3ecd52cab467f09b88a904b6e176a',
        'log_sha256': 'c6f7b6c055085ffab8a027b224778e0d27adf904ddadc2645bbf7866f9298cd4',
        'paid_wall_seconds': 3.440489402040839},
    'e22': {
        'study_sha256': '7a3f7c1be32d95c65c95dede31f11c0663f815c09e27ae00dfda39ec6d87b661',
        'request_sha256': '7afeaaf39ccf9b00d317b06099c0b9a6e53896a6e3c7f347f13b9d883afd0684',
        'log_sha256': 'c6f7b6c055085ffab8a027b224778e0d27adf904ddadc2645bbf7866f9298cd4',
        'paid_wall_seconds': 3.3781889399979264},
}
KNOBS = {'lr': .0053125, 'prior_lr_mult': 1.5}
NPZ_SHA = 'c188d10dd45d8263443b4c085833d39aa0e5fe3ef92ae01f3cc910e369f4b28a'
PARENTS = {
    'atlas': {'candidate_id': 'atlas--abe34fb6f0c5f7802cec80e5a2d88aaffb4aa85e55e4021f646457630abec4f0',
              'receipt_sha256': 'e2bfc5fdc66ea2c32e77b22d16a00346a83852b3192d9f4b4ee2da2619da20a6',
              'state_sha256': '95401e3063a5f0bba29cae59835f50bd1e7f8387776944b09aceea5ba4738075'},
    'e22': {'candidate_id': 'e22--5376bd11b0af6ba9ffe3151f3fbc6604e09b8df896869241676f2559e4f730be',
            'receipt_sha256': 'bf6005d2118b3f6994ca240e706a5e8a4e5ef94818f8b54ecc695a6fb00a03dd',
            'state_sha256': 'c68b983b89db276b3e1bc3352c2e176066d84e3d9eef54ac9c49c8201a1d89b8'},
}
FRAME_CONTRACT = {'kind': 'retained_original_and_actual_appended_states',
                  'original_steps': [0, 150, 300, 450, 600, 750, 900, 1050, 1200],
                  'appended_steps': list(BOUNDARIES), 'new_draws_for_old_frames': False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def original(family, *, archive=ARCHIVE):
    if family not in PARENTS:
        raise ValueError('only the two named C6 parents are admitted')
    pin = PARENTS[family]
    directory = Path(archive) / family / pin['candidate_id'] / CASE
    path = directory / 'receipt.json'
    if sha(path) != pin['receipt_sha256']:
        raise ValueError('original C6 receipt changed')
    result = read(path)
    if (result.get('source', {}).get('commit') != ORIGINAL_COMMIT or result.get('seed') != SEED
            or result.get('requested_recipe_overrides') != KNOBS or result.get('recipe', {}).get('name') != family
            or result['recipe'].get('total_steps') is not None or result.get('case', {}).get('id') != CASE
            or result.get('completed_updates') != START or result.get('status') != 'COMPLETE'
            or not all(result.get(k) is True for k in ('passed', 'metric_passed', 'sustained_metric_passed',
                                                      'default_protocol_complete', 'source_unchanged'))
            or result.get('verdict') != 'PASS' or result.get('failed_bounds') != []):
        raise ValueError('original C6 complete PASS/recipe/clock contract differs')
    protocol = result['protocol']
    expected = list(range(0, START + 1, 50))
    if (protocol['updates'] != START or protocol['default_updates'] != START
            or protocol['evaluation_samples'] != SAMPLES or protocol['metric_observations'] != 24
            or protocol['metric_evaluation_steps'] != expected
            or [o['step'] for o in result['observations']] != expected):
        raise ValueError('original full cadence/draw count differs')
    compound_gate(result['observations'], [], completed=START)
    artifacts = {}
    for name in ('final-state.pt', 'observations.npz', 'goal.gif'):
        p = directory / name; declaration = result['artifacts'][name]
        if sha(p) != declaration['sha256'] or p.stat().st_size != declaration['bytes']:
            raise ValueError(f'original artifact changed: {name}')
        artifacts[name] = {'path': str(p), **declaration}
    if artifacts['final-state.pt']['sha256'] != pin['state_sha256'] or artifacts['observations.npz']['sha256'] != NPZ_SHA:
        raise ValueError('original fixed checkpoint/array identity differs')
    return {'family': family, **pin, 'path': str(path), 'artifacts': artifacts, 'receipt': result,
            'original_gate': 'PASS', 'original_study_gate': 'INCOMPLETE'}


def compound_gate(prefix, appended, *, completed):
    """Retain the original first five-pass acquisition; never restart it later."""
    if [o.get('step') for o in prefix] != list(range(0, START + 1, 50)):
        raise ValueError('the original 24 post-update observations must be retained')
    if any(type(o.get('passed')) is not bool for o in prefix + appended):
        raise ValueError('observation pass flags must be binary')
    run = 0; confirmed = None
    for observation in prefix[1:]:
        run = run + 1 if observation['passed'] else 0
        if run == 5:
            confirmed = observation['step']; break
    if confirmed != 1100 or not all(o['passed'] for o in prefix if o['step'] >= confirmed):
        raise ValueError('the named original acquisition/hold prefix differs')
    if [o.get('step') for o in appended] != list(BOUNDARIES[:len(appended)]) or len(appended) > 3:
        raise ValueError('only the three declared appended boundaries are admitted')
    if type(completed) is not int or not START <= completed <= END or any(o['step'] > completed for o in appended):
        raise ValueError('invalid continuation cursor')
    full = completed == END and len(appended) == 3
    status = 'INCOMPLETE' if not full else 'PASS' if all(o['passed'] for o in appended) else 'FAIL'
    return {'status': status, 'passed': status == 'PASS', 'first_confirmed_step': confirmed,
            'original_hold_checks': 2, 'new_hold_checks': len(appended),
            'hold_checks': 2 + len(appended), 'hold_passed': 2 + sum(o['passed'] for o in appended),
            'required_hold_checks': 5, 'full_protocol_complete': full, 'speed_eligible': False,
            'original_gate': 'PASS', 'original_study_gate': 'INCOMPLETE'}


def verify_sources(directory, source, *, check_head=False):
    directory = Path(directory).resolve()
    if source.get('commit') != ORIGINAL_COMMIT or not source.get('files_sha256'):
        raise ValueError('only the exact original 8021 source is admitted')
    if check_head:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory, text=True).strip()
        if head != ORIGINAL_COMMIT:
            raise ValueError('original scientific checkout HEAD differs')
    for name, expected in source['files_sha256'].items():
        p = (directory / name).resolve()
        if not p.is_relative_to(directory) or not p.is_file() or sha(p) != expected:
            raise ValueError(f'original scientific source changed: {name}')


def guard_imports(directory, source, *, modules=None):
    """Same bytes at the wrong path cannot replace the frozen scientific source."""
    directory = Path(directory).resolve()
    protected = ('particlegan', 'benchmarks', 'lib')
    for name, module in (sys.modules if modules is None else modules).items():
        if name.split('.', 1)[0] not in protected:
            continue
        filename = getattr(module, '__file__', None)
        if filename is None:
            # Original lib is a PEP420 namespace, with no __init__.py. Its
            # search path must still identify exactly the exported source;
            # every loaded descendant is checked independently below.
            expected = directory / name.replace('.', '/')
            paths = list(getattr(module, '__path__', ()))
            spec = getattr(module, '__spec__', None)
            spec_paths = list(getattr(spec, 'submodule_search_locations', ()) or ())
            descendants = [key for key in source['files_sha256']
                           if key.startswith(name.replace('.', '/') + '/')]
            if (name.split('.', 1)[0] != 'lib' or len(paths) != 1 or len(spec_paths) != 1
                    or spec is None or spec.origin is not None
                    or Path(paths[0]).resolve() != expected or Path(spec_paths[0]).resolve() != expected
                    or not expected.is_dir() or (expected / '__init__.py').exists()
                    or not descendants or any(not (directory / key).is_file()
                                              or sha(directory / key) != source['files_sha256'][key]
                                              for key in descendants)):
                raise ValueError(f'unbound scientific namespace: {name}')
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f'wrong scientific import path: {name}: {path}')
        key = path.relative_to(directory).as_posix()
        if source['files_sha256'].get(key) != sha(path):
            raise ValueError(f'unbound scientific module bytes: {key}')


def activate_original(directory, source):
    if any(n.split('.', 1)[0] in {'particlegan', 'benchmarks', 'lib'} for n in sys.modules):
        raise ValueError('continuation must start in a fresh scientific namespace')
    verify_sources(directory, source)
    # Remove the maintained snapshot/worktree from scientific module lookup.
    old = Path(directory).resolve()
    sys.path[:] = [str(old)] + [p for p in sys.path if p and not any(
        (Path(p).resolve() / n).is_dir() for n in ('particlegan', 'benchmarks', 'lib'))]


def engineering_history(directory=PREVIOUS_OUTPUT, *, pins=PREVIOUS_STUDIES):
    """Retain the two source-admission failures and debit their measured cost."""
    records = []
    for family in PARENTS:
        study_path = Path(directory) / family / 'study.json'
        pin = pins[family]
        if sha(study_path) != pin['study_sha256']:
            raise ValueError('previous engineering study changed')
        study = read(study_path)
        if (study.get('source') != PREVIOUS_SOURCE or len(study.get('rows', [])) != 1):
            raise ValueError('previous engineering source/row identity differs')
        row = study['rows'][0]; name = f'c6-{family}-broad-hold-1200-to-1350'
        if (row.get('id') != name or row.get('status') != 'INCOMPLETE'
                or row.get('child_returncode') != 1 or row.get('full_protocol_complete') is not False
                or row.get('result_path') is not None or row.get('result_sha256') is not None
                or row.get('timeout_seconds') != CAP or row.get('allowance_seconds') != CAP + GRACE
                or verify_cost(row) != pin['paid_wall_seconds']
                or study.get('spent_seconds') != pin['paid_wall_seconds']):
            raise ValueError('previous engineering completion/cost contract differs')
        artifacts = {}
        for filename, key in (('request.json', 'request_sha256'), ('run.log', 'log_sha256')):
            path = study_path.parent / name / filename
            if sha(path) != pin[key]:
                raise ValueError('previous engineering request/log changed')
            artifacts[filename] = {'path': str(path), 'sha256': pin[key]}
        records.append({'family': family, 'study_path': str(study_path),
                        'study_sha256': pin['study_sha256'], 'source': PREVIOUS_SOURCE,
                        'artifacts': artifacts, 'status': 'INCOMPLETE',
                        'scientific_updates': 0, 'paid_wall_seconds': pin['paid_wall_seconds']})
    return {'reason': 'New explicit engineering cohort repairs strict PEP420 namespace admission only; original scientific source and parents unchanged.',
            'records': records, 'paid_seconds': math.fsum(r['paid_wall_seconds'] for r in records),
            'ordinary_updates': 0, 'automatic_retry': False}


def can_reserve(previous_paid, new_paid):
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in (previous_paid, new_paid)):
        raise ValueError('invalid combined continuation cost')
    return previous_paid + new_paid + CAP + GRACE <= TOTAL_CAP


def plan(root=ROOT, *, original_root=ORIGINAL_ROOT):
    root = Path(root).resolve()
    history = engineering_history()
    parents = {family: original(family) for family in PARENTS}
    source = parents['atlas']['receipt']['source']
    if parents['e22']['receipt']['source'] != source:
        raise ValueError('the two original source manifests differ')
    verify_sources(original_root, source, check_head=True)
    definitions = {}
    for family, parent in parents.items():
        definitions[f'c6-{family}-broad-hold-1200-to-1350'] = {
            'id': f'c6-{family}-broad-hold-1200-to-1350', 'family': family,
            'original_case': CASE, 'original_recipe': parent['receipt']['recipe'],
            'original_receipt_sha256': parent['receipt_sha256'], 'parent_artifacts': parent['artifacts'],
            'original_source': source, 'seed': SEED, 'evaluation_seed': EVAL_SEED,
            'evaluation_samples': SAMPLES, 'start_completed_updates': START,
            'total_execution_limit': END, 'new_updates': END - START,
            'new_observation_steps': list(BOUNDARIES), 'first_confirmed_step': 1100,
            'minimum_post_confirmation_hold_checks': 5,
            'sampling': parent['receipt']['case']['sampling'], 'frame_contract': FRAME_CONTRACT,
            'scope': 'Named persistence extension only; original1200 PASS/studyINCOMPLETE unchanged; no ordinary eight-case, family/default or speed qualification.',
        }
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    return {'schema': 'c6_policy_hold_continuation_v2',
            'spec': {'id': 'c6-two-family-broad-hold-extension-20261003-v2',
                     'representation_card': {'path': parents['atlas']['path'], 'sha256': parents['atlas']['receipt_sha256']},
                     'total_paid_cap_seconds': TOTAL_CAP, 'export_grace_seconds': GRACE, 'frames': 12,
                     'previous_paid_seconds': history['paid_seconds'],
                     'remaining_paid_cap_seconds': TOTAL_CAP - history['paid_seconds'],
                     'engineering_recovery': history,
                     'resources': {'host_memory_mb': 2048}, 'no_automatic_retry': True,
                     'baseline_prerequisite': 'verified original Atlas img_intensity2 control PASS; one-host scope only'},
            'source': {'commit': head, 'files_sha256': {SELF: sha(root / SELF)}},
            'original_source': source, 'original_root': str(Path(original_root).resolve()),
            'case_definitions': definitions, 'parents': parents,
            'rows': [{'id': key, 'family': d['family'], 'status': 'UNKNOWN',
                      'timeout_seconds': CAP, 'allowance_seconds': CAP + GRACE,
                      'case_sha256': digest(d)} for key, d in definitions.items()],
            'status': 'READY', 'spent_seconds': history['paid_seconds'], 'qualification_input': False,
            'default_adoption': False, 'ordinary_eight_case_qualification': False, 'speed_ranking': False}


def prepare(output, *, root=ROOT, queue_root=None):
    from experiments.forge.__main__ import queue_location
    from experiments.forge.policy_execution import freeze_source
    from experiments.forge.sources import snapshot_source, verify_snapshot
    output = Path(output).resolve(); queue_root = queue_location(Path(root), queue_root)
    preparation = output.parent / f'.{output.name}.hold-preparation.json'
    if preparation.exists():
        saved = read(preparation)
        verify_snapshot(Path(saved['execution_source']['snapshot_path']), saved['execution_source'])
        return saved
    packet = plan(root)
    base = freeze_source(root, queue_root, packet['source'])
    with tempfile.TemporaryDirectory(prefix='hold-source-', dir=queue_root) as temporary:
        staging = Path(temporary); files = dict(base['files'])
        for relative in files:
            target = staging / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(base['snapshot_path']) / relative, target)
        for relative, expected in packet['original_source']['files_sha256'].items():
            path = Path(packet['original_root']) / relative
            if sha(path) != expected:
                raise ValueError('original source changed during immutable export')
            name = f'hold-original-source/{relative}'
            target = staging / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target); files[name] = expected
        manifest = {'schema_version': 1, 'digest': digest(files), 'files': files,
                    'origin_commit': packet['source']['commit']}
        snapshot = snapshot_source(staging, queue_root / 'policy/hold-extension', manifest)
    packet.update(execution_source={**manifest, 'snapshot_path': str(snapshot)},
                  scientific_snapshot=str(snapshot / 'hold-original-source'), queue_root=str(queue_root))
    packet['source']['execution_digest'] = manifest['digest']
    write(preparation, packet)
    return packet


def tree_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape:
            return False
        if left.is_floating_point() or left.is_complex():
            return bool(torch.allclose(left.cpu(), right.cpu(), rtol=0, atol=0, equal_nan=True))
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tree_equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    if type(left) is float and type(right) is float and math.isnan(left) and math.isnan(right):
        return True
    return type(left) is type(right) and left == right


def finite_state(value, path=()):
    """Finite trained state, with source-defined missing LR evidence retained.

    continuous.py stores NaN for masked/no-evidence displacement/cosine entries
    and unavailable test statistics; t_b/t_2b may be infinite at zero variance.
    log_bf_b/log_bf_2b use negative infinity before sufficient early evidence.
    These named diagnostic leaves are not model/optimizer health metrics. Their
    values are preserved and compared, never replaced or interpreted as PASS.
    The original public loader still validates the entire controller schema.
    """
    import torch
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point() and not value.is_complex():
            return True
        masked = (len(path) >= 5 and path[:2] == ('trainer', 'lr_settle')
                  and type(path[2]) is int and type(path[3]) is int
                  and path[4] in {'r_b', 'r_2b', 'blocks', 'last_block'})
        return bool((~torch.isinf(value)).all()) if masked else bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(v, path + (k,)) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return all(finite_state(v, path + (i,)) for i, v in enumerate(value))
    if not isinstance(value, float) or math.isfinite(value):
        return True
    diagnostic = (len(path) >= 6 and path[:2] == ('trainer', 'lr_settle')
                  and type(path[2]) is int and type(path[3]) is int
                  and path[4] in {'last', 'last_look', 'log'}
                  and path[-1] in {'t_b', 't_2b', 'mean_r_b', 'mean_r_2b', 'log_bf_b', 'log_bf_2b'})
    return diagnostic and (math.isnan(value) or path[-1] in {'t_b', 't_2b'}
                           or value == -math.inf and path[-1] in {'log_bf_b', 'log_bf_2b'})


def restore(fixture, state, *, initial=START, final=END):
    """Restore under the original cap first; change only the two external caps."""
    if (type(initial) is not int or type(final) is not int or final <= initial
            or fixture.execution_steps != initial or fixture.trainer.max_steps != initial
            or state.get('completed_steps') != initial or state.get('trainer', {}).get('completed_steps') != initial
            or state['trainer'].get('max_steps') != initial or fixture.recipe.total_steps is not None
            or state.get('recipe') != fixture.recipe.to_dict() or not finite_state(state)):
        raise ValueError('original complete state/recipe/external cap differs')
    # The original bytes are immutable; the deserialized input is also kept
    # separate from any controller/optimizer tensor ownership during restore.
    fixture.load_state_dict(deepcopy(state))
    if not tree_equal(fixture.state_dict(), state):
        raise ValueError('complete public fixture restore differs from the checkpoint')
    restored = fixture.state_dict()
    fixture.trainer.extend_execution(final)
    fixture.execution_steps = final
    expected = deepcopy(restored); expected['trainer']['max_steps'] = final
    if not tree_equal(fixture.state_dict(), expected):
        raise ValueError('extension changed state beyond the external trainer cap')
    return {'complete_checkpoint_restore': True, 'only_external_cap_changed': True,
            'original_completed_updates': initial, 'appended_update_limit': final - initial,
            'recipe_schedule_changed': False}


def observe_pure(fixture, *, n=SAMPLES, seed=EVAL_SEED):
    import torch
    from benchmarks.toy_audit import api_contract, api_run
    before = fixture.state_dict()
    modes = {name: model.training for name, model in
             ((name, getattr(fixture.trainer, name)) for name in ('G', 'D', 'prior', 'ema_G', 'ema_prior'))}
    with api_run.isolated_evaluation():
        observed = api_contract.validate_observation(fixture.observe(n=n, seed=seed))
    if not tree_equal(before, fixture.state_dict()) or modes != {
            name: getattr(fixture.trainer, name).training for name in modes}:
        raise ValueError('evaluation changed complete fixture/RNG/controller/model-mode state')
    if fixture.device.type == 'cuda':
        torch.cuda.synchronize(fixture.device)
    return observed


def _runtime(torch, device):
    return {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': str(device),
            'cuda': torch.version.cuda, 'torch_threads': torch.get_num_threads(),
            **({'cuda_device_model': torch.cuda.get_device_name(device)} if str(device).startswith('cuda') else {})}


def validate_arrays(result, arrays, case, score):
    """Recompute the unchanged numerical question from the three actual clouds."""
    import numpy as np
    import torch
    observations = result['observations']
    if [o.get('step') for o in observations] != list(BOUNDARIES):
        raise ValueError('complete three-boundary trace is required')
    expected_keys = {f'step{s}_view{i}_{role}' for s in BOUNDARIES
                     for i in range(3) for role in ('target', 'samples')}
    if set(arrays) != expected_keys:
        raise ValueError('new array/view cadence differs')
    previous = -1.
    for observation in observations:
        step = observation['step']; elapsed = observation.get('elapsed_seconds')
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not previous < elapsed <= CAP:
            raise ValueError('nonfinite/nonincreasing/over-cap new observation timing')
        previous = elapsed
        if len(observation.get('views', [])) != 3:
            raise ValueError('new goal-view metadata differs')
        for index in range(3):
            for role in ('target', 'samples'):
                values = arrays[f'step{step}_view{index}_{role}']
                shape = (2,) if index == 1 else (SAMPLES, 2)
                if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
                    raise ValueError('new view has wrong draw count/dtype or nonfinite values')
        scored = score(case, torch.from_numpy(arrays[f'step{step}_view0_samples']), step)
        if (observation.get('metrics') != scored['metrics'] or observation.get('passed') is not scored['passed']
                or observation.get('failed_bounds') != sorted(set(scored['failed_bounds']))):
            raise ValueError('retained new arrays and original gate/metrics disagree')


def verify_result(request_path):
    """Fresh CPU-only retained-artifact check; no model sampling or updates."""
    request = read(request_path); packet = request['packet']; row = request['row']
    source_root = Path(packet['scientific_snapshot'])
    activate_original(source_root, packet['original_source'])
    import numpy as np
    import torch
    from benchmarks.toy_audit import api_vectors, api_run
    torch.set_num_threads(1)
    parent = original(row['family'])
    if parent != packet['parents'][row['family']]:
        raise ValueError('original parent changed before retained verification')
    result = read(Path(request['target']) / 'result.json')
    for name, artifact in result['artifacts'].items():
        p = Path(request['target']) / name
        if artifact.get('path') != str(p) or sha(p) != artifact['sha256'] or p.stat().st_size != artifact['bytes']:
            raise ValueError('retained child artifact changed')
    with np.load(Path(request['target']) / 'appended-observations.npz', allow_pickle=False) as arrays:
        validate_arrays(result, arrays, parent['receipt']['case'], api_vectors.score_case)
    state = torch.load(Path(request['target']) / 'continued-state.pt', map_location='cpu', weights_only=False)
    old = torch.load(parent['artifacts']['final-state.pt']['path'], map_location='cpu', weights_only=False)
    if (state.get('version') != old['version'] or state.get('case_id') != CASE
            or state.get('completed_steps') != END or state.get('trainer', {}).get('completed_steps') != END
            or state['trainer'].get('max_steps') != END
            or api_run.json_value(state.get('recipe')) != parent['receipt']['recipe']
            or state['trainer'].get('recipe') != old['trainer']['recipe']
            or set(state['trainer'].get('streams', {})) != set(old['trainer']['streams'])
            or set(state.get('trainer', {})) != set(old['trainer'])
            or state['trainer'].get('device') != old['trainer']['device']
            or state['trainer'].get('dtype') != old['trainer']['dtype']
            or state['trainer'].get('serial_backward') is not True
            or state['trainer'].get('optimizer_options') != old['trainer']['optimizer_options']
            or not finite_state(state)):
        raise ValueError('continued complete checkpoint clock/recipe/streams/schema is invalid')
    guard_imports(source_root, packet['original_source'])
    print(json.dumps({'retained_arrays_verified': True, 'optimizer_updates': 0,
                      'sampler_calls': 0, 'scientific_source': ORIGINAL_COMMIT}), flush=True)
    return 0


def child(request_path):
    request = read(request_path); packet = request['packet']; row = request['row']
    target = Path(request['target']); family = row['family']; parent = packet['parents'][family]
    source_root = Path(packet['scientific_snapshot'])
    activate_original(source_root, packet['original_source'])
    import numpy as np
    import torch
    from benchmarks.toy_audit import api_vectors, api_contract, api_run
    guard_imports(source_root, packet['original_source'])
    # Match original api_run.main: one intra-op thread, with its other backend
    # execution settings left at this same-runtime fresh process's defaults.
    torch.set_num_threads(1)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('the scientific child requires physical GPU1 as logical cuda:0')
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    actual_runtime = _runtime(torch, device)
    if actual_runtime != parent['receipt']['runtime']:
        raise ValueError('checkpoint continuation requires its original Python/Torch/CUDA/device/thread runtime')
    target.mkdir(parents=True, exist_ok=True)
    # Whole original envelopes and inputs are fixed hashes, never editable flags.
    verified_parent = original(family)
    if verified_parent != parent:
        raise ValueError('original parent changed between plan and child')
    case = deepcopy(parent['receipt']['case'])
    result = {'schema': 'c6_policy_hold_child_v1', 'family': family, 'case_id': row['id'],
              'scope': packet['case_definitions'][row['id']]['scope'], 'original_gate': 'PASS',
              'original_study_gate': 'INCOMPLETE', 'historical_results_changed': False,
              'original_receipt_sha256': parent['receipt_sha256'], 'original_source': packet['original_source'],
              'maintained_source': packet['source'], 'runtime': actual_runtime,
              'protocol': {'initial_completed_updates': START, 'total_execution_limit': END,
                           'new_updates': 150, 'new_observation_steps': list(BOUNDARIES),
                           'evaluation_samples': SAMPLES, 'evaluation_seed': EVAL_SEED,
                           'acquisition_cap_seconds': CAP, 'export_grace_seconds': GRACE},
              'original_observation_flags': [{'step': o['step'], 'passed': o['passed']} for o in parent['receipt']['observations']],
              'observations': [], 'qualification_input': False, 'default_adoption': False,
              'speed_ranking': False, 'ordinary_eight_case_qualification': False}
    completed = START; fixture = None; arrays = {}; media_records = []
    started = time.monotonic()
    try:
        fixture = api_vectors.build_case(CASE, device=device, seed=SEED, recipe_name=family,
                                          max_steps=START, recipe_overrides=KNOBS)
        if api_run.json_value(fixture.recipe.to_dict()) != parent['receipt']['recipe']:
            raise ValueError('resolved original public recipe differs')
        state = torch.load(parent['artifacts']['final-state.pt']['path'], map_location='cpu', weights_only=False)
        result['restore'] = restore(fixture, state)
        # The single initial draw is a pure checkpoint parity control, not a new
        # observation or a replacement for the original initial/terminal score.
        parity = observe_pure(fixture)
        with np.load(parent['artifacts']['observations.npz']['path'], allow_pickle=False) as old_arrays:
            if parity['metrics'] != parent['receipt']['observations'][-1]['metrics'] or parity['passed'] is not True:
                raise ValueError('restored original1200 metrics differ')
            for index, view in enumerate(parity['views']):
                for role in ('target', 'samples'):
                    if not np.array_equal(api_contract.array(view[role]), old_arrays[f'step1200_view{index}_{role}']):
                        raise ValueError('restored original1200 public sampler differs')
            for old_step in FRAME_CONTRACT['original_steps']:
                record = deepcopy(next(o for o in parent['receipt']['observations'] if o['step'] == old_step))
                for index, view in enumerate(record['views']):
                    for role in ('target', 'samples'):
                        view[role] = old_arrays[f'step{old_step}_view{index}_{role}'].copy()
                media_records.append(record)
        result['checkpoint_sampler_parity'] = {'passed': True, 'evaluations': 1, 'completed_steps': START,
                                               'seed': EVAL_SEED, 'samples': SAMPLES,
                                               'new_scientific_observations': 0}
        while completed < END:
            if time.monotonic() - started >= CAP:
                raise TimeoutError('fixed acquisition allowance exhausted')
            statistics = fixture.step(); completed += 1
            if fixture.completed_steps != completed or fixture.trainer.completed_steps != completed or statistics['step'] != completed:
                raise ValueError('one public step must append exactly one update to the restored cursor')
            if completed not in BOUNDARIES:
                continue
            record = observe_pure(fixture)
            record.update(step=completed, elapsed_seconds=time.monotonic() - started)
            for index, view in enumerate(record['views']):
                for role in ('target', 'samples'):
                    value = api_contract.array(view[role]).copy()
                    arrays[f'step{completed}_view{index}_{role}'] = value; view[role] = value
            media_records.append(record)
            compact = {k: api_run.json_value(v) for k, v in record.items() if k != 'views'}
            compact['views'] = [{k: v for k, v in view.items() if k not in ('target', 'samples')} for view in record['views']]
            result['observations'].append(compact)
            print(json.dumps({'event': 'hold_observation', 'family': family, 'step': completed,
                              'passed': record['passed'], 'elapsed_seconds': record['elapsed_seconds'],
                              'failed_bounds': record['failed_bounds']}, allow_nan=False), flush=True)
        torch.cuda.synchronize(device)
        if time.monotonic() - started > CAP:
            raise TimeoutError('complete acquisition exceeded the fixed allowance')
        result.update(status='COMPLETE', completed_steps=completed, new_updates_completed=completed - START,
                      observer_purity=True, full_protocol_complete=True)
    except Exception as error:
        result.update(status='INCOMPLETE' if isinstance(error, TimeoutError) else 'ERROR',
                      reason=f'{type(error).__name__}: {error}', completed_steps=completed,
                      new_updates_completed=completed - START, full_protocol_complete=False,
                      observer_purity=None if not result['observations'] else True)
    result['acquisition_seconds'] = time.monotonic() - started
    result['compound_hold'] = compound_gate(parent['receipt']['observations'], result['observations'], completed=completed)
    result['passed'] = result['status'] == 'COMPLETE' and result['compound_hold']['passed']
    result['verdict'] = 'PASS' if result['passed'] else 'FAIL'
    result['artifacts'] = {}
    export_start = time.monotonic()
    try:
        if arrays:
            np.savez_compressed(target / 'appended-observations.npz', **arrays)
        if fixture is not None:
            torch.save(fixture.state_dict(), target / 'continued-state.pt')
        if media_records:
            media_case = {**case, 'title': f'{family.upper()} C6 broad: named1200→1350 hold extension',
                          'goal': 'Does the original first acquisition at1100 remain passing for five later checks?',
                          'scope': result['scope'], 'default_steps': END}
            for record in media_records:
                for view in record['views']:
                    view['caption'] = (view.get('caption', '') + '\nOriginal0–1200 retained;1250/1300/1350 are appended actual draws. '
                                       'Original1200 PASS / studyINCOMPLETE unchanged; diagnostic extension only.')
            api_run.render_gif(media_case, media_records, target / 'goal.gif',
                               full_budget=result['full_protocol_complete'], requested_steps=END,
                               final_verdict=f"Named hold {result['verdict']} ({result['status']}); original1200 PASS/studyINCOMPLETE retained")
        result['export_seconds'] = time.monotonic() - export_start
        if result['export_seconds'] > GRACE:
            raise TimeoutError('fixed export allowance exhausted')
        for name in ('appended-observations.npz', 'continued-state.pt', 'goal.gif'):
            path = target / name
            if path.is_file():
                result['artifacts'][name] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
        result['gif_frames'] = len(media_records)
        verify_sources(source_root, packet['original_source']); guard_imports(source_root, packet['original_source'])
        for artifact in parent['artifacts'].values():
            if sha(artifact['path']) != artifact['sha256']:
                raise ValueError('original artifact changed during continuation')
        result['original_inputs_unchanged'] = True
    except Exception as error:
        result.update(status='INCOMPLETE' if isinstance(error, TimeoutError) else 'ERROR',
                      reason=f'export/source verification: {type(error).__name__}: {error}', passed=False, verdict='FAIL')
    write(target / 'result.json', result)
    return 0 if result['status'] == 'COMPLETE' else 1


def baseline_control(path):
    """The one-host positive prerequisite does not claim original19 qualification."""
    ledger_path = Path(path); ledger = read(ledger_path)
    candidates = [row for row in ledger.get('rows', [])
                  if row['id'] == 'atlas-original19-portability-img_intensity2']
    if len(candidates) != 1:
        raise ValueError('verified Atlas original intensity2 control is required before continuation')
    row = candidates[0]
    if row.get('original_gate') != 'PASS' or row.get('status') != 'PASS' or row.get('full_protocol_complete') is not True:
        raise ValueError('the original Atlas baseline control has no complete PASS')
    result_path = Path(row['result_path'])
    if sha(result_path) != row['result_sha256']:
        raise ValueError('baseline original result identity changed')
    for name, artifact in row.get('artifacts', {}).items():
        if sha(result_path.parent / name) != artifact['sha256']:
            raise ValueError('baseline original artifact changed')
    if not row.get('artifacts') or ledger.get('qualification_input') is not False:
        raise ValueError('unbound baseline control')
    # Use the same maintained validator as the baseline owner. A status word or
    # exit0 in a hand-written ledger does not establish the control protocol.
    from experiments.forge.sources import verify_snapshot
    snapshot = Path(ledger['execution_source']['snapshot_path'])
    verify_snapshot(snapshot, ledger['execution_source'])
    owner = _baseline_module(snapshot)
    verified = owner.certify(ledger, row, result_path.parent, row['child_returncode'])
    if (verified['status'] != 'PASS' or verified['original_gate'] != 'PASS'
            or verified['result_sha256'] != row['result_sha256'] or verified['completed_steps'] != 600):
        raise ValueError('baseline original public control verification failed')
    return {'path': str(result_path), 'sha256': row['result_sha256'],
            'ledger_path': str(ledger_path), 'ledger_sha256': sha(ledger_path),
            'maintained_source': ledger['source'],
            'scope': 'one verified original Atlas intensity2 control, not19/19 or current clean-MoG qualification'}


def certify(packet, row, target, returncode):
    """Require exact completion, arrays/state/media and binary compound outcome."""
    path = Path(target) / 'result.json'
    if not path.is_file():
        return {'status': 'INCOMPLETE', 'reason': 'child result unavailable; no retry',
                'original_gate': 'PASS', 'original_study_gate': 'INCOMPLETE', 'full_protocol_complete': False}
    result = read(path); family = row['family']; parent = packet['parents'][family]
    if (result.get('case_id') != row['id'] or result.get('family') != family
            or result.get('original_receipt_sha256') != parent['receipt_sha256']
            or result.get('original_source') != packet['original_source']
            or result.get('maintained_source') != packet['source']
            or result.get('runtime') != packet['lane_runtime']):
        raise ValueError('child source/parent/runtime identity differs')
    if result.get('status') != 'COMPLETE':
        return {'status': 'INCOMPLETE' if result.get('status') == 'INCOMPLETE' else 'ERROR',
                'reason': result.get('reason'), 'result_path': str(path), 'result_sha256': sha(path),
                'original_gate': 'PASS', 'original_study_gate': 'INCOMPLETE', 'full_protocol_complete': False}
    expected_protocol = {'initial_completed_updates': START, 'total_execution_limit': END,
                         'new_updates': 150, 'new_observation_steps': list(BOUNDARIES),
                         'evaluation_samples': SAMPLES, 'evaluation_seed': EVAL_SEED,
                         'acquisition_cap_seconds': CAP, 'export_grace_seconds': GRACE}
    compound = compound_gate(parent['receipt']['observations'], result['observations'], completed=result['completed_steps'])
    if (returncode != 0 or result.get('new_updates_completed') != 150 or result.get('protocol') != expected_protocol
            or result.get('compound_hold') != compound or not compound['full_protocol_complete']
            or result.get('passed') is not compound['passed'] or result.get('verdict') != compound['status']
            or result.get('observer_purity') is not True or result.get('original_inputs_unchanged') is not True
            or result.get('restore', {}).get('complete_checkpoint_restore') is not True
            or result['restore'].get('only_external_cap_changed') is not True
            or result.get('checkpoint_sampler_parity', {}).get('passed') is not True):
        raise ValueError('complete continuation resources/restore/compound gate differ')
    if result.get('original_observation_flags') != [
            {'step': o['step'], 'passed': o['passed']} for o in parent['receipt']['observations']]:
        raise ValueError('original observation flags were rewritten')
    for field, limit in (('acquisition_seconds', CAP), ('export_seconds', GRACE)):
        value = result.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError('missing/nonfinite/over-cap continuation timing')
    if set(result.get('artifacts', {})) != {'appended-observations.npz', 'continued-state.pt', 'goal.gif'} or result.get('gif_frames') != 12:
        raise ValueError('complete continuation arrays/checkpoint/actual goal media unavailable')
    for name, artifact in result['artifacts'].items():
        p = Path(target) / name
        if artifact['path'] != str(p) or sha(p) != artifact['sha256'] or p.stat().st_size != artifact['bytes']:
            raise ValueError('continuation artifact identity differs')
    from PIL import Image
    with Image.open(Path(target) / 'goal.gif') as gif:
        if gif.n_frames != 12:
            raise ValueError('the actual GIF must contain twelve distinct captured states')
    request = Path(target) / 'request.json'
    if not request.is_file() or read(request) != {'packet': packet, 'row': row, 'target': str(Path(target))}:
        # The row gains command/status while running; the immutable scientific
        # fields in the original request still must match the registered row.
        if not request.is_file():
            raise ValueError('immutable continuation request unavailable')
        saved = read(request)
        if (saved.get('packet', {}).get('source') != packet['source']
                or saved['packet'].get('execution_source') != packet['execution_source']
                or saved['packet'].get('scientific_snapshot') != packet['scientific_snapshot']
                or saved['packet'].get('case_definitions') != packet['case_definitions']
                or saved['packet'].get('parents') != packet['parents']
                or saved['packet'].get('original_source') != packet['original_source']
                or saved['row'].get('id') != row['id'] or saved['row'].get('family') != row['family']
                or saved['row'].get('timeout_seconds') != CAP or saved['row'].get('allowance_seconds') != CAP + GRACE
                or saved.get('target') != str(Path(target))):
            raise ValueError('retained continuation request/source/cap identity differs')
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, '-B', str(Path(packet['execution_source']['snapshot_path']) / SELF), '--verify-result', str(request)]
    checked = subprocess.run(command, cwd=packet['execution_source']['snapshot_path'], env=environment,
                             capture_output=True, text=True, timeout=30)
    if checked.returncode != 0:
        raise ValueError('retained numerical/state/source verification failed: ' + checked.stderr[-2000:])
    return {'status': compound['status'], 'original_gate': 'PASS', 'original_study_gate': 'INCOMPLETE',
            'study_gate': compound['status'], 'full_protocol_complete': True,
            'completed_steps': END, 'new_updates': 150, 'result_path': str(path), 'result_sha256': sha(path),
            'artifacts': result['artifacts'], 'compound_hold': compound, 'media': result['artifacts']['goal.gif'],
            'acquisition_seconds': result['acquisition_seconds'], 'export_seconds': result['export_seconds'],
            'qualification_input': False, 'default_adoption': False, 'speed_eligible': False}


def verify_cost(result):
    paid = result.get('paid_wall_seconds', 0.)
    reserved = result.get('unmeasured_interrupt_reserved_seconds', 0.)
    charged = result.get('charged_seconds', paid + reserved)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (paid, reserved, charged)):
        raise ValueError('nonfinite or negative retained continuation cost')
    if not math.isclose(charged, paid + reserved, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError('paid and interrupt reservation do not match charged cost')
    if reserved and (result.get('status') not in {'INCOMPLETE', 'ERROR'}
                     or charged != CAP + GRACE or reserved != CAP + GRACE - paid):
        raise ValueError('an interruption must conservatively retain the exact original total allowance')
    if result.get('full_protocol_complete'):
        minimum = result['acquisition_seconds'] + result['export_seconds']
        if paid + 1e-6 < minimum:
            raise ValueError('retained paid cost is below actual complete child time')
    return charged


def recertify(packet, row):
    if row.get('result_path'):
        if sha(row['result_path']) != row['result_sha256']:
            raise ValueError('retained continuation result changed')
        verified = certify(packet, row, Path(row['result_path']).parent, row.get('child_returncode', 0))
        for field in ('status', 'original_gate', 'original_study_gate', 'full_protocol_complete'):
            if row.get(field) != verified.get(field):
                raise ValueError('retained continuation status differs from verified evidence')
    return verify_cost(row)


def _baseline_module(root):
    path = Path(root) / 'reports/forge/continuous-baseline-20261003/run_atlas_baseline.py'
    spec = importlib.util.spec_from_file_location('maintained_atlas_baseline_owner', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def run(output, *, baseline_ledger, root=ROOT, queue_root=None, max_new_attempts=None):
    """Root-invoked only; each family has its own maintained exclusive study lease."""
    from experiments.forge.policy_execution import PolicyCoordinator
    packet = prepare(output, root=root, queue_root=queue_root)
    if engineering_history() != packet['spec']['engineering_recovery']:
        raise ValueError('preserved engineering prefix differs from prepared recovery cohort')
    baseline = baseline_control(baseline_ledger)
    owner = _baseline_module(root)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('physical GPU1 numeric mask required')
    actual_runtime = owner.runtime()
    if any(actual_runtime != p['receipt']['runtime'] for p in packet['parents'].values()):
        raise ValueError('continuation runtime differs from the original state cohort')
    coordinator = PolicyCoordinator(packet['queue_root'], report_root=Path(root) / 'reports/forge')
    launched = 0; output = Path(output).resolve(); summaries = []
    for family in PARENTS:
        lane = deepcopy(packet)
        lane.update(rows=[r for r in packet['rows'] if r['family'] == family], lane_runtime=actual_runtime,
                    baseline_control=baseline, spec_sha256=digest(packet['spec']))
        key, canonical = coordinator.register(lane, output / family, family, actual_runtime)
        with coordinator.study_lease(key) as study_lease:
            if study_lease is None:
                summaries.append(coordinator.publish_attachment(key, canonical, output / family)); continue
            lane = read(canonical / 'study.json'); row = lane['rows'][0]
            expected = next(r for r in packet['rows'] if r['family'] == family)
            if (len(lane['rows']) != 1 or any(row.get(k) != expected[k] for k in
                    ('id', 'family', 'timeout_seconds', 'allowance_seconds', 'case_sha256'))
                    or lane['spec'] != packet['spec'] or lane['source'] != packet['source']
                    or lane['case_definitions'] != packet['case_definitions']):
                raise ValueError('retained finite continuation source/spec/row cap differs')
            if row['status'] not in {'UNKNOWN', 'RUNNING'}:
                lane['spent_seconds'] = recertify(lane, row)
                summaries.append(lane); continue
            trial = {'family': family, 'recipe_overrides': KNOBS}
            attempt = coordinator.attempt_key(lane, trial, row)
            row['attempt_key'] = attempt
            if not coordinator.retained(attempt) and max_new_attempts is not None and launched >= max_new_attempts:
                lane['waiting_reason'] = 'explicit dispatch limit reached'; summaries.append(lane); break
            if not coordinator.retained(attempt) and not can_reserve(
                    packet['spec']['previous_paid_seconds'], sum(s.get('spent_seconds', 0.) for s in summaries)):
                lane['waiting_reason'] = 'remaining two-attempt cap cannot reserve complete allowance'
                write(canonical / 'study.json', lane); summaries.append(lane); break
            if not coordinator.retained(attempt) and not owner.gpu_readiness()['ready']:
                lane['waiting_reason'] = owner.gpu_readiness()['reason']; write(canonical / 'study.json', lane); summaries.append(lane); break
            with coordinator.admit(attempt, lane, row, 'cuda:0') as (admission, lease):
                if admission['status'] == 'busy':
                    lane['waiting_reason'] = admission['reason']; write(canonical / 'study.json', lane); summaries.append(lane); break
                if admission['status'] == 'completed':
                    result = deepcopy(admission['result']); result['reused_physical_attempt'] = True
                    recertify(lane, {**row, **result})
                elif admission['status'] == 'interrupted':
                    result = {'status': 'INCOMPLETE', 'reason': admission['reason'],
                              'paid_wall_seconds': admission.get('terminal', {}).get('paid_wall_seconds', 0.) if admission.get('terminal') else 0.,
                              'charged_seconds': admission['charged_seconds'], 'reused_physical_attempt': True}
                    result['unmeasured_interrupt_reserved_seconds'] = max(0., result['charged_seconds'] - result['paid_wall_seconds'])
                elif admission['status'] == 'awaiting_certification':
                    request = read(admission['command'][-1]); target = Path(request['target'])
                    result = certify(lane, row, target, admission['terminal']['child_returncode'])
                    result.update(paid_wall_seconds=admission['terminal']['paid_wall_seconds'],
                                  charged_seconds=admission['terminal']['paid_wall_seconds'],
                                  child_returncode=admission['terminal']['child_returncode'])
                    verify_cost(result)
                    coordinator.complete(attempt, result)
                else:
                    target = canonical / row['id']
                    if target.exists():
                        result = {'status': 'INCOMPLETE', 'reason': 'orphan artifacts retained; no retry',
                                  'paid_wall_seconds': 0., 'charged_seconds': CAP + GRACE,
                                  'unmeasured_interrupt_reserved_seconds': CAP + GRACE}
                    else:
                        target.mkdir(parents=True); request = target / 'request.json'
                        write(request, {'packet': lane, 'row': row, 'target': str(target)})
                        command = [sys.executable, '-u', '-B', str(Path(lane['execution_source']['snapshot_path']) / SELF), '--child', str(request)]
                        row.update(status='RUNNING', command=command, log_path=str(target / 'run.log'))
                        write(canonical / 'study.json', lane); done = None
                        print(json.dumps({'event': 'start', 'family': family, 'new_updates': 150,
                                          'log': row['log_path']}), flush=True)
                        try:
                            done = coordinator.launch(command, lane, target / 'run.log', (study_lease, lease), CAP + GRACE)
                            result = certify(lane, row, target, done.returncode)
                            result.update(child_returncode=done.returncode, paid_wall_seconds=done.paid_wall_seconds,
                                          charged_seconds=done.paid_wall_seconds)
                        except subprocess.TimeoutExpired as error:
                            paid = getattr(error, 'paid_wall_seconds', 0.)
                            result = {'status': 'INCOMPLETE', 'reason': 'supervised cap exhausted; no retry',
                                      'paid_wall_seconds': paid, 'charged_seconds': paid}
                        except Exception as error:
                            terminal_path = Path(admission['lease_path']).parent / 'supervisor-terminal.json'
                            terminal = read(terminal_path) if terminal_path.exists() else None
                            paid = done.paid_wall_seconds if done is not None else terminal['paid_wall_seconds'] if terminal and terminal.get('token') == admission['token'] else None
                            reserve = CAP + GRACE if paid is None else 0.
                            result = {'status': 'ERROR', 'reason': f'{type(error).__name__}: {error}',
                                      'paid_wall_seconds': paid or 0., 'charged_seconds': (paid or 0.) + reserve,
                                      'unmeasured_interrupt_reserved_seconds': reserve}
                        launched += 1
                    coordinator.complete(attempt, result)
                row.update(result)
                lane.update(status=row['status'], spent_seconds=verify_cost(row),
                            completed=int(row.get('full_protocol_complete', False)))
                write(canonical / 'study.json', lane)
                print(json.dumps({'event': 'complete', 'family': family, 'status': row['status'],
                                  'charged_seconds': lane['spent_seconds']}), flush=True)
            summaries.append(lane)
    new_paid = sum(s.get('spent_seconds', 0.) for s in summaries)
    summary = {'schema': 'c6_hold_continuation_summary_v2', 'status': 'COMPLETE' if len(summaries) == 2 and all(
               s['rows'][0]['status'] not in {'UNKNOWN', 'RUNNING'} for s in summaries) else 'INCOMPLETE',
               'family_studies': summaries, 'total_paid_cap_seconds': TOTAL_CAP,
               'previous_paid_seconds': packet['spec']['previous_paid_seconds'],
               'new_paid_seconds': new_paid,
               'spent_seconds': packet['spec']['previous_paid_seconds'] + new_paid,
               'engineering_recovery': packet['spec']['engineering_recovery'],
               'qualification_input': False, 'default_adoption': False, 'speed_ranking': False}
    if summary['spent_seconds'] > TOTAL_CAP:
        raise ValueError('continuation paid cost exceeded its fixed two-attempt cap')
    write(output / 'hold-continuation.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--output', type=Path); parser.add_argument('--queue-root', type=Path)
    parser.add_argument('--baseline-ledger', type=Path); parser.add_argument('--max-new-attempts', type=int)
    parser.add_argument('--child', type=Path)
    parser.add_argument('--verify-result', type=Path, help='fresh zero-update source-bound retained-artifact verification')
    args = parser.parse_args(argv)
    if args.child:
        return child(args.child)
    if args.verify_result:
        return verify_result(args.verify_result)
    # Direct parent CLI execution must find maintained Forge from any cwd.
    # Fresh scientific child/verification dispatch above owns its original
    # namespace and must never receive this maintained checkout import path.
    sys.path.insert(0, str(ROOT))
    if args.plan:
        print(json.dumps(plan(), indent=2, allow_nan=False)); return 0
    if args.output is None:
        parser.error('--output is required')
    if args.max_new_attempts is not None and args.max_new_attempts < 1:
        parser.error('dispatch limit must be positive')
    if args.prepare_only:
        result = prepare(args.output, queue_root=args.queue_root)
    else:
        if args.baseline_ledger is None:
            parser.error('--baseline-ledger with verified original Atlas control PASS is required')
        owner = _baseline_module(ROOT)
        for key, value in owner.ENVIRONMENT.items():
            if key == 'CUDA_VISIBLE_DEVICES' and os.environ.get(key) not in (None, '1'):
                parser.error('physical GPU1 numeric mask required')
            os.environ[key] = value
        result = run(args.output, baseline_ledger=args.baseline_ledger, queue_root=args.queue_root,
                     max_new_attempts=args.max_new_attempts)
    print(json.dumps({'status': result['status'], 'spent_seconds': result.get('spent_seconds', 0.),
                      'qualification_input': False}, allow_nan=False))
    return 0 if args.prepare_only or result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())

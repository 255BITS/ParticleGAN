"""Read-only all-step applicability proof for the upstream noise-floor change."""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parents[1]
REPO = Path('/ml2/hypergan/ParticleGAN-ra11-pr155')
OLD = 'f459cb6d6aaaabeb1af076ec53ad7a963618de90'
NEW = 'cabe2084284db923d525918cbf3e18de6f20faac'
PACKAGE = STUDY / 'pkg-RA16-portability/particlegan'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):h.update(block)
    return h.hexdigest()


def blob(commit, name):
    return subprocess.check_output(['git', 'show', f'{commit}:particlegan/{name}'], cwd=REPO).decode()


assert not torch.cuda.is_initialized()
torch.set_default_device('cpu')
torch.set_num_threads(1)
old_policy, new_policy = blob(OLD, 'policy.py'), blob(NEW, 'policy.py')
old_continuous, new_continuous = blob(OLD, 'continuous.py'), blob(NEW, 'continuous.py')
for name, value in [('upstream-old-policy.py', old_policy), ('upstream-new-policy.py', new_policy),
                    ('upstream-old-continuous.py', old_continuous), ('upstream-new-continuous.py', new_continuous)]:
    (ROOT / name).write_text(value)


def named(tree, kind, name):
    return next(n for n in ast.walk(tree) if isinstance(n, kind) and n.name == name)


# The floor change does not alter the testers' evolution/counter semantics.
for name in ('SettleTest', 'SequentialSettleTest', 'StationarityLR'):
    a = named(ast.parse(old_continuous), ast.ClassDef, name)
    b = named(ast.parse(new_continuous), ast.ClassDef, name)
    assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    c = named(ast.parse((PACKAGE / 'continuous.py').read_text()), ast.ClassDef, name)
    assert ast.dump(a, include_attributes=False) == ast.dump(c, include_attributes=False)


def output_sigma(source):
    node = named(ast.parse(source), ast.FunctionDef, '_output_sigma')
    env = dict(torch=torch, math=math)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<upstream output_sigma>', 'exec'), env)
    return env['_output_sigma']


old_sigma, new_sigma = output_sigma(old_policy), output_sigma(new_policy)


def evaluate(fn, *, generator, table, noise=1.):
    log_sigma = torch.tensor(math.log(.01), dtype=torch.float64, device='cpu', requires_grad=True)
    owner = SimpleNamespace(recipe=SimpleNamespace(output_noise_mode='learnable'),
        log_output_sigma=log_sigma, controller=SimpleNamespace(mobility=.2),
        roles=[['generator', 'table', 'noise'], ['critic']],
        lr_settle=SimpleNamespace(testers=[
            [SimpleNamespace(s=generator), SimpleNamespace(s=table), SimpleNamespace(s=noise)],
            [SimpleNamespace(s=1.)]]))
    sigma = fn(owner, .029, detach=False)
    return dict(sigma=float(sigma.detach()), derivative=float(torch.autograd.grad(sigma, log_sigma)[0]))


witnesses = []
for table in (1., .5, .25, 1. / 64.):
    a, b = evaluate(old_sigma, generator=1. / 64., table=table), evaluate(new_sigma, generator=1. / 64., table=table)
    if table > 1. / 64.:assert a == b
    else:assert a != b
    witnesses.append(dict(generator=1. / 64., table=table, noise=1., old=a, new=b, equal=a == b))

ports = ['stationary', 'mode_hold', 'ring_shift', 'vector_two_broad', 'vector_overlap',
    'vector_spiral', 'vector_anisotropic', 'vector_unequal_mass', 'vector_unequal_width',
    'img_blobs4', 'img_stripes2', 'img_bars4', 'img_intensity2']
selected = []
for task in ports:
    lane = 'validation-ra15' if task == 'ring_shift' else 'validation-ra14'
    selected.append((f'ported/{task}', STUDY / lane / 'runs' / task / 'final-state.pt'))
for task in ('grid100', 'rotated100', 'staggered100'):
    selected.append((f'native/{task}', STUDY / 'validation-ra14-r2/runs' / task / 'final-state.pt'))
    selected.append((f'moving/{task}', STUDY / 'validation-ra15/moving' / task / 'checkpoint-001500.pt'))
for problem in ('toy', 'mnist'):
    selected.append((f'learned/{problem}', STUDY / 'mnist/ra13-settled/training' / problem /
                     'RA13-settled/checkpoint-2000.pt'))
records = []
inputs = {}
for task, path in selected:
    obj = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    state = obj.get('trainer', obj)
    assert state['schema'] == 4 and state['recipe']['lr_control'] == 'stationarity'
    assert state['recipe']['output_noise_mode'] == 'learnable'
    roles = state['policy']['roles']
    assert roles == [['generator', 'table', 'noise'], ['critic']]
    tester_records = []
    for i, (row, owner_roles) in enumerate(zip(state['lr_settle'], roles)):
        if i == 1:continue
        for j, (tester, role) in enumerate(zip(row, owner_roles)):
            assert tester is not None
            count = tester['counts']['stationary']
            assert type(count) is int and count >= 0
            bound = 2. ** (-count)
            assert tester['s'] >= bound
            tester_records.append(dict(owner=[i, j], role=role, final_s=tester['s'],
                lifetime_stationary_decisions=count, all_step_s_lower_bound=bound,
                final_population_expiries=tester['counts'].get('population_expiries'),
                final_reopens=tester['counts']['reopens']))
    retained = [x for x in tester_records if x['role'] != 'noise']
    witnesses_for_run = [x for x in retained if x['all_step_s_lower_bound'] > 1. / 64.]
    assert witnesses_for_run
    table_record = next(x for x in retained if x['role'] == 'table')
    assert table_record['lifetime_stationary_decisions'] < 6
    checkpoint_sha = sha(path)
    inputs[str(path)] = checkpoint_sha
    records.append(dict(task=task, path=str(path), checkpoint_sha256=checkpoint_sha,
        completed_steps=state['completed_steps'], source_backend=state['backend_selection']['actual_backend'],
        recipe_sha256=hashlib.sha256(json.dumps(state['recipe'], sort_keys=True).encode()).hexdigest(),
        base_sigma=state['recipe']['output_noise_std'], roles=roles, tester_records=tester_records,
        all_step_proof='cumulative stationary-decision bound from fresh s=1, not final-scale interpolation',
        witness_role='table', witness_lifetime_stationary_count=table_record['lifetime_stationary_decisions'],
        all_step_table_s_lower_bound=table_record['all_step_s_lower_bound'],
        old_and_new_settle_override_every_step=1.,
        unaffected_by_noise_floor_change=True, fresh_training_required_for_noise_floor_change=False))
    del state, obj

assert len(records) == 21 and sum(not x['task'].startswith('learned/') for x in records) == 19
# Original replay origins and all four branch ranges lie inside the proved
# 2000-update learned prefix; do not infer a new post-2000 horizon.
replay = (STUDY / 'mnist/ra16-replay/replay.py').read_text()
assert '\nSTART = 1000\n' in replay and '\nCOUNT = 10\n' in replay
learned = [x for x in records if x['task'].startswith('learned/')]
assert all(x['completed_steps'] == 2000 for x in learned)
result = dict(status='PROVEN_UNAFFECTED_FOR_ORIGINAL_FINITE_HORIZONS',
    original_base=OLD, advanced_base=NEW, scoped_change='learnable output-noise floor ignores noise tester',
    scope_limit='Noise-floor applicability only; latest combined-source replay/full suite remain required for compatibility.',
    invariant=dict(initial_s=1., stationary_multiplier=.5,
        stationary_counter_cumulative_and_never_cleared_by_restart=True,
        only_stationary_decision_decreases_s=True,
        drift_reopen_and_population_expiry_do_not_decrease_s=True,
        all_step_bound='s(t) >= 2^(-C_stationary(T)) for every 0 <= t <= T',
        floor_override_equal_if_any_retained_nonnoise_role_above='1/64',
        exactly_six_stationary_decisions_necessary_from_initial_s_one=True),
    actual_upstream_function_boundary_witnesses=witnesses,
    standard_and_sequential_settle_AST_unchanged_across_bases=True,
    original_quality_tasks_proven_unaffected=19, learned_fixture_prefixes_proven_unaffected=2,
    minimum_all_step_table_s_bound=min(x['all_step_table_s_lower_bound'] for x in records),
    tasks_requiring_fresh_training_for_noise_floor_change=[],
    original_learned_replay=dict(start=1000, updates_per_branch=10, branches_per_fixture=2,
        fixtures=2, total_updates=40, within_proved_training_prefix=True,
        latest_combined_source_replay_required=True),
    records=records, inputs=inputs, CUDA_initialized=False,
    history_relabelled=False, GPU_launched=False, models_constructed=False,
    new_training_or_seed_experiments=False)
assert not torch.cuda.is_initialized()
(ROOT / 'receipt.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('records', 'inputs')}, sort_keys=True), flush=True)
for record in records:
    print(json.dumps(dict(task=record['task'], updates=record['completed_steps'],
        stationary=record['witness_lifetime_stationary_count'],
        all_step_table_s_bound=record['all_step_table_s_lower_bound'], status='PROVEN_UNAFFECTED'),
        sort_keys=True), flush=True)

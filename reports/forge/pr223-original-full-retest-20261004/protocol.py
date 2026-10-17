"""Pure original19 retest declarations; importing this module creates no learner."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = 'reports/forge/pr223-original-full-retest-20261004'
PROTOCOL = DIRECTORY + '/protocol.json'
REFERENCE = 'reports/develop-gates-20261001/atlas19-replay.json'
CONFIG = 'configs/100gaussians/atlas.json'
LEGACY = 'reports/forge/continuous-baseline-20261003/run_atlas_baseline.py'
SUCCESSFUL_COMMIT = 'a0d6d89fb470f551b3f790016a237c40a377e1e8'
OPTIONS = dict(eval_output_noise=True, save_final_state=True, strict_streams=True,
               diagnostics=True, evaluation_generate='indexed', serial_backward_argument=True,
               initialization='batch_feature_zero', image_prior_perturb=False, ring_frozen_control=False)
ORDER = [('portability', t) for t in ('img_intensity2', 'mode_hold', 'img_blobs4', 'img_bars4',
         'img_stripes2', 'vector_two_broad', 'vector_unequal_mass', 'vector_unequal_width',
         'vector_anisotropic', 'vector_overlap', 'vector_spiral', 'stationary', 'ring_shift')] + [
         (g, t) for g in ('moving', 'native') for t in ('grid100', 'rotated100', 'staggered100')]
CAPS = [150,180,150,150,150,180,180,180,180,180,180,1470,960,390,420,420,1470,1440,1380]
ENVIRONMENT = dict(CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
    CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
    PYTHONUNBUFFERED='1', MPLBACKEND='Agg')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(root=ROOT):
    return validate(json.loads((Path(root) / PROTOCOL).read_text()))


def validate(card):
    if (card.get('schema') != 'pg_pr223_original_full_retest_protocol_v1'
            or card.get('status') != 'DECLARED_NOT_EXECUTED' or card.get('family') != 'atlas'):
        raise ValueError('unknown retest protocol')
    budget = dict(case_caps_sum_seconds=9810, shared_metadata_and_finalization_seconds=180,
                  planned_maximum_seconds=9990, aggregate_cap_seconds=10800, export_grace_seconds=0, retries=0)
    if card.get('budget') != budget:
        raise ValueError('original finite retest budget changed')
    if card.get('runtime') != dict(physical_gpu=1, device='cuda:0', threads=1, memory_fraction=.2,
            min_free_mib=12288, max_temperature_c=82, deterministic=True, tf32=False, autocast=False):
        raise ValueError('physical/runtime contract changed')
    if card.get('claims') != dict(scope='fresh full original19 recipe and served law retest',
            current26_qualification=False, default_adoption=False, speed_ranking=False, old_passes_are_new_credit=False):
        raise ValueError('retest cannot borrow qualification or old outcomes')
    rows = card.get('rows', [])
    if len(rows) != 19 or [(r.get('group'),r.get('task')) for r in rows] != ORDER:
        raise ValueError('all19 original ordered questions required')
    for i, (row, cap) in enumerate(zip(rows, CAPS), 1):
        group, task = ORDER[i-1]
        if row.get('id') != f'pr223-faithful19-retest-v1-{group}-{task}' or row.get('ordinal') != i:
            raise ValueError('original ordered case identity changed')
        original = row['original_definition']
        if digest(original) != row['historical_case_sha256']:
            raise ValueError('original scientific case fingerprint differs')
        if (original['id'] != f'atlas-original19-{group}-{task}' or original['group'] != group
                or original['task'] != task or original['original_options'] != OPTIONS
                or original['added_policy_hold_gate'] is not False or original['original_terminal_gate'] is not True):
            raise ValueError('original host/serving/gates changed')
        if row['proposed_inclusive_allowance_seconds'] != cap:
            raise ValueError('case cap changed')
        paid = row['historical_paid_seconds']
        if type(paid) not in (int,float) or not math.isfinite(paid) or paid < 0 or math.ceil((1.5*paid+90)/30)*30 != cap:
            raise ValueError('fixed cap rationale not bound to historical paid cost')
        clocks = original['observation_steps']
        if clocks != sorted(set(clocks)) or original['original_host']['steps'] != clocks[-1]:
            raise ValueError('original complete cadence changed')
        steps = ([0,500,1000,1500] if group == 'moving' else
                 [10,1000,2000,2400,2410,2500,3000,4000,4600] if task == 'ring_shift' else
                 [clocks[j*(len(clocks)-1)//8] for j in range(9)])
        if row['media_steps'] != steps or (group != 'moving' and not set(steps) <= set(clocks)):
            raise ValueError('goal media must use fixed existing original observation clocks')
        expected = deepcopy(card['common_resolved_recipe'])
        host = original['original_host']
        expected.update({k:host[k] for k in ('num_particles','z_dim','batch_size')})
        if row['resolved_recipe'] != expected or expected['total_steps'] is not None:
            raise ValueError('complete original Recipe/resources required; no injected horizon')
    if sum(r['original_definition']['original_host']['steps'] for r in rows) != 48800:
        raise ValueError('original full update budget differs')
    return card


def source_equivalence(root=ROOT, *, read_blob=None):
    """Compare all package bytes and the two scientific RA15 files, not branding."""
    root = Path(root)
    blob = read_blob or (lambda ref: subprocess.check_output(['git','show',ref], cwd=root))
    names = subprocess.check_output(['git','ls-tree','-r','--name-only',SUCCESSFUL_COMMIT,'particlegan'], cwd=root, text=True).splitlines()
    current = sorted(p.relative_to(root).as_posix() for p in (root/'particlegan').rglob('*.py'))
    if sorted(n for n in names if n.endswith('.py')) != current:
        raise ValueError('successful and current complete package file lists differ')
    names = current + [CONFIG] + [
        'reports/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930/validation-ra15/'+p
        for p in ('screen_current.py','current_api_fixtures.py')]
    pins = {}
    for name in names:
        old = hashlib.sha256(blob(f'{SUCCESSFUL_COMMIT}:{name}')).hexdigest()
        if sha(root/name) != old:
            raise ValueError('successful scientific source changed: '+name)
        pins[name] = old
    # The maintained legacy driver can have reporting-only repairs. Bind its
    # exact scientific functions structurally, excluding the old metric renderer.
    old_tree = ast.parse(blob(f'{SUCCESSFUL_COMMIT}:{LEGACY}'))
    new_tree = ast.parse((root/LEGACY).read_text())
    wanted = ('original_inputs','native_requirements','task_definition','moving_source','child','certify')
    def functions(tree):
        return {n.name:ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted}
    if functions(old_tree) != functions(new_tree) or set(functions(new_tree)) != set(wanted):
        raise ValueError('legacy science/gate functions drifted from successful replay')
    return {'successful_commit':SUCCESSFUL_COMMIT,'files_sha256':pins,
            'legacy_scientific_functions':list(wanted),'old_metric_renderer_excluded':True}


def metadata_preflight(root, legacy, card=None):
    """Original source/JSON checks only: no Torch, scorer, snapshot or queue call."""
    card = validate(card or load(root))
    if sha(Path(root)/CONFIG) != card['config_sha256']:
        raise ValueError('original full config bytes differ')
    if json.loads((Path(root)/CONFIG).read_text()) != card['original_config']:
        raise ValueError('full original config differs')
    inputs = legacy.original_inputs(root)
    derived = {}
    for row in card['rows']:
        actual = legacy.task_definition(row['group'],row['task'],inputs)
        # The historical field hashes a dictionary whose keys are absolute
        # inspection paths. A new worktree changes those keys, not the science.
        # Keep both identities; never claim the old fingerprint binds this path.
        previous = deepcopy(row['original_definition'])
        previous['external_inputs_sha256'] = actual['external_inputs_sha256']
        if actual != previous:
            raise ValueError('source-derived original case/seed/gate/cadence differs: '+row['id'])
        derived[row['id']] = actual
    parity = source_equivalence(root)
    return {'status':'PASS_METADATA_ONLY','protocol_sha256':digest(card),'source_parity':parity,
            'cases':19,'updates':48800,'models':0,'sampler_calls':0,'scorer_calls':0,
            'preparation_calls':0,'queue_calls':0,'inputs':inputs,'source_derived_definitions':derived,
            'location_only_external_fingerprints':{
                row['id']:{'historical':row['original_definition']['external_inputs_sha256'],
                           'current':derived[row['id']]['external_inputs_sha256']}
                for row in card['rows']}}

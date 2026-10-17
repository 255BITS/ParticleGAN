"""Execution-blocked RA10 API skeleton; finalize only after owner source freeze."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
API = ROOT / 'quality/ra8/integration-contract/check_api.py'
NATIVE_HOST = Path('/ml2/hypergan/lrfree-20260926/harness/hosts/native100/toy_models.py')
API_SHA = '36ab993ffcd9408ae84f9c35a1d35c41cb6d40176e42f54121878d416cddac61'
EXECUTION_READY = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def guard_map(mapping):
    for path, expected in mapping.items():
        assert sha(path) == expected, path


def load_api_after_guard():
    assert sha(API) == API_SHA
    spec = importlib.util.spec_from_file_location('ra10_frozen_api_util', API)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api


def construct(api, package, saved, case):
    """Build a new host from copied bytes; never adapt a legacy checkpoint."""
    if case == 'toy':
        return api.construct(package, saved)
    assert case == 'grid'
    torch, nn = api.torch, api.nn
    host = ast.parse(NATIVE_HOST.read_text())
    critic = next(n for n in host.body if isinstance(n, ast.ClassDef)
                  and n.name == 'SimpleMLPDiscriminator')
    namespace = dict(torch=torch, nn=nn)
    exec(compile(ast.Module(body=[critic], type_ignores=[]), str(NATIVE_HOST), 'exec'), namespace)
    cls = namespace['SimpleMLPDiscriminator']
    D = cls.__new__(cls)
    nn.Module.__init__(D)
    weights = saved['models']['D']
    D.fourier = len(weights['freqs'])
    D.register_buffer('freqs', weights['freqs'].clone())
    layer_ids = sorted(int(k.split('.')[1]) for k in weights if k.startswith('net.') and k.endswith('.weight'))
    layers = []
    for index, layer in enumerate(layer_ids):
        layers.append(api.linear(weights[f'net.{layer}.weight'], weights[f'net.{layer}.bias']))
        if index + 1 < len(layer_ids):
            layers.append(nn.LeakyReLU(.2, inplace=True))
    D.net = nn.Sequential(*layers)
    G = api.linear(saved['models']['G']['weight'], saved['models']['G']['bias'])
    prior = package.ParticlePrior.__new__(package.ParticlePrior)
    nn.Module.__init__(prior)
    prior.z = nn.Parameter(saved['models']['prior']['z'].clone())
    with torch.random.fork_rng(devices=[]):
        return package.GANTrainer(package.Recipe(**deepcopy(saved['recipe'])), G, D,
            prior=prior, seed=314159, optimizer_options=deepcopy(saved['optimizer_options']),
            penalty_options=deepcopy(saved['penalty_options']),
            serial_backward=saved.get('serial_backward', False))


def owned_view_without_swaps(api, trainer):
    """Atomic-control observations must not invoke public save's view swap."""
    roots = (trainer.G, trainer.D, trainer.prior, trainer.ema_G, trainer.ema_prior)
    geometry = trainer.birth_death.latent_geometry
    entries = {key: (id(value[0]()), value[1], value[2]) for key, value in geometry._entries.items()}
    return dict(state=trainer._state_dict(), fast=deepcopy(trainer._fast),
        parameter_identity_versions=[(id(p), p._version) for root in roots for p in root.parameters()],
        gradients=[(id(p.grad), None if p.grad is None else p.grad.clone())
                   for root in roots for p in root.parameters()],
        modes=[m.training for root in roots for m in root.modules()],
        buffer_identity_versions=[(id(v), v._version) for root in roots for v in root.buffers()],
        cpu_rng=api.torch.get_rng_state().clone(), streams=api.all_streams(trainer),
        snapshot_identity=id(trainer.birth_death.snapshot), geometry_entries=entries,
        geometry_work=dict(geometry.work), moved_rows=trainer.birth_death.moved_rows)


def reject_atomic(api, trainer, good, label, mutate):
    bad = deepcopy(good)
    mutate(bad)
    before = api.fingerprint(owned_view_without_swaps(api, trainer))
    try:
        trainer.load_state_dict(bad)
    except ValueError as error:
        reason = str(error)
    else:
        raise AssertionError(f'{label}: accepted malformed metadata')
    assert before == api.fingerprint(owned_view_without_swaps(api, trainer)), label
    return dict(case=label, rejected=True, atomic=True, reason=reason)


def phase_balances(trainer):
    bd = trainer.birth_death
    last = bd.last
    moment = last['mean_transport']  # Final owner field mapping remains pending.
    assert bd.BACKEND_SCHEMA == 9 and moment['status'] in ('invalid', 'veto', 'firing')
    assert last['count_categories'] == 2 * last['cells']
    assert last['count_multiplicity'] == 3 * last['cells'] + 3
    assert last['count_cutoff'] == .05 / (3 * last['cells'] + 3)
    assert moment['alpha'] == last['count_cutoff']
    assert all(moment[key] == last[target] for key, target in
               (('step', 'step'), ('snapshot', 'snapshot'), ('cells', 'cells'), ('rank', 'metric_rank')))
    assert moment['observations'] == last['calibration_rows']
    phases = [last[f'ordinary_{name}_moves'] for name in ('mass', 'support', 'global', 'mean')]
    assert last['ordinary_copy_moves'] == sum(phases)
    assert last['ordinary_moves'] == last['ordinary_copy_moves'] + last['ordinary_novel_birth_moves']
    assert last['moves'] == last['ordinary_moves'] + last['iso_moves']
    assert moment['moves'] == last['ordinary_mean_moves']
    assert last['ordinary_moves'] <= int(.05 * bd.N)
    json.dumps(bd.diagnostics(), allow_nan=False)
    return dict(mean_status=moment['status'], copy_phase_moves=phases,
                ordinary_moves=last['ordinary_moves'], all_moves=last['moves'])


def cold_roundtrip(api, package, case, state, output):
    subject = construct(api, package, state, case)
    initial = subject.state_dict()
    assert initial['schema'] == 5 and initial['birth_death']['backend_schema'] == 9
    assert initial['birth_death']['last']['mean_transport']['status'] == 'initial'
    subject.load_state_dict(state)  # Authentic owner-created backend9 state only.
    checkpoint = subject.state_dict()
    path = output / 'roundtrip.pt'
    api.torch.save(dict(trainer=checkpoint), path)
    clone = construct(api, package, state, case)
    clone.load_state_dict(api.torch.load(path, map_location='cpu', weights_only=False)['trainer'])
    assert api.fingerprint(api.checkpoint_served_view(subject)) == api.fingerprint(api.checkpoint_served_view(clone))
    assert clone.birth_death.snapshot is None and not clone.birth_death.latent_geometry._entries
    assert clone.birth_death.moved_rows is None
    return subject, clone


def observed_sample(api, trainer):
    rows = []
    previous = trainer.birth_death.perturb_latent
    def trace(latent, stream, controller=None, record=False, *, prior=None, rows=None):
        observed_rows.append(None if rows is None else rows.clone())
        return previous(latent, stream, controller, record, prior=prior, rows=rows)
    observed_rows = rows
    trainer.birth_death.perturb_latent = trace
    try:
        output = trainer.sample(17)
    finally:
        trainer.birth_death.perturb_latent = previous
    assert len(rows) == 1 and rows[0] is not None
    return output, rows[0], trainer.eval_generator.get_state().clone()


def positive_serving_continuation(api, package, case, state, subject, output):
    if not subject._serve_settled():
        assert subject._fast is None
        return dict(status='NOT_ELIGIBLE', synthetic_stamp=False, sample_calls=0)
    assert subject._fast is not None
    observed_sample(api, subject)  # Warm cache and advance the actual saved evaluation cursor.
    checkpoint = subject.state_dict()
    path = output / 'after-first-sample.pt'
    api.torch.save(dict(trainer=checkpoint), path)
    clone = construct(api, package, state, case)
    clone.load_state_dict(api.torch.load(path, map_location='cpu', weights_only=False)['trainer'])
    assert not clone.birth_death.latent_geometry._entries
    left = observed_sample(api, subject)
    right = observed_sample(api, clone)
    assert all(api.torch.equal(a, b) for a, b in zip(left, right))
    assert api.fingerprint(api.checkpoint_served_view(subject)) == api.fingerprint(api.checkpoint_served_view(clone))
    return dict(status='PASS', actual_eligible=True, warm_cold_continuation_exact=True,
                rows_outputs_eval_cursor_and_full_state_exact=True, sample_calls=3, rows_per_call=17)


def two_updates(api, package, state):
    torch = api.torch
    subject = construct(api, package, state, 'toy')
    clone = construct(api, package, state, 'toy')
    subject.load_state_dict(state)
    clone.load_state_dict(state)
    assert state['completed_steps'] < state['recipe']['total_steps']
    assert subject.birth_death.rows_since_eval == 0
    batch = state['birth_death']['reservoir'][:128]
    torch.set_rng_state(state['cpu_rng'].clone())
    left = subject.step(batch)
    left_state = api.served_view(subject)
    torch.set_rng_state(state['cpu_rng'].clone())
    right = clone.step(batch)
    assert api.fingerprint(left) == api.fingerprint(right)
    assert api.fingerprint(left_state) == api.fingerprint(api.served_view(clone))
    assert subject.completed_steps == clone.completed_steps == state['completed_steps'] + 1
    assert subject.birth_death.snapshot_serial == state['birth_death']['snapshot_serial']
    return dict(status='PASS', CPU_updates_total=2, full_state_losses_gradients_and_cursors_exact=True)


def planned_malformed_controls():
    """Pin final names/types once owner fields are frozen; do not run this draft."""
    def bd(state):
        return state['birth_death']
    def moment(state):
        return bd(state)['last']['mean_transport']
    return [
        ('old_backend8', lambda s: bd(s).update(backend_schema=8)),
        ('wrong_trainer_schema', lambda s: s.update(schema=4)),
        ('backend_schema_strict_int', lambda s: bd(s).update(backend_schema=9.0)),
        ('missing_mean_policy', lambda s: bd(s)['settings'].pop('mean_policy')),
        ('old_common_family', lambda s: bd(s)['last'].update(count_multiplicity=3*bd(s)['last']['cells']+2)),
        ('multiplicity_strict_int', lambda s: bd(s)['last'].update(count_multiplicity=float(3*bd(s)['last']['cells']+3))),
        ('old_cutoff', lambda s: bd(s)['last'].update(count_cutoff=.05/(3*bd(s)['last']['cells']+2))),
        ('missing_mean_map', lambda s: bd(s)['last'].pop('mean_transport')),
        ('mean_observations_strict_int', lambda s: moment(s).update(observations=True)),
        ('nonfinite_mean_scalar', lambda s: moment(s).update(lower_bound=float('nan'))),
        ('future_mean_stamp', lambda s: moment(s).update(step=moment(s)['step']+1)),
        ('old_mean_alpha', lambda s: moment(s).update(alpha=.05/(3*moment(s)['cells']+2))),
        ('mean_sign_status_conflict', lambda s: moment(s).update(status='veto' if moment(s)['status']=='firing' else 'firing')),
        ('mean_move_balance', lambda s: moment(s).update(moves=moment(s)['moves']+1)),
    ]


def main():
    if not EXECUTION_READY:
        raise SystemExit('SOURCE-ONLY SKELETON: await owner fields/source freeze, genuine fixtures and final helper seal')
    raise NotImplementedError('Finalize guard/provenance/output orchestration in a NEW helper after stable source')


if __name__ == '__main__':
    main()

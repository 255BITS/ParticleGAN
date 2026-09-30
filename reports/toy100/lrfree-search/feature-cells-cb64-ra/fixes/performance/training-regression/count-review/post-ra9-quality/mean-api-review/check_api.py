"""Independent composed RA10 CPU checkpoint/API contracts after raw guards."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import traceback

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
OWNER = ROOT / 'integration/review/training-regression/post-ra9-quality/mean-category-production'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def guard(mapping):
    for path, digest in mapping.items():
        assert sha(path) == digest, path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def log(event, **fields):
    print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), event=event, **fields)), flush=True)


def raw_guards(inputs):
    assert inputs['status'] == 'PRE_EXECUTION_API_SOURCE_INPUT_FROZEN'
    guard(inputs['protected_sha256'])
    assert inputs['backend_schema'] == 9 and inputs['trainer_schema'] == 5
    owner_ready = json.loads(Path(inputs['owner_ready']).read_text())
    composed = json.loads(Path(inputs['root_composition']).read_text())
    assert owner_ready['status'] == 'FROZEN_CPU_QUALIFIED'
    assert owner_ready['package_source_sha256'] == composed['source_sha256']
    assert owner_ready['package_sha256'] == composed['package_sha256'] == inputs['package_sha256']
    package = Path(inputs['package_root'])
    actual = {str(p.relative_to(package / 'particlegan')): sha(p)
              for p in sorted((package / 'particlegan').rglob('*.py'))}
    assert actual == composed['source_sha256'] and len(actual) == 30
    h = hashlib.sha256()
    for name in sorted(actual):
        h.update(name.encode() + b'\0' + (package / 'particlegan' / name).read_bytes() + b'\0')
    assert h.hexdigest() == inputs['package_sha256']
    assert Path(inputs['config_path']).read_bytes() == (ROOT / 'configs/overrides-CB64-RA9.json').read_bytes()
    assert sha(inputs['config_path']) == inputs['config_sha256']
    hooks = json.loads(Path(inputs['lineage_hook_receipt']).read_text())
    assert hooks['status'] == 'PASS'
    for case, bindings in inputs['fixtures'].items():
        assert case in ('grid', 'toy')
        provenance = json.loads(Path(bindings['provenance']).read_text())
        assert provenance['case'] == case and provenance['device'] == 'cpu'
        assert provenance['construction'] == 'fresh native GANTrainer/backend9; no backend8/full-state load or relabel'
        assert provenance['seed'] == 314159
        assert provenance['no_optimizer_or_GAN_gradient_step'] is True
        assert provenance['completed_steps_before'] == 0 and provenance['completed_steps_after'] == 1
        assert sha(provenance['original_checkpoint']) == provenance['original_checkpoint_sha256']
        guard(provenance['input_binding'])
        for name, digest in provenance['output_sha256'].items():
            assert sha(Path(bindings['provenance']).parent / name) == digest
    return package


def owned_view(api, trainer):
    """No public-save swaps while observing an atomic rejection."""
    roots = (trainer.G, trainer.D, trainer.prior, trainer.ema_G, trainer.ema_prior)
    geometry = trainer.birth_death.latent_geometry
    entries = {key: (id(value[0]()), value[1], value[2]) for key, value in geometry._entries.items()}
    return dict(state=trainer._state_dict(), fast=deepcopy(trainer._fast),
        parameter_ids=[id(p) for root in roots for p in root.parameters()],
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
    served = trainer._fast is not None
    roots = (trainer.G, trainer.D, trainer.prior, trainer.ema_G, trainer.ema_prior)
    parameters = [p for root in roots for p in root.parameters()]
    versions = {id(p): p._version for p in parameters}
    swapped_ids = {id(p) for p in trainer._served_parameters()} if served else set()
    before = api.fingerprint(owned_view(api, trainer))
    try:
        trainer.load_state_dict(bad)
    except ValueError as error:
        reason = str(error)
    else:
        raise AssertionError(f'{label}: accepted malformed metadata')
    assert before == api.fingerprint(owned_view(api, trainer)), label
    for p in parameters:
        # Existing trainer5 releases/reapplies the served view before/after a
        # rejected load. The corresponding cache-version advance is declared.
        expected_delta = 2 if id(p) in swapped_ids else 0
        assert p._version - versions[id(p)] == expected_delta, (label, expected_delta)
    return dict(case=label, rejected=True, semantic_state_view_RNG_and_cache_contents_atomic=True,
        inherited_served_parameter_version_delta=2 if served else 0, reason=reason)


def mutated_bound(stamp, outside_mean):
    """Keep EB arithmetic self-consistent to target only the known-range guard."""
    stamp['mean'] = float(outside_mean)
    stamp['lower_bound'] = stamp['mean'] - stamp['variance_penalty'] - stamp['range_penalty']


def malformed_controls(case):
    bd = lambda s: s['birth_death']
    mean = lambda s: bd(s)['last']['mean_transport']
    common = [
        ('old_backend8', lambda s: bd(s).update(backend_schema=8)),
        ('wrong_trainer_schema', lambda s: s.update(schema=4)),
        ('backend_schema_strict_int', lambda s: bd(s).update(backend_schema=9.0)),
        ('missing_mean_policy', lambda s: bd(s)['settings'].pop('mean_policy')),
        ('old_common_family', lambda s: bd(s)['last'].update(count_multiplicity=3*bd(s)['last']['cells']+2)),
        ('multiplicity_strict_int', lambda s: bd(s)['last'].update(count_multiplicity=float(3*bd(s)['last']['cells']+3))),
        ('old_cutoff', lambda s: bd(s)['last'].update(count_cutoff=.05/(3*bd(s)['last']['cells']+2))),
        ('missing_mean_map', lambda s: bd(s)['last'].pop('mean_transport')),
        ('mean_observations_strict_int', lambda s: mean(s).update(observations=True)),
        ('nonfinite_mean_scalar', lambda s: mean(s).update(lower_bound=float('nan'))),
        ('future_mean_stamp', lambda s: mean(s).update(step=mean(s)['step']+1)),
        ('old_mean_alpha', lambda s: mean(s).update(alpha=.05/(3*mean(s)['cells']+2))),
        ('mean_sign_status_conflict', lambda s: mean(s).update(status='veto' if mean(s)['status']=='firing' else 'firing')),
        ('mean_move_balance', lambda s: mean(s).update(moves=mean(s)['moves']+1)),
    ]
    if case == 'grid':
        common += [
            ('mean_known_score_range', lambda s: mutated_bound(mean(s), 3*mean(s)['radius'])),
            ('mean_evaluation_counter', lambda s: bd(s)['counters'].update(mean_evals=bd(s)['counters']['mean_evals']+1)),
        ]
    else:
        # No second full malformed suite: the toy-specific branch targets the
        # real veto state, which must never claim an action objective.
        common = [('veto_objective_forbidden', lambda s: mean(s).update(objective_before_mean=0., objective_after_mean=0.))]
    return common


def persisted_hook_state(api, trainer, trace, hook):
    torch = api.torch
    moved = torch.tensor(trace['moved_rows'], dtype=torch.long)
    if hook['evidence_after'] is not None:
        ev = trainer.row_evidence.state_dict()
        assert api.fingerprint(ev) == api.fingerprint(hook['evidence_after'])
        for name in ('M', 'Qs', 'W', 'S', 'flag'):
            assert not bool(ev[name][moved].any()), name
    if hook['tester_after'] is not None:
        test = trainer._table_tester().state_dict()
        assert api.fingerprint(test) == api.fingerprint(hook['tester_after'])
        mask = test['stationary_rows']
        assert mask is not None and not bool(mask[moved].any())
        assert test['population_active'] is False
    return dict(actual_complete_moved_rows=len(moved), persisted_evidence_and_participation_exact=True,
        optimizer_history_inheritance_is_not_own_participation=True)


def returned_state_independence(api, trainer):
    returned = trainer.state_dict()
    before = api.fingerprint(owned_view(api, trainer))
    returned['birth_death']['last']['mean_transport']['step'] += 1
    returned['models']['prior']['z'].add_(1.)
    returned['birth_death']['lineage_neighbors'].fill_(-1)
    assert before == api.fingerprint(owned_view(api, trainer))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'retain previous attempts'
    inputs = json.loads(args.inputs.read_text())
    package_path = raw_guards(inputs)  # Before Torch/import/constructor/PT interpretation.
    assert sha(Path(__file__)) == inputs['protected_sha256'][str(Path(__file__).resolve())]
    args.output.mkdir(parents=True)
    records = []
    try:
        skeleton = module(HERE / 'check_api_skeleton.py', 'ra10_preserved_api_skeleton')
        api = skeleton.load_api_after_guard()
        torch = api.torch
        assert torch.cuda.is_initialized() is False
        package = api.load_package(package_path, 'ra10_composed_api')
        global_before = torch.get_rng_state().clone()
        states = {}
        for case in ('grid', 'toy'):
            log('case_start', case=case)
            bindings = inputs['fixtures'][case]
            before = torch.load(bindings['before'], map_location='cpu', weights_only=False)
            after = torch.load(bindings['after'], map_location='cpu', weights_only=False)
            hook = torch.load(bindings['hook'], map_location='cpu', weights_only=False)
            trace = json.loads(Path(bindings['trace']).read_text())
            source_fingerprints = {name: api.fingerprint(value) for name, value in (('before',before),('after',after),('hook',hook))}
            assert before['schema'] == after['schema'] == 5
            assert before['birth_death']['backend_schema'] == after['birth_death']['backend_schema'] == 9
            assert before['completed_steps'] == 0 and after['completed_steps'] == 1
            assert before['birth_death']['last']['mean_transport']['status'] == 'initial'
            assert after['device'] == 'cpu' and after['cuda_rng'] is None
            initial = skeleton.construct(api, package, before, case)
            initial.load_state_dict(before)
            assert api.fingerprint(initial.state_dict()) == api.fingerprint(before)
            out = args.output / case
            out.mkdir()
            subject, clone = skeleton.cold_roundtrip(api, package, case, after, out)
            phase = skeleton.phase_balances(subject)
            assert phase['mean_status'] == ('firing' if case == 'grid' else 'veto')
            persisted = persisted_hook_state(api, subject, trace, hook)
            independent = returned_state_independence(api, subject)
            serving = skeleton.positive_serving_continuation(api, package, case, after, subject, out)
            good = subject.state_dict()
            controls = [reject_atomic(api, subject, good, label, mutate)
                        for label, mutate in malformed_controls(case)]
            assert source_fingerprints == {name: api.fingerprint(value) for name, value in (('before',before),('after',after),('hook',hook))}
            states[case] = after
            record = dict(case=case, status='PASS', initial_state_and_genuine_backend9_load=True,
                full_state_cold_roundtrip=True, returned_checkpoint_independent=independent,
                discarded_derived_chart_geometry_packets_and_moved_rows=True,
                phase_balances=phase, persisted_hook_state=persisted,
                serving=serving, rejected_controls=controls)
            records.append(record)
            log('case_pass', case=case, mean_status=phase['mean_status'], serving=serving['status'], rejected=len(controls))
        updates = skeleton.two_updates(api, package, states['toy'])
        torch.set_rng_state(global_before)
        assert not torch.cuda.is_initialized()
        guard(inputs['protected_sha256'])
        outputs = {str(path): sha(path) for path in sorted(args.output.rglob('*')) if path.is_file()}
        receipt = dict(status='PASS', scope='Independent CPU full checkpoint/API/resume/typed-state contracts; no quality score or CUDA',
            backend_schema=9, trainer_schema=5, package_sha256=inputs['package_sha256'],
            config_sha256=inputs['config_sha256'], records=records, continuation=updates,
            input_seal_sha256=sha(args.inputs), source_and_input_sha256=inputs['protected_sha256'],
            private_output_sha256=outputs, cuda_initialized=False,
            API_sample_calls=sum(x['serving']['sample_calls'] for x in records),
            API_rows_per_sample=17, new_quality_emissions=0, quality_verdict=None,
            limitation='CPU mechanics are explicit fresh-law fixtures, not historical CUDA replay; inherited served-load rejection advances parameter versions while retaining semantic/view state.')
        (args.output / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n')
        log('api_complete', status='PASS', receipt_sha256=sha(args.output / 'receipt.json'), CPU_updates_total=2)
    except Exception as error:
        failure = dict(status='FAIL', error=repr(error), traceback=traceback.format_exc(), completed_records=records,
            input_seal_sha256=sha(args.inputs), source_and_input_sha256=inputs['protected_sha256'])
        (args.output / 'FAILURE.json').write_text(json.dumps(failure, sort_keys=True, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()

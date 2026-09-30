"""Affected backend10 API controls; execute only with sealed genuine fixtures."""
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
OLD = ROOT / 'performance/training-regression/count-review/post-ra9-quality/mean-api-review'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def guard(mapping):
    for path, digest in mapping.items():
        assert sha(path) == digest, path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def log(event, **fields):
    print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=event, **fields)), flush=True)


def raw_guards(inputs):
    assert inputs['status'] == 'PRE_EXECUTION_AFFECTED_API_FROZEN'
    assert inputs['backend_schema'] == 10 and inputs['trainer_schema'] == 5
    assert inputs['API_sample_calls'] == 3 and inputs['CPU_updates_total'] == 2
    guard(inputs['protected_sha256'])  # All bytes before Torch/PT/constructors.
    assert sha(__file__) == inputs['protected_sha256'][str(Path(__file__).resolve())]
    package = Path(inputs['package_root'])
    composition = json.loads(Path(inputs['root_composition']).read_text())
    owner = json.loads(Path(inputs['owner_ready']).read_text())
    actual = {str(p.relative_to(package / 'particlegan')): sha(p)
              for p in sorted((package / 'particlegan').rglob('*.py'))}
    assert len(actual) == 31 and actual == composition['source_sha256']
    assert actual == owner['package_source_sha256']
    h = hashlib.sha256()
    for name in sorted(actual):
        h.update(name.encode() + b'\0' + (package / 'particlegan' / name).read_bytes() + b'\0')
    assert h.hexdigest() == owner['package_sha256'] == composition['package_sha256'] == inputs['package_sha256']
    assert Path(inputs['config_path']).read_bytes() == (ROOT / 'configs/overrides-CB64-RA9.json').read_bytes()
    assert sha(inputs['config_path']) == inputs['config_sha256']
    mechanics = json.loads(Path(inputs['mechanics_receipt']).read_text())
    assert mechanics['status'] == 'PASS'
    for case, binding in inputs['fixtures'].items():
        assert case in ('grid', 'toy')
        provenance = json.loads(Path(binding['provenance']).read_text())
        assert provenance['case'] == case and provenance['device'] == 'cpu'
        assert provenance['construction'] == inputs['genuine_construction']
        assert provenance['seed'] == 314159 and provenance['no_optimizer_or_GAN_gradient_step'] is True
        assert provenance['completed_steps_before'] == 0 and provenance['completed_steps_after'] == 1
        guard(provenance['input_binding'])
        assert sha(provenance['original_checkpoint']) == provenance['original_checkpoint_sha256']
        for name, digest in provenance['output_sha256'].items():
            assert sha(Path(binding['provenance']).parent / name) == digest
    return package


def phase_and_frame(trainer):
    bd, last = trainer.birth_death, trainer.birth_death.last
    stamp = last['mean_transport']
    assert bd.BACKEND_SCHEMA == 10 and stamp['schema'] == 2
    assert stamp['chart_rank'] == last['metric_rank'] == bd.paired_average['rank']
    assert stamp['cells'] == last['cells'] and stamp['snapshot'] == bd.snapshot_serial
    assert stamp['output_dim'] == math.prod(bd.sample_shape)
    assert stamp['moment_rank'] == len(stamp['selected_axes']) <= min(8, stamp['output_dim'])
    assert stamp['fitted_rows'] == (bd.N + 1) // 2
    assert last['count_categories'] == 2 * last['cells']
    assert last['count_multiplicity'] == 3 * last['cells'] + 3
    assert last['count_cutoff'] == stamp['alpha'] == .05 / (3 * last['cells'] + 3)
    assert stamp['radius'] == math.sqrt(stamp['moment_rank'] / .05)
    phases = [last[f'ordinary_{name}_moves'] for name in ('mass', 'support', 'global', 'mean')]
    assert sum(phases) == last['ordinary_copy_moves']
    assert last['ordinary_moves'] == last['ordinary_copy_moves'] + last['ordinary_novel_birth_moves']
    assert last['moves'] == last['ordinary_moves'] + last['iso_moves']
    assert stamp['moves'] == last['ordinary_mean_moves']
    assert last['ordinary_moves'] <= math.floor(.05 * bd.N)
    json.dumps(bd.diagnostics(), allow_nan=False)
    return dict(chart_rank=stamp['chart_rank'], moment_rank=stamp['moment_rank'],
        output_dim=stamp['output_dim'], selected_axes=list(stamp['selected_axes']),
        mean_status=stamp['status'], mean_moves=stamp['moves'], copy_phase_moves=phases,
        all_moves=last['moves'], common_count_multiplicity=last['count_multiplicity'])


def cold_roundtrip(api, skeleton, package, case, state, output):
    subject = skeleton.construct(api, package, state, case)
    subject.load_state_dict(state)  # Genuine backend10 only.
    path = output / 'roundtrip.pt'
    api.torch.save(dict(trainer=subject.state_dict()), path)
    clone = skeleton.construct(api, package, state, case)
    clone.load_state_dict(api.torch.load(path, map_location='cpu', weights_only=False)['trainer'])
    assert api.fingerprint(api.checkpoint_served_view(subject)) == api.fingerprint(api.checkpoint_served_view(clone))
    assert clone.birth_death.snapshot is None and not clone.birth_death.latent_geometry._entries
    assert clone.birth_death.moved_rows is None
    return subject


def axes_ownership(api, old, trainer, state):
    # Direct backend outbound boundary: trainer deepcopy must not hide an alias.
    before = api.fingerprint(old.owned_view(api, trainer))
    returned = trainer.birth_death.state_dict()
    returned['last']['mean_transport']['selected_axes'].append(-1)
    assert api.fingerprint(old.owned_view(api, trainer)) == before
    incoming = deepcopy(state)
    trainer.load_state_dict(incoming)
    before = api.fingerprint(old.owned_view(api, trainer))
    incoming['birth_death']['last']['mean_transport']['selected_axes'].append(-1)
    assert api.fingerprint(old.owned_view(api, trainer)) == before
    return dict(returned_backend_axes_owned=True, caller_load_axes_owned=True)


def malformed_controls():
    mean = lambda s: s['birth_death']['last']['mean_transport']
    def axis(s, value):
        mean(s)['selected_axes'][0] = value
    return [
        ('old_mean_schema1', lambda s: mean(s).update(schema=1)),
        ('missing_moment_rank', lambda s: mean(s).pop('moment_rank')),
        ('moment_rank_float_alias', lambda s: mean(s).update(moment_rank=float(mean(s)['moment_rank']))),
        ('axis_bool_alias', lambda s: axis(s, bool(mean(s)['selected_axes'][0]))),
        ('duplicate_axes', lambda s: mean(s)['selected_axes'].__setitem__(-1, mean(s)['selected_axes'][0])),
        ('out_of_bounds_axis', lambda s: axis(s, mean(s)['output_dim'])),
        ('output_dim_disagrees_with_shape', lambda s: mean(s).update(output_dim=mean(s)['output_dim'] + 1)),
        ('small_output_axis_order', lambda s: mean(s)['selected_axes'].reverse()),
        ('wrong_projection_policy', lambda s: mean(s).update(projection_policy='legacy_critic_frame')),
        ('critic_rank_radius', lambda s: mean(s).update(radius=math.sqrt(mean(s)['chart_rank'] / .05))),
        ('firing_zero_moment_rank', lambda s: mean(s).update(moment_rank=0, selected_axes=[], fitted_rows=0)),
    ]


def observed_sample(api, trainer):
    bd = trainer.birth_death
    existed = 'perturb_latent' in vars(bd)
    original_instance = vars(bd).get('perturb_latent')
    original = bd.perturb_latent
    observations = []
    def trace(latent, stream, controller=None, record=False, *, prior=None, rows=None):
        raw = latent.detach().clone()
        perturbed = original(latent, stream, controller, record, prior=prior, rows=rows)
        observations.append(dict(rows=None if rows is None else rows.clone(), raw_latent=raw,
                                 perturbed_latent=perturbed.detach().clone()))
        return perturbed
    bd.perturb_latent = trace
    try:
        output = trainer.sample(17)
    finally:
        if existed:
            bd.perturb_latent = original_instance
        else:
            del bd.perturb_latent
    assert len(observations) == 1 and observations[0]['rows'] is not None
    assert observations[0]['rows'].numel() == 17
    return dict(output=output, observation=observations[0], eval_cursor=trainer.eval_generator.get_state().clone())


def positive_continuation(api, skeleton, package, state, subject, output):
    assert subject._serve_settled() and subject._fast is not None, 'real geometry lease required'
    observed_sample(api, subject)  # Exactly one warm chunk, original sampling law.
    path = output / 'after-first-sample.pt'
    api.torch.save(dict(trainer=subject.state_dict()), path)
    clone = skeleton.construct(api, package, state, 'grid')
    clone.load_state_dict(api.torch.load(path, map_location='cpu', weights_only=False)['trainer'])
    assert not clone.birth_death.latent_geometry._entries
    left, right = observed_sample(api, subject), observed_sample(api, clone)
    assert api.fingerprint(left) == api.fingerprint(right)
    assert api.fingerprint(api.checkpoint_served_view(subject)) == api.fingerprint(api.checkpoint_served_view(clone))
    return dict(status='PASS', actual_positive_lease=True, synthetic_stamp=False,
        rows_raw_perturbed_latents_outputs_cursor_full_state_exact=True, sample_calls=3, rows_per_call=17)


def two_updates(api, skeleton, package, state):
    # Retain the qualified continuation body and the already corrected None guard.
    subject = skeleton.construct(api, package, state, 'toy')
    clone = skeleton.construct(api, package, state, 'toy')
    subject.load_state_dict(state)
    clone.load_state_dict(state)
    assert state['completed_steps'] < 2000 and (state['recipe']['total_steps'] is None
        or state['completed_steps'] < state['recipe']['total_steps'])
    assert subject.birth_death.rows_since_eval == 0
    batch = state['birth_death']['reservoir'][:128]
    assert len(batch) == 128
    api.torch.set_rng_state(state['cpu_rng'].clone())
    left = subject.step(batch)
    left_state = api.served_view(subject)
    api.torch.set_rng_state(state['cpu_rng'].clone())
    right = clone.step(batch)
    assert api.fingerprint(left) == api.fingerprint(right)
    assert api.fingerprint(left_state) == api.fingerprint(api.served_view(clone))
    assert subject.completed_steps == clone.completed_steps == state['completed_steps'] + 1
    assert subject.birth_death.snapshot_serial == state['birth_death']['snapshot_serial']
    return dict(status='PASS', CPU_updates_total=2, original_batch_rows=128,
        full_state_losses_gradients_streams_exact=True, extra_chart_reaction=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'retain all previous attempts'
    inputs = json.loads(args.inputs.read_text())
    package_path = raw_guards(inputs)
    args.output.mkdir(parents=True)
    records = []
    try:
        old = load_module(OLD / 'check_api.py', 'qualified_ra10_atomic_util')
        skeleton = load_module(OLD / 'check_api_skeleton.py', 'qualified_ra10_constructor_util')
        api = skeleton.load_api_after_guard()
        torch = api.torch
        assert not torch.cuda.is_initialized()
        package = api.load_package(package_path, 'composed_output_mean_api')
        global_before = torch.get_rng_state().clone()
        states = {}
        for case in ('grid', 'toy'):
            log('case_start', case=case)
            binding = inputs['fixtures'][case]
            before = torch.load(binding['before'], map_location='cpu', weights_only=False)
            after = torch.load(binding['after'], map_location='cpu', weights_only=False)
            hook = torch.load(binding['hook'], map_location='cpu', weights_only=False)
            trace = json.loads(Path(binding['trace']).read_text())
            source_before = api.fingerprint((before, after, hook))
            assert before['schema'] == after['schema'] == 5
            assert before['birth_death']['backend_schema'] == after['birth_death']['backend_schema'] == 10
            assert before['completed_steps'] == 0 and after['completed_steps'] == 1
            assert after['device'] == 'cpu' and after['cuda_rng'] is None
            if case == 'grid':
                initial = skeleton.construct(api, package, before, case)
                initial.load_state_dict(before)
                assert api.fingerprint(initial.state_dict()) == api.fingerprint(before)
                assert before['birth_death']['last']['mean_transport']['status'] == 'initial'
            out = args.output / case
            out.mkdir()
            subject = cold_roundtrip(api, skeleton, package, case, after, out)
            phase = phase_and_frame(subject)
            assert phase['mean_status'] == ('firing' if case == 'grid' else 'veto')
            persisted = old.persisted_hook_state(api, subject, trace, hook)
            record = dict(case=case, status='PASS', full_state_cold_roundtrip=True,
                          phase_and_frame=phase, persisted_hook_state=persisted)
            if case == 'grid':
                assert phase['mean_moves'] > 0 and phase['moment_rank'] == 2 and phase['chart_rank'] == 8
                record['axes_ownership'] = axes_ownership(api, old, subject, after)
                good = subject.state_dict()
                controls = [old.reject_atomic(api, subject, good, label, mutate)
                            for label, mutate in malformed_controls()]
                legacy = torch.load(inputs['old9_grid_state'], map_location='cpu', weights_only=False)
                assert legacy['birth_death']['backend_schema'] == 9
                legacy_before = api.fingerprint(legacy)
                controls.append(old.reject_atomic(api, subject, good, 'genuine_old_backend9',
                    lambda s: (s.clear(), s.update(deepcopy(legacy)))))
                assert api.fingerprint(legacy) == legacy_before
                record['rejected_controls'] = controls
                record['serving'] = positive_continuation(api, skeleton, package, after, subject, out)
            assert api.fingerprint((before, after, hook)) == source_before
            states[case] = after
            records.append(record)
            log('case_pass', case=case, mean_moves=phase['mean_moves'])
        updates = two_updates(api, skeleton, package, states['toy'])
        torch.set_rng_state(global_before)
        assert not torch.cuda.is_initialized()
        guard(inputs['protected_sha256'])
        receipt = dict(status='PASS', scope='Affected genuine backend10/schema2 CPU API and continuation',
            backend_schema=10, trainer_schema=5, package_sha256=inputs['package_sha256'],
            records=records, continuation=updates, input_seal_sha256=sha(args.inputs),
            source_and_input_sha256=inputs['protected_sha256'], API_sample_calls=3,
            API_rows_per_sample=17, CPU_updates_total=2, old_generic_metadata_cases_repeated=False,
            cuda_initialized=False, quality_verdict=None, new_quality_emissions=0,
            limit='Fresh-law CPU mechanics; no historical CUDA replay, repeated significance, group equivalence or serving quality claim.')
        (args.output / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n')
        log('complete', status='PASS', receipt_sha256=sha(args.output / 'receipt.json'))
    except Exception as error:
        (args.output / 'FAILURE.json').write_text(json.dumps(dict(status='FAIL', error=repr(error),
            traceback=traceback.format_exc(), completed_records=records), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()

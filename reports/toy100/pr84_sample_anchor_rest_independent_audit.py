"""Exact success-path and forced-failure audit of the sample-anchor rest guard."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    if path.is_file():
        return json.loads(path.read_bytes())
    return json.loads(gzip.decompress(path.with_name(path.name + '.gz').read_bytes()))


def success_parity(base_path, rest_path):
    base_raw = ((base_path / 'summary.json').read_bytes() if (base_path / 'summary.json').is_file()
                else gzip.decompress((base_path / 'summary.json.gz').read_bytes()))
    rest_raw = ((rest_path / 'summary.json').read_bytes() if (rest_path / 'summary.json').is_file()
                else gzip.decompress((rest_path / 'summary.json.gz').read_bytes()))
    base, rest = json.loads(base_raw), json.loads(rest_raw)
    assert base['status'] == rest['status'] == 'PASS'
    assert len(base['branches']) == len(rest['branches']) == 3
    for path, declaration in ((base_path, base['declaration']),
                              (rest_path, rest['declaration'])):
        for name, expected in declaration['sources'].items():
            assert sha((path / 'source' / name).read_bytes()) == expected
    compared = 0
    for original, guarded in zip(base['branches'], rest['branches']):
        assert (original['start'], original['end']) == (guarded['start'], guarded['end'])
        for arm in ('original', 'reallocation'):
            before = deepcopy(original['variants'][arm])
            after = deepcopy(guarded['variants'][arm])
            for receipt in (before['dynamics'], after['dynamics']):
                for key in ('method', 'scratch_optimizer_policy', 'source_sha256',
                            'nonconverged_fit_policy', 'conditional_invariance'):
                    receipt.pop(key, None)
            for correction in after['dynamics'].get('corrections', []):
                assert correction.pop('nonconverged_fit_rested') is False
                assert correction['fit']['status'] == 'CONVERGED'
            assert before == after
            assert read_json(base_path / f'{arm}-{original["start"]}-{original["end"]}.json') == original['variants'][arm]
            assert read_json(rest_path / f'{arm}-{guarded["start"]}-{guarded["end"]}.json') == guarded['variants'][arm]
            compared += guarded['end'] - guarded['start'] + 1
    assert compared == 88  # 44 ordinary + 44 guarded update records.
    return dict(base_summary_sha256=sha(base_raw), rest_summary_sha256=sha(rest_raw),
                compared_updates=compared, guarded_success_updates=44,
                equality='complete branch JSON after method/source metadata and new false flag only',
                all_converged=True)


class StopAfterOne(Exception):
    pass


def forced_one(root, rest_path):
    sys.path.insert(0, str(root))
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
    from reports.toy100 import sample_anchor_candidate as base_module
    from reports.toy100.pr84_critic_refinement_capture import snapshot, _sha
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    from reports.toy100.sample_anchor_rest_candidate import sample_anchor_rest_candidate, METHOD

    declaration = json.loads((rest_path / 'declaration.json').read_text())
    assert declaration['method'] == METHOD
    for name, expected in declaration['sources'].items():
        assert sha((root / name).read_bytes()) == expected, name
    config = json.loads((root / 'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name=METHOD, lr_floor=1., lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap')
    config.pop('network_lr_floor')
    recipe, noise, _ = declared_recipe(config)
    spec = next(row['spec'] for row in plan() if row['spec']['name'] == 'mode_hold')
    assert spec['steps'] == 1200
    ordinary_fit = base_module.fit_output_targets
    changed = {}

    def forced_failure(generator, prior_z, target, **kwargs):
        receipt = ordinary_fit(generator, prior_z, target, **kwargs)
        changed['natural_status'] = receipt['status']
        with torch.no_grad():
            network = next(generator.parameters())
            network_before = network.detach().clone()
            prior_before = prior_z.detach().clone()
            network.flatten()[0].add_(.125)
            prior_z.flatten()[0].add_(.125)
            changed['network_injected'] = not torch.equal(network, network_before)
            changed['prior_injected'] = not torch.equal(prior_z, prior_before)
        receipt['status'] = 'FORCED_NONCONVERGENCE'
        return receipt

    states, rows = {}, {}
    for arm in ('original', 'forced'):
        factory = (lambda: pr84_smoothed_candidate(task='mode_hold')) if arm == 'original' else (
            lambda: sample_anchor_rest_candidate(task='mode_hold', correction=True))
        ordinary_adam = torch.optim.Adam.step
        calls = {}

        def audited_step(optimizer, closure=None):
            calls.setdefault(optimizer, []).append(tuple(group['lr'] for group in optimizer.param_groups))
            return ordinary_adam(optimizer, closure=closure)

        fit_patch = patch.object(base_module, 'fit_output_targets', forced_failure) if arm == 'forced' else patch.object(base_module, 'fit_output_targets', ordinary_fit)
        with fit_patch, patch.object(torch.optim.Adam, 'step', audited_step), factory() as (recorder, generated):
            recorder.accounting = lambda d_calls, outer: (_ for _ in ()).throw(StopAfterOne()) if outer == 1 else None
            try:
                run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
            except StopAfterOne:
                pass
            else:
                raise AssertionError('one-update stop callback did not fire')
            assert recorder.outer_steps == 1 and len(recorder.records) == 1
            assert recorder._local['noise_policy']._step_calls == 1
            assert recorder.rng_replay_verified == 2
            applied = {}
            for role, optimizer, expected in zip(('d', 'g_prior'), recorder.optimizers,
                                                 ((.00425,), (.00425, .0085))):
                assert calls[optimizer] == [expected]
                moments = {int(optimizer.state[p]['step']) for group in optimizer.param_groups
                           for p in group['params']}
                assert moments == {1}
                applied[role] = dict(rates=list(expected), calls=1, moments=1)
            if arm == 'forced':
                assert all(torch.equal(current, previous) for current, previous in zip(
                    recorder._params(recorder.optimizers[1]), recorder.g_base))
                correction = recorder.corrections[0]
                assert correction['fit']['status'] == 'FORCED_NONCONVERGENCE'
                assert correction['nonconverged_fit_rested'] is True
                assert correction['selected'] == recorder.row['reallocation']['selected'] == 'rest'
                assert correction['final_cost'] == correction['pre_cost']
                assert correction['final_cost'] == recorder.row['reallocation']['final_cost']
                assert recorder.rng_checks == recorder.owner_checks == 1
                assert recorder.batch_checks == 3
            states[arm] = snapshot(recorder._local)
            rows[arm] = dict(state_sha256=_sha(states[arm]), applied=applied,
                             record=deepcopy(recorder.records[0]),
                             correction=deepcopy(recorder.corrections[0]) if arm == 'forced' else None,
                             generated_host_sha256=sha(generated.encode()))
    assert changed['natural_status'] == 'CONVERGED'
    assert changed['network_injected'] and changed['prior_injected']
    assert rows['original']['generated_host_sha256'] == rows['forced']['generated_host_sha256']
    native_record = deepcopy(rows['forced']['record'])
    correction_record = native_record.pop('reallocation')
    assert native_record == rows['original']['record']
    assert correction_record['selected'] == 'rest'
    for key in ('critic', 'optimizer_d', 'optimizer_g', 'ema_g', 'ema_z', 'rng', 'noise'):
        assert _sha(states['original'][key]) == _sha(states['forced'][key]), key
    assert _sha(states['original']['generator']) != _sha(states['forced']['generator'])
    assert _sha(states['original']['prior']) != _sha(states['forced']['prior'])
    return dict(scope='one actual native cold update; stopped before EMA/checkpoint',
                source_rest_sha256=sha((root / 'reports/toy100/sample_anchor_rest_candidate.py').read_bytes()),
                generated_host_sha256=rows['forced']['generated_host_sha256'],
                natural_fit_status=changed['natural_status'],
                forced_network_and_prior_perturbations=True,
                all_g_prior_parameters_exactly_restored_to_pre_g=True,
                d_both_adam_ema_rng_noise_identical_to_original=True,
                native_record_identical_to_original_except_added_correction=True,
                no_extra_adam_calls=True, rows=rows)


def forced_favorable_native(root, rest_path, capture, source_dir):
    """At archived step 1332, prove the guard overrides a beneficial native G step."""
    sys.path.insert(0, str(root))
    from reports.toy100 import pr84_prediction_state_filter as replay
    from reports.toy100 import sample_anchor_candidate as base_module
    from reports.toy100.sample_anchor_rest_candidate import sample_anchor_rest_candidate

    declaration = json.loads((rest_path / 'declaration.json').read_text())
    assert sha((capture / 'selected-states.pt').read_bytes()) == declaration['states_sha256']
    states = torch.load(capture / 'selected-states.pt', weights_only=True, map_location='cpu')
    archived = read_json(rest_path / 'reallocation-1324-1335.json')
    natural = next(row for row in archived['dynamics']['corrections'] if row['step'] == 1332)
    assert natural['native_cost'] < natural['pre_cost']
    ordinary_fit = base_module.fit_output_targets
    observed = {}

    @contextmanager
    def selected(*, task, prediction):
        with sample_anchor_rest_candidate(task=task, correction=prediction) as value:
            observed['recorder'] = value[0]
            yield value

    def force_at_1332(generator, prior_z, target, **kwargs):
        recorder = observed['recorder']
        step = recorder._local['step'] + 1
        receipt = ordinary_fit(generator, prior_z, target, **kwargs)
        if step == 1332:
            assert receipt['status'] == 'CONVERGED'
            with torch.no_grad():
                next(generator.parameters()).flatten()[0].add_(.125)
                prior_z.flatten()[0].add_(.125)
            receipt['status'] = 'FORCED_NONCONVERGENCE'
            observed['injected'] = True
        return receipt

    source_dir.mkdir()
    with patch.object(base_module, 'fit_output_targets', force_at_1332), patch.object(
            replay.prediction_module, 'pr84_opponent_prediction', selected):
        config = json.loads((root / 'configs/toy100/constraints_simple_regularization.json').read_text())
        result, _ = replay.run_local(config, states[1324]['pre_step'], start=1324, end=1332,
                                     opponent='predicted', source_dir=source_dir)
    recorder = observed['recorder']
    assert observed['injected']
    rows = recorder.corrections
    assert len(rows) == 9 and [row['step'] for row in rows] == list(range(1324, 1333))
    assert rows[:8] == archived['dynamics']['corrections'][:8]
    failed = rows[-1]
    assert failed['fit']['status'] == 'FORCED_NONCONVERGENCE'
    assert failed['nonconverged_fit_rested'] is True
    assert failed['selected'] == 'rest'
    assert failed['native_cost'] < failed['pre_cost']
    assert failed['pre_cost'] == natural['pre_cost']
    assert failed['native_cost'] == natural['native_cost']
    assert failed['final_cost'] == failed['pre_cost']
    assert all(torch.equal(p, old) for p, old in zip(
        recorder._params(recorder.optimizers[1]), recorder.g_base))
    assert result['moment_steps'] == {'d': [1332], 'g': [1332]}
    assert result['dynamics']['correction_rng_checks'] == 9
    assert result['dynamics']['correction_owner_checks'] == 9
    assert result['dynamics']['native_batch_checks'] == 27
    return dict(scope='archived 1324 pre-step, local continuation through favorable native step 1332 only',
                capture_sha256=declaration['states_sha256'],
                natural_native_improves=True,
                prior_eight_corrections_exact=True,
                forced_fit_nonconverged_rested=True,
                all_g_prior_parameters_exactly_pre_g=True,
                own_full_state_sha256=result['final_state_sha256'],
                own_rng_sha256=result['rng_final_sha256'],
                no_extra_adam_updates=True,
                native_sample_checks=27,
                pre_cost=failed['pre_cost'], native_cost=failed['native_cost'],
                final_cost=failed['final_cost'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--rest', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    parity = success_parity(args.base, args.rest)
    (args.output / 'success-parity.json').write_text(json.dumps(parity, indent=2) + '\n')
    print(json.dumps(dict(event='SUCCESS_PATH_PARITY', **parity)), flush=True)
    forced = forced_one(args.root.resolve(), args.rest)
    (args.output / 'forced-nonconvergence.json').write_text(json.dumps(forced, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(event='FORCED_NONCONVERGENCE',
                          restored=forced['all_g_prior_parameters_exactly_restored_to_pre_g'],
                          non_owner_parity=forced['d_both_adam_ema_rng_noise_identical_to_original'])), flush=True)
    favorable = forced_favorable_native(args.root.resolve(), args.rest, args.capture,
                                        args.output / 'source')
    (args.output / 'forced-favorable-native.json').write_text(
        json.dumps(favorable, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(event='FORCED_FAVORABLE_NATIVE',
                          step=1332, restored=favorable['all_g_prior_parameters_exactly_pre_g'],
                          prior_eight_exact=favorable['prior_eight_corrections_exact'])), flush=True)


if __name__ == '__main__':
    main()

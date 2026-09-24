"""Declared single-convolution implementation repair of PR84's rest gate."""

from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100.pr84_rest_repair_scratch import alternating_curvature

METHOD = 'pr84_single_convolution_rest_slope_repair'
CANDIDATE = 'single_slope'
OPTIONS = dict(bound_d=True, curvature_bound=.25, d_curvature_bound=3., smooth_critic=True, smooth_cap=.15)


@contextmanager
def audited_repair(task='mode_hold', **options):
    import torch
    with alternating_curvature(task=task, **options) as (recorder, source):
        rows = []
        delegate = torch.optim.Adam.step
        receipt = recorder.receipt

        def step(optimizer, closure=None):
            if recorder.phase is not None and not recorder.passthrough:
                role = 'd' if optimizer is recorder.optimizers[0] else 'g'
                groups = [dict(role='prior' if group.get('_comparison_prior') else role, lr=float(group['lr']))
                          for group in optimizer.param_groups]
                expected = {'g': .00425, 'd': .00425, 'prior': .0085}
                if any(group['lr'] != expected[group['role']] for group in groups):
                    raise RuntimeError('repaired rest gate changed a nominal role learning rate')
                rows.append(dict(outer_step=recorder.outer_steps + 1, phase=recorder.phase, role=role, groups=groups))
            return delegate(optimizer, closure=closure)

        def measured_receipt():
            value = receipt()
            value.update(method=METHOD, shared_gate_eligible=False, scratch_optimizer_policy=METHOD,
                         implementation_repair='manual rest slope stencil calls captured unpatched critic forward',
                         rest_slope_threshold=.2, rest_slope_convolutions=1, actual_role_rates=rows)
            return value

        recorder.receipt = measured_receipt
        with patch.object(torch.optim.Adam, 'step', step):
            yield recorder, source


def archive(output, phase):
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source_archive'
    source.mkdir()
    files = [Path(__file__), ROOT / 'reports/toy100/pr84_rest_repair_scratch.py',
             ROOT / 'reports/toy100/extra_adam_scratch.py',
             ROOT / 'benchmarks/toy100/warm_equilibrium_probe.py', ROOT / 'benchmarks/toy100/continuous_probe.py',
             ROOT / 'configs/toy100/constraints_simple_regularization.json', ROOT / 'tests/test_pr84_rest_repair.py',
             ROOT / 'reports/toy100/continuous-evidence/pr84-rest-repair/source/pr84_head_original.py',
             ROOT / 'reports/toy100/continuous-evidence/pr84-rest-repair/source/implementation-repair.patch']
    for path in files:
        (source / path.name).write_bytes(path.read_bytes())
    declaration = dict(method=METHOD, phase=phase, options=OPTIONS, seed=0, candidate_count=1,
                       original_pr84_head='468ad2619a0536f8c69ef878b86bc514d296338a',
                       change='Single implementation repair: remove second convolution from measured rest slope',
                       rest_slope_threshold=.2, gradient_width_cap=.15, change_threshold_or_bounds=False,
                       order=['warm200', 'trajectory400', 'mode_hold1200', 'fixed_target_hold2400'],
                       consumed_preceding_pass_required=True, stop_at_first_failed_host=True,
                       nominal_rates=dict(g=.00425, d=.00425, prior=.0085),
                       noise_horizons=dict(mode_hold=1200, trajectory=400),
                       shared_gate_eligible=False, scratch_optimizer_policy=METHOD,
                       oracle_options_disabled=['mode_loosen', 'boost_cap', 'latent_nudge', 'stray_step'],
                       occupied_modes='existing read-only diagnostic; never controls this arm',
                       source={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (output / 'declaration.json').write_text(json.dumps(declaration, indent=2) + '\n')


def variants():
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context

    def factory(method):
        @contextmanager
        def activate(state, prefix):
            recorder, _ = prefix
            recorder.enabled = method == CANDIDATE
            completed, target = state['completed_steps'], state['target_steps']

            def accounting(calls, outer):
                state['declare_optimizer_accounting'](calls=completed + calls + target - completed - outer,
                                                       moment_updates=target)
                if outer % 20 == 0:
                    print(json.dumps(dict(event='WARM_PROGRESS', method=method, update=completed + outer,
                                          field_calls=calls)), flush=True)

            recorder.accounting = accounting
            receipt = dict(method=method, shared_gate_eligible=False, scratch_optimizer_policy=METHOD)
            if method == 'identity':
                yield receipt
            else:
                with constant_rate_context(state) as rates:
                    receipt.update(rates)
                    yield receipt
                if recorder.enabled:
                    receipt.update(recorder.receipt())
        return activate
    return {name: factory(name) for name in ('identity', 'constant', CANDIDATE)}


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['warm', 'cold', 'hold'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    if args.phase != 'warm' and args.previous is None:
        raise ValueError('consumed preceding gate evidence required')
    if args.phase == 'cold':
        prior = json.loads(args.previous.read_text())
        if not prior['identity_cold_parity'] or prior['variants'][CANDIDATE]['status'] != 'PASS':
            raise RuntimeError('warm gate failed; cold forbidden')
    elif args.phase == 'hold':
        prior = json.loads(args.previous.read_text())
        if len(prior['stages']) != 2 or any(not row.get('verdict', {}).get('passed') for row in prior['stages']):
            raise RuntimeError('both original cold hosts must pass before extended hold')
    archive(args.output, args.phase)
    if args.previous is not None:
        (args.output / 'previous-gate.json').write_bytes(args.previous.read_bytes())
    if args.phase == 'warm':
        from benchmarks.toy100.warm_equilibrium_probe import run_warm_variants
        with audited_repair(start_step=1000, **OPTIONS) as (_, source):
            (args.output / 'source_archive/mode_hold_transformed.py').write_text(source)
        result = run_warm_variants(config, variants(), output_dir=args.output / 'run',
                                  prefix_context=lambda: audited_repair(start_step=1000, **OPTIONS))
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.phase == 'hold':
        from benchmarks.toy100.continuous_probe import run_probe
        with audited_repair(**OPTIONS) as (recorder, source):
            (args.output / 'source_archive/mode_hold_transformed.py').write_text(source)

            def hook(state):
                recorder.accounting = lambda calls, outer: state['declare_optimizer_accounting'](
                    calls=calls + 2400 - outer, moment_updates=2400)
                recorder.accounting(recorder.rows[recorder.optimizers[0]]['calls'], recorder.outer_steps)

            result = run_probe(config, mode='constant', steps=2400, noise_horizon=1200, diagnostic_every=10,
                               dense_after=999, dense_until=1200, checkpoint_hook_step=1, checkpoint_hook=hook,
                               log=lambda event: print(json.dumps(event), flush=True))
        result.update(dynamics_receipt=recorder.receipt(), shared_gate_eligible=False, scratch_optimizer_policy=METHOD)
        (args.output / 'hold.json').write_text(json.dumps(result, allow_nan=False) + '\n')
        print(json.dumps(dict(status=result['status'], stationary=result['stationary'],
                              continued_hold=result['continued_hold'])), flush=True)
        return
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy
    config.update(name='pr84_rest_repair', lr_floor=1., lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap')
    config.pop('network_lr_floor')
    recipe, noise, _ = declared_recipe(config)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    stages, started = [], time.perf_counter()
    for task in ('trajectory', 'mode_hold'):
        spec = next(job['spec'] for job in plan() if job['spec']['name'] == task)
        try:
            with audited_repair(task=task, **OPTIONS) as (recorder, source):
                (args.output / 'source_archive' / (task + '_transformed.py')).write_text(source)

                def progress(calls, outer):
                    if outer % 100 == 0:
                        print(json.dumps(dict(event='COLD_PROGRESS', task=task, update=outer, field_calls=calls)), flush=True)

                recorder.accounting = progress
                result, context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
            verdict = test_verdict(spec, result)
            raw = dict(result=result, verdict=verdict, dynamics=recorder.receipt(), noise=context['noise_receipt'],
                       applied=context['applied'], spec=spec, shared_gate_eligible=False, scratch_optimizer_policy=METHOD)
            (args.output / (task + '.json')).write_text(json.dumps(raw, allow_nan=False) + '\n')
            stages.append(dict(task=task, verdict=verdict, live=result['live']))
            print(json.dumps(dict(event='STAGE_DONE', **stages[-1])), flush=True)
            if not verdict['passed']:
                break
        except Exception as error:
            import traceback
            raw = dict(error=repr(error), traceback=traceback.format_exc(), receipt=recorder.receipt(),
                       shared_gate_eligible=False, scratch_optimizer_policy=METHOD)
            (args.output / (task + '.error.json')).write_text(json.dumps(raw, indent=2) + '\n')
            stages.append(dict(task=task, error=repr(error)))
            break
    status = dict(stages=stages, seconds=time.perf_counter() - started)
    (args.output / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(dict(event='DONE', **status)), flush=True)


if __name__ == '__main__':
    main()

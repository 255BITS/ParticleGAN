"""One alternating-field implicit arm, using the frozen fail-fast host gates."""

from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100.alternating_implicit_scratch import alternating_implicit, METHOD

OPTIONS = dict(krylov_dim=8, linear_tolerance=.1, nonlinear_tolerance=.5,
               fd_relative=1e-4, correction_limit=2., max_backtracks=8)
CANDIDATE = 'alternating_implicit'


def archive(output, phase):
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source_archive'
    source.mkdir()
    files = [Path(__file__), ROOT / 'reports/toy100/alternating_implicit_scratch.py',
             ROOT / 'reports/toy100/implicit_extra_scratch.py',
             ROOT / 'reports/toy100/fixed_metric_extra_scratch.py',
             ROOT / 'reports/toy100/extra_adam_scratch.py',
             ROOT / 'benchmarks/toy100/warm_equilibrium_probe.py',
             ROOT / 'benchmarks/toy100/continuous_probe.py',
             ROOT / 'configs/toy100/constraints_simple_regularization.json',
             ROOT / 'tests/test_alternating_implicit_scratch.py']
    for path in files:
        (source / path.name).write_bytes(path.read_bytes())
    declaration = dict(method=METHOD, phase=phase, options=OPTIONS, seed=0,
                       shared_gate_eligible=False, scratch_optimizer_policy=METHOD,
                       candidate_count=1,
                       change='Replace simultaneous base/query field with anchored alternating-map field; solver unchanged',
                       field='FD(D,G), FG(round(D1+(D-D0)-PD*(FD(D,G)-FD0)),G)',
                       ordinary_control='Exact active phase-0 ordinary Adam D-then-G path, checked against unwrapped constant control',
                       order=['warm200', 'trajectory400', 'mode_hold1200', 'fixed_target_hold2400'],
                       consumed_preceding_pass_required=True, stop_at_first_failed_host=True,
                       nominal_rates=dict(g=.00425, d=.00425, prior=.0085),
                       noise_horizons=dict(mode_hold=1200, trajectory=400),
                       source={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in files},
                       references=['https://f-t-s.github.io/projects/cgd/'],
                       reference_scope='Implicit strategic-response motivation only; anchored alternating field is a new scratch variant, no convergence theorem claimed')
    (output / 'declaration.json').write_text(json.dumps(declaration, indent=2) + '\n')


def variants():
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context

    def factory(method):
        @contextmanager
        def activate(state, prefix):
            recorder, _ = prefix
            recorder.enabled = method in ('ordinary_control', CANDIDATE)
            recorder.explicit_control = method == 'ordinary_control'
            completed, target = state['completed_steps'], state['target_steps']

            def accounting(calls, outer):
                state['declare_optimizer_accounting'](
                    calls=completed + calls + target - completed - outer, moment_updates=target)
                if outer % 20 == 0:
                    print(json.dumps(dict(event='WARM_PROGRESS', method=method, update=completed + outer,
                                          field_calls=calls, scale=recorder.last_scale)), flush=True)

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
    return {name: factory(name) for name in ('identity', 'constant', 'ordinary_control', CANDIDATE)}


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
        if not (prior['identity_cold_parity'] and prior['ordinary_control_parity'] and
                prior['variants'][CANDIDATE]['status'] == 'PASS'):
            raise RuntimeError('warm gate did not pass; cold acquisition forbidden')
    elif args.phase == 'hold':
        prior = json.loads(args.previous.read_text())
        if len(prior['stages']) != 2 or any(not row.get('verdict', {}).get('passed') for row in prior['stages']):
            raise RuntimeError('both original cold hosts must pass before extended hold')
    archive(args.output, args.phase)
    if args.previous is not None:
        (args.output / 'previous-gate.json').write_bytes(args.previous.read_bytes())
    if args.phase == 'warm':
        from benchmarks.toy100.warm_equilibrium_probe import run_warm_variants, _metrics_without_time
        with alternating_implicit(start_step=1000, **OPTIONS) as (_, source):
            (args.output / 'source_archive/mode_hold_transformed.py').write_text(source)
        result = run_warm_variants(config, variants(), output_dir=args.output / 'run',
                                  prefix_context=lambda: alternating_implicit(start_step=1000, **OPTIONS))
        control, plain = [json.loads((args.output / 'run' / (name + '.json')).read_text())
                          for name in ('ordinary_control', 'constant')]
        if (control['final_state_sha256'] != plain['final_state_sha256'] or
                _metrics_without_time(control) != _metrics_without_time(plain)):
            raise RuntimeError('active ordinary alternating control differs from original constant host')
        result['ordinary_control_parity'] = True
        (args.output / 'run/summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.phase == 'hold':
        from benchmarks.toy100.continuous_probe import run_probe
        with alternating_implicit(**OPTIONS) as (recorder, source):
            (args.output / 'source_archive/mode_hold_transformed.py').write_text(source)

            def hook(state):
                recorder.accounting = lambda calls, outer: state['declare_optimizer_accounting'](
                    calls=calls + 2400 - outer, moment_updates=2400)
                recorder.accounting(recorder.rows[recorder.optimizers[0]]['calls'], recorder.outer_steps)

            result = run_probe(config, mode='constant', steps=2400, noise_horizon=1200, diagnostic_every=10,
                               dense_after=999, dense_until=1200, checkpoint_hook_step=1, checkpoint_hook=hook,
                               log=lambda event: print(json.dumps(event), flush=True))
        result.update(dynamics_receipt=recorder.receipt(), shared_gate_eligible=False,
                      scratch_optimizer_policy=METHOD)
        (args.output / 'hold.json').write_text(json.dumps(result, allow_nan=False) + '\n')
        print(json.dumps(dict(status=result['status'], stationary=result['stationary'],
                              continued_hold=result['continued_hold'])), flush=True)
        return
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy
    config.update(name=CANDIDATE, lr_floor=1., lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap')
    config.pop('network_lr_floor')
    recipe, noise, _ = declared_recipe(config)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    stages, started = [], time.perf_counter()
    for task in ('trajectory', 'mode_hold'):
        try:
            spec = next(job['spec'] for job in plan() if job['spec']['name'] == task)
            with alternating_implicit(task=task, **OPTIONS) as (recorder, source):
                (args.output / 'source_archive' / (task + '_transformed.py')).write_text(source)

                def progress(calls, outer):
                    if outer % 20 == 0:
                        print(json.dumps(dict(event='COLD_PROGRESS', task=task, update=outer,
                                              field_calls=calls, scale=recorder.last_scale)), flush=True)

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

"""Fail-fast cold acquisition after the frozen critic-refinement dense hold."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.pr84_critic_refinement import METHOD as WARM_METHOD
from reports.toy100.pr84_critic_refinement_cold import METHOD, pr84_critic_refinement_cold
from reports.toy100.pr84_smoothed_parity import _state_sha


def require_hold(path):
    previous = json.loads(path.read_text())
    candidate = previous['variants']['refinement']
    if (previous['method'] != WARM_METHOD or previous['phase'] != 'hold'
            or not previous['identity_cold_parity']
            or not previous['original_control_exact_parity']
            or not previous['first200_parity'] or candidate['status'] != 'PASS'
            or candidate['local_stability']['checks'] != 200
            or not candidate['local_stability']['pass_all']
            or candidate['long_hold']['checks'] != 1200
            or not candidate['long_hold']['pass_all']):
        raise RuntimeError('cold requires the complete, source-bound passing warm and dense hold')
    for name, digest in previous['source'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'passing hold source changed: {name}')
    return previous


def run(output, source):
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
    config = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name='pr84_critic_refinement_cold', lr_floor=1., lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap')
    config.pop('network_lr_floor')
    recipe, noise, _ = declared_recipe(config)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    stages = []
    for task, steps in (('trajectory', 400), ('mode_hold', 1200)):
        spec = next(job['spec'] for job in plan() if job['spec']['name'] == task)
        adam_step = torch.optim.Adam.step
        calls = {}
        def audited_step(optimizer, closure=None):
            entry = calls.setdefault(optimizer, [])
            entry.append(tuple(group['lr'] for group in optimizer.param_groups))
            return adam_step(optimizer, closure=closure)
        with patch.object(torch.optim.Adam, 'step', audited_step), \
             pr84_critic_refinement_cold(task=task) as (recorder, _):
            def progress(calls, outer):
                if outer % 20 == 0:
                    print(json.dumps(dict(event='COLD_PROGRESS', task=task, update=outer,
                        fit_gradient_evaluations=recorder.fit_gradient_evaluations)), flush=True)
            recorder.accounting = progress
            result, context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
        rates = {}
        for role, optimizer, expected in zip(('d', 'g'), recorder.optimizers,
                                              ((.00425,), (.00425, .0085))):
            observed = calls[optimizer]
            moments = [int(optimizer.state[p]['step']) for group in optimizer.param_groups
                       for p in group['params']]
            if len(observed) != steps or any(row != expected for row in observed):
                raise RuntimeError(f'{task}: actual {role} rates or Adam callback counts changed')
            if any(step != steps for step in moments):
                raise RuntimeError(f'{task}: actual {role} moment count changed')
            rates[role] = dict(applied_rates=expected, actual_adam_calls=len(observed),
                               moment_steps_min=min(moments), moment_steps_max=max(moments))
        if recorder.outer_steps != steps or recorder.bank_rng_verified != steps or recorder.fit_rng_verified != steps:
            raise RuntimeError('missing active steps or RNG verification')
        verdict = test_verdict(spec, result)
        data = dict(method=METHOD, result=result, applied=context['applied'],
                    noise=context['noise_receipt'], dynamics=recorder.receipt(),
                    actual_adam_accounting=rates, spec=spec, verdict=verdict,
                    final_training_state_sha256=_state_sha(recorder),
                    shared_gate_eligible=False, source=source)
        (output / f'{task}.json').write_text(json.dumps(data, allow_nan=False) + '\n')
        row = dict(task=task, verdict=verdict, live=result['live'], seconds=result['seconds'],
                   fit_gradient_evaluations=recorder.fit_gradient_evaluations)
        stages.append(row)
        print(json.dumps(dict(event='STAGE_DONE', **row)), flush=True)
        if not verdict['passed']:
            break
    summary = dict(method=METHOD, phase='cold', source=source, stages=stages,
                   status='PASS' if len(stages) == 2 and all(r['verdict']['passed'] for r in stages) else 'FAIL',
                   own_acquired_continuation_tested=False, shared_gate_eligible=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(event='COLD_DONE', **summary)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    previous = require_hold(args.previous)
    names = set(previous['source']) | {
        'reports/toy100/pr84_critic_refinement_cold.py',
        'reports/toy100/pr84_critic_refinement_cold_probe.py',
        'benchmarks/transfer_suite/legacy_noise_adapters.py',
        'benchmarks/transfer_suite/toy100_compatibility.py',
        'benchmarks/transfer_suite/protocol.py',
        'tests/test_pr84_critic_refinement_cold.py',
    }
    source = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}
    declaration = dict(method=METHOD, phase='cold', source=source, shared_gate_eligible=False,
        previous_hold_sha256=hashlib.sha256(args.previous.read_bytes()).hexdigest(),
        order=['trajectory400', 'mode_hold1200'], stop_at_first_failed_host=True,
        noise_horizon='original host budget', lr_decay=False, seed=0,
        rates=dict(g=.00425, d=.00425, prior=.0085),
        rule='same40-iteration/80-closure penalized critic refinement; eight native D batches',
        native_fit_pairs=dict(trajectory=96, mode_hold=1024),
        scope='cold acquisition only; own-acquired-state hold and shared gates still required')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in source:
        target = args.output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (args.output / 'previous-gate.json').write_bytes(args.previous.read_bytes())
    (args.output / 'declaration.json').write_text(json.dumps(declaration, indent=2) + '\n')
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    torch.set_num_threads(1)
    run(args.output, source)


if __name__ == '__main__':
    main()

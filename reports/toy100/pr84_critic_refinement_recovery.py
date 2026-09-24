"""Source-bound failed-fit reproduction and twenty-update numerical recovery.

This is a saved cold-state filter, not a cold acquisition or stability gate.
No target-quality value selects a fit, a retry, a scale, or an update.
"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100 import pr84_critic_relaxation as fit
from reports.toy100 import pr84_critic_refinement_cold as cold
from reports.toy100 import pr84_critic_refinement_finite as safe
from reports.toy100.pr84_critic_refinement_after_step import (
    LocalContinuationComplete, resume_after_set_step,
)
from reports.toy100.pr84_critic_refinement_capture import _sha, snapshot


CAPTURE_SHA = '19ceb38a92ba95ffaccd3aafbda55bbd77917765612666d923e986e5523d6965'
START, END = 472, 491


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(item) for item in value)
    return True


def exact_fit(payload, recipe):
    _, critic, _ = fit.modules(payload['post_accepted_d'])
    initial = deepcopy(critic.state_dict())
    original_records = None
    before_rng = torch.get_rng_state().clone()
    with patch.object(fit, 'd_loss', cold.cached_d_loss):
        try:
            fit.relax(critic, payload['bank'], recipe.make_loss(),
                      recipe.make_gradient_penalty(), START, payload['metric'])
        except FloatingPointError as error:
            if str(error) != 'nonfinite local critic fit':
                raise
            tb = error.__traceback__
            while tb is not None:
                if tb.tb_frame.f_code.co_name == 'relax':
                    original_records = deepcopy(tb.tb_frame.f_locals['records'])
                tb = tb.tb_next
        if original_records != payload['finite_records']:
            raise RuntimeError('original failed fit did not reproduce every finite closure exactly')
        critic.load_state_dict(initial)
        result = safe.finite_trial_fit(critic, payload['bank'], recipe.make_loss(),
                    recipe.make_gradient_penalty(), START, payload['metric'])
    checks = dict(original_failure_reproduced=original_records is not None,
                  finite_records_exact=result['records'] == payload['finite_records'],
                  recovered_best_state_exact=_sha(critic.state_dict()) == _sha(payload['best']),
                  all_parameters_finite=finite_tree(critic.state_dict()),
                  invalid_gradients_cleared=all(p.grad is None for p in critic.parameters()),
                  rng_unchanged=torch.equal(before_rng, torch.get_rng_state()),
                  closure_calls=result['closure_calls'], finite_closure_calls=result['finite_closure_calls'],
                  nonfinite_closure_calls=result['nonfinite_closure_calls'],
                  best_loss=min(row['total_loss'] for row in result['records']),
                  restored_best_state_sha256=_sha(critic.state_dict()))
    if (any(checks[key] is not True for key in ('original_failure_reproduced', 'finite_records_exact',
            'recovered_best_state_exact', 'all_parameters_finite', 'invalid_gradients_cleared', 'rng_unchanged'))
            or (checks['closure_calls'], checks['finite_closure_calls'], checks['nonfinite_closure_calls']) != (49,48,1)):
        raise RuntimeError(f'failed exact numerical recovery: {checks}')
    return checks, result


def continuation(payload, recipe, noise, output):
    from benchmarks import learned_lr_evaluation as bridge
    from benchmarks.locked_shared import mode_hold
    from benchmarks.smart_descent import evaluate
    from benchmarks.toy100.continuous_probe import _noise_policy
    from benchmarks.transfer_suite import vector_tasks
    from benchmarks.transfer_suite.compare_defaults import candidate, optimizer_defaults
    policy = _noise_policy(noise, 1200)
    finite_checks, first_fit = [], {}
    with ExitStack() as stack:
        stack.enter_context(optimizer_defaults(recipe, []))
        control = evaluate.FixedControl(vector_tasks.fixed_policy('cosine'), 1200)
        stack.enter_context(bridge.control_host_schedules(control))
        recorder, source = stack.enter_context(safe.pr84_critic_refinement_finite())
        call_count = 0
        safe_fit = fit.relax
        def audited_fit(critic, bank, gan, regularizer, step, metric):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_fit.update(accepted_d_state_exact=_sha(snapshot(recorder._local)) == _sha(payload['post_accepted_d']),
                                 bank_exact=_sha(bank) == _sha(payload['bank']),
                                 metric_exact=_sha(metric) == _sha(payload['metric']))
                if not all(first_fit.values()):
                    raise RuntimeError(f'first replayed accepted-D/bank/metric differs: {first_fit}')
            result = safe_fit(critic, bank, gan, regularizer, step, metric)
            if call_count == 1:
                first_fit.update(recovered_best_state_exact=_sha(critic.state_dict()) == _sha(payload['best']),
                                 finite_records_exact=result['records'] == payload['finite_records'])
                if not all(first_fit.values()):
                    raise RuntimeError('first host recovered fit differs from its standalone reproduction')
            return result
        stack.enter_context(patch.object(fit, 'relax', audited_fit))
        def observe(step, measure):
            state = snapshot(recorder._local)
            valid = finite_tree(state)
            finite_checks.append(dict(step=step, finite=valid,
                critic_parameter_norm=float(torch.cat([p.detach().double().flatten()
                    for p in recorder._local['critic'].parameters()]).norm())))
            if not valid:
                raise FloatingPointError(f'nonfinite accepted outer state at {step}')
        stack.enter_context(patch.object(mode_hold, 'checkpoint', observe))
        replay = stack.enter_context(resume_after_set_step(recorder, source, payload['pre_step'],
            completed_steps=START-1, target_steps=END, host_steps=1200, log=emit))
        (output/'transformed-mode-hold.py').write_text(replay.source)
        common = candidate(recipe)
        try:
            mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=1200, particle_l2=0.,
                vicreg_weight=recipe.prior_reg), gan_factory=common.make_loss,
                cap_factory=common.make_penalty, noise_policy=policy, diagnostics=True)
        except LocalContinuationComplete:
            pass
        else:
            raise RuntimeError('bounded continuation did not stop at its requested checkpoint')
    receipt = replay.receipt()
    dynamics = recorder.receipt()
    if ([row['step'] for row in receipt['checkpoints']] != list(range(START, END+1))
            or recorder.outer_steps != 20 or recorder.bank_rng_verified != 20
            or recorder.fit_rng_verified != 20 or not receipt['completed']
            or not all(row['finite'] for row in finite_checks)):
        raise RuntimeError('incomplete finite20-update continuation')
    torch.save(replay.final_state, output/'final-state.pt')
    result = dict(start=START, end=END, steps=20, finite_state_pass=True,
                  quality_observations=receipt['checkpoints'], first_fit=first_fit,
                  finite_checks=finite_checks, replay=receipt, dynamics=dynamics,
                  final_state_sha256=_sha(replay.final_state),
                  final_state_file_sha256=hashlib.sha256((output/'final-state.pt').read_bytes()).hexdigest(),
                  quality_gate_pass_claim=False, shared_gate_eligible=False)
    return result


def main():
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.capture.read_bytes()).hexdigest() != CAPTURE_SHA:
        raise RuntimeError('this gate is declared only for the captured update472 failure')
    names = ('reports/toy100/pr84_critic_refinement_finite.py',
             'reports/toy100/pr84_critic_refinement_after_step.py',
             'reports/toy100/pr84_critic_refinement_recovery.py',
             'reports/toy100/pr84_critic_refinement_resume.py',
             'reports/toy100/pr84_critic_refinement_capture.py',
             'reports/toy100/pr84_critic_refinement_cold.py',
             'reports/toy100/pr84_critic_refinement.py',
             'reports/toy100/pr84_critic_relaxation.py',
             'reports/toy100/pr84_smoothed_candidate.py',
             'reports/toy100/alternating_curvature_scratch.py',
             'benchmarks/locked_shared/mode_hold.py',
             'benchmarks/toy100/continuous_probe.py',
             'benchmarks/transfer_suite/toy100_compatibility.py',
             'configs/toy100/constraints_simple_regularization.json',
             'particlegan/gan_loss.py', 'particlegan/grad_regularizers.py',
             'tests/test_pr84_critic_refinement_finite.py',
             'tests/test_pr84_critic_refinement_after_step.py')
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    declaration = dict(method=safe.METHOD, capture_sha256=CAPTURE_SHA, source_hashes=hashes,
        order=['exact original failed fit', 'same fit with later-invalid rejection', 'updates472 through491'],
        initial_snapshot_stage='before gradient, AFTER set_step; skip only first set_step',
        original_host_budget=1200, noise_horizon=1200, start=START, end=END,
        configuration_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        numerical_criteria='exact failed-fit recovery;20 finite accepted states with constant rates and once-only moments',
        no_quality_threshold_for_this_cold_state_filter=True, shared_gate_eligible=False)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in names:
        target = args.output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    (args.output/'config.json').write_bytes(args.config.read_bytes())
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    emit(dict(event='DECLARED', **declaration))
    torch.set_num_threads(1)
    payload = torch.load(args.capture, weights_only=True)
    recipe, noise, _ = declared_recipe(json.loads(args.config.read_text()))
    try:
        exact, fit_result = exact_fit(payload, recipe)
        (args.output/'exact-fit.json').write_text(json.dumps(dict(checks=exact, result=fit_result), allow_nan=False)+'\n')
        emit(dict(event='EXACT_FIT_PASS', **exact))
        result = continuation(payload, recipe, noise, args.output)
        (args.output/'continuation.json').write_text(json.dumps(result, allow_nan=False)+'\n')
        gate = dict(status='PASS_NUMERICAL_RECOVERY', method=safe.METHOD,
            capture_sha256=CAPTURE_SHA, source_hashes=hashes, exact_failed_fit=exact,
            finite_path_parity=dict(passed=exact['finite_records_exact'],
                scope='all48 original actual finite closures and first replayed bank/metric/state exact',
                focused_unit_tests=7, complete_noise_clock_resume_tests=3),
            after_set_step_parity_exact=all(result['first_fit'].values()),
            continuation={key:result[key] for key in ('start','end','steps','finite_state_pass','quality_observations')},
            cold_acquisition_pass_claim=False, stationary_hold_pass_claim=False, shared_gate_eligible=False)
        (args.output/'gate.json').write_text(json.dumps(gate, indent=2, allow_nan=False)+'\n')
        emit(dict(event='NUMERICAL_RECOVERY_DONE', status=gate['status'],
            minimum_hq=min(row['hq'] for row in result['quality_observations']),
            minimum_modes=min(row['modes'] for row in result['quality_observations']),
            nonfinite_trials=result['dynamics']['nonfinite_fit_gradient_evaluations']))
    except BaseException as error:
        (args.output/'error.json').write_text(json.dumps(dict(status='ERROR_INCOMPLETE', error=repr(error),
            shared_gate_eligible=False), indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()

"""Gated own-state stationary continuation through update 12000.

This resumes the unperturbed own-acquired update2400 snapshot only after cold,
the dense own-state hold, and the paired same-target response all pass. The
native objective, factory, rates, Adam moments, and 1200-step noise horizon
remain unchanged. Every-update quality is observed; the first miss stops.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch

from reports.toy100 import sample_anchor_own_state_probe as own


ROOT = Path(__file__).resolve().parents[2]
START = 2400
END = 12000
SUMMARY_CADENCE = 200
LOG_CADENCE = 100


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_archived_sources(directory, expected, root):
    for name, digest in expected.items():
        if (sha((root / name).read_bytes()) != digest
                or sha((directory / 'source' / name).read_bytes()) != digest):
            raise RuntimeError(f'archived or current source changed: {name}')


def require_ready(cold, hold, response, root, entrypoint):
    """Reject every incomplete predecessor before creating any output."""
    from reports.toy100.pr84_critic_refinement_capture import _sha

    factory, method, filename = own.load_factory(root, entrypoint)
    _, cold_decl, ring, config = own.require_cold(cold, root, factory,
                                                 entrypoint, method, filename)
    own.verify_loaded_sources(root, cold_decl['source'])
    hold_decl = json.loads((hold / 'declaration.json').read_text())
    if (hold_decl.get('factory') != entrypoint or hold_decl.get('method') != method
            or hold_decl.get('cold_ring_state_file_sha256') != ring['final_state_file_sha256']
            or hold_decl.get('cold_declaration_sha256') != sha((cold / 'declaration.json').read_bytes())):
        raise RuntimeError('hold is not bound to this qualified cold factory')
    saved, hold_result = own.require_hold(hold, cold, hold_decl)
    verify_archived_sources(hold, hold_decl['source'], root)
    response_decl = json.loads((response / 'declaration.json').read_text())
    response_result = json.loads((response / 'response.json').read_text())
    if (response_decl.get('factory') != entrypoint or response_decl.get('method') != method
            or response_decl.get('source') != hold_decl['source']
            or response_decl.get('cold_ring_state_file_sha256') != ring['final_state_file_sha256']
            or response_decl.get('cold_declaration_sha256') != sha((cold / 'declaration.json').read_bytes())
            or response_decl.get('own_hold_state_file_sha256') != hold_result['final_state_file_sha256']
            or response_result.get('factory') != entrypoint
            or response_result.get('method') != method
            or response_result.get('source') != response_decl['source']
            or response_result.get('cold_ring_state_file_sha256') != ring['final_state_file_sha256']
            or response_result.get('own_hold_state_file_sha256') != hold_result['final_state_file_sha256']):
        raise RuntimeError('response is not bound to this own-state hold')
    verify_archived_sources(response, response_decl['source'], root)
    verdict = response_result.get('verdict', {})
    if (verdict.get('status') != 'PASS_LOCAL_RESPONSE'
            or verdict.get('control_all_pass') is not True
            or verdict.get('frozen_all_fail') is not True
            or verdict.get('perturbed_final_five_pass') is not True
            or verdict.get('steps') != 50 or verdict.get('first') != 2401
            or verdict.get('last') != 2450
            or response_result.get('same_target') is not True):
        raise RuntimeError('paired same-target response did not pass its declared filter')
    for role in ('control', 'perturbed'):
        branch = response_result[role]
        receipt = branch['receipt']
        if (receipt['restored_snapshot_sha256'] != _sha(saved)
                or receipt['completed_before_resume'] != START
                or receipt['target_steps'] != 2450
                or receipt['updates'] != 50 or receipt['completed'] is not True
                or receipt['actual_adam_updates'] != {'d': 50, 'g': 50}
                or receipt['optimizer_callbacks'] != {'d': 150, 'g': 150}
                or receipt['noise_horizon'] != 1200
                or [row['step'] for row in receipt['checkpoints']] != list(range(2401, 2451))):
            raise RuntimeError(f'{role} response did not use the complete own update2400 state')
    if (saved['noise']['step_calls'] != START
            or saved['noise_policy']['total_steps'] != 1200
            or hold_result['final_snapshot_sha256'] != _sha(saved)):
        raise RuntimeError('own-state continuation does not start at completed update2400')
    return factory, method, saved, config, cold_decl, hold_decl, response_decl, response_result


def _tensor_stats(values):
    tensors = [value.detach().double() for value in values]
    if not tensors:
        return dict(elements=0, l2=0., abs_max=0., finite=True)
    finite = all(bool(torch.isfinite(value).all()) for value in tensors)
    if not finite:
        raise FloatingPointError('nonfinite model or Adam tensor in passive summary')
    return dict(elements=sum(value.numel() for value in tensors),
                l2=float(sum(value.square().sum() for value in tensors).sqrt()),
                abs_max=max(float(value.abs().max()) for value in tensors), finite=True)


def _finite_state(local):
    result = {}
    for role in ('generator', 'critic', 'prior'):
        module = local[role]
        result[role] = dict(parameters=_tensor_stats(module.parameters()),
                            buffers=_tensor_stats(module.buffers()))
    for role in ('d', 'g'):
        optimizer = local['opt_' + role]
        states = [optimizer.state[p] for group in optimizer.param_groups for p in group['params']]
        steps = [int(state['step']) for state in states]
        result['adam_' + role] = dict(
            steps_min=min(steps), steps_max=max(steps),
            rates=[group['lr'] for group in optimizer.param_groups],
            first_moment=_tensor_stats(state['exp_avg'] for state in states),
            second_moment=_tensor_stats(state['exp_avg_sq'] for state in states))
    return result


def _fit_condition(rows):
    records = [record for row in rows for record in row['fit']['records']]
    conditions = []
    invalid = 0
    for record in records:
        minimum, maximum = record['singular_min'], record['singular_max']
        if minimum > 0 and math.isfinite(minimum) and math.isfinite(maximum):
            ratio = maximum / minimum
            if math.isfinite(ratio):
                conditions.append(ratio)
            else:
                invalid += 1
        else:
            invalid += 1
    conditions.sort()
    ranks = [record['rank'] for record in records]
    return dict(corrections=len(rows), jacobians=len(records),
                nonlinear_trials=sum(len(record['trials']) for record in records),
                rank_min=min(ranks) if ranks else None,
                rank_max=max(ranks) if ranks else None,
                raw_svd_condition_min=conditions[0] if conditions else None,
                raw_svd_condition_median=conditions[len(conditions)//2] if conditions else None,
                raw_svd_condition_max=conditions[-1] if conditions else None,
                zero_or_nonfinite_singular_min=invalid,
                interpretation='existing GN thin-SVD receipts; raw max/min, not active-subspace condition')


class PassiveObserver:
    def __init__(self, output):
        self.output = output
        self.state = None
        self.pending_step = None
        self.last_pre_step = None
        self.last_complete_step = START
        self.summaries = []
        self.exception = None
        self.partial_state = None
        self.quality_failure = None

    def summary(self, step):
        from reports.toy100.pr84_critic_refinement_capture import _rng, _sha, snapshot
        local = self.state.local
        before = _sha(snapshot(local))
        rng = _sha(_rng(local))
        rows = [row for row in self.state.recorder.corrections
                if step - SUMMARY_CADENCE < row['step'] <= step]
        if len(rows) != SUMMARY_CADENCE:
            raise RuntimeError('periodic fit receipts omitted a completed correction')
        result = dict(step=step, state=_finite_state(local), fit=_fit_condition(rows))
        if _sha(snapshot(local)) != before or _sha(_rng(local)) != rng:
            raise RuntimeError('passive model/Adam/Jacobian summary changed training state or RNG')
        self.summaries.append(result)
        print(json.dumps(dict(event='LONG_STATE_SUMMARY', step=step,
                              fit=result['fit'], adam_steps={role:result['state']['adam_'+role]['steps_max']
                                                               for role in ('d','g')})), flush=True)

    @contextmanager
    def resume(self, delegate, *args, **kwargs):
        from benchmarks.locked_shared import mode_hold
        from reports.toy100.pr84_critic_refinement_capture import snapshot
        from reports.toy100.pr84_critic_refinement_resume import FirstHoldFailure

        with delegate(*args, **kwargs) as state:
            self.state = state
            normal_before = state.before_step
            normal_checkpoint = state.checkpoint

            def before_step(step, local):
                self.pending_step = step + 1
                normal_before(step, local)
                self.last_pre_step = state.before_clock

            def checkpoint(step, measure):
                previous = len(state.points)
                try:
                    normal_checkpoint(step, measure)
                finally:
                    if len(state.points) > previous:
                        self.last_complete_step = step
                        point = state.points[-1]
                        passed = point['modes'] == 8 and point['hq'] >= .9
                        if step % LOG_CADENCE == 0 or not passed:
                            print(json.dumps(dict(event='LONG_PROGRESS', step=step,
                                                  modes=point['modes'], hq=point['hq'],
                                                  passed=passed)), flush=True)
                        if passed and step % SUMMARY_CADENCE == 0:
                            self.summary(step)

            state.before_step = before_step
            with patch.object(mode_hold, 'checkpoint', checkpoint):
                try:
                    yield state
                except BaseException as error:
                    if isinstance(error, FirstHoldFailure):
                        self.quality_failure = state.failure
                    else:
                        self.exception = repr(error)
                        if state.local is not None:
                            try:
                                self.partial_state = snapshot(state.local)
                            except BaseException as capture_error:
                                self.exception += f'; partial snapshot failed: {capture_error!r}'
                    raise


def _save_state(path, state):
    from reports.toy100.pr84_critic_refinement_capture import _sha
    if state is None:
        return None
    torch.save(state, path)
    return dict(file=path.name, file_sha256=sha(path.read_bytes()), snapshot_sha256=_sha(state))


def main():
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    from reports.toy100 import pr84_model_error_recovery as recovery
    from reports.toy100.pr84_critic_refinement_capture import _sha
    parser = argparse.ArgumentParser()
    parser.add_argument('--factory', required=True)
    parser.add_argument('--cold', type=Path, required=True)
    parser.add_argument('--hold', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.root.resolve()
    factory, method, saved, config, cold_decl, hold_decl, response_decl, response_result = require_ready(
        args.cold, args.hold, args.response, root, args.factory)
    recipe, noise, _ = declared_recipe(config)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source = own.bind_sources(args.output, cold_decl['source'], root)
    for name in set(hold_decl['source']) | set(response_decl['source']) | {
            'reports/toy100/sample_anchor_long_stationary_probe.py'}:
        raw = (root / name).read_bytes()
        if name in hold_decl['source'] and sha(raw) != hold_decl['source'][name]:
            raise RuntimeError(f'hold source changed: {name}')
        if name in response_decl['source'] and sha(raw) != response_decl['source'][name]:
            raise RuntimeError(f'response source changed: {name}')
        source[name] = sha(raw)
        target = args.output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    own.verify_loaded_sources(root, source)
    with factory(task='mode_hold', start_step=0, correction=True) as (_, generated):
        from reports.toy100.pr84_critic_refinement_resume import resumed_source
        _, continued = resumed_source(generated, START, END)
    (args.output / 'generated-resumed-mode_hold.py').write_text(continued)
    declaration = dict(method=method, factory=args.factory, source=source,
        cold_ring_state_file_sha256=response_result['cold_ring_state_file_sha256'],
        hold_state_file_sha256=response_result['own_hold_state_file_sha256'],
        response_file_sha256=sha((args.response / 'response.json').read_bytes()),
        input_own2400_snapshot_sha256=_sha(saved),
        generated_resume_source_sha256=sha(continued.encode()),
        completed_before_resume=START, target_steps=END, updates=END-START,
        quality_gate='every completed update: all8 modes and HQ>=.9; stop at first miss',
        summaries_every=SUMMARY_CADENCE, log_every=LOG_CADENCE,
        noise_horizon=1200, nominal_rates={'d':[.00425], 'g_prior':[.00425,.0085]},
        same_target=True, shared_gate_eligible=False,
        scope='finite stationary continuation through12000; no claim of infinite stability')
    own.write_json(args.output / 'declaration.json', declaration)
    print(json.dumps(dict(event='LONG_DECLARED', **declaration)), flush=True)
    observer = PassiveObserver(args.output)
    original_resume = recovery.resume_mode_hold

    @contextmanager
    def monitored(*a, **kw):
        with observer.resume(original_resume, *a, **kw) as state:
            yield state

    try:
        with patch.object(recovery, 'resume_mode_hold', monitored):
            branch, actual = own.run_bound(saved, recipe, noise, factory,
                completed=START, target=END, fail_fast=True)
        if actual != continued or branch['receipt']['restored_snapshot_sha256'] != _sha(saved):
            raise RuntimeError('long continuation source or restored state changed')
        receipt = branch['receipt']
        status = 'PASS_FINITE_12000' if (receipt['pass_all'] and receipt['completed']
                  and receipt['updates'] == END-START) else 'FAIL_FIRST_CHECK'
        final = _save_state(args.output / 'last-complete-state.pt', branch['state'])
        quality_failure = None
        if observer.quality_failure is not None:
            quality_failure = dict(step=observer.quality_failure['step'],
                before_set_step=_save_state(args.output/'failure-before-set-step.pt',
                                            observer.quality_failure['before_set_step']),
                after_checkpoint=_save_state(args.output/'failure-after-checkpoint.pt',
                                             observer.quality_failure['after_checkpoint']))
        result = dict(status=status, method=method, factory=args.factory,
            receipt=receipt, dynamics=branch['dynamics'], result=branch['result'],
            applied=branch['applied'], policy=branch['policy'],
            generated_resume_source_sha256=sha(actual.encode()),
            input_own2400_snapshot_sha256=_sha(saved), final_state=final,
            quality_failure=quality_failure, periodic=observer.summaries,
            first_failure_step=receipt['first_failure_step'],
            no_distribution_shift=True, no_extra_training_queries=True,
            shared_gate_eligible=False)
        own.write_json(args.output/'long.json', result)
        print(json.dumps(dict(event='LONG_DONE', status=status,
                              updates=receipt['updates'], failure=receipt['first_failure_step'])), flush=True)
    except BaseException as error:
        pre = _save_state(args.output/'numerical-last-before-set-step.pt', observer.last_pre_step or saved)
        partial = _save_state(args.output/'numerical-partial-state.pt', observer.partial_state)
        own.write_json(args.output/'error.json', dict(status='ERROR_INCOMPLETE',
            error=repr(error), observer_error=observer.exception,
            pending_step=observer.pending_step, last_complete_step=observer.last_complete_step,
            safe_before_set_step=pre, partial_state=partial,
            partial_stage='mid-update; not a restart boundary', periodic=observer.summaries,
            shared_gate_eligible=False))
        print(json.dumps(dict(event='LONG_ERROR', step=observer.pending_step,
                              last_complete=observer.last_complete_step,
                              error=repr(error))), flush=True)
        raise


if __name__ == '__main__':
    main()

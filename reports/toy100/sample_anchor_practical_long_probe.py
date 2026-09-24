"""Finite same-target continuation through quality dips; frozen anchor update.

This is a separate observer/grade epoch from the strict first-failure driver.
It resumes the same qualified own-acquired update-2400 state, runs the
unchanged prestart anchor factory to update 12000, and records every quality
failure and recovery. Numerical failures still stop immediately.
"""

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch


START, END = 2400, 12000
LOG_CADENCE, SUMMARY_CADENCE = 100, 200
FACTORY = 'reports.toy100.sample_anchor_prestart_candidate:sample_anchor_prestart_candidate'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def grade(row):
    return dict(step=row['step'], modes=row['modes'], hq=row['hq'],
                passed=row['modes'] == 8 and row['hq'] >= .9)


def summarize_quality(rows):
    if [row['step'] for row in rows] != list(range(START + 1, END + 1)):
        raise RuntimeError('missing or duplicated practical continuation checkpoint')
    failures = [row for row in rows if not row['passed']]
    runs = []
    active = None
    for row in rows:
        if row['passed']:
            if active is not None:
                active['length'] = active['last'] - active['first'] + 1
                active['recovered_at'] = row['step']
                runs.append(active)
                active = None
            continue
        if active is None:
            active = dict(first=row['step'], last=row['step'], min_modes=row['modes'],
                          min_hq=row['hq'])
        else:
            active['last'] = row['step']
            active['min_modes'] = min(active['min_modes'], row['modes'])
            active['min_hq'] = min(active['min_hq'], row['hq'])
    if active is not None:
        active['length'] = active['last'] - active['first'] + 1
        active['recovered_at'] = None
        runs.append(active)
    return dict(checks=len(rows), passing_checks=len(rows)-len(failures),
                failing_checks=len(failures), first_failure_step=(failures[0]['step'] if failures else None),
                min_modes=min(row['modes'] for row in rows),
                min_hq=min(row['hq'] for row in rows),
                failure_runs=runs, longest_failure_run=max((run['length'] for run in runs), default=0),
                endpoint=rows[-1], last_50=rows[-50:], last_200_passing=sum(row['passed'] for row in rows[-200:]))


def observer_class(strict):
    class PracticalObserver(strict.PassiveObserver):
        def __init__(self, output):
            super().__init__(output)
            self.quality = []
            self.open_failure = None

        @contextmanager
        def resume(self, delegate, *args, **kwargs):
            from benchmarks.locked_shared import mode_hold
            from reports.toy100.pr84_critic_refinement_capture import snapshot

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
                    normal_checkpoint(step, measure)
                    if len(state.points) == previous:
                        return
                    self.last_complete_step = step
                    row = grade(state.points[-1])
                    self.quality.append(row)
                    if not row['passed'] and self.open_failure is None:
                        self.open_failure = step
                        print(json.dumps(dict(event='PRACTICAL_FAILURE_START', **row)), flush=True)
                    elif row['passed'] and self.open_failure is not None:
                        print(json.dumps(dict(event='PRACTICAL_RECOVERY', first_failure=self.open_failure,
                                              recovered_at=step, failed_updates=step-self.open_failure)), flush=True)
                        self.open_failure = None
                    if step % LOG_CADENCE == 0:
                        print(json.dumps(dict(event='PRACTICAL_PROGRESS', **row,
                                              checks=len(self.quality),
                                              failing_checks=sum(not value['passed'] for value in self.quality))),
                              flush=True)
                    if step % SUMMARY_CADENCE == 0:
                        self.summary(step)

                state.before_step = before_step
                with patch.object(mode_hold, 'checkpoint', checkpoint):
                    try:
                        yield state
                    except BaseException as error:
                        self.exception = repr(error)
                        if state.local is not None:
                            try:
                                self.partial_state = snapshot(state.local)
                            except BaseException as capture_error:
                                self.exception += f'; partial snapshot failed: {capture_error!r}'
                        raise

    return PracticalObserver


def save_state(path, value):
    if value is None:
        return None
    from reports.toy100.pr84_critic_refinement_capture import _sha
    torch.save(value, path)
    return dict(file=path.name, file_sha256=sha(path.read_bytes()), snapshot_sha256=_sha(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cold', type=Path, required=True)
    parser.add_argument('--hold', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from reports.toy100 import sample_anchor_long_stationary_probe as strict
    from reports.toy100 import sample_anchor_own_state_probe as own
    from reports.toy100 import pr84_model_error_recovery as recovery
    from reports.toy100.pr84_critic_refinement_capture import _sha
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe

    torch.set_num_threads(1)
    factory, method, saved, config, cold_decl, hold_decl, response_decl, response_result = strict.require_ready(
        args.cold, args.hold, args.response, root, FACTORY)
    if args.output.exists():
        raise FileExistsError(args.output)
    recipe, noise, _ = declared_recipe(config)
    args.output.mkdir(parents=True)
    sources = own.bind_sources(args.output, cold_decl['source'], root)
    for name in set(hold_decl['source']) | set(response_decl['source']) | {
            'reports/toy100/sample_anchor_long_stationary_probe.py'}:
        raw = (root / name).read_bytes()
        if name in hold_decl['source'] and sha(raw) != hold_decl['source'][name]:
            raise RuntimeError(f'hold source changed: {name}')
        if name in response_decl['source'] and sha(raw) != response_decl['source'][name]:
            raise RuntimeError(f'response source changed: {name}')
        sources[name] = sha(raw)
        target = args.output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    own.verify_loaded_sources(root, sources)
    this_source = Path(__file__).read_bytes()
    source_target = args.output / 'source/reports/toy100/sample_anchor_practical_long_probe.py'
    source_target.parent.mkdir(parents=True, exist_ok=True)
    source_target.write_bytes(this_source)
    sources['reports/toy100/sample_anchor_practical_long_probe.py'] = sha(this_source)
    with factory(task='mode_hold', start_step=0, correction=True) as (_, generated):
        from reports.toy100.pr84_critic_refinement_resume import resumed_source
        _, continued = resumed_source(generated, START, END)
    (args.output / 'generated-resumed-mode_hold.py').write_text(continued)
    declaration = dict(method=method, factory=FACTORY, source=sources,
        cold_ring_state_file_sha256=response_result['cold_ring_state_file_sha256'],
        hold_state_file_sha256=response_result['own_hold_state_file_sha256'],
        response_file_sha256=sha((args.response/'response.json').read_bytes()),
        input_own2400_snapshot_sha256=_sha(saved),
        generated_resume_source_sha256=sha(continued.encode()),
        completed_before_resume=START, target_steps=END, updates=END-START,
        quality_policy='continue through all finite failures; record every update and all failure/recovery runs',
        numerical_policy='stop and archive last before-step plus marked partial state',
        summaries_every=SUMMARY_CADENCE, log_every=LOG_CADENCE,
        noise_horizon=1200, nominal_rates={'d':[.00425], 'g_prior':[.00425,.0085]},
        same_target=True, method_source_unchanged=True, shared_gate_eligible=False,
        strict_first_failure_gate_unchanged=True,
        scope='finite stationary continuation through12000; descriptive practical persistence, no infinity claim')
    save_json(args.output/'declaration.json', declaration)
    print(json.dumps(dict(event='PRACTICAL_DECLARED', input_snapshot=_sha(saved),
                          generated_source=declaration['generated_resume_source_sha256'],
                          updates=END-START, factory=FACTORY)), flush=True)

    observer = observer_class(strict)(args.output)
    original_resume = recovery.resume_mode_hold

    @contextmanager
    def monitored(*a, **kw):
        with observer.resume(original_resume, *a, **kw) as state:
            yield state

    started = time.perf_counter()
    try:
        with patch.object(recovery, 'resume_mode_hold', monitored):
            branch, actual = own.run_bound(saved, recipe, noise, factory,
                                           completed=START, target=END, fail_fast=False)
        if (actual != continued or branch['receipt']['restored_snapshot_sha256'] != _sha(saved)
                or branch['receipt']['updates'] != END-START
                or not branch['receipt']['completed']):
            raise RuntimeError('practical continuation source, input or completion changed')
        receipt = branch['receipt']
        if [grade(row) for row in receipt['checkpoints']] != observer.quality:
            raise RuntimeError('passive every-update quality record differs from host receipt')
        reference = response_result['control']['receipt']['checkpoints']
        if receipt['checkpoints'][:50] != reference:
            raise RuntimeError('first50 unperturbed updates differ from prior paired response control')
        quality = summarize_quality(observer.quality)
        status = 'PASS_FINITE_12000' if quality['failing_checks'] == 0 else 'COMPLETE_WITH_STRICT_FAILURES'
        final_state = save_state(args.output/'final-state.pt', branch['state'])
        with gzip.open(args.output/'raw.json.gz', 'wt') as file:
            json.dump(dict(receipt=receipt, dynamics=branch['dynamics'], result=branch['result'],
                           applied=branch['applied'], policy=branch['policy']), file, allow_nan=False)
        result = dict(status=status, method=method, factory=FACTORY,
            quality=quality, every_update=observer.quality, periodic=observer.summaries,
            first50_unperturbed_response_control_exact=True,
            input_own2400_snapshot_sha256=_sha(saved), final_state=final_state,
            generated_resume_source_sha256=sha(actual.encode()),
            actual_adam_updates=receipt['actual_adam_updates'],
            optimizer_callbacks=receipt['optimizer_callbacks'],
            noise_horizon=receipt['noise_horizon'],
            final_noise_step_calls=branch['state']['noise']['step_calls'],
            applied=branch['applied'], elapsed_seconds=time.perf_counter()-started,
            strict_gate_passed=quality['failing_checks']==0,
            shared_gate_eligible=False)
        save_json(args.output/'result.json', result)
        print(json.dumps(dict(event='PRACTICAL_DONE', status=status,
                              passing=quality['passing_checks'], failing=quality['failing_checks'],
                              first_failure=quality['first_failure_step'],
                              endpoint=quality['endpoint'], elapsed_seconds=result['elapsed_seconds'])), flush=True)
    except BaseException as error:
        before = save_state(args.output/'numerical-last-before-set-step.pt', observer.last_pre_step or saved)
        partial = save_state(args.output/'numerical-partial-state.pt', observer.partial_state)
        save_json(args.output/'error.json', dict(status='ERROR_INCOMPLETE', error=repr(error),
            observer_error=observer.exception, pending_step=observer.pending_step,
            last_complete_step=observer.last_complete_step,
            safe_before_set_step=before, partial_state=partial,
            partial_stage='mid-update; not a restart boundary',
            every_update=observer.quality, periodic=observer.summaries,
            elapsed_seconds=time.perf_counter()-started, shared_gate_eligible=False))
        print(json.dumps(dict(event='PRACTICAL_ERROR', pending=observer.pending_step,
                              last_complete=observer.last_complete_step, error=repr(error))), flush=True)
        raise


if __name__ == '__main__':
    main()

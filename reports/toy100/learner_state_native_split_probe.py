"""Exact two-update native mode-hold split check for confirmed support memory.

The full path executes updates 1 and 2. The split path stops after update 1,
then restores the complete host-plus-learner envelope before update 2's
set_step and real bank. Both use the original 1200-step recipe/noise horizon;
the stop occurs after the ordinary checkpoint and EMA, before optional final
host diagnostics. This is a bounded correctness check, not an acquisition run.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = ROOT
sys.path.insert(0, str(LOCAL_ROOT))
CONFIG = 'configs/toy100/constraints_simple_regularization.json'
FACTORY = 'reports.toy100.sample_anchor_memory_candidate:sample_anchor_memory_candidate'
SOURCE_NAMES = (
    'reports/toy100/sample_anchor_memory_candidate.py',
    'reports/toy100/sample_group_two_bank_memory.py',
    'reports/toy100/sample_anchor_prestart_candidate.py',
    'reports/toy100/sample_anchor_rest_candidate.py',
    'reports/toy100/sample_anchor_candidate.py',
    'reports/toy100/sample_group_anchor.py',
    'reports/toy100/reallocation_smoothed_candidate.py',
    'reports/toy100/joint_output_pullback.py',
    'reports/toy100/pr84_smoothed_candidate.py',
    'reports/toy100/alternating_curvature_scratch.py',
    'reports/toy100/extra_adam_scratch.py',
    'reports/toy100/pr84_critic_refinement_capture.py',
    'reports/toy100/pr84_critic_refinement_resume.py',
    'reports/toy100/learner_state_envelope.py',
    'reports/toy100/learner_state_split_harness.py',
    'reports/toy100/learner_state_native_split_probe.py',
    'benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/observation.py',
    'benchmarks/locked_shared/mlp.py',
    'benchmarks/transfer_suite/legacy_noise_adapters.py',
    'benchmarks/transfer_suite/compare_defaults.py',
    'benchmarks/transfer_suite/toy100_compatibility.py',
    'benchmarks/toy100/continuous_probe.py',
    'benchmarks/learned_lr_evaluation.py',
    'particlegan/gan_loss.py',
    'particlegan/grad_regularizers.py',
    'particlegan/particle_prior.py',
    CONFIG,
)


class StopAtBoundary(Exception):
    """The declared completed update and its ordinary checkpoint finished."""


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sources(root, *, archive=None):
    from reports.toy100.learner_state_envelope import source_fingerprint
    paths = {}
    for name in SOURCE_NAMES:
        path = ((LOCAL_ROOT if name in {
            'reports/toy100/learner_state_envelope.py',
            'reports/toy100/learner_state_split_harness.py',
            'reports/toy100/learner_state_native_split_probe.py'} else root) / name)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        paths[name] = sha(raw)
        if archive is not None:
            target = archive / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    return paths, source_fingerprint(paths)


def setup(root):
    from benchmarks.toy100.continuous_probe import _noise_policy, prepared_config
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    from reports.toy100.sample_anchor_memory_candidate import METHOD, sample_anchor_memory_candidate
    config = json.loads((root / CONFIG).read_text())
    recipe, noise, _ = declared_recipe(prepared_config(config, 'constant'))
    if (recipe.lr_floor != 1. or recipe.lr_anneal_start != 0.
            or recipe.prior_reg != 0.):
        raise RuntimeError('native split configuration is not the constant unregularized ring')
    return recipe, noise, METHOD, sample_anchor_memory_candidate, _noise_policy


def _host_call(recipe, policy, *, steps):
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.compare_defaults import candidate
    common = candidate(recipe)
    return mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(
        steps=steps, particle_l2=0., vicreg_weight=recipe.prior_reg),
        gan_factory=common.make_loss, cap_factory=common.make_penalty,
        noise_policy=policy, diagnostics=True)


def _stack(recipe):
    from benchmarks import learned_lr_evaluation as bridge
    from benchmarks.smart_descent import evaluate
    from benchmarks.transfer_suite import vector_tasks
    from benchmarks.transfer_suite.compare_defaults import optimizer_defaults
    stack = ExitStack()
    applied = []
    stack.enter_context(optimizer_defaults(recipe, applied))
    control = evaluate.FixedControl(vector_tasks.fixed_policy('cosine'), 1200)
    stack.enter_context(bridge.control_host_schedules(control))
    return stack, applied


def _measure(step, measure, local):
    from reports.toy100.pr84_critic_refinement_capture import _rng, _sha
    before = _sha(_rng(local))
    with torch.random.fork_rng(devices=[]):
        point = dict(step=step, **measure())
    if _sha(_rng(local)) != before:
        raise RuntimeError('short exact observation changed training RNG')
    return point


def fresh(stop, *, root, recipe, noise, method, factory, make_policy,
          source_sha256):
    from benchmarks.locked_shared import mode_hold
    from reports.toy100.learner_state_envelope import capture
    from reports.toy100.pr84_critic_refinement_capture import _sha, snapshot
    if stop not in (1, 2):
        raise ValueError('fresh branch is limited to one or two completed updates')
    policy = make_policy(noise, 1200)
    stack, applied = _stack(recipe)
    result = {}
    with stack:
        recorder, generated = stack.enter_context(factory(task='mode_hold',
            start_step=0, correction=True))
        original_checkpoint = mode_hold.checkpoint
        initial_host = None
        initial_learner = None
        points = []

        original_phases = recorder.phases

        def phases(step, opt_d, opt_g, local):
            nonlocal initial_host, initial_learner
            if step == 0:
                if initial_host is not None:
                    raise RuntimeError('first native phase was entered twice')
                # The phase hook follows set_step(0) and precedes the first
                # real draw. Its complete host and empty learner hashes must
                # agree between independently initialized fresh branches.
                initial_host = _sha(snapshot(local))
                initial_learner = _sha(recorder.learner_state_dict())
            yield from original_phases(step, opt_d, opt_g, local)

        def checkpoint(step, measure):
            local = recorder._local
            point = _measure(step, measure, local)
            points.append(point)
            original_checkpoint(step, lambda: {k:v for k,v in point.items() if k!='step'})
            if step == stop:
                result['envelope'] = capture(local, recorder, method=method,
                    source_sha256=source_sha256, completed_step=step)
                raise StopAtBoundary()

        with patch.object(recorder, 'phases', phases), patch.object(mode_hold, 'checkpoint', checkpoint):
            try:
                _host_call(recipe, policy, steps=1200)
            except StopAtBoundary:
                pass
            else:
                raise RuntimeError('short fresh host did not stop at its declared checkpoint')
        if (len(points) != stop or recorder.learner_bank_count != stop
                or initial_host is None or initial_learner is None):
            raise RuntimeError('fresh branch missed a native bank or checkpoint')
        result.update(source_sha256=source_sha256, steps=list(range(1,stop+1)),
            observations=points, initial_host_sha256=initial_host,
            initial_learner_sha256=initial_learner,
            generated_host_source_sha256=sha(generated.encode()),
            final_host_sha256=_sha(snapshot(recorder._local)),
            learner_bank_count=recorder.learner_bank_count,
            learner_last_observed_step=recorder.learner_last_observed_step,
            applied=applied,
            dynamics=recorder.receipt())
    if result['final_host_sha256'] != result['envelope']['host_sha256']:
        raise RuntimeError('fresh envelope is not the completed live host state')
    return result


def resume(prefix, target, *, root, recipe, noise, method, factory,
           make_policy, source_sha256):
    from benchmarks.locked_shared import mode_hold
    from reports.toy100.learner_state_envelope import (
        attach_before_next_bank, capture, validate)
    from reports.toy100.pr84_critic_refinement_capture import _sha, snapshot
    from reports.toy100.pr84_critic_refinement_resume import resume_mode_hold
    if target != 2:
        raise ValueError('short memory resume only checks update two')
    validate(prefix, method=method, source_sha256=source_sha256,
             completed_step=1)
    policy = make_policy(noise, 1200)
    stack, applied = _stack(recipe)
    result = {}
    with stack:
        recorder, generated = stack.enter_context(factory(task='mode_hold',
            start_step=0, correction=True))
        state = stack.enter_context(resume_mode_hold(recorder, generated,
            prefix['host'], completed_steps=1, target_steps=2,
            fail_fast=False))
        stack.enter_context(attach_before_next_bank(state, recorder, prefix,
            method=method, source_sha256=source_sha256))
        ordinary_checkpoint = mode_hold.checkpoint

        def checkpoint(step, measure):
            ordinary_checkpoint(step, measure)
            if step == target:
                result['envelope'] = capture(recorder._local, recorder,
                    method=method, source_sha256=source_sha256,
                    completed_step=step)
                raise StopAtBoundary()

        with patch.object(mode_hold, 'checkpoint', checkpoint):
            try:
                _host_call(recipe, policy, steps=target)
            except StopAtBoundary:
                pass
            else:
                raise RuntimeError('short resumed host did not stop at checkpoint two')
        result.update(source_sha256=source_sha256, steps=[2],
            observations=list(state.points),
            restored_host_sha256=prefix['host_sha256'],
            loaded_envelope_sha256=prefix['envelope_sha256'],
            learner_loaded_before_next_bank=state.restored,
            generated_host_source_sha256=sha(generated.encode()),
            generated_resume_source_sha256=state.source_sha256,
            learner_bank_count=recorder.learner_bank_count,
            learner_last_observed_step=recorder.learner_last_observed_step,
            applied=applied, dynamics=recorder.receipt())
    if (_sha(state.final_state) != result['envelope']['host_sha256']
            or _sha(snapshot(recorder._local)) != result['envelope']['host_sha256']):
        raise RuntimeError('resumed envelope is not the exact finalized host state')
    return result


def main():
    from reports.toy100.learner_state_envelope import source_fingerprint
    from reports.toy100.learner_state_split_harness import run_short_split
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(LOCAL_ROOT))
    recipe, noise, method, factory, make_policy = setup(root)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    binding, fingerprint = sources(root, archive=args.output)
    declaration = dict(scope='native mode_hold update1→2 exact split only',
        method=method, factory=FACTORY, source=binding,
        source_fingerprint=fingerprint, noise_horizon=1200,
        nominal_rates=dict(d=.00425,g=.00425,prior=.0085),
        expected_learner_transition='pending after1, confirmed after2',
        shared_gate_eligible=False)
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='SPLIT_DECLARED',method=method,
                          source_fingerprint=fingerprint)), flush=True)

    def full(end):
        return fresh(end,root=root,recipe=recipe,noise=noise,method=method,
            factory=factory,make_policy=make_policy,source_sha256=fingerprint)

    def prefix(end):
        return fresh(end,root=root,recipe=recipe,noise=noise,method=method,
            factory=factory,make_policy=make_policy,source_sha256=fingerprint)

    def resumed(saved,end):
        return resume(saved,end,root=root,recipe=recipe,noise=noise,
            method=method,factory=factory,make_policy=make_policy,
            source_sha256=fingerprint)

    runs={}
    def traced(name, fn):
        def call(*args):
            value=fn(*args)
            runs[name]=value
            print(json.dumps(dict(event='SPLIT_BRANCH',name=name,
                steps=value['steps'], learner_banks=value['learner_bank_count'],
                envelope_sha256=value['envelope']['envelope_sha256'])),flush=True)
            return value
        return call
    try:
        first_full=traced('full',full)(2)
        verdict=run_short_split(lambda end: first_full,traced('prefix',prefix),
            traced('resumed',resumed),method=method,source_sha256=fingerprint,
            initial_host_sha256=first_full['initial_host_sha256'],
            start_step=0,split_step=1,target_step=2)
        if (runs['prefix']['envelope']['learner']['confirmed'] is not False
                or runs['resumed']['envelope']['learner']['confirmed'] is not True
                or runs['full']['generated_host_source_sha256'] !=
                   runs['resumed']['generated_host_source_sha256']):
            raise RuntimeError('native split did not test the same pending→confirmed candidate')
        torch.save({name: value['envelope'] for name,value in runs.items()},
                   args.output/'complete-envelopes.pt')
        slim={name: {key:value for key,value in branch.items()
                     if key not in ('envelope','dynamics')}
              for name,branch in runs.items()}
        result=dict(status='EXACT_SPLIT_PASS',verdict=verdict,branches=slim,
            envelope_file_sha256=sha((args.output/'complete-envelopes.pt').read_bytes()),
            source_fingerprint=fingerprint,shared_gate_eligible=False)
        (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(event='SPLIT_DONE',status=result['status'])),flush=True)
    except BaseException as error:
        (args.output/'error.json').write_text(json.dumps(dict(status='ERROR_INCOMPLETE',
            error=repr(error),completed_branches=list(runs)))+'\n')
        raise


if __name__ == '__main__':
    main()

"""Exact-head PR93 cold acquisition and same-process PR84 controls.

The submitted adapter is not edited. An unsupported trajectory factory is
retained as a setup error and does not prevent the independently requested
ring acquisition measurement. No hold, coefficient or radius sweep.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

HEAD = 'e958a81c49b4dc3c2b3eaad2a17000d70a091831'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    code = args.checkout.resolve()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip() != HEAD:
        raise RuntimeError('wrong PR93 checkout')
    subprocess.run(['git', 'diff', '--quiet', HEAD, '--'], cwd=code, check=True)
    sys.path.insert(0, str(code))
    import torch
    from benchmarks.toy100.warm_equilibrium_probe import _feed_hash
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    from reports.toy100.exit_aware_step_clip import exit_aware_candidate, METHOD
    from reports.toy100.pr84_smoothed_parity import _without_runtime_timing
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((code/'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name='pr93_exact_cold_independent_audit', lr_floor=1., lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap'); config.pop('network_lr_floor')
    recipe, noise, _ = declared_recipe(config)
    model_policy = declared_model_policy(config)
    specs = {row['spec']['name']: row['spec'] for row in plan()}
    declaration = dict(method=METHOD, head=HEAD, shared_gate_eligible=False,
        pinned_environment=dict(torch=torch.__version__, python=sys.version, device='cpu', threads=1),
        submitter_environment='torch2.14.0+cpu; not the locally pinned2.13.0+cu126',
        tasks=['trajectory','mode_hold'], budgets={task:specs[task]['steps'] for task in ('trajectory','mode_hold')},
        variants=['pr84','pr93_disabled','pr93'],
        scope='cold acquisition only; PR93 source/constants unchanged; no warm or hold rerun',
        setup_failure='retain unsupported trajectory setup error, continue independently requested ring audit',
        config=config, driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    (args.output/'source').mkdir()
    (args.output/'source/pr93_cold_audit.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)

    def digest(value):
        h = hashlib.sha256(); _feed_hash(h, value); return h.hexdigest()

    def snapshot(recorder):
        local = recorder._local
        policy = local['noise_policy']
        noise_state = {name: value.get_state().clone() if isinstance(value, torch.Generator) else deepcopy(value)
                       for name, value in vars(policy).items()}
        return dict(generator=deepcopy(local['generator'].state_dict()),
            critic=deepcopy(local['critic'].state_dict()), prior=deepcopy(local['prior'].state_dict()),
            optimizer_g=deepcopy(local['opt_g'].state_dict()), optimizer_d=deepcopy(local['opt_d'].state_dict()),
            ema_g=deepcopy(local.get('ema_g')), ema_z=deepcopy(local.get('ema_z')),
            data_stream=None if local.get('stream') is None else local['stream'].get_state().clone(),
            torch_rng=torch.get_rng_state().clone(), noise_policy=noise_state,
            snapshot_scope='after host final evaluation and diagnostics; full current fixed-noise policy')

    raw, stages = {}, []
    started = time.perf_counter()
    for task in declaration['tasks']:
        for variant in declaration['variants']:
            print(json.dumps(dict(event='STAGE_START', task=task, variant=variant)), flush=True)
            start = time.perf_counter(); recorder = None
            try:
                context = (pr84_smoothed_candidate(task=task) if variant=='pr84'
                           else exit_aware_candidate(task=task))
                with context as (recorder, generated):
                    if variant != 'pr84':
                        recorder.clip_enabled = variant == 'pr93'
                    result, details = run_legacy(specs[task], recipe, noise, model_policy=model_policy)
                state = snapshot(recorder)
                moments = {}
                for role in ('d','g'):
                    states = state['optimizer_'+role]['state'].values()
                    moments[role] = sorted(set(float(item['step']) for item in states))
                    if moments[role] != [float(specs[task]['steps'])]:
                        raise AssertionError(f'{task}/{variant}: moment ownership differs')
                if recorder.outer_steps != specs[task]['steps'] or recorder.rng_replay_verified != 2*specs[task]['steps']:
                    raise AssertionError('outer count or same-batch replay failed')
                for optimizer in recorder.optimizers:
                    if recorder.rows[optimizer]['calls'] != 3*specs[task]['steps']:
                        raise AssertionError('native gradient callback count differs')
                verdict = test_verdict(specs[task], result)
                value = dict(task=task, variant=variant, result=result, verdict=verdict,
                    applied=details['applied'], noise=details['noise_receipt'], dynamics=recorder.receipt(),
                    final_state_sha256=digest(state), final_moments=moments,
                    actual_gradient_blocks_per_role=3*specs[task]['steps'],
                    additional_prior_projection='PR93 per-clipped-particle output Jacobians; uncounted by inherited3-field receipt',
                    generated_source_sha256=hashlib.sha256(generated.encode()).hexdigest(),
                    shared_gate_eligible=False, elapsed_seconds=time.perf_counter()-start)
                torch.save(state, args.output/f'{task}-{variant}-state.pt')
                (args.output/f'{task}-{variant}.json').write_text(json.dumps(value, allow_nan=False)+'\n')
                (args.output/f'source/generated-{task}-{variant}.py').write_text(generated)
                raw[task,variant] = value
                stages.append(dict(task=task, variant=variant, verdict=verdict, live=result['live'],
                    elapsed_seconds=value['elapsed_seconds'], clips=value['dynamics'].get('exit_clip_updates',0),
                    state_sha256=value['final_state_sha256']))
                if variant=='pr93_disabled':
                    control = raw[task,'pr84']
                    checks = dict(full_state=value['final_state_sha256']==control['final_state_sha256'],
                        observations=_without_runtime_timing(result)==_without_runtime_timing(control['result']),
                        rates=value['applied']==control['applied'], noise=value['noise']==control['noise'],
                        records=value['dynamics']['records']==control['dynamics']['records'],
                        generated_source=value['generated_source_sha256']==control['generated_source_sha256'])
                    stages[-1]['disabled_exact_controls'] = checks
                    if not all(checks.values()):
                        raise AssertionError(f'disabled PR93 differs from PR84: {checks}')
            except Exception as error:
                failure = dict(task=task, variant=variant, error=repr(error), traceback=traceback.format_exc(),
                    completed_outer_steps=None if recorder is None else recorder.outer_steps,
                    elapsed_seconds=time.perf_counter()-start)
                (args.output/f'{task}-{variant}.error.json').write_text(json.dumps(failure, indent=2)+'\n')
                stages.append(failure)
                # A known unsupported context is a setup error, not a quality result.
                if not (task=='trajectory' and variant!='pr84' and isinstance(error, AttributeError)
                        and "sample_ring" in str(error) and recorder is None):
                    raise
            print(json.dumps(dict(event='STAGE_DONE', **stages[-1])), flush=True)
            (args.output/'progress.json').write_text(json.dumps(stages, indent=2)+'\n')
    hashes = {}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path is None: continue
        path = Path(path).resolve()
        if path.suffix == '.py' and path.is_relative_to(code):
            name = str(path.relative_to(code)); data = path.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            target = args.output/'source/checkout'/name
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    for name in ('reports/toy100/exit_aware_step_clip.py','configs/toy100/constraints_simple_regularization.json'):
        data = (code/name).read_bytes(); hashes[name] = hashlib.sha256(data).hexdigest()
        target = args.output/'source/checkout'/name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    subprocess.run(['git', 'diff', '--quiet', HEAD, '--'], cwd=code, check=True)
    summary = dict(declaration=declaration, stages=stages, executed_sources=hashes,
                   seconds=time.perf_counter()-started, shared_gate_eligible=False)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(event='DONE', stages=stages, seconds=summary['seconds'])), flush=True)


if __name__ == '__main__':
    main()

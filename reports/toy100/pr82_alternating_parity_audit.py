"""Independently compare PR82's unbounded adapter to the frozen alternating host.

This is a read-only, fixed-seed audit driver. It records full optimizer tensors,
host metrics, training RNG, and noise receipts for both original hosts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy, run_legacy
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
from reports.toy100.alternating_curvature_scratch import alternating_curvature
from reports.toy100.extra_adam_scratch import extra_adam


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def normalized(value):
    if isinstance(value, torch.Tensor):
        array = value.detach().contiguous().cpu().numpy()
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    return value


def untimed(value):
    if isinstance(value, dict):
        return {k: untimed(v) for k, v in value.items() if k not in ('seconds', 'stable_from_seconds', 'confirmed_seconds')}
    if isinstance(value, list):
        return [untimed(v) for v in value]
    return value


def one(task, recipe, noise, model_policy, wrapped):
    spec = next(row['spec'] for row in plan() if row['spec']['name'] == task)
    created = []
    policies = []
    original_init = torch.optim.Adam.__init__
    original_policy_init = NoisePolicy.__init__

    def capture_init(optimizer, *args, **kwargs):
        original_init(optimizer, *args, **kwargs)
        created.append(optimizer)

    def capture_policy(policy, *args, **kwargs):
        original_policy_init(policy, *args, **kwargs)
        policies.append(policy)

    with patch.object(torch.optim.Adam, '__init__', capture_init), patch.object(NoisePolicy, '__init__', capture_policy):
        if wrapped:
            with alternating_curvature(task=task, bound_d=True, curvature_bound=1e9,
                                       d_curvature_bound=1e9) as (recorder, _):
                result, context = run_legacy(spec, recipe, noise, model_policy=model_policy)
            adapter = dict(outer_steps=recorder.outer_steps,
                           rng_replay_verified=recorder.rng_replay_verified,
                           all_factors_one=all(row['d']['factor'] == row['g']['factor'] == 1.
                                               for row in recorder.records),
                           generated_source_sha256=recorder.host_source['generated_function_sha256'])
        else:
            result, context = run_legacy(spec, recipe, noise, model_policy=model_policy)
            adapter = None
    if len(created) != 2:
        raise RuntimeError(f'expected D and G optimizers, found {len(created)}')
    if len(policies) != 1:
        raise RuntimeError(f'expected one noise policy, found {len(policies)}')
    policy = policies[0]
    stream_digest = {name: hashlib.sha256(stream.get_state().numpy().tobytes()).hexdigest()
                     for name in ('input_stream', 'output_stream')
                     if (stream := getattr(policy, name, None)) is not None}
    optimizer_digest = digest(normalized([dict(groups=[
        dict(hyperparameters={key: value for key, value in group.items() if key != 'params'},
             parameters=[parameter.detach().clone() for parameter in group['params']])
        for group in optimizer.param_groups], state=optimizer.state_dict()) for optimizer in created]))
    return dict(verdict=test_verdict(spec, result), result_sha256=digest(untimed(result)),
                optimizer_state_sha256=optimizer_digest,
                global_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                noise_stream_state_sha256=stream_digest,
                noise_receipt_sha256=digest(context['noise_receipt']),
                applied=context['applied'], live=result['live'], adapter=adapter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name='alternating_curvature_response', lr_floor=1., lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap')
    config.pop('network_lr_floor')
    recipe, noise, _ = declared_recipe(config)
    model_policy = declared_model_policy(config)
    tasks = {}
    for task in ('trajectory', 'mode_hold'):
        original = one(task, recipe, noise, model_policy, False)
        wrapped = one(task, recipe, noise, model_policy, True)
        # Replays intentionally increase the policy's counting receipts; its
        # final training streams and single outside-the-block noise clock must
        # still agree with the original host.
        fields = ('result_sha256', 'optimizer_state_sha256', 'global_rng_sha256',
                  'noise_stream_state_sha256', 'applied', 'live')
        parity = {field: original[field] == wrapped[field] for field in fields}
        tasks[task] = dict(original=original, wrapped=wrapped, parity=parity)
        print(json.dumps(dict(task=task, parity=parity,
                              original_live=original['live'], wrapped_live=wrapped['live'])), flush=True)
        if not all(parity.values()) or not wrapped['adapter']['all_factors_one']:
            raise AssertionError(f'full-host parity failed for {task}')
    trajectory_spec = next(row['spec'] for row in plan() if row['spec']['name'] == 'trajectory')
    with extra_adam(task='trajectory', method='sim_adam') as (sim_recorder, _):
        simultaneous_result, simultaneous_context = run_legacy(
            trajectory_spec, recipe, noise, model_policy=model_policy)
    simultaneous = dict(verdict=test_verdict(trajectory_spec, simultaneous_result),
                        live=simultaneous_result['live'],
                        result_sha256=digest(untimed(simultaneous_result)),
                        applied=simultaneous_context['applied'],
                        outer_steps=sim_recorder.outer_steps,
                        joint_points_verified=sim_recorder.joint_points_verified,
                        generated_source_sha256=sim_recorder.host_source['generated_function_sha256'])
    print(json.dumps(dict(task='trajectory', method='simultaneous_adam_control',
                          verdict=simultaneous['verdict']['status'], live=simultaneous['live'])), flush=True)
    output = dict(commit='c1515197', tasks=tasks, simultaneous_trajectory=simultaneous,
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__),
                                              ROOT / 'reports/toy100/alternating_curvature_scratch.py',
                                              ROOT / 'reports/toy100/extra_adam_scratch.py',
                                              ROOT / 'benchmarks/locked_shared/trajectory.py',
                                              ROOT / 'benchmarks/locked_shared/mode_hold.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()

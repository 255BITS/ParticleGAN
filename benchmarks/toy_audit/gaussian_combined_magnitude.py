"""Combined fixed-scale prior and G/D gradient clipping using the established public-API runner.

The shared runner's protocol and initial-proof hooks are bound only within this
adapter's calls. Its archived files stay immutable; no training loop is copied.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import torch
from . import bcap_past_extrapolation as shared
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.state import state_digest

ROOT = shared.ROOT
PROTOCOL = ROOT / 'reports/forge/gaussian-combined-magnitude/protocol.json'
host = shared
checkpoints, summarize, scorer = shared.checkpoints, shared.summarize, shared.scorer
ARMS = ('alternating', 'simultaneous', 'extrapolation_from_past')


def initial_proof(context, task_id, protocol):
    path = ROOT / protocol['tasks'][task_id]['baseline_initial']
    saved, current = torch.load(path, weights_only=True, map_location='cpu'), context.state_dict()
    strip = lambda values: {k:v for k,v in values.items() if k not in ('game_update', 'network_update', 'network_gradient_scale', 'prior_update', 'prior_gradient_scale')}
    if strip(current['recipe']) != strip(saved['recipe']):
        raise ValueError('initial recipe changed beyond combined clipping and timing')
    for key in ('models', 'initial_lrs', 'streams'):
        if state_digest(current['trainer'][key]) != state_digest(saved['trainer'][key]):
            raise ValueError('initial trainer differs: '+key)
    actual = deepcopy(current['trainer']['optimizers'])
    for optimizer in actual:
        for group in optimizer['param_groups']:
            if group.pop('network_update', 'dualnorm') != ('spectral_capped' if group['algorithm'] == 'dualnorm' else 'dualnorm'):
                raise ValueError('unexpected network direction')
            if 'network_gradient_scale' in group and group.pop('network_gradient_scale') != .1:
                raise ValueError('network scale differs from hypothesis')
            if group['algorithm'] == 'row_capped':
                group['algorithm'] = 'rownorm'
                if group.pop('row_gradient_scale') != .001:
                    raise ValueError('prior scale differs from hypothesis')
    if state_digest(actual) != state_digest(saved['trainer']['optimizers']):
        raise ValueError('initial optimizer changed beyond combined clipping')
    for key in ('initialization', 'prior', 'streams'):
        if state_digest(current[key]) != state_digest(saved[key]):
            raise ValueError('initial context differs: '+key)
    return dict(matched=True, baseline_sha256=file_hash(path),
                allowed_delta='G/D scale .1, prior row cap .001, declared game timing',
                model_hashes={k:state_digest(v) for k,v in current['trainer']['models'].items()})


@contextmanager
def bound_runner():
    previous = shared.PROTOCOL, shared.initial_proof
    shared.PROTOCOL, shared.initial_proof = PROTOCOL, initial_proof
    try:
        yield shared
    finally:
        shared.PROTOCOL, shared.initial_proof = previous


def declaration():
    with bound_runner() as runner:
        return runner.declaration()


def build(arm, task_id, device, *, cap=4000):
    with bound_runner() as runner:
        return runner.build(arm, task_id, device, cap=cap)


def trial(arm, task_id, phase, raw, *, device):
    with bound_runner() as runner:
        return runner.trial(arm, task_id, phase, raw, device=device)


def execute(raw, *, device):
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('study requires CUDA; no CPU fallback')
    protocol = declaration()
    raw.mkdir(parents=True, exist_ok=False)
    results = []
    plan = [(arm, task, 'stationary') for arm in ARMS for task in protocol['tasks']]
    plan += [(arm, protocol['adaptation']['task'], 'shift') for arm in ARMS]
    for arm, task, phase in plan:
        name = f'{arm}-{task}-{phase}'
        log = raw / (name+'.log')
        print(json.dumps(dict(event='start', arm=arm, task=task, phase=phase, log=str(log))), flush=True)
        timeout = protocol['budget']['per_stationary_trial_seconds' if phase=='stationary' else 'per_shift_trial_seconds']
        command = [sys.executable, '-u', '-m', __spec__.name, 'trial', '--arm', arm, '--task', task,
                   '--phase', phase, '--output', str(raw), '--device', device]
        with log.open('w') as stdout:
            finished = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT, timeout=timeout+60)
        if finished.returncode:
            raise RuntimeError('trial failed; no retry: '+str(log))
        result = json.loads((raw/name/'receipt.json').read_text())
        results.append(result)
        print(json.dumps(dict(event='complete', arm=arm, task=task, phase=phase,
                              acquisition=result['acquisition_verdict'], hold=result['hold_verdict'], seconds=result['loop_seconds'])), flush=True)
    if sum(r['additional_updates'] for r in results) != protocol['budget']['new_training_updates']:
        raise ValueError('study update accounting differs')
    if len({r['data_sha256'] for r in results if r['phase']=='shift'}) != 1:
        raise ValueError('shift real batch sequence differs')
    atomic_json(raw/'results.json', dict(schema_version=1, protocol_sha256=file_hash(PROTOCOL), results=results,
        new_training_updates=sum(r['additional_updates'] for r in results),
        new_training_loop_seconds=sum(r['loop_seconds'] for r in results)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run','trial'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--task', choices=('gaussian1d_acquisition','ring16_acquisition'))
    parser.add_argument('--phase', choices=('stationary','shift'))
    args = parser.parse_args()
    if args.mode=='run':
        execute(args.output, device=args.device)
    else:
        trial(args.arm, args.task, args.phase, args.output, device=args.device)


if __name__=='__main__':
    main()

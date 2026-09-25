"""Screen global cosine hold/floor changes with the frozen shared-c6 D profile.

Only two additional global Recipe fields are accepted beyond the existing
shared-default search: lr_anneal_start and lr_floor. Both the historical host
schedule and the recipe-aware optimizer bridge are instrumented, so receipts
count the actual calls that set optimizer rates.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
from particlegan import get_recipe, learning_rate_scale
from benchmarks import learned_lr_evaluation as bridge

from . import compare_defaults, shared_default_search as reference, shared_profile_search as profile, suite
from .compare_defaults import write


SCHEDULE_FIELDS = {'lr_anneal_start', 'lr_floor'}


def prepare(declaration):
    with patch.object(reference, 'OPTIONS', reference.OPTIONS | SCHEDULE_FIELDS):
        jobs, recipes, discriminators = profile.prepare(declaration)
    base = get_recipe(lr=.00425, d_lr_mult=1., prior_lr_mult=2.,
                                betas=(0., .99), prior_betas=None, reg_coeff=6.,
                                reg_kappa=1.25, prior_reg=.05).replace(name='shared_c6')
    for _, recipe in recipes:
        normalized = recipe.replace(name=base.name, lr_anneal_start=base.lr_anneal_start,
                                    lr_floor=base.lr_floor)
        if normalized != base:
            raise ValueError('all non-schedule Recipe fields must equal shared_c6')
    return jobs, recipes, discriminators


@contextmanager
def configured_schedule(recipe):
    receipt = dict(kind='cosine', hold=recipe.lr_anneal_start, floor=recipe.lr_floor,
                   bridge_calls=0, bridge_total_steps=[], first_call=None, last_call=None)

    def scale(step, total_steps, original_hold, original_floor):
        if (original_hold, original_floor) not in (
            (.6, .05), (recipe.lr_anneal_start, recipe.lr_floor),
        ):
            raise ValueError('unexpected host or recipe schedule arguments')
        value = learning_rate_scale(step, total_steps, recipe.lr_anneal_start, recipe.lr_floor)
        receipt['bridge_calls'] += 1
        if total_steps not in receipt['bridge_total_steps']:
            receipt['bridge_total_steps'].append(total_steps)
        item = dict(step=step, total_steps=total_steps, multiplier=value)
        if receipt['first_call'] is None:
            receipt['first_call'] = item
        receipt['last_call'] = item
        return value

    with patch.object(bridge, 'learning_rate_scale', scale), \
            patch.object(compare_defaults, 'learning_rate_scale', scale):
        yield receipt


def episode(job, recipe, card=None):
    with configured_schedule(recipe) as schedule:
        payload = profile.episode(job, recipe, deepcopy(card))
    result = payload['result']
    actions = result.get('actions', [])
    errors = []
    for action in actions:
        if set(action) != {'step', 'role', 'multiplier'} or action['role'] not in ('g', 'd'):
            raise ValueError('unexpected fixed schedule trace entry')
        expected = learning_rate_scale(action['step'], job['spec']['steps'],
                                       recipe.lr_anneal_start, recipe.lr_floor)
        errors.append(abs(action['multiplier'] - expected))
    schedule.update(total_steps=job['spec']['steps'], trace_points=len(actions),
                    trace_roles=sorted({action['role'] for action in actions}),
                    max_abs_trace_error=max(errors, default=None),
                    trace_equation_verified=bool(actions) and all(math.isclose(error, 0., abs_tol=1e-14)
                                                                  for error in errors))
    if not result.get('error'):
        if not schedule['bridge_calls'] or not schedule['trace_equation_verified']:
            raise AssertionError('candidate did not use the declared global cosine schedule')
        if schedule['bridge_total_steps'] != [job['spec']['steps']]:
            raise AssertionError('controller budget differs from frozen host budget')
    payload['schedule'] = schedule
    return payload


def run(declaration, output):
    jobs, recipes, discriminators = prepare(declaration)
    output.mkdir(parents=True, exist_ok=False)
    (output/'episodes').mkdir()
    torch.set_num_threads(1)
    protocol = suite.snapshot(output)
    protocol.update(version='shared-schedule-v1', seed=0, jobs=jobs,
                    declaration=declaration,
                    selection='One unchanged global recipe and D profile for every host; '
                    '24 live observations and a final five-check PASS. EMA separate.',
                    schedule_bridge='Instrument the host and recipe optimizer bridge schedule calls '
                    'while their controllers set native and legacy optimizer rates.')
    write(output/'protocol.json', protocol)
    write(output/'plan.json', declaration)
    records = []
    for _, recipe in recipes:
        for job in jobs:
            suite.verify_source(protocol)
            name = job['spec']['name']
            print(f'START {recipe.name} {name}', flush=True)
            payload = episode(job, recipe, discriminators.get(name))
            payload['source_sha256'] = protocol['source_sha256']
            raw = (json.dumps(payload, sort_keys=True, allow_nan=False)+'\n').encode()
            artifact = f'episodes/{recipe.name}__{name}.json.gz'
            (output/artifact).write_bytes(gzip.compress(raw, mtime=0))
            result = payload['result']
            records.append({k: v for k, v in payload.items() if k not in ('result', 'source_sha256')} |
                           dict(artifact=artifact, uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                                live=result.get('live'), ema=result.get('ema'), seconds=result['seconds']))
            write(output/'index.json', dict(records=records))
            reference.render(records, output)
            print(json.dumps(dict(event='DONE', candidate=recipe.name, task=name,
                                  architecture=payload['architecture'], status=payload['verdict']['status'],
                                  suffix=payload['verdict'].get('convergence', {}).get('passing_suffix'),
                                  schedule=payload['schedule'], live=result.get('live'),
                                  error=result.get('error'))), flush=True)
    suite.verify_source(protocol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    run(json.loads(args.plan.read_text()), args.output)

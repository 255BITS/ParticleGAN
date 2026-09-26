"""Search one unchanged numerical recipe across the frozen 19 behavioral hosts.

Plans may select screening tasks, but never specify per-task optimizer overrides.
Only a complete 19-case row can qualify as an overall PASS.
"""
from benchmarks.locked_shared.recorded_recipes import GAN_V2
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import torch

from benchmarks import learned_lr_evaluation as bridge
from benchmarks.locked_shared import baseline
from benchmarks.smart_descent import evaluate
from . import suite, vector_tasks
from .compare_defaults import candidate, effective_spec, ema_verdict, optimizer_defaults, plan, write
from .linear_skip_refinement_research import constructor as skip_constructor
from .smooth_critic_research import constructor as smooth_constructor
from .protocol import test_verdict
from benchmarks.gan_v3 import legacy_dict

OPTIONS = {'lr', 'd_lr_mult', 'prior_lr_mult', 'betas', 'prior_betas', 'reg_coeff', 'reg_kappa', 'prior_reg'}


def prepare(declaration):
    if set(declaration) - {'candidates', 'tasks', 'purpose'}:
        raise ValueError('unsupported plan field; per-task overrides are prohibited')
    jobs = plan()
    names = {j['spec']['name'] for j in jobs}
    selected = declaration.get('tasks', list(names))
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= names:
        raise ValueError('invalid task selection')
    jobs = [j for j in jobs if j['spec']['name'] in selected]
    recipes = []
    seen = set()
    for card in declaration['candidates']:
        if set(card) != {'name', 'overrides'} or set(card['overrides']) - OPTIONS:
            raise ValueError('candidate must declare only shared recipe overrides')
        name = card['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name) or name in seen:
            raise ValueError('candidate names must be unique lowercase identifiers')
        seen.add(name)
        # This archived search began from the previous public GAN preset.
        # New package defaults must not change its unspecified fields.
        recipe = GAN_V2.replace(**card['overrides']).replace(name=name)
        recipes.append((card, recipe))
    if not recipes:
        raise ValueError('at least one candidate required')
    return jobs, recipes


def episode(job, recipe):
    spec = effective_spec(job['spec'], recipe)
    policy = vector_tasks.fixed_policy('cosine')
    applied = []
    start = time.perf_counter()
    try:
        with optimizer_defaults(recipe, applied), ExitStack() as stack:
            if spec['runner'] == 'legacy':
                control = evaluate.FixedControl(policy, spec['steps'])
                with bridge.control_host_schedules(control):
                    result = baseline.run_toy(spec['name'], candidate(recipe))
                result['actions'] = control.trace
                result['seconds'] = time.perf_counter() - start
            else:
                architecture = spec.get('research_discriminator')
                if architecture:
                    create = skip_constructor(architecture) if architecture.get('skip') == 'raw_linear' else smooth_constructor(architecture)
                    stack.enter_context(patch.object(vector_tasks, 'SimpleMLPDiscriminator', create))
                result = suite.run_episode(spec, policy, fixed=True, allow_reserved=True)
        json.dumps(result, allow_nan=False)
    except Exception:
        result = dict(error=traceback.format_exc(), seconds=time.perf_counter() - start)
    return dict(recipe=legacy_dict(recipe), candidate=asdict(candidate(recipe)), original_spec=job['spec'],
                spec=spec, architecture=job['architecture'], reference=job['reference'],
                reference_sha256=job['reference_sha256'], applied=applied,
                verdict=test_verdict(spec, result), ema_verdict=ema_verdict(spec, result), result=result)


def render(records, output):
    candidates = {}
    for record in records:
        candidates.setdefault(record['recipe']['name'], []).append(record)
    lines = ['# Shared-recipe search progress', '',
             'One recipe per candidate; seed 0; fixed hosts and budgets. Only 19/19 complete passes qualify. '
             'Screening counts cannot be combined across different recipes. EMA is separate.', '',
             '| Candidate | Live passes | Attempted / 19 | Final metric shortfall | Overall |',
             '| --- | ---: | ---: | ---: | --- |']
    for name, rows in sorted(candidates.items(), key=lambda kv: (-sum(r['verdict']['passed'] for r in kv[1]),
                           sum(r['verdict']['shortfall'] for r in kv[1]))):
        passed = sum(r['verdict']['passed'] for r in rows)
        status = 'PASS' if passed == len(rows) == 19 else 'FAIL' if len(rows) == 19 else 'INCOMPLETE'
        lines.append(f"| {name} | {passed} | {len(rows)}/19 | {sum(r['verdict']['shortfall'] for r in rows):.5g} | {status} |")
    lines += ['', '| Candidate | Test | Live | Final passing streak |', '| --- | --- | --- | ---: |']
    for row in records:
        lines.append(f"| {row['recipe']['name']} | {row['spec']['name']} | {row['verdict']['status']} | "
                     f"{row['verdict'].get('convergence', {}).get('passing_suffix', 0)} |")
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def run(declaration, output):
    jobs, recipes = prepare(declaration)
    output.mkdir(parents=True, exist_ok=False)
    (output/'episodes').mkdir()
    torch.set_num_threads(1)
    protocol = suite.snapshot(output)
    protocol.update(version='unadjusted-default-search-v1', seed=0, jobs=jobs,
                    declaration=declaration, selection='All 19 live behavioral metrics sustained; EMA separate. '
                    'One unchanged recipe across hosts; architecture and test budgets fixed.')
    write(output/'protocol.json', protocol)
    write(output/'plan.json', declaration)
    records = []
    for card, recipe in recipes:
        for job in jobs:
            suite.verify_source(protocol)
            name = job['spec']['name']
            print(f'START {recipe.name} {name}', flush=True)
            payload = episode(job, recipe)
            payload['source_sha256'] = protocol['source_sha256']
            raw = (json.dumps(payload, sort_keys=True, allow_nan=False)+'\n').encode()
            artifact = f'episodes/{recipe.name}__{name}.json.gz'
            (output/artifact).write_bytes(gzip.compress(raw, mtime=0))
            result = payload['result']
            records.append({k: v for k, v in payload.items() if k not in ('result', 'source_sha256')} |
                           dict(artifact=artifact, uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                                live=result.get('live'), ema=result.get('ema'), seconds=result['seconds']))
            write(output/'index.json', dict(records=records))
            render(records, output)
            print(json.dumps(dict(event='DONE', candidate=recipe.name, task=name,
                                  status=payload['verdict']['status'],
                                  suffix=payload['verdict'].get('convergence', {}).get('passing_suffix'),
                                  live=result.get('live'), error=result.get('error'))), flush=True)
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

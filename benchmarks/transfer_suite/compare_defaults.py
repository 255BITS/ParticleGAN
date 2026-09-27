"""Matched behavioral comparison of the old and promoted public GAN recipes.

Keep each toy's data, architecture, support and budget fixed. Apply each public
recipe's loss, regularization and absolute G/D/prior optimizer settings. This
is distinct from the historical 19/19 row with per-host optimizer settings.

Vector and image hosts take the recipe through their spec (``effective_spec``)
and train on the shared runner, whose recipe-built optimizers own the rates and
schedule. Custom hosts go through ``run_custom_host``. Nothing here constructs
an optimizer or writes a learning rate.
"""
from benchmarks.locked_shared.recorded_recipes import GAN_V1, GAN_V2
import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import torch

from benchmarks.locked_shared import baseline
from . import suite, vector_tasks
from .formulations import axes
from .linear_skip_refinement_research import constructor as skip_constructor
from .smooth_critic_research import constructor as smooth_constructor
from .protocol import test_verdict
from benchmarks.gan_v3 import legacy_dict

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'reports/transfer_suite'
RECIPES = {'current': GAN_V1, 'proposed': GAN_V2}


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def plan():
    """Load frozen test definitions without requiring historical run outputs.

    Extracted from the published passing architectures at d77e9e8. Reference
    paths and hashes retain their provenance; the archives are local artifacts.
    """
    jobs = read(Path(__file__).with_name('plans') / 'default_comparison.json')
    assert len(jobs) == 19 and len({j['spec']['name'] for j in jobs}) == 19
    return jobs


def candidate(recipe):
    return baseline.Candidate(
        name=recipe.name, loss_type=recipe.loss_type, gan_mode=recipe.gan_mode,
        reg_arm=recipe.reg_arm, reg_coeff=recipe.reg_coeff, reg_kappa=recipe.reg_kappa,
        particle_l2=0., vicreg_weight=recipe.prior_reg, lr_multiplier=1.)


def effective_spec(original, recipe):
    spec = deepcopy(original)
    if spec['runner'] == 'vector':
        spec.update(lr=recipe.lr, d_lr_mult=recipe.d_lr_mult, prior_lr_mult=recipe.prior_lr_mult,
                    betas=list(recipe.betas), loss_type=recipe.loss_type, gan_mode=recipe.gan_mode,
                    reg_arm=recipe.reg_arm, reg_coeff=recipe.reg_coeff, reg_kappa=recipe.reg_kappa,
                    prior_reg=recipe.prior_reg, ema_decay=recipe.ema_decay)
    elif spec['runner'] == 'image':
        spec.update(lr_g=recipe.lr, lr_d=recipe.lr * recipe.d_lr_mult,
                    prior_lr_multiplier=recipe.prior_lr_mult, adam_betas=list(recipe.betas),
                    loss_type=recipe.loss_type, gan_mode=recipe.gan_mode,
                    gradient_penalty=recipe.reg_arm, penalty_coeff=recipe.reg_coeff,
                    kappa=recipe.reg_kappa, prior_weight=recipe.prior_reg, ema_decay=recipe.ema_decay)
    if spec['runner'] != 'legacy':
        before, after = axes(original, spec['runner']), axes(spec, spec['runner'])
        for key in ('architecture', 'resources', 'target'):
            assert before[key] == after[key], key
    return spec


def run_custom_host(spec, recipe):
    """One frozen custom host under ``recipe``.

    A migrated host (``problem_hosts``) trains on the shared runner, whose
    recipe-built optimizers own the rates and schedule. Any other host still
    owns its optimizers and runs unchanged (``recipe_owned`` False).
    """
    from . import problem_hosts
    start = time.perf_counter()
    if problem_hosts.is_migrated(spec['name']):
        result, _ = problem_hosts.run_problem(spec, recipe)
        result['recipe_owned'] = True
    else:
        result = baseline.run_toy(spec['name'], candidate(recipe))
        result['recipe_owned'] = False
    result['seconds'] = time.perf_counter() - start
    return result


def run_host(spec, recipe, policy, critic=None):
    """One frozen host under ``recipe`` (already applied to ``spec``); errors become a result.

    ``critic`` (or else a declared ``research_discriminator`` card) only swaps
    the vector host's critic class; the recipe, optimizers and schedule are
    untouched.
    """
    start = time.perf_counter()
    try:
        if spec['runner'] == 'legacy':
            result = run_custom_host(spec, recipe)
        else:
            card = spec.get('research_discriminator')
            if critic is None and card:
                critic = skip_constructor(card) if card.get('skip') == 'raw_linear' else smooth_constructor(card)
            with ExitStack() as stack:
                if critic is not None:
                    stack.enter_context(patch.object(vector_tasks, 'SimpleMLPDiscriminator', critic))
                result = suite.run_episode(spec, policy, fixed=True, allow_reserved=True)
        json.dumps(result, allow_nan=False)
    except Exception:
        result = dict(error=traceback.format_exc(), seconds=time.perf_counter() - start)
    return result


def ema_verdict(spec, result):
    observations = result.get('observations', [])
    if len(observations) != 24 or not all(isinstance(p.get('ema'), dict) for p in observations):
        return dict(status='N/A', reason='Host does not record a complete EMA curve')
    ema = dict(live=result['ema'], observations=[dict(p['ema'], step=p['step']) for p in observations])
    return test_verdict(spec, ema)


def run(arm, output, tasks=None):
    jobs = plan()
    if tasks:
        if not set(tasks) <= {j['spec']['name'] for j in jobs}:
            raise ValueError('unknown task')
        jobs = [j for j in jobs if j['spec']['name'] in tasks]
    output.mkdir(parents=True, exist_ok=False)
    (output / 'episodes').mkdir()
    torch.set_num_threads(1)
    # Retain the archived arm names as well as their exact historical settings.
    recipe = RECIPES[arm].replace(name='gan' if arm == 'proposed' else 'gan_legacy')
    assert recipe.lr_anneal_start == .6 and recipe.lr_floor == .05
    policy = vector_tasks.fixed_policy('cosine')
    protocol = suite.snapshot(output)
    protocol.update(comparison='public-gan-defaults-v1', arm=arm, recipe=legacy_dict(recipe),
                    seed=0, jobs=jobs, adaptation='Apply absolute G/D/prior LRs, betas and core loss weights. '
                    'Retain each host architecture, particle support, batch, steps, initialization, auxiliary losses and data. '
                    'Preserve legacy host EMA settings; native vector/image EMA uses the recipe decay.')
    write(output / 'protocol.json', protocol)
    records = []
    for job in jobs:
        suite.verify_source(protocol)
        spec = effective_spec(job['spec'], recipe)
        name = spec['name']
        print(f'START {arm} {name} steps={spec["steps"]}', flush=True)
        result = run_host(spec, recipe, policy)
        verdict = test_verdict(spec, result)
        ema = ema_verdict(spec, result)
        record = dict(arm=arm, recipe=legacy_dict(recipe), original_spec=job['spec'], spec=spec,
                      architecture=job['architecture'], reference=job['reference'],
                      reference_sha256=job['reference_sha256'], candidate=asdict(candidate(recipe)),
                      applied=result.get('applied', []), verdict=verdict, ema_verdict=ema, result=result,
                      source_sha256=protocol['source_sha256'])
        raw = (json.dumps(record, sort_keys=True, allow_nan=False) + '\n').encode()
        artifact = f'episodes/{arm}__{name}.json.gz'
        (output / artifact).write_bytes(gzip.compress(raw, mtime=0))
        records.append({k: v for k, v in record.items() if k not in ('result', 'source_sha256')} |
                       dict(artifact=artifact, uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                            live=result.get('live'), ema=result.get('ema'), seconds=result['seconds']))
        write(output / 'index.json', dict(records=records))
        print(json.dumps(dict(event='DONE', arm=arm, task=name, status=verdict['status'],
                              passing_suffix=verdict.get('convergence', {}).get('passing_suffix'),
                              live=result.get('live'), ema_status=ema['status'], error=result.get('error'))), flush=True)
    suite.verify_source(protocol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=RECIPES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+')
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    run(args.arm, args.output, args.tasks)

"""Matched behavioral comparison of the old and promoted public GAN recipes.

Keep each toy's data, architecture, support and budget fixed. Apply each public
recipe's loss, regularization and absolute G/D/prior optimizer settings. This
is distinct from the historical 19/19 row with per-host optimizer settings.
"""
import argparse
from contextlib import contextmanager, ExitStack
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

from particlegan import ParticlePrior, get_recipe
from benchmarks import learned_lr_evaluation as bridge
from benchmarks.locked_shared import baseline
from benchmarks.smart_descent import evaluate
from . import image_tasks, suite, vector_tasks
from .formulations import axes
from .linear_skip_refinement_research import constructor as skip_constructor
from .smooth_critic_research import constructor as smooth_constructor
from .protocol import required_tasks, test_verdict

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'reports/transfer_suite'
RECIPES = {'current': 'gan_legacy', 'proposed': 'gan'}


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def plan():
    """Choose the already published passing architecture before either arm runs."""
    leaderboard = read(REPORTS / 'formulations/leaderboard.json')['rows'][0]
    jobs = []
    for spec in required_tasks():
        reference = leaderboard['required'][spec['name']]['artifact']
        jobs.append(dict(spec=spec, reference=reference, architecture='original host'))
    chosen = {'vector_unequal_mass': 'linear_skip_d96_beta5',
              'vector_unequal_width': 'axis_softplus5', 'vector_overlap': 'd128_l3_f3'}
    for name, cell in leaderboard['cases'].items():
        label = 'residual16' if cell['runner'] == 'image' else chosen.get(name, 'original architecture')
        trial = next(t for t in cell['trials'] if t['label'] == label)
        assert trial['verdict']['passed']
        payload = read(REPORTS / trial['artifact'])
        spec = deepcopy(payload.get('effective_spec', payload.get('spec', payload.get('task'))))
        spec['runner'] = cell['runner']
        jobs.append(dict(spec=spec, reference=trial['artifact'], architecture=label))
    assert len(jobs) == 19 and len({j['spec']['name'] for j in jobs}) == 19
    for job in jobs:
        job['reference_sha256'] = hashlib.sha256((REPORTS / job['reference']).read_bytes()).hexdigest()
    return jobs


def candidate(recipe):
    return baseline.Candidate(
        name=recipe.name, loss_type=recipe.loss_type, gan_mode=recipe.gan_mode,
        reg_arm=recipe.reg_arm, reg_coeff=recipe.reg_coeff, reg_kappa=recipe.reg_kappa,
        particle_l2=0., vicreg_weight=recipe.prior_reg, lr_multiplier=1.,
        lr_schedule='host', lr_anneal_start=recipe.lr_anneal_start, lr_floor=recipe.lr_floor)


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


@contextmanager
def optimizer_defaults(recipe, applied):
    """Apply absolute recipe rates to every group, including mixed/AE priors.

    The phase bridge identifies existing opt_p direct-particle optimizers. Prior
    instances identify their parameters even when mixed with generator weights.
    Both optimizer-level and explicit per-group Adam betas are replaced.
    """
    prior_ids = set()
    original_prior = ParticlePrior.__init__
    original_adam = torch.optim.Adam.__init__
    original_role = bridge.optimizer_role
    original_control = evaluate.FixedControl

    def prior_init(self, *args, **kwargs):
        original_prior(self, *args, **kwargs)
        prior_ids.update(id(p) for p in self.parameters())

    def adam_init(self, params, *args, **kwargs):
        params = list(params)
        if not params or not isinstance(params[0], dict):
            params = [dict(params=params)]
        groups = []
        for group in params:
            values = list(group['params'])
            for is_prior in (False, True):
                selected = [p for p in values if (id(p) in prior_ids) == is_prior]
                if selected:
                    groups.append(dict(group, params=selected, _comparison_prior=is_prior,
                                       betas=recipe.prior_betas or recipe.betas if is_prior else recipe.betas))
        args = list(args)
        if len(args) >= 2:
            args[1] = recipe.betas
        else:
            kwargs['betas'] = recipe.betas
        original_adam(self, groups, *args, **kwargs)

    def role(optimizer, locals_):
        result = original_role(optimizer, locals_)
        if locals_.get('opt_p') is optimizer:
            for group in optimizer.param_groups:
                group['_comparison_prior'] = True
        return result

    class RecipeControl(original_control):
        def step(self, optimizer, completed_updates, role):
            if optimizer not in self.base_rates:
                for group in optimizer.param_groups:
                    kind = 'd' if role == 'd' else 'prior' if group['_comparison_prior'] else 'g'
                    old_rate = group['lr']
                    group['lr'] = recipe.lr * {'g': 1., 'd': recipe.d_lr_mult, 'prior': recipe.prior_lr_mult}[kind]
                    group['betas'] = (recipe.prior_betas or recipe.betas) if kind == 'prior' else recipe.betas
                    applied.append(dict(role=kind, host_lr=old_rate, lr=group['lr'], betas=list(group['betas']),
                                        parameters=sum(p.numel() for p in group['params'])))
            super().step(optimizer, completed_updates, role)

    with ExitStack() as stack:
        stack.enter_context(patch.object(ParticlePrior, '__init__', prior_init))
        stack.enter_context(patch.object(torch.optim.Adam, '__init__', adam_init))
        stack.enter_context(patch.object(bridge, 'optimizer_role', role))
        for module in (evaluate, vector_tasks, image_tasks):
            stack.enter_context(patch.object(module, 'FixedControl', RecipeControl))
        yield


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
    recipe = get_recipe(RECIPES[arm])
    assert recipe.lr_anneal_start == .6 and recipe.lr_floor == .05
    policy = vector_tasks.fixed_policy('cosine')
    protocol = suite.snapshot(output)
    protocol.update(comparison='public-gan-defaults-v1', arm=arm, recipe=recipe.to_dict(),
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
        applied = []
        start = time.perf_counter()
        try:
            with optimizer_defaults(recipe, applied), ExitStack() as stack:
                if spec['runner'] == 'legacy':
                    control = evaluate.FixedControl(policy, spec['steps'])
                    with bridge.control_host_schedules(control):
                        result = baseline.run_toy(name, candidate(recipe))
                    result['actions'] = control.trace
                    result['seconds'] = time.perf_counter() - start
                else:
                    card = spec.get('research_discriminator')
                    if card:
                        create = skip_constructor(card) if card.get('skip') == 'raw_linear' else smooth_constructor(card)
                        stack.enter_context(patch.object(vector_tasks, 'SimpleMLPDiscriminator', create))
                    result = suite.run_episode(spec, policy, fixed=True, allow_reserved=True)
            json.dumps(result, allow_nan=False)
        except Exception:
            result = dict(error=traceback.format_exc(), seconds=time.perf_counter() - start)
        verdict = test_verdict(spec, result)
        ema = ema_verdict(spec, result)
        record = dict(arm=arm, recipe=recipe.to_dict(), original_spec=job['spec'], spec=spec,
                      architecture=job['architecture'], reference=job['reference'],
                      reference_sha256=job['reference_sha256'], candidate=asdict(candidate(recipe)),
                      applied=applied, verdict=verdict, ema_verdict=ema, result=result,
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
    args = parser.parse_args()
    run(args.arm, args.output, args.tasks)

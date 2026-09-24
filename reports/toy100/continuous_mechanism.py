"""Scratch observational/causal mode-hold diagnostics; never a gate promotion."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import inspect
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run_one(options):
    import torch
    from particlegan.grad_regularizers import GradRegularizer
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy

    torch.set_num_threads(1)
    base = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    config = dict(base, name=options['name'])
    if options.get('constant'):
        config.update(lr_floor=1.0, lr_anneal_start=0.0)
        config.pop('network_lr_horizon_cap', None)
        config.pop('network_lr_floor', None)
    if 'lr' in options:
        config['lr'] = options['lr']
    if 'arm' in options:
        config.update(reg_arm=options['arm'], reg_coeff=options['coeff'])
    recipe, noise, _ = declared_recipe(config)
    spec = next(job['spec'] for job in plan() if job['spec']['name'] == 'mode_hold')
    original_step = torch.optim.Adam.step
    original_penalty = GradRegularizer.penalty
    original_norm = GradRegularizer._grad_norm
    state = {'step': 0, 'host': None}
    rows, cap_rows = [], []

    def penalty(self, *args, **kwargs):
        state['step'] = kwargs.get('step', args[3] if len(args) > 3 else 1)
        return original_penalty(self, *args, **kwargs)

    def grad_norm(self, *args, **kwargs):
        value = original_norm(self, *args, **kwargs)
        step = state['step']
        if step >= 700 and step % 10 == 0:
            values = value.detach()
            if kwargs.get('squared', False):
                values = values.sqrt()
            cap_rows.append(dict(step=step, n_mean=float(values.mean()),
                                 n_max=float(values.max()), active=float((values > self.kappa).float().mean())))
        return value

    def step(self, *args, **kwargs):
        if state['host'] is None:
            frame = inspect.currentframe().f_back
            while frame and not all(k in frame.f_locals for k in ('generator', 'prior', 'opt_g', 'opt_d')):
                frame = frame.f_back
            if frame is None:
                raise RuntimeError('Cannot locate unchanged host frame')
            state['host'] = {k: frame.f_locals[k] for k in ('generator', 'prior', 'opt_g', 'opt_d')}
            del frame
        host = state['host']
        current = state['step']
        role = 'd' if self is host['opt_d'] else 'g'
        if options.get('resume_constant') and current > options.get('branch_at', 1000):
            for group in self.param_groups:
                group['lr'] = recipe.lr * (recipe.d_lr_mult if role == 'd' else recipe.prior_lr_mult if group['_comparison_prior'] else 1)
                kind = 'd' if role == 'd' else 'prior' if group['_comparison_prior'] else 'g'
                if options.get('freeze') == kind:
                    for parameter in group['params']:
                        parameter.grad = None
        observe = current >= 700 and current % 10 == 0
        network = host['generator'].model
        prior = host['prior']
        if observe:
            before = [[p.detach().clone() for p in group['params']] for group in self.param_groups]
            row = dict(step=current, role=role, groups=[])
            for group in self.param_groups:
                grads = [p.grad.detach() for p in group['params'] if p.grad is not None]
                row['groups'].append(dict(lr=group['lr'], prior=group['_comparison_prior'],
                    grad_rms=(sum(float(g.square().sum()) for g in grads) / max(1, sum(g.numel() for g in grads))) ** .5))
            if role == 'g':
                with torch.no_grad():
                    z_before = prior.z.clone()
                    x_before = network(z_before)
        answer = original_step(self, *args, **kwargs)
        if observe:
            for group, old, record in zip(self.param_groups, before, row['groups']):
                record['update_rms'] = (sum(float((p.detach()-q).square().sum()) for p,q in zip(group['params'],old)) / sum(p.numel() for p in group['params'])) ** .5
                denominators=[]
                for parameter in group['params']:
                    moment=self.state.get(parameter,{})
                    if 'exp_avg_sq' in moment:
                        correction=1-group['betas'][1] ** float(moment['step'])
                        denominators.append((moment['exp_avg_sq']/correction).sqrt().flatten())
                if denominators:
                    values=torch.cat(denominators)
                    record['denominator_quantiles']=torch.quantile(values,torch.tensor([0.,.1,.5,.9,1.])).tolist()
            if role == 'g':
                with torch.no_grad():
                    x_g = network(z_before)
                    x_after = network(prior.z)
                    for key, diff in (('g_output', x_g-x_before), ('prior_output',x_after-x_g),('total_output',x_after-x_before)):
                        row[key + '_rms'] = float(diff.square().sum(1).mean().sqrt())
                        row[key + '_coherent_share'] = float(diff.mean(0).square().sum()/diff.square().sum(1).mean().clamp_min(1e-20))
            rows.append(row)
        return answer

    started = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Adam, 'step', step))
        stack.enter_context(patch.object(GradRegularizer, 'penalty', penalty))
        stack.enter_context(patch.object(GradRegularizer, '_grad_norm', grad_norm))
        result, context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
    out = dict(options=options, config=config, result=result, cap=cap_rows, updates=rows,
               seconds=time.perf_counter()-started, applied=context['applied'],
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               interpretation='Scratch observation or explicit late freeze/restart diagnostic; no shared gate eligibility')
    output = Path(options['output'])
    output.mkdir(parents=True, exist_ok=True)
    (output / (options['name']+'.json')).write_text(json.dumps(out, indent=2)+'\n')
    return dict(name=options['name'], seconds=out['seconds'], curve=[dict(step=p['step'],modes=p['modes'],hq=p['hq']) for p in result['observations'][-7:]])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--suite', choices=['observe','branch','epsmeasure','r1r2'], default='observe')
    args=parser.parse_args()
    options = ([dict(name='constant_denom',constant=True),dict(name='resume_none',resume_constant=True,branch_at=1000)] if args.suite=='epsmeasure' else
               [dict(name='scheduled'),dict(name='constant',constant=True),dict(name='constant_0025',constant=True,lr=.0025)]
               if args.suite == 'observe' else
               [dict(name='resume_'+str(freeze).lower(),resume_constant=True,branch_at=1000,freeze=freeze) for freeze in (None,'g','prior','d')])
    if args.suite=='r1r2':
        options=[dict(name=f'r1r2_{i:02}',constant=True,lr=lr,arm='a_r1r2',coeff=coeff)
                 for i,(lr,coeff) in enumerate(( (lr,coeff) for lr in (.001,.0025,.00425) for coeff in (.01,.1,1.) ))]
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'declaration.json').write_text(json.dumps(dict(rows=options,seed=0,budget=1200,
            base='configs/toy100/constraints_simple_regularization.json',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            task_order=['mode_hold','trajectory'],
            criteria='24 frozen live checkpoints, final 5 require eight modes/HQ>=.9; stop at first failed complete host'),indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending=[pool.submit(run_one, dict(row,output=str(args.output))) for row in options]
        for future in as_completed(pending):
            print(json.dumps(future.result()),flush=True)


if __name__ == '__main__':
    main()

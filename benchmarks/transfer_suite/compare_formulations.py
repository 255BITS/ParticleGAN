"""Isolate old versus new loss/regularization under the published host recipes.

The proposed reference is the archived 19/19 row. This is explicitly separate
from compare_defaults, which applies each public numerical optimizer preset.
"""
from benchmarks.locked_shared.recorded_recipes import GAN_V1
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from benchmarks import learned_lr_evaluation as bridge
from benchmarks.locked_shared import baseline
from benchmarks.smart_descent import evaluate, study
from . import image_tasks, suite, vector_tasks
from .compare_defaults import plan, read, write, ema_verdict
from .linear_skip_refinement_research import constructor as skip_constructor
from .smooth_critic_research import constructor as smooth_constructor
from .protocol import test_verdict


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'episodes').mkdir()
    torch.set_num_threads(1)
    recipe = GAN_V1
    config = replace(study.BASE, name='current_core_host_recipe',
                     reg_coeff=recipe.reg_coeff, reg_kappa=recipe.reg_kappa,
                     vicreg_weight=recipe.prior_reg, particle_l2=0.)
    policy = vector_tasks.fixed_policy('cosine')
    protocol = suite.snapshot(output)
    jobs = plan()
    protocol.update(comparison='matched-core-formulation-v1', jobs=jobs, config=asdict(config), seed=0,
                    scope='Replace only b_cap coefficient, kappa and spread weight in the published host recipes. '
                    'Preserve their LRs, Adam, schedules, resources, architectures, EMA and auxiliary objectives. '
                    'This is not the full public numerical optimizer preset.')
    write(output / 'protocol.json', protocol)
    records = []
    for job in jobs:
        suite.verify_source(protocol)
        spec = dict(job['spec'])
        if spec['runner'] == 'vector':
            spec.update(reg_coeff=recipe.reg_coeff, reg_kappa=recipe.reg_kappa, prior_reg=recipe.prior_reg)
        elif spec['runner'] == 'image':
            spec.update(penalty_coeff=recipe.reg_coeff, kappa=recipe.reg_kappa, prior_weight=recipe.prior_reg)
        name = spec['name']
        print(f'START current_core {name} steps={spec["steps"]}', flush=True)
        started = time.perf_counter()
        applied = []
        original_control = evaluate.FixedControl

        class AuditControl(original_control):
            def step(self, optimizer, completed_updates, role):
                if optimizer not in self.base_rates:
                    for group in optimizer.param_groups:
                        applied.append(dict(role=role, lr=group['lr'], betas=list(group['betas']),
                                            parameters=sum(p.numel() for p in group['params'])))
                super().step(optimizer, completed_updates, role)

        with ExitStack() as stack:
            for module in (evaluate, vector_tasks, image_tasks):
                stack.enter_context(patch.object(module, 'FixedControl', AuditControl))
            if spec['runner'] == 'legacy':
                control = evaluate.FixedControl(policy, spec['steps'])
                with bridge.control_host_schedules(control):
                    result = baseline.run_toy(name, config)
                result['actions'] = control.trace
                result['seconds'] = time.perf_counter() - started
            else:
                card = spec.get('research_discriminator')
                if card:
                    create = skip_constructor(card) if card.get('skip') == 'raw_linear' else smooth_constructor(card)
                    stack.enter_context(patch.object(vector_tasks, 'SimpleMLPDiscriminator', create))
                result = suite.run_episode(spec, policy, fixed=True, allow_reserved=True)
        verdict, ema = test_verdict(spec, result), ema_verdict(spec, result)
        record = dict(arm='current_core', original_spec=job['spec'], spec=spec, candidate=asdict(config),
                      architecture=job['architecture'], reference=job['reference'], reference_sha256=job['reference_sha256'],
                      applied=applied, verdict=verdict, ema_verdict=ema, result=result,
                      source_sha256=protocol['source_sha256'])
        raw = (json.dumps(record, sort_keys=True, allow_nan=False)+'\n').encode()
        artifact = f'episodes/current_core__{name}.json.gz'
        (output / artifact).write_bytes(gzip.compress(raw, mtime=0))
        records.append({k: v for k, v in record.items() if k not in ('result', 'source_sha256')} |
                       dict(artifact=artifact, uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                            live=result.get('live'), ema=result.get('ema'), seconds=result['seconds']))
        write(output / 'index.json', dict(records=records))
        print(json.dumps(dict(event='DONE', arm='current_core', task=name, status=verdict['status'],
                              passing_suffix=verdict.get('convergence', {}).get('passing_suffix'),
                              live=result.get('live'), ema_status=ema['status'], error=result.get('error'))), flush=True)
    suite.verify_source(protocol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)

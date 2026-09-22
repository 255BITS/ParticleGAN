"""Verify the public default recipe, trainer and critic against the rare winner."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from particlegan import LinearSkipDiscriminator, get_recipe, learning_rate_scale
from lib.toy_models import SimpleMLPGenerator
from . import suite, vector_tasks
from .protocol import test_verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    reference_path = root/'reports/transfer_suite/rare_focus/linear_refinement/screen/episodes/linear_skip_d96_beta5__vector_unequal_mass.json.gz'
    reference = json.loads(gzip.decompress(reference_path.read_bytes()))
    spec = reference['spec']
    protocol = suite.snapshot(args.output)

    def write(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    write('protocol.json', protocol)
    # Only resources differ from get_recipe(); all formulation/optimizer fields
    # are the promoted defaults. Architecture, data units and init remain host choices.
    recipe = get_recipe(num_particles=spec['particles'], batch_size=spec['batch'], total_steps=spec['steps'])
    torch.set_num_threads(1)
    torch.manual_seed(0)
    prior = recipe.make_prior(init_std=.5, generator=torch.Generator().manual_seed(0))
    g = SimpleMLPGenerator(recipe.z_dim, spec['hidden'], spec['layers'], 2)
    d = LinearSkipDiscriminator()
    trainer = recipe.make_trainer(g, d, prior=prior,
                                 latent_generator=torch.Generator().manual_seed(1),
                                 penalty_generator=torch.Generator().manual_seed(2))
    write('recipe.json', recipe.to_dict())
    original_groups = [[dict(lr=group['lr'], betas=group['betas']) for group in opt.param_groups]
                       for opt in (trainer.opt_g, trainer.opt_d)]
    write('optimizer_groups.json', original_groups)
    rng = torch.Generator().manual_seed(0)
    observations = []
    actions = []
    expected = {math.ceil(i*spec['steps']/24) for i in range(1, 25)}
    start = time.perf_counter()
    for step in range(spec['steps']):
        completed = step+1
        real = vector_tasks.sample_target(spec, spec['batch'], rng, completed)
        trainer.step(real, generator_real=lambda: vector_tasks.sample_target(spec, spec['batch'], rng, completed))
        scale = learning_rate_scale(step, recipe.total_steps, recipe.lr_anneal_start, recipe.lr_floor)
        for opt, groups in zip((trainer.opt_g, trainer.opt_d), original_groups):
            assert all(group['lr'] == base['lr']*scale for group, base in zip(opt.param_groups, groups))
        if step % 20 == 0:
            actions.extend(dict(step=step, role=role, multiplier=scale) for role in ('d', 'g'))
        if completed in expected:
            metrics = []
            for ema in (False, True):
                samples = trainer.sample(vector_tasks.EVAL_SAMPLES, ema=ema, generator=torch.Generator().manual_seed(990))
                metrics.append(vector_tasks.score_samples(samples, spec, completed))
            live, ema = metrics
            observations.append(dict(**live, ema=ema, step=completed, seconds=time.perf_counter()-start))
            print(f'STEP {completed}: live spread={live["component_min_eigen_ratio"]:.6f}', flush=True)
    result = dict(live=live, ema=ema, observations=observations, actions=actions)
    verdict = test_verdict(spec, result)
    write('result.json', result)
    write('verdict.json', verdict)
    reference_curve = deepcopy(reference['result']['observations'])
    actual_curve = deepcopy(observations)
    for curve in (reference_curve, actual_curve):
        for point in curve:
            point.pop('seconds')
    parity = dict(all_24_live_ema_exact=actual_curve == reference_curve,
                  all_actions_exact=actions == reference['result']['actions'],
                  final_live_exact=live == reference['result']['live'], final_ema_exact=ema == reference['result']['ema'])
    write('parity.json', dict(**parity, reference=str(reference_path.relative_to(root)),
                             reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                             live_pass=verdict['passed'], passing_suffix=verdict['convergence']['passing_suffix']))
    assert all(parity.values()) and verdict['passed'] and verdict['convergence']['passing_suffix'] == 6
    suite.verify_source(protocol)
    print('PASS: public default/trainer/critic exactly reproduce all live/EMA checkpoints and actions.', flush=True)


if __name__ == '__main__':
    main()

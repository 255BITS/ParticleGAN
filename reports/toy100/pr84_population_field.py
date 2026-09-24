"""Offline population density-ratio comparison at exact saved host states.

The analytic score is the unrestricted, unregularized population Rp critic.
It is not the optimum of the finite host critic with b_cap, and is never used
to train. Its q mixture is frozen when differentiating generator samples.
"""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.locked_shared import mode_hold
from reports.toy100 import pr84_heldout_signal as heldout
from reports.toy100.pr84_stationary_field_analysis import smoothed_score

STEPS = (1325, 1389, 1530, 1540)
ARMS = ('learned_sharp', 'learned_stencil', 'population_sharp', 'population_stencil')


def mixture_log_density(x, means, sigma):
    import math
    squared = (x.double()[:, None, :] - means.detach().double()[None, :, :]).square().sum(-1)
    return torch.logsumexp(-squared / (2 * sigma**2), dim=1) - (
        math.log(len(means)) + x.shape[-1] * math.log(sigma * math.sqrt(2 * math.pi)))


def ratio_score(x, real_means, real_sigma, fake_means, fake_sigma):
    return (mixture_log_density(x, real_means, real_sigma)
            - mixture_log_density(x, fake_means, fake_sigma)).unsqueeze(-1)


def analyze(step, snapshot, capture, config):
    real_means = mode_hold.ring_means()
    width = capture['stages']['record']['critic_width']
    result = dict(step=step, width=width, arms={})
    all_batch_hashes = []
    for arm in ARMS:
        recipe, policy, prior, generator, critic, optimizer_g, _, data = heldout._construct(snapshot, config)
        loss = recipe.make_loss()
        clean = heldout._clean(generator, prior)
        assert torch.equal(clean, torch.tensor(capture['stages']['pre_step']))
        parameters = [p for group in optimizer_g.param_groups for p in group['params']]
        before_parameters = [p.detach().clone() for p in parameters]
        roles = ['prior' if group.get('_comparison_prior') else 'network'
                 for group in optimizer_g.param_groups for _ in group['params']]
        quality_gradient = heldout._quality_gradient(generator, prior, real_means, parameters)
        frozen_metric = []
        for group in optimizer_g.param_groups:
            for p in group['params']:
                state = optimizer_g.state[p]
                denom = (state['exp_avg_sq'] / (1 - group['betas'][1] ** float(state['step']))).sqrt() + group['eps']
                frozen_metric.append((group['lr'] / denom).detach().double())
        population = lambda x: ratio_score(x, real_means, mode_hold.SIGMA, clean, .029)
        base_score = population if arm.startswith('population') else critic.model
        score = ((lambda x: smoothed_score(base_score, x, width))
                 if arm.endswith('stencil') else base_score)
        x = clean.detach().clone().requires_grad_(True)
        direct_score = score(x)
        direct_gradient = torch.autograd.grad(direct_score.sum(), x)[0].detach()
        rows, hashes = [], []
        network_gradients, output_directions = [], []
        for batch_index in range(17):
            latent, indices = prior.sample(mode_hold.BATCH, generator=data)
            fake = generator(latent)
            real = mode_hold.sample_ring(real_means, mode_hold.BATCH, mode_hold.SIGMA, data)
            if batch_index == 0:
                captured_prior = [b for b in capture['stages']['batches']
                                  if b['kind'] == 'prior' and b['phase'] == 1][-1]
                captured_real = [b for b in capture['stages']['batches']
                                 if b['kind'] == 'real' and b['phase'] == 1][-1]
                assert indices.tolist() == captured_prior['indices']
                assert torch.equal(real, torch.tensor(captured_real['values']))
            digest = hashlib.sha256()
            for tensor in (indices, fake.detach(), real):
                digest.update(tensor.contiguous().numpy().tobytes())
            hashes.append(digest.hexdigest())
            objective = loss.g_loss(score(fake), score(real))
            values = torch.autograd.grad(objective, [fake] + parameters)
            sample_direction, gradient = -values[0].detach(), [v.detach() for v in values[1:]]
            output_direction = torch.zeros_like(clean).index_add_(0, indices, sample_direction)
            row = dict(batch=batch_index, loss=float(objective.detach()),
                       quality_direction=heldout._quality_direction(
                           quality_gradient, gradient, frozen_metric, roles),
                       independent_output_direction=heldout._directional(clean, output_direction, real_means))
            rows.append(row)
            if batch_index:
                network_gradients.append(torch.cat([g.flatten() for g, role in zip(gradient, roles) if role == 'network']))
                output_directions.append(output_direction.flatten())
        assert all(torch.equal(p.detach(), old) for p, old in zip(parameters, before_parameters))
        all_batch_hashes.append(hashes)
        result['arms'][arm] = dict(
            direct_score=direct_score.detach().flatten().tolist(),
            direct_score_ascent=heldout._directional(clean, direct_gradient, real_means),
            direct_score_gradient=direct_gradient.tolist(),
            network_gradient_coherence=heldout._energy_fraction(network_gradients),
            independent_output_coherence=heldout._energy_fraction(output_directions),
            rows=rows)
    assert all(hashes == all_batch_hashes[0] for hashes in all_batch_hashes)
    result['paired_batch_sha256'] = all_batch_hashes[0]
    result['base_support'] = clean.tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    evidence = ROOT / 'reports/toy100/continuous-evidence/pr84-stationary-failure-diagnosis'
    names = ('compact-states.pt.gz', 'diagnosis.json.gz')
    manifest = json.loads((evidence / 'manifest.json').read_text())
    # The committed capture manifest binds the portable state subset and diagnosis.
    entries = manifest['files']
    for name in names:
        item = entries[name]
        assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == item['sha256']
    states = torch.load(io.BytesIO(gzip.decompress((evidence / names[0]).read_bytes())), weights_only=True)
    diagnosis = json.loads(gzip.decompress((evidence / names[1]).read_bytes()))
    captures = {row['step']: row for row in diagnosis['rows']}
    config = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    sources = ('reports/toy100/pr84_population_field.py', 'reports/toy100/pr84_heldout_signal.py',
               'reports/toy100/pr84_stationary_field_analysis.py',
               'configs/toy100/constraints_simple_regularization.json',
               'benchmarks/locked_shared/mode_hold.py')
    declaration = dict(scope='offline_population_field_only_not_training', steps=STEPS,
        batches='actual next G batch plus 16 consecutive held-out batches; paired across arms',
        arms=ARMS, q_frozen=True, training_updates=0,
        ideal_scope='unrestricted unregularized population Rp; not finite penalized host optimum',
        source={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        inputs={name: hashlib.sha256((evidence / name).read_bytes()).hexdigest() for name in names})
    (args.output / 'declaration.json').write_text(json.dumps(declaration, indent=2) + '\n')
    for name in sources:
        target = args.output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    result = []
    for step in STEPS:
        row = analyze(step, states[step]['post_accepted_d'], captures[step], config)
        result.append(row)
        (args.output / f'step-{step}.json').write_text(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(dict(event='STEP_DONE', step=step)), flush=True)
    (args.output / 'summary.json').write_text(json.dumps(dict(declaration=declaration, rows=result), allow_nan=False) + '\n')


if __name__ == '__main__':
    main()

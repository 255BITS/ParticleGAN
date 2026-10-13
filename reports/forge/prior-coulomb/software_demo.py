"""Bounded public BCAP integration demo; no Tier 2 qualification or seed search."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn

from particlegan.coulomb_prior import CENTER_BUFFER, RADIUS_BUFFER
from experiments.forge.api import FormulationContext

ROOT = Path(__file__).resolve().parents[3]
STEPS = 64
WEIGHT = .01


def context(weight):
    leader = json.loads((ROOT / 'configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json').read_text())
    settings = deepcopy(leader['recipe_overrides'])
    settings.update(num_particles=64, z_dim=2, batch_size=32, total_steps=STEPS)
    if weight:
        settings.update(prior_regularizer='coulomb', prior_reg=weight)
    return FormulationContext(recipe_preset='bcap', recipe_overrides=settings,
        prior=dict(kind='mog', sigma=.025, standardize=False),
        prior_contract='recipe_owned_v1', seed=0)


def trainer(ctx):
    g = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 2)), component='generator')
    d = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1)), component='discriminator')
    return ctx.build_trainer(g, d)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def data(ctx):
    return torch.randn(32, 2, generator=ctx.streams.generator('data', component='software_demo', purpose='batches'))


def main():
    output = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'runs/prior-coulomb/software-demo')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    baseline_ctx, active_ctx = context(0), context(WEIGHT)
    baseline, active = trainer(baseline_ctx), trainer(active_ctx)
    initial_equal = all(equal(getattr(baseline, k).state_dict(),
        {name: value for name, value in getattr(active, k).state_dict().items()
         if name not in (CENTER_BUFFER, RADIUS_BUFFER)}) for k in ('G', 'D', 'prior'))
    initial_locations = active.prior.z.detach().clone()
    force = torch.autograd.grad(active.prior_regularizer(active.prior.z), active.prior.z)[0]
    magnitudes = force.norm(dim=1)
    force_bracket = []
    for coefficient in (.001, .01, .1):
        scaled = magnitudes * coefficient
        normalized = .03 * scaled / (scaled.square() + .001**2).sqrt()
        force_bracket.append(dict(prior_reg=coefficient, max_auxiliary_row_gradient=float(scaled.max()),
            median_auxiliary_row_gradient=float(scaled.median()),
            median_auxiliary_only_rownorm_displacement=float(normalized.median()),
            max_auxiliary_only_rownorm_displacement=float(normalized.max())))
    target = torch.randn(512, 2, generator=torch.Generator().manual_seed(0))
    frames, events, before = [], [], active.prior.z.detach().clone()
    data_hash = hashlib.sha256()
    saved = None
    for update in range(1, STEPS + 1):
        a, b = data(baseline_ctx), data(active_ctx)
        assert torch.equal(a, b)
        data_hash.update(a.numpy().tobytes())
        baseline.step(a)
        values = active.step(b)
        displacement = (active.prior.z.detach() - before).norm(dim=1)
        before = active.prior.z.detach().clone()
        if update % 8 == 0:
            event = dict(step=update, finite=all(torch.isfinite(p).all().item()
                for model in (active.G, active.D, active.prior) for p in model.parameters()),
                prior_regularization=float(values['prior_regularization']),
                mean_actual_row_displacement=float(displacement.mean()),
                max_actual_row_displacement=float(displacement.max()))
            events.append(event)
            print(json.dumps(event), flush=True)
            outputs = [target, baseline.sample(512), active.sample(512), active.prior.z.detach()]
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            titles = ['Target N(0,I)', 'BCAP leader outputs', 'Coulomb outputs', 'Coulomb latent rows']
            for ax, points, title in zip(axes, outputs, titles):
                ax.scatter(points[:, 0], points[:, 1], s=4, alpha=.55)
                ax.set(xlim=(-4, 4), ylim=(-4, 4), title=title, aspect='equal')
            axes[-1].scatter(initial_locations[:, 0], initial_locations[:, 1], s=8, facecolors='none', edgecolors='gray')
            fig.suptitle(f'Public API integration, update {update}/64; finite={event["finite"]}; no quality qualification')
            fig.tight_layout()
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=85)
            plt.close(fig)
            frames.append(Image.open(buffer).convert('RGB'))
        if update == STEPS // 2:
            saved = active_ctx.state_dict()
    expected = active_ctx.state_dict()
    restored_ctx = context(WEIGHT)
    restored = trainer(restored_ctx)
    restored_ctx.load_state_dict(saved)
    # Evaluation calls are part of the checkpointed stream history too.
    for update in range(STEPS // 2 + 1, STEPS + 1):
        restored.step(data(restored_ctx))
        if update % 8 == 0:
            restored.sample(512)
    exact_restore = equal(expected, restored_ctx.state_dict())
    finite = all(event['finite'] for event in events)
    loss_positive = all(event['prior_regularization'] > 0 for event in events)
    report = dict(scope='64-update public API software integration; no scientific qualification', seed=0,
        target_law='two independent standard Gaussian coordinates', steps=STEPS,
        recipe=active.recipe.to_dict(), sampling='clean live MoG sigma=.025, uniform rows',
        gate=dict(initial_tensors_equal=initial_equal, finite_all_8_observations=finite,
            positive_auxiliary_energy_all_8_observations=loss_positive, exact_checkpoint_continuation=exact_restore),
        measurements=dict(final_raw_prior_energy=events[-1]['prior_regularization'],
            final_mean_actual_row_displacement=events[-1]['mean_actual_row_displacement'],
            final_max_actual_row_displacement=events[-1]['max_actual_row_displacement'],
            max_initial_unweighted_auxiliary_row_gradient=float(magnitudes.max())),
        initial_force_bracket=force_bracket, data_batch_sha256=data_hash.hexdigest(),
        input_source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('particlegan/coulomb_prior.py', 'particlegan/recipes.py', 'particlegan/training.py',
                         'reports/forge/prior-coulomb/software_demo.py')})
    report['status'] = 'PASS' if all(report['gate'].values()) else 'FAIL'
    (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'events.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events))
    frames[0].save(output / 'actual-training.gif', save_all=True, append_images=frames[1:], duration=300, loop=0)
    print(json.dumps(dict(status=report['status'], measurements=report['measurements'])), flush=True)
    assert report['status'] == 'PASS'


if __name__ == '__main__':
    main()

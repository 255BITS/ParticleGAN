"""Public MoG contracts, independent of experiment-only dependencies."""
import builtins
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import MoGParticlePrior, ParticlePrior, ParticleRegularizer, Recipe, get_recipe


def test_default_mog_and_recipe_match_the_selected_compact_experiment():
    prior = MoGParticlePrior()
    recipe = get_recipe("mog")
    assert prior.z.shape == (400, 4) and prior.sigma_rel == 1/40 and prior.standardize
    assert prior.sigma > 0 and prior.sigma == prior.d0 * prior.sigma_rel
    assert recipe.num_particles == 400 and recipe.total_steps == 28000
    assert recipe.prior_lr_mult == 100 and recipe.prior_betas == (.5, .999)
    assert isinstance(recipe.make_prior(), MoGParticlePrior)
    assert type(get_recipe().make_prior()) is ParticlePrior
    assert Recipe(**json.loads(json.dumps(recipe.to_dict()))) == recipe

    from experiments.config import read_config
    from experiments.train_100gaussians import DEFAULTS
    root = Path(__file__).resolve().parents[1]
    cfg = {**DEFAULTS, **read_config(root / "configs/mog/default.toml")}
    reference = read_config(root / "configs/mog/scale_longer/n400_r1over40_28k_s1.yaml")
    assert {k: v for k, v in cfg.items() if k != "out_dir"} == {
        k: v for k, v in reference.items() if k != "out_dir"}
    assert cfg['epochs'] * cfg['steps_per_epoch'] == recipe.total_steps
    assert cfg['particle_lr_multiplier'] * 10 == recipe.prior_lr_mult
    assert cfg['particle_beta1'] == recipe.prior_betas[0]


def test_ddgan_mog_recipe_matches_the_100k_study_and_supports_overrides():
    from experiments.config import read_config
    from experiments.train_denoising import training_recipe

    recipe = get_recipe("ddgan_mog")
    root = Path(__file__).resolve().parents[1]
    cfg = read_config(root / "configs/denoising/mog_capacity_100k/ddgan_mog.yaml")
    assert recipe.replace(name="ddgan") == training_recipe(cfg)
    assert recipe.model == "ddgan" and recipe.conditioning == "ucd"
    assert recipe.total_steps == 100_000 and recipe.lr_floor == 1.
    prior = recipe.make_prior()
    assert isinstance(prior, MoGParticlePrior)
    assert prior.z.shape == (400, 4) and prior.sigma > 0
    assert Recipe(**json.loads(json.dumps(recipe.to_dict()))) == recipe
    changed = get_recipe("ddgan_mog", num_classes=8, z_dim=6, num_particles=32,
                         total_steps=2000, lr_floor=.05)
    assert changed.num_classes == 8 and changed.make_prior().z.shape == (32, 6)
    assert changed.total_steps == 2000 and changed.lr_floor == .05
    assert get_recipe("ddgan_mog") == recipe
    assert get_recipe("ddgan").total_steps == 56_000
    assert get_recipe("ddgan").prior_kind == "particles"


def test_forward_matches_sampling_rng_and_flows_through_module_hooks():
    global_rng = torch.get_rng_state().clone()
    prior = MoGParticlePrior(12, 3, generator=torch.Generator().manual_seed(7)).double()
    direct_rng = torch.Generator().manual_seed(17)
    sample_rng = torch.Generator().manual_seed(17)
    calls = []
    handle = prior.register_forward_hook(lambda *args: calls.append(True))
    indices = prior.sample_indices(20, direct_rng)
    direct = prior(indices, generator=direct_rng)
    sampled, sampled_indices = prior.sample(20, sample_rng)
    handle.remove()
    assert len(calls) == 2
    assert torch.equal(indices, sampled_indices) and torch.equal(direct, sampled)
    assert torch.equal(direct_rng.get_state(), sample_rng.get_state())
    assert torch.equal(global_rng, torch.get_rng_state())
    assert not torch.equal(sampled, prior.means()[indices])
    sampled.square().mean().backward()
    assert prior.z.grad is not None and torch.isfinite(prior.z.grad).all()
    assert set(dict(prior.named_parameters())) == {'z'}


def test_fixed_epsilon_snapshot_and_zero_sigma_rng_contract():
    prior = MoGParticlePrior(8, 2, standardize=False)
    eps = torch.arange(6, dtype=prior.z.dtype).reshape(3, 2)
    rng = torch.Generator().manual_seed(3)
    state = rng.get_state()
    z, ids = prior.sample(3, rng, fixed_first_n=True, offset=2, eps=eps)
    assert torch.equal(z, prior.z[2:5] + prior.sigma * eps)
    assert torch.equal(ids, torch.arange(2, 5)) and torch.equal(state, rng.get_state())
    for wrong in (eps[:2], eps.double()):
        with pytest.raises(ValueError, match='eps'):
            prior.sample(3, fixed_first_n=True, eps=wrong)
    atoms = ParticlePrior(8, 2, generator=torch.Generator().manual_seed(19))
    zero = MoGParticlePrior(8, 2, sigma_rel=0, standardize=False,
                            generator=torch.Generator().manual_seed(19))
    for fixed in (False, True):
        a, b = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
        x, i = atoms.sample(5, a, fixed_first_n=fixed)
        y, j = zero.sample(5, b, fixed_first_n=fixed)
        assert torch.equal(x, y) and torch.equal(i, j) and torch.equal(a.get_state(), b.get_state())
    before = torch.get_rng_state().clone()
    assert torch.equal(zero(torch.tensor([0, 1])), atoms(torch.tensor([0, 1])))
    assert torch.equal(before, torch.get_rng_state())


def test_checkpoint_restores_read_settings_fixed_sigma_and_seeded_samples(tmp_path):
    prior = MoGParticlePrior(12, 2, sigma_rel=.125, standardize=False).double()
    with torch.no_grad():
        prior.z.mul_(2).add_(3)  # Do not recalibrate after training/EMA updates.
    sigma = prior.sigma.clone()
    path = tmp_path / 'mog.pt'
    torch.save(prior.state_dict(), path)
    restored = MoGParticlePrior(12, 2, sigma_rel=0, standardize=True).double()
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert not restored.standardize and restored.sigma_rel == .125
    assert torch.equal(sigma, restored.sigma)
    x, idx = prior.sample(32, torch.Generator().manual_seed(21))
    y, ids = restored.eval().sample(32, torch.Generator().manual_seed(21))
    assert torch.equal(x, y) and torch.equal(idx, ids)

    # A parent module also restores the MoG's extra state and noisy forward path.
    parent = nn.ModuleDict({'prior': MoGParticlePrior(12, 2, sigma_rel=0)})
    parent.load_state_dict({'prior.' + k: v for k, v in prior.state_dict().items()})
    assert not parent['prior'].standardize and parent['prior']._noise_enabled

    # Historical checkpoints contain tensors only; caller supplies read standardization.
    legacy = {k: v for k, v in prior.state_dict().items() if k != '_extra_state'}
    old = MoGParticlePrior(12, 2, sigma_rel=0, standardize=False).double()
    old.load_state_dict(legacy, strict=True)
    y, _ = old.sample(32, torch.Generator().manual_seed(21))
    assert torch.equal(x, y)


@pytest.mark.parametrize('use_scipy', [False, True])
def test_calibration_exact_even_median_and_torch_only_fallback(monkeypatch, use_scipy):
    if use_scipy:
        pytest.importorskip('scipy')
    else:
        original_import = builtins.__import__
        def torch_only(name, *args, **kwargs):
            if name.startswith(('scipy', 'numpy')):
                raise ImportError('optional dependency unavailable')
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', torch_only)
    prior = MoGParticlePrior(4, 1, sigma_rel=.5, standardize=False, dtype=torch.float64)
    with torch.no_grad():
        prior.z.copy_(torch.tensor([[0.], [1.], [4.], [10.]]))
    before = torch.get_rng_state().clone()
    prior.calibrate()
    # NN distances [1,1,3,6]: midpoint median is 2, not the lower middle value 1.
    assert prior.d0.item() == 2 and prior.sigma.item() == 1
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('recipe_name', ['mog', 'ddgan_mog'])
def test_mog_recipe_optimizer_updates_raw_means_and_preserves_fixed_buffers(recipe_name):
    recipe = get_recipe(recipe_name, num_particles=12, z_dim=2, prior_betas=[.5, .999])
    prior = recipe.make_prior()
    generator, critic = nn.Linear(2, 2), nn.Linear(2, 1)
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior, foreach=False)
    assert [group['lr'] for group in opt_g.param_groups] == [.0006, .06]
    assert [group['betas'] for group in opt_g.param_groups] == [(0., .999), (.5, .999)]
    assert opt_d.param_groups[0]['betas'] == (0., .999)
    before, sigma = prior.z.detach().clone(), prior.sigma.clone()
    eps = torch.ones(6, 2)
    z, _ = prior.sample(6, fixed_first_n=True, eps=eps)
    loss = generator(z).square().mean() + ParticleRegularizer()(prior.z)
    loss.backward()
    opt_g.step()
    assert not torch.equal(before, prior.z) and torch.equal(sigma, prior.sigma)
    ema = copy.deepcopy(prior)
    with torch.no_grad():
        prior.z[0].add_(2)
        ema.z.lerp_(prior.z, .1)
    torch.testing.assert_close(ema.means().mean(0), torch.zeros(2), atol=1e-6, rtol=0)
    assert not torch.equal(ema.means(), prior.means()) and torch.equal(sigma, ema.sigma)
    frozen = recipe.make_prior(learnable=False)
    assert not list(frozen.parameters())
    assert len(recipe.make_optimizers(generator, critic, frozen)[0].param_groups) == 1


@pytest.mark.parametrize('kwargs', [dict(sigma_rel=-1), dict(sigma_rel=float('nan')),
                                     dict(standardize='false'), dict(num_particles=1), dict(init_std=0)])
def test_invalid_mog_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        MoGParticlePrior(**kwargs)


@pytest.mark.parametrize('kwargs', [dict(prior_kind='typo'), dict(sigma_rel=-1),
                                     dict(standardize='false'), dict(prior_betas=[1., .9]),
                                     dict(prior_kind='particles', sigma_rel=.1)])
def test_invalid_recipe_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        get_recipe('mog', **kwargs)


def test_factory_overrides_are_local_and_cannot_silently_discard_noise():
    original = get_recipe()
    mog = original.make_prior(prior_kind='mog', sigma_rel=.1, num_particles=8)
    assert isinstance(mog, MoGParticlePrior) and original.prior_kind == 'particles'
    with pytest.raises(ValueError, match='nonzero sigma_rel'):
        original.make_prior(sigma_rel=.1)
    with pytest.raises(ValueError, match='prior_kind'):
        original.make_prior(prior_kind='typo')

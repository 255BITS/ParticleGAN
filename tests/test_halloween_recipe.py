"""Historical formula, role ownership and checkpoint compatibility controls."""
from copy import deepcopy
from dataclasses import asdict
import math

import pytest
import torch
from torch import nn

from particlegan import GANLoss, GANTrainer, Recipe, TensorFlowV1Adam, get_recipe, learning_rate_scales
from experiments.forge.configuration_search import recipe_identity_fields
from particlegan.recipe_schedules import apply_training_schedules


def module():
    model = nn.Linear(2, 2, dtype=torch.float64)
    from particlegan import init
    init.deterministic_orthogonal_(model, seed=0)
    return model


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def test_distinct_generator_critic_and_prior_role_settings_are_consumed():
    recipe = get_recipe("bcap", betas=(.4, .8), d_betas=(.1, .7), prior_betas=(.2, .6),
                        eps=.03, d_eps=.005, prior_eps=.0001, num_particles=8)
    g, d = module(), module()
    prior = recipe.make_prior()
    og, od = recipe.make_optimizers(g, d, prior)
    assert [group["betas"] for group in og.param_groups] == [(.4, .8), (.2, .6)]
    assert [group["eps"] for group in og.param_groups] == [.03, .0001]
    assert od.param_groups[0]["betas"] == (.1, .7)
    assert od.param_groups[0]["eps"] == .005
    # Default inheritance remains shared, and explicit factory kwargs win.
    shared = recipe.replace(d_betas=None, d_eps=None, prior_eps=None)
    shared_g, shared_d = shared.make_optimizers(g, d, prior)
    assert all(group["eps"] == .03 for group in shared_g.param_groups + shared_d.param_groups)
    assert shared_d.param_groups[0]["betas"] == (.4, .8)
    assert recipe.make_critic_optimizer(d, eps=.02).param_groups[0]["eps"] == .02


def test_role_beta2_schedule_starts_from_each_roles_own_initial_moment():
    recipe = get_recipe("bcap", total_steps=20, num_particles=8, betas=(.4, .8), d_betas=(.1, .7),
                        prior_betas=(.2, .6), beta2_end=.95, beta2_anneal_end=.2)
    og, od = recipe.make_optimizers(module(), module(), recipe.make_prior())
    apply_training_schedules(2, recipe, (og, od))
    for group, expected in zip(og.param_groups, [(.4, .875), (.2, .775)]):
        assert group["betas"] == pytest.approx(expected)
    assert od.param_groups[0]["betas"] == pytest.approx((.1, .825))


def test_explicit_shared_d_moments_preserve_signature_and_d_only_schedule_is_active():
    from experiments.forge.techniques import recipe_field_active, validate_same_technique
    base = get_recipe("bcap", beta2_end=.999)
    validate_same_technique(base, base.replace(d_betas=base.betas))
    distinct = base.replace(d_betas=(0, .8))
    assert recipe_field_active("beta2_end", distinct)
    validate_same_technique(distinct, distinct.replace(d_betas=(0, .7)))


def test_historical_labels_and_joint_stream_derivatives():
    loss = GANLoss("least_squares", labels=(-1, 1, 1))
    real = torch.tensor([0., 1., 2.], requires_grad=True)
    fake = torch.tensor([-1., 0., 1.], requires_grad=True)
    assert loss.d_loss(real, fake).item() == pytest.approx(7/6)
    dr, df = torch.autograd.grad(loss.d_loss(real, fake), (real, fake))
    assert torch.allclose(dr, (real - 1) / 3)
    assert torch.allclose(df, (fake + 1) / 3)
    gr, gf = torch.autograd.grad(loss.joint_g_loss(fake, real), (real, fake))
    assert torch.allclose(gr, (real + 1) / 3)
    assert torch.allclose(gf, (fake - 1) / 3)


def test_dense_tf_formula_and_skipped_gradients_use_optimizer_application_clock():
    p, skipped = nn.Parameter(torch.tensor([1.], dtype=torch.float64)), nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    optimizer = TensorFlowV1Adam([p, skipped], lr=.008, betas=(.61, .56), eps=.308)
    expected, m, v = 1., 0., 0.
    for step, grad in enumerate([.001, -.2, .04, .7], 1):
        p.grad = torch.tensor([grad], dtype=torch.float64)
        m, v = .61 * m + .39 * grad, .56 * v + .44 * grad**2
        expected -= .008 * math.sqrt(1 - .56**step) / (1 - .61**step) * m / (math.sqrt(v) + .308)
        optimizer.step()
        assert p.item() == pytest.approx(expected, abs=1e-15)
    skipped.grad = torch.tensor([.03], dtype=torch.float64)
    optimizer.step()
    assert optimizer.state[skipped]["step"].item() == 5
    assert optimizer.param_groups[0]["_tf_step"] == 5
    q, r = nn.Parameter(p.detach().clone()), nn.Parameter(skipped.detach().clone())
    restored = TensorFlowV1Adam([q, r], lr=.008, betas=(.61, .56), eps=.308)
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    for a, b in [(p, q), (skipped, r)]:
        a.grad = torch.ones_like(a) * .12
        b.grad = a.grad.clone()
    optimizer.step(); restored.step()
    assert torch.equal(p, q) and torch.equal(skipped, r)
    equal(optimizer.state_dict(), restored.state_dict())


def test_tf_checkpoint_rejects_bad_powers_or_native_state_before_mutation():
    p = nn.Parameter(torch.ones(2))
    opt = TensorFlowV1Adam([p], lr=.01)
    original = deepcopy(opt.state_dict())
    corrupt = deepcopy(original)
    corrupt["param_groups"][0]["_tf_beta2_power"] = .1
    with pytest.raises(ValueError, match="power"):
        opt.load_state_dict(corrupt)
    equal(original, opt.state_dict())
    with pytest.raises(ValueError, match="checkpoint"):
        opt.load_state_dict(torch.optim.Adam([p]).state_dict())
    p.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.]), (2,), check_invariants=True)
    with pytest.raises(ValueError, match="dense"):
        opt.step()
    equal(original, opt.state_dict())


def test_native_recipe_optimizer_rejects_legacy_backend_state():
    legacy = get_recipe("halloween")
    native = legacy.replace(adam_variant="pytorch")
    for factory in ("make_generator_optimizer", "make_critic_optimizer"):
        def parameters():
            model = module()
            return model.parameters() if factory == "make_generator_optimizer" else model
        old = getattr(legacy, factory)(parameters())
        new = getattr(native, factory)(parameters())
        before = deepcopy(new.state_dict())
        with pytest.raises(ValueError, match="checkpoint"):
            new.load_state_dict(old.state_dict())
        equal(before, new.state_dict())


def test_negative_tf_variance_checkpoint_is_rejected_before_mutation():
    p = nn.Parameter(torch.ones(2))
    opt = TensorFlowV1Adam([p])
    p.grad = torch.ones_like(p)
    opt.step()
    before = deepcopy(opt.state_dict())
    corrupt = deepcopy(before)
    next(iter(corrupt["state"].values()))["exp_avg_sq"].fill_(-1)
    with pytest.raises(ValueError, match="moment"):
        opt.load_state_dict(corrupt)
    equal(before, opt.state_dict())


def test_default_additions_preserve_archived_recipe_identity_and_packets():
    recipe = get_recipe("bcap")
    old = asdict(recipe)
    additions = {"d_betas", "d_eps", "prior_eps", "loss_labels", "adam_variant", "lr_schedule",
                 "lr_decay_rate", "lr_decay_steps", "lr_decay_staircase"}
    old = {k: v for k, v in old.items() if k not in additions}
    assert recipe_identity_fields(old) == recipe_identity_fields(asdict(recipe))
    assert not additions & recipe.to_dict().keys()
    assert Recipe(**recipe.to_dict()) == recipe
    changed = recipe.replace(d_eps=.2)
    assert recipe_identity_fields(asdict(changed)) != recipe_identity_fields(old)


def test_explicit_decay_clock_does_not_rescale_when_training_budget_changes():
    recipe = get_recipe("halloween", lr_schedule="exponential", lr_decay_rate=.96, lr_decay_steps=50)
    assert learning_rate_scales(25, recipe) == pytest.approx((.96**.5, .96**.5))
    assert learning_rate_scales(25, recipe.replace(total_steps=12)) == learning_rate_scales(25, recipe)
    assert learning_rate_scales(25, recipe.replace(lr_decay_staircase=True)) == (1., 1.)
    assert learning_rate_scales(500, get_recipe("halloween")) == (1., 1.)


def test_public_halloween_trainer_restores_all_optimizer_clocks_and_streams():
    from experiments.forge.rng import NamedStreams
    recipe = get_recipe("halloween", num_particles=8, batch_size=4, total_steps=8,
                        prior_kind="mog", sigma_rel=.025, standardize=False)
    def make():
        from experiments.forge.api import TRAINER_STREAM_BINDINGS
        streams = NamedStreams(0)
        return GANTrainer(recipe, module(), nn.Linear(2, 1, dtype=torch.float64),
                          seed=0, **{name: streams.generator(family, component=component, purpose=purpose)
                                     for name, (family, component, purpose) in TRAINER_STREAM_BINDINGS.items()})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        left = make()
        batch = torch.ones(4, 2, dtype=torch.float64)
        left.step(batch)
        saved = deepcopy(left.state_dict())
        right = make()
        right.load_state_dict(saved)
        left.step(batch); right.step(batch)
        equal(left.state_dict(), right.state_dict())


@pytest.mark.parametrize("changes", [
    {"d_betas": [True, .9]}, {"d_betas": [float("nan"), .9]}, {"d_eps": 0}, {"prior_eps": False},
    {"loss_labels": [-1, 1, True]}, {"adam_variant": "unknown"}, {"lr_decay_steps": 1.5},
    {"lr_decay_rate": 0}, {"lr_schedule": "unknown"}, {"beta2_end": .8}, {"amsgrad": True},
])
def test_historical_recipe_rejects_invalid_or_unsupported_laws(changes):
    with pytest.raises(ValueError):
        get_recipe("halloween", **changes)

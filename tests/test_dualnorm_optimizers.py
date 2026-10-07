"""Analytic optimizer and public checkpoint contracts, not research qualification.

The tiny API updates below verify execution and continuation. Scientific toy
acquisition gates and actual-training media belong to the registered Forge study.
"""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe, scale_learning_rates
from particlegan.init import deterministic_orthogonal_
from particlegan.optim.dualnorm import NormalizedOptimizer, polar_factor


FAMILIES = (
    "sgda", "nsgda_global", "nsgda_layer", "ada_nsgda", "dualnorm",
    "dualnorm_D_only", "particle_rownorm_only",
)


@pytest.fixture(autouse=True)
def cuda_optimizer_contract():
    """Run numerical and tiny public-API software checks on the GPU only."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for optimizer contract checks")
    with torch.device("cuda:0"), torch.autograd.set_multithreading_enabled(False):
        yield


@pytest.mark.parametrize("family", FAMILIES)
def test_standardized_role_hyperparameters_reach_normalized_and_native_hybrid_groups(family):
    recipe = get_recipe("bcap", optimizer_family=family, lr=.01, d_lr_mult=1.5,
                        prior_lr_mult=3., betas=(0., .9), d_betas=(0., .8),
                        prior_betas=(0., .7), eps=.03, d_eps=.02, prior_eps=.01,
                        num_particles=8, z_dim=2, standardize=False,
                        optimizer_adam_lr=.00425 if family in {
                            "dualnorm_D_only", "particle_rownorm_only"} else None)
    generator, encoder, critic = (nn.Linear(2, 2) for _ in range(3))
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior, encoder=encoder)
    assert [group["betas"] for group in opt_g.param_groups] == [(0., .9), (0., .9), (0., .7)]
    assert [group["eps"] for group in opt_g.param_groups] == [.03, .03, .01]
    assert opt_d.param_groups[0]["betas"] == (0., .8)
    assert opt_d.param_groups[0]["eps"] == .02
    if family == "dualnorm":
        assert opt_g._adam is None and opt_d._adam is None
    if family == "dualnorm_D_only":
        assert [group["lr"] for group in opt_g.param_groups] == [.00425, .00425, .00425 * 3.]
        assert opt_d.param_groups[0]["lr"] == .015
        assert type(opt_g._adam) is torch.optim.Adam and opt_d._adam is None
    if family == "particle_rownorm_only":
        assert [group["lr"] for group in opt_g.param_groups] == [.00425, .00425, .03]
        assert opt_d.param_groups[0]["lr"] == .00425 * 1.5
        assert type(opt_g._adam) is torch.optim.Adam and type(opt_d._adam) is torch.optim.Adam


def test_graft_rejects_nonzero_critic_first_moment_from_standardized_role_override():
    with pytest.raises(ValueError, match="beta1=0"):
        get_recipe("bcap", optimizer_family="ada_nsgda", d_betas=(.5, .9))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("schedule,expected", [("constant", 1.), ("exponential", .25)])
def test_explicit_schedule_multiplies_all_normalized_and_native_hybrid_rates(family, schedule, expected):
    recipe = get_recipe("bcap", optimizer_family=family, lr=.01, d_lr_mult=1.5,
                        prior_lr_mult=3., num_particles=8, z_dim=2, standardize=False,
                        lr_schedule=schedule, lr_decay_rate=.5, lr_decay_steps=10,
                        optimizer_adam_lr=.00425 if family in {
                            "dualnorm_D_only", "particle_rownorm_only"} else None)
    prior = recipe.make_prior()
    optimizers = recipe.make_optimizers(nn.Linear(2, 2), nn.Linear(2, 1), prior)
    rates = [[group["lr"] for group in opt.param_groups] for opt in optimizers]
    scale_learning_rates(20, recipe, optimizers, rates, prior)
    for optimizer, initial_rates in zip(optimizers, rates):
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
            [rate * expected for rate in initial_rates])


def test_inactive_critic_betas_do_not_invent_adam_mechanism_in_dualnorm_search():
    from experiments.forge.techniques import recipe_field_active, technique_signature, validate_same_technique
    recipe = get_recipe("bcap", optimizer_family="dualnorm")
    alternative = recipe.replace(d_betas=(.5, .9))
    assert not recipe_field_active("d_betas", alternative)
    assert "critic_moments" not in technique_signature(recipe)["mechanisms"]
    validate_same_technique(recipe, alternative)


def parameter(values):
    return nn.Parameter(torch.tensor(values, dtype=torch.float64))


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for actual, expected in zip(left, right):
            assert_state_equal(actual, expected)
    else:
        assert left == right


def test_sgda_uses_each_players_scheduled_step_size_without_normalization():
    generator, prior, critic = parameter([1., -2.]), parameter([[1., -2.]]), parameter([1., -2.])
    groups = [dict(params=[generator], role="generator", lr=.2),
              dict(params=[prior], role="prior", lr=.7),
              dict(params=[critic], role="critic", lr=.4)]
    optimizer = NormalizedOptimizer(groups, family="sgda", lr=.1)
    for group in optimizer.param_groups:
        tensor = group["params"][0]
        tensor.grad = torch.tensor([3., 4.], dtype=torch.float64).reshape(tensor.shape)
    optimizer.step()
    for tensor, rate in ((generator, .2), (prior, .7), (critic, .4)):
        torch.testing.assert_close(tensor, tensor.new_tensor([1., -2.]).reshape(tensor.shape) - rate * tensor.grad,
                                   rtol=0, atol=1e-15)


def test_global_norm_combines_generator_and_encoder_but_separates_prior_and_critic():
    generator = parameter([0., 0.])
    encoder = parameter([0., 0., 0.])
    prior = parameter([[0., 0.]])
    critic = parameter([0., 0.])
    eps = 1e-7
    optimizer = NormalizedOptimizer([
        dict(params=[generator], role="generator", lr=.2),
        dict(params=[encoder], role="encoder", lr=.2),
        dict(params=[prior], role="prior", lr=.7),
        dict(params=[critic], role="critic", lr=.4),
    ], family="nsgda_global", eps=eps)
    generator.grad = torch.tensor([3., 4.], dtype=torch.float64)
    encoder.grad = torch.tensor([0., 0., 12.], dtype=torch.float64)
    prior.grad = torch.tensor([[6., 8.]], dtype=torch.float64)
    critic.grad = torch.tensor([5., 12.], dtype=torch.float64)
    optimizer.step()
    for tensor, rate, norm in ((generator, .2, 13.), (encoder, .2, 13.),
                              (prior, .7, 10.), (critic, .4, 13.)):
        torch.testing.assert_close(tensor, -rate * tensor.grad / (norm + eps),
                                   rtol=0, atol=1e-15)


def test_layer_normalization_budgets_each_tensor_independently():
    weight, bias = parameter([[0., 0.]]), parameter([0., 0., 0.])
    weight.grad = torch.tensor([[3., 4.]], dtype=torch.float64)
    bias.grad = torch.tensor([0., 0., 12.], dtype=torch.float64)
    optimizer = NormalizedOptimizer([dict(params=[weight, bias], role="generator")],
                                    family="nsgda_layer", lr=.3, eps=1e-8)
    optimizer.step()
    torch.testing.assert_close(weight, -.3 * weight.grad / (5. + 1e-8), rtol=0, atol=1e-15)
    torch.testing.assert_close(bias, -.3 * bias.grad / (12. + 1e-8), rtol=0, atol=1e-15)


def test_adam_magnitude_graft_uses_sgd_direction_and_applies_lr_once():
    weight = parameter([1., -2.])
    eta, beta2, eps = .2, .5, 1e-7
    optimizer = NormalizedOptimizer([dict(params=[weight], role="generator")],
                                    family="ada_nsgda", lr=eta, betas=(0., beta2), eps=eps)
    expected = weight.detach().clone()
    second_moment = torch.zeros_like(expected)
    for step, values in enumerate(([3., 4.], [-1., 2.]), start=1):
        gradient = torch.tensor(values, dtype=torch.float64)
        second_moment = beta2 * second_moment + (1 - beta2) * gradient.square()
        adam_direction = gradient / ((second_moment / (1 - beta2 ** step)).sqrt() + eps)
        expected -= eta * adam_direction.norm() * gradient / (gradient.norm() + eps)
        weight.grad = gradient
        optimizer.step()
        torch.testing.assert_close(weight, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("shape", [(4, 2), (2, 4)])
def test_rectangular_dualnorm_matrix_step_has_declared_spectral_budget(shape):
    weight = nn.Parameter(torch.zeros(shape, dtype=torch.float64))
    gradient = torch.zeros_like(weight)
    gradient[0, 0], gradient[1, 1] = 3., 4.
    weight.grad = gradient
    eta = .3
    optimizer = NormalizedOptimizer([dict(params=[weight], role="critic")],
                                    family="dualnorm", lr=eta)
    optimizer.step()
    factor = math.sqrt(max(1., shape[0] / shape[1]))
    expected = torch.zeros_like(weight)
    expected[0, 0] = expected[1, 1] = -eta * factor
    torch.testing.assert_close(weight, expected, rtol=0, atol=1e-15)
    torch.testing.assert_close(torch.linalg.svdvals(weight),
                               torch.full((2,), eta * factor, dtype=torch.float64),
                               rtol=0, atol=1e-15)


def test_rank_deficient_dualnorm_is_finite_and_zero_gradient_matrix_is_skipped():
    weight = parameter([[0., 0.], [0., 0.]])
    optimizer = NormalizedOptimizer([dict(params=[weight], role="critic")],
                                    family="dualnorm", lr=.2, momentum=.5)
    weight.grad = torch.tensor([[3., 0.], [0., 0.]], dtype=torch.float64)
    optimizer.step()
    assert torch.isfinite(weight).all()
    assert torch.linalg.matrix_norm(weight, ord=2).item() == pytest.approx(.2)
    previous = weight.detach().clone()
    weight.grad = torch.zeros_like(weight)
    optimizer.step()
    assert torch.equal(weight, previous)


@pytest.mark.parametrize("mu", [0., .5, .9])
def test_dualnorm_vector_momentum_is_accumulated_before_normalizing(mu):
    bias = parameter([0., 0.])
    optimizer = NormalizedOptimizer([dict(params=[bias], role="generator")],
                                    family="dualnorm", lr=.2, momentum=mu)
    expected = torch.zeros_like(bias)
    momentum = torch.zeros_like(bias)
    for values in ([3., 4.], [-1., 0.]):
        bias.grad = torch.tensor(values, dtype=torch.float64)
        momentum = mu * momentum + bias.grad
        expected -= .2 * momentum / (momentum.norm() + optimizer.defaults["eps"])
        optimizer.step()
        torch.testing.assert_close(bias, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("family", ["dualnorm", "particle_rownorm_only"])
def test_particle_rownorm_moves_sampled_unique_rows_only_without_momentum(family):
    table = nn.Parameter(torch.zeros((5, 2), dtype=torch.float64))
    optimizer = NormalizedOptimizer([dict(params=[table], role="prior")],
                                    family=family, lr=.4, momentum=.5 if family == "dualnorm" else 0.)
    # Dense gradients model a whole-table VICReg contribution: nonzero values
    # on an unsampled row still must not grant it update ownership.
    table.grad = torch.tensor([[1., 2.], [3., 4.], [8., 9.], [-5., 0.], [6., 7.]],
                              dtype=torch.float64)
    optimizer.set_sampled_rows(table, torch.tensor([1, 1, 3]))
    optimizer.step()
    expected = torch.zeros_like(table)
    expected[1] = -.4 * table.grad[1] / (5. + optimizer.defaults["eps"])
    expected[3] = -.4 * table.grad[3] / (5. + optimizer.defaults["eps"])
    torch.testing.assert_close(table, expected, rtol=0, atol=1e-15)
    previous = table.detach().clone()
    table.grad = torch.tensor([[0., 5.], [-4., 3.], [1., 1.], [1., 1.], [1., 1.]],
                              dtype=torch.float64)
    optimizer.set_sampled_rows(table, torch.tensor([0, 0]))
    optimizer.step()
    expected[0] = -.4 * table.grad[0] / (5. + optimizer.defaults["eps"])
    torch.testing.assert_close(table, expected, rtol=0, atol=1e-15)
    assert torch.equal(table[1:], previous[1:])
    # A prior row-normalized state must never carry historical momentum.
    assert "momentum_buffer" not in optimizer.state.get(table, {})


def test_missing_sample_ownership_fails_before_any_other_parameter_moves():
    weight = parameter([1., 2.])
    table = nn.Parameter(torch.zeros((5, 2), dtype=torch.float64))
    optimizer = NormalizedOptimizer([
        dict(params=[weight], role="generator"), dict(params=[table], role="prior"),
    ], family="dualnorm", lr=.2)
    weight.grad, table.grad = torch.ones_like(weight), torch.ones_like(table)
    before = [p.detach().clone() for p in (weight, table)]
    with pytest.raises(ValueError, match="sampled rows"):
        optimizer.step()
    for actual, expected in zip((weight, table), before):
        assert torch.equal(actual, expected)


def test_standalone_checkpoint_preserves_pending_sample_ownership():
    table = nn.Parameter(torch.zeros((5, 2), dtype=torch.float64))
    restored_table = nn.Parameter(table.detach().clone())
    optimizer = NormalizedOptimizer([dict(params=[table], role="prior")], family="dualnorm", lr=.2)
    restored = NormalizedOptimizer([dict(params=[restored_table], role="prior")], family="dualnorm", lr=.2)
    optimizer.set_sampled_rows(table, torch.tensor([2, 2, 4]))
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    table.grad = torch.arange(1, 11, dtype=torch.float64).reshape(5, 2)
    restored_table.grad = table.grad.clone()
    optimizer.step()
    restored.step()
    assert torch.equal(table, restored_table)
    assert torch.equal(table[[0, 1, 3]], torch.zeros((3, 2), dtype=torch.float64))
    assert_state_equal(optimizer.state_dict(), restored.state_dict())


@pytest.mark.parametrize("family", ["dualnorm_D_only", "particle_rownorm_only"])
def test_isolation_arms_preserve_native_adam_updates_for_baseline_players(family):
    recipe = get_recipe("bcap", optimizer_family=family, lr=.03,
                        optimizer_adam_lr=.0006, d_lr_mult=1.5, prior_lr_mult=3.,
                        z_dim=2, num_particles=8, standardize=False)
    generator, encoder, critic = (nn.Linear(2, 2, dtype=torch.float64) for _ in range(3))
    prior = recipe.make_prior().to(dtype=torch.float64)
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior, encoder=encoder)
    groups = [(list(generator.parameters()) + list(encoder.parameters()), .0006)]
    if family == "dualnorm_D_only":
        groups.append((list(prior.parameters()), .0006 * 3.))
    else:
        groups.append((list(critic.parameters()), .0006 * 1.5))
    references = [([nn.Parameter(p.detach().clone()) for p in params], rate)
                  for params, rate in groups]
    baseline = torch.optim.Adam([dict(params=params, lr=rate) for params, rate in references],
                                betas=recipe.betas, eps=recipe.eps)
    for step in range(2):
        for (params, _), (copies, _) in zip(groups, references):
            for index, (actual, copy) in enumerate(zip(params, copies)):
                gradient = torch.arange(1, actual.numel() + 1, dtype=actual.dtype).reshape(actual.shape)
                actual.grad = gradient * (index + 1) * (1. if step == 0 else -.3)
                copy.grad = actual.grad.clone()
        if family == "particle_rownorm_only":
            prior.z.grad = torch.ones_like(prior.z)
            opt_g.set_sampled_rows(prior.z, torch.tensor([0, 2]))
        else:
            for p in critic.parameters():
                p.grad = torch.ones_like(p)
        opt_g.step()
        opt_d.step()
        baseline.step()
        for (params, _), (copies, _) in zip(groups, references):
            for actual, copy in zip(params, copies):
                torch.testing.assert_close(actual, copy, rtol=0, atol=1e-15)


@pytest.mark.parametrize("family", FAMILIES)
def test_existing_cosine_multiplier_scales_new_units_and_hybrid_baseline_rates(family):
    recipe = get_recipe("bcap", optimizer_family=family, lr=.03,
                        optimizer_adam_lr=.0006 if family in ("dualnorm_D_only", "particle_rownorm_only") else None,
                        d_lr_mult=1.5, prior_lr_mult=3., z_dim=2, num_particles=8,
                        total_steps=100, lr_anneal_start=.6, lr_floor=.05,
                        network_lr_floor=None, network_lr_horizon_cap=None,
                        standardize=False)
    generator, critic = nn.Linear(2, 2), nn.Linear(2, 1)
    prior = recipe.make_prior()
    optimizers = recipe.make_optimizers(generator, critic, prior)
    base_rates = [[group["lr"] for group in opt.param_groups] for opt in optimizers]
    scale_learning_rates(80, recipe, optimizers, base_rates, prior)
    # Halfway through the declared last 40% cosine: .05 + .95 * .5.
    for optimizer, initial_rates in zip(optimizers, base_rates):
        for group, rate in zip(optimizer.param_groups, initial_rates):
            assert group["lr"] == pytest.approx(.525 * rate)


def make_trainer(family):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        recipe = get_recipe("bcap", optimizer_family=family,
                            optimizer_momentum=.5 if family in ("dualnorm", "dualnorm_D_only") else 0.,
                            optimizer_adam_lr=.0006 if family in ("dualnorm_D_only", "particle_rownorm_only") else None,
                            lr=.01, d_lr_mult=1.5, prior_lr_mult=3.,
                            z_dim=2, num_particles=8, batch_size=4, total_steps=8,
                            prior_kind="mog", sigma_rel=.025, standardize=False)
        generator = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
        critic = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        prior = recipe.make_prior()
        deterministic_orthogonal_(generator, seed=0)
        deterministic_orthogonal_(critic, seed=1)
        deterministic_orthogonal_(prior, seed=2)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                          model_generator=torch.Generator(device=generator[0].weight.device).manual_seed(17))


@pytest.mark.parametrize("family", FAMILIES)
def test_public_optimizer_checkpoint_restores_exact_next_update(family):
    trainer = make_trainer(family)
    batch = torch.tensor([[-1., -.5], [-.2, .1], [.4, .8], [1., 1.5]])
    trainer.step(batch)
    checkpoint = trainer.state_dict()
    trainer.step(batch)
    expected = trainer.state_dict()
    restored = make_trainer(family)
    restored.load_state_dict(checkpoint)
    restored.step(batch)
    assert_state_equal(expected, restored.state_dict())


@pytest.mark.parametrize("family", FAMILIES)
def test_public_optimizer_checkpoint_rejects_wrong_state_without_mutating_live_state(family):
    trainer = make_trainer(family)
    batch = torch.tensor([[-1., -.5], [-.2, .1], [.4, .8], [1., 1.5]])
    trainer.step(batch)
    before = trainer.state_dict()
    checkpoint = deepcopy(before)
    checkpoint["optimizers"][0]["param_groups"][0]["role"] = "prior"
    with pytest.raises(ValueError, match="optimizer"):
        trainer.load_state_dict(checkpoint)
    assert_state_equal(before, trainer.state_dict())


def test_later_native_adam_group_updates_and_resumes_exactly():
    table, generator = parameter([[0., 0.], [0., 0.]]), parameter([0., 0.])
    optimizer = NormalizedOptimizer([dict(params=[table], role="prior")],
                                    family="particle_rownorm_only", lr=.3, betas=(0., .9))
    assert optimizer._adam is None
    optimizer.add_param_group(dict(params=[generator], role="generator", lr=.02))
    reference = parameter([0., 0.])
    native = torch.optim.Adam([reference], lr=.02, betas=(0., .9))
    generator.grad = reference.grad = torch.tensor([3., -4.], dtype=torch.float64)
    table.grad = torch.ones_like(table)
    optimizer.set_sampled_rows(table, torch.tensor([0]))
    optimizer.step()
    native.step()
    assert torch.equal(generator, reference)
    assert optimizer._adam is not None

    restored_table = parameter(table.detach().tolist())
    restored_generator = parameter(generator.detach().tolist())
    restored = NormalizedOptimizer([dict(params=[restored_table], role="prior")],
                                   family="particle_rownorm_only", lr=.3, betas=(0., .9))
    restored.add_param_group(dict(params=[restored_generator], role="generator", lr=.02))
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    for current, current_table, current_generator in (
            (optimizer, table, generator), (restored, restored_table, restored_generator)):
        current_table.grad = torch.full_like(current_table, 2.)
        current_generator.grad = torch.tensor([-1., 2.], dtype=torch.float64)
        current.set_sampled_rows(current_table, torch.tensor([1]))
        current.step()
    reference.grad = generator.grad.clone()
    native.step()
    assert torch.equal(generator, reference)
    assert torch.equal(generator, restored_generator)
    assert torch.equal(table, restored_table)
    assert_state_equal(optimizer.state_dict(), restored.state_dict())


def test_later_dualnorm_group_keeps_the_full_arm_free_of_adam():
    table, weight = parameter([[0., 0.], [0., 0.]]), parameter([[0., 0.], [0., 0.]])
    optimizer = NormalizedOptimizer([dict(params=[table], role="prior")],
                                    family="dualnorm", lr=.3, momentum=.5)
    optimizer.add_param_group(dict(params=[weight], role="generator", lr=.02))
    assert optimizer._adam is None
    table.grad = torch.ones_like(table)
    weight.grad = torch.diag(torch.tensor([3., 4.], dtype=torch.float64))
    optimizer.set_sampled_rows(table, torch.tensor([0]))
    optimizer.step()
    assert optimizer._adam is None
    torch.testing.assert_close(weight, -.02 * torch.eye(2, dtype=torch.float64), rtol=0, atol=1e-15)


@pytest.mark.parametrize("small_singular_value", [1e-7, 0.])
@pytest.mark.parametrize("transpose", [False, True])
def test_large_polar_preserves_resolved_directions_and_removes_null_completion(
        small_singular_value, transpose):
    gradient = torch.zeros((2, 1025), dtype=torch.float64)
    gradient[0, 0], gradient[1, 1] = 1., small_singular_value
    gradient = gradient.T if transpose else gradient
    factor = polar_factor(gradient)
    assert torch.isfinite(factor).all()
    expected = factor.new_tensor([1., float(small_singular_value > 0)])
    torch.testing.assert_close(torch.linalg.svdvals(factor), expected,
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(4, 4), (4, 7), (7, 4), (2, 1025), (1025, 2)])
def test_polar_suppresses_numerical_null_update_without_changing_resolved_direction(dtype, shape):
    # Explicit diagonal fixture: the weak direction is below the numerical
    # rank floor, while a well-resolved small direction remains normalized.
    gradient = torch.zeros(shape, dtype=dtype)
    computation_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    floor = max(shape) * torch.finfo(computation_dtype).eps
    gradient[0, 0], gradient[1, 1] = 1., floor / 4
    expected = torch.zeros_like(gradient)
    expected[0, 0] = 1.
    before_rng = torch.cuda.get_rng_state().clone()
    actual = polar_factor(gradient)
    assert actual.device.type == "cuda" and actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(before_rng, torch.cuda.get_rng_state())
    gradient[1, 1] = floor * 4
    expected[1, 1] = 1.
    torch.testing.assert_close(polar_factor(gradient), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3, 5), (5, 3), (0, 3), (3, 0)])
def test_zero_polar_has_no_null_space_update(shape):
    gradient = torch.zeros(shape)
    assert torch.equal(polar_factor(gradient), gradient)


def test_default_dualnorm_does_not_move_along_a_rounding_scale_direction():
    weight = nn.Parameter(torch.zeros((2, 2)))
    optimizer = NormalizedOptimizer([dict(params=[weight], role="critic")],
                                    family="dualnorm", lr=.2)
    weight.grad = torch.diag(weight.new_tensor([3., 1e-8]))
    optimizer.step()
    torch.testing.assert_close(weight, weight.new_tensor([[-.2, 0.], [0., 0.]]), rtol=0, atol=0)


@pytest.mark.parametrize("family", ["ada_nsgda", "dualnorm", "dualnorm_D_only", "particle_rownorm_only"])
@pytest.mark.parametrize("corruption", ["shape", "nonfinite", "missing"])
def test_public_optimizer_checkpoint_rejects_invalid_update_history_atomically(family, corruption):
    trainer = make_trainer(family)
    trainer.step(torch.tensor([[-1., -.5], [-.2, .1], [.4, .8], [1., 1.5]]))
    before = trainer.state_dict()
    checkpoint = deepcopy(before)
    key = "momentum_buffer" if family == "dualnorm" else "exp_avg_sq"
    history = next(values for values in checkpoint["optimizers"][0]["state"].values() if key in values)
    if corruption == "shape":
        history[key] = history[key].new_zeros((1,))
    elif corruption == "nonfinite":
        history[key].reshape(-1)[0] = float("nan")
    else:
        del history[key]
    with pytest.raises(ValueError, match="optimizer"):
        trainer.load_state_dict(checkpoint)
    assert_state_equal(before, trainer.state_dict())

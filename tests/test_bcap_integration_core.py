"""Analytical and public-API software contracts for the opt-in BCAP repair.

These bounded updates check arithmetic, routing and continuation; they are not
toy quality experiments or substitutes for Forge's full-budget task gates.
"""
from copy import deepcopy

import pytest
import torch
from torch import nn

from experiments.forge.api import TRAINER_STREAM_BINDINGS
from experiments.forge.rng import NamedStreams
from particlegan import GANTrainer, Recipe, get_recipe
from particlegan.init import deterministic_orthogonal_
from particlegan.kinetic_transport import kinetic_transport_loss, kinetic_transport_local_loss
from particlegan.optim.constraint_geometry import constraint_geometry_backward, project_nonascent
from particlegan.optim.direction_blend import DirectionBlendOptimizer
from particlegan.optim.dualnorm import NormalizedOptimizer
from particlegan.training import _normalized_recipe


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _trainer(*, active=True, convolution=False, device="cpu"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        recipe = get_recipe("bcap", z_dim=2, num_particles=8, batch_size=4,
                            total_steps=4, prior_kind="mog", sigma_rel=.025,
                            standardize=False,
                            constraint_geometry_mode="direction_blend" if active else "none",
                            kinetic_transport_weight=1. if active else 0.,
                            kinetic_transport_local_weight=1. if active else 0.)
        if convolution:
            generator = nn.Sequential(nn.Unflatten(1, (2, 1, 1)),
                                      nn.ConvTranspose2d(2, 1, 2), nn.Tanh())
            critic = nn.Sequential(nn.Conv2d(1, 2, 2), nn.Flatten(), nn.Linear(2, 1))
        else:
            generator = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
            critic = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        prior = recipe.make_prior()
        for component in (generator, critic, prior):
            deterministic_orthogonal_(component, seed=0)
            component.to(device)
        streams = NamedStreams(0, device=device)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                          **{name: streams.generator(family, component=component, purpose=purpose)
                             for name, (family, component, purpose) in TRAINER_STREAM_BINDINGS.items()})


def _batch(device="cpu", convolution=False):
    if convolution:
        return torch.arange(16, dtype=torch.float32, device=device).reshape(4, 1, 2, 2) / 8 - 1
    return torch.tensor([[-1., -.5], [-.2, .1], [.4, .8], [1., 1.5]], device=device)


def test_sliced_transport_has_exact_scalar_quantile_cost_and_detached_targets():
    fake = torch.tensor([[-2.], [-1.], [0.], [2.]], dtype=torch.float64, requires_grad=True)
    real = torch.tensor([[-2.], [0.], [1.], [2.]], dtype=torch.float64, requires_grad=True)
    before = torch.get_rng_state().clone()
    value = kinetic_transport_loss(fake, real)
    assert value.item() == pytest.approx(.5 / 2.1875)
    gradient, target_gradient = torch.autograd.grad(value, (fake, real), allow_unused=True)
    assert target_gradient is None
    assert gradient[1].item() < 0 and gradient[2].item() < 0
    assert gradient[0].item() == gradient[3].item() == 0
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("objective", [kinetic_transport_loss, kinetic_transport_local_loss])
def test_transport_permutation_units_null_and_fake_pullback(objective):
    real = torch.tensor([[-1.3, .2], [-.7, -.4], [.6, .8], [1.1, -.5],
                         [.3, .15], [1.7, .1]], dtype=torch.float64, requires_grad=True)
    fake = (real.detach() + torch.tensor([.21, -.13], dtype=real.dtype)).requires_grad_()
    before = torch.get_rng_state().clone()
    value = objective(fake, real)
    assert value > 0
    assert objective(fake.flip(0), real.roll(2, 0)).item() == pytest.approx(value.item())
    assert objective(7 * fake + 11, 7 * real + 11).item() == pytest.approx(value.item())
    exact = real.detach().clone().requires_grad_()
    null = objective(exact, real)
    assert null.item() == 0
    assert torch.equal(torch.autograd.grad(null, exact)[0], torch.zeros_like(exact))
    assert torch.autograd.gradcheck(lambda x: objective(x, real.detach()), (fake,))
    assert torch.equal(before, torch.get_rng_state())


def test_local_transport_expands_contracted_cloud_and_handles_duplicate_anchors():
    real = torch.tensor([[-1.], [-.6], [.6], [1.]], dtype=torch.float64)
    fake = torch.tensor([[-.06], [-.02], [.02], [.06]], dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad(kinetic_transport_local_loss(fake, real), fake)
    assert bool((gradient * fake < 0).all())
    duplicate = torch.zeros(6, 2, dtype=torch.float64, requires_grad=True)
    null = kinetic_transport_local_loss(duplicate, duplicate.detach())
    assert null.item() == 0
    assert bool(torch.isfinite(torch.autograd.grad(null, duplicate)[0]).all())


def test_geometry_two_constraints_and_direction_blend_joint_sampled_ownership_resume():
    displacement = torch.tensor([2., 1.], dtype=torch.float64)
    normals = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
    torch.testing.assert_close(project_nonascent(displacement, normals), displacement.new_tensor([0., 1.]))

    def build():
        vector = nn.Parameter(torch.tensor([1., .5], dtype=torch.float64))
        table = nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
        opt = DirectionBlendOptimizer([dict(params=[vector], role="generator"),
                                       dict(params=[table], role="prior", lr=.03)],
                                      lr=.012, smoothing=.001)
        return vector, table, opt

    vector, table, optimizer = build()
    rows = torch.tensor([0, 2, 2])
    optimizer.set_sampled_rows(table, rows)
    protected = vector[0] + table[rows].sum()
    total = -protected + vector[1]
    def forbidden():
        raise AssertionError("direction blend must not evaluate finite losses")
    constraint_geometry_backward(total, optimizer, (protected,), protected_evaluator=forbidden)
    checkpoint = deepcopy(optimizer.state_dict())
    restored_vector, restored_table, restored = build()
    restored_vector.grad = vector.grad.clone()
    restored_table.grad = table.grad.clone()
    restored.load_state_dict(checkpoint)
    before = torch.get_rng_state().clone()
    optimizer.step()
    restored.step()
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(vector, restored_vector) and torch.equal(table, restored_table)
    assert torch.equal(table[1], torch.zeros_like(table[1]))
    assert vector[0] + table[rows].sum() <= 1.
    assert optimizer.direction_blend_stats["conflict_steps"] == 1
    assert optimizer.direction_blend_stats["blended_steps"] == 1
    _equal(optimizer.state_dict(), restored.state_dict())
    with pytest.raises(ValueError, match="protected-loss"):
        optimizer.step()


def test_direction_opposition_stalls_without_second_optimizer_clock():
    parameter = nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    optimizer = DirectionBlendOptimizer([parameter], lr=.1, smoothing=.001)
    constraint_geometry_backward(-parameter.sum(), optimizer, (parameter.sum(), -parameter.sum()))
    optimizer.step()
    assert parameter.item() == 1.
    assert optimizer.state[parameter]["step"] == 1
    assert optimizer.direction_blend_stats["pareto_stalls"] == 1
    assert optimizer.constraint_geometry_stats["steps"] == 1


def test_joint_direction_masks_each_sampled_prior_independently():
    left = nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
    right = nn.Parameter(torch.zeros(4, 2, dtype=torch.float64))
    vector = nn.Parameter(torch.tensor([1., .5], dtype=torch.float64))
    optimizer = DirectionBlendOptimizer([dict(params=[left], role="prior"),
                                        dict(params=[right], role="prior"),
                                        dict(params=[vector], role="generator")],
                                       lr=.012, smoothing=.001)
    optimizer.set_sampled_rows(left, torch.tensor([0, 2]))
    optimizer.set_sampled_rows(right, torch.tensor([1, 3]))
    protected = vector[0] + left[[0, 2]].sum() + right[[1, 3]].sum()
    constraint_geometry_backward(-protected + vector[1], optimizer, (protected,))
    optimizer.step()
    assert torch.equal(left[1], torch.zeros_like(left[1]))
    assert torch.equal(right[[0, 2]], torch.zeros_like(right[[0, 2]]))
    assert vector[0] + left[[0, 2]].sum() + right[[1, 3]].sum() <= 1.
    assert optimizer.sampled_rows_for(left) is optimizer.sampled_rows_for(right) is None


def test_strict_finite_variant_preserves_declared_backtracking_and_one_clock():
    from particlegan.optim.strict_progress import StrictProgressOptimizer
    parameter = nn.Parameter(torch.tensor([.1], dtype=torch.float64))
    optimizer = StrictProgressOptimizer([parameter], lr=1., smoothing=.001)
    protected = parameter.square().sum()
    before = protected.detach().clone()
    constraint_geometry_backward(-parameter.sum(), optimizer, (protected,),
                                 protected_evaluator=lambda: (parameter.square().sum(),))
    optimizer.step()
    assert parameter.square().sum() < before
    assert optimizer.strict_progress_stats["accepted_steps"] == 1
    assert optimizer.strict_progress_stats["backtracks"] == 2
    assert optimizer.strict_progress_stats["min_accepted_scale"] == .25
    assert optimizer.state[parameter]["step"] == 1


def test_nonconflicting_direction_keeps_actual_rounded_base_update_bitwise():
    original = torch.tensor([.1234567], dtype=torch.float32)
    plain, guarded = (nn.Parameter(original.clone()) for _ in range(2))
    options = dict(lr=.12345676, smoothing=.001)
    baseline = NormalizedOptimizer([plain], **options)
    candidate = DirectionBlendOptimizer([guarded], **options)
    plain.sum().backward()
    baseline.step()
    loss = guarded.sum()
    constraint_geometry_backward(loss, candidate, (loss,))
    candidate.step()
    assert not torch.equal(original + (plain.detach() - original), plain.detach())
    assert torch.equal(plain, guarded)
    assert candidate.constraint_geometry_stats["projected_steps"] == 0
    saved = candidate.state_dict()
    saved.pop("constraint_geometry")
    saved.pop("direction_blend")
    _equal(saved, baseline.state_dict())


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("convolution", [False, True])
def test_active_public_trainer_transport_joint_projection_and_full_checkpoint_resume(device, convolution):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = _trainer(device=device, convolution=convolution)
    assert type(trainer.opt_g) is DirectionBlendOptimizer
    assert type(trainer.opt_d) is NormalizedOptimizer
    if convolution:
        assert any("dualnorm_convolution" in group for group in trainer.opt_g.param_groups)
        assert any("dualnorm_convolution" in group for group in trainer.opt_d.param_groups)
    batch = _batch(device, convolution)
    result = trainer.step(batch)
    assert result["kinetic_transport"] > 0 and result["kinetic_transport_local"] > 0
    torch.testing.assert_close(result["loss_g"], result["loss_gan"] + result["kinetic_transport"]
                               + result["kinetic_transport_local"])
    assert trainer.opt_g.constraint_geometry_stats["steps"] == 1
    saved = trainer.state_dict()
    trainer.step(batch)
    expected = trainer.state_dict()
    restored = _trainer(device=device, convolution=convolution)
    restored.load_state_dict(saved)
    restored.step(batch)
    _equal(expected, restored.state_dict())


def test_active_signal_protects_only_original_adversarial_loss_and_preserves_d_and_streams(monkeypatch):
    from particlegan import training
    control, candidate = _trainer(active=False), _trainer(active=True)
    original = training.constraint_geometry_backward
    observed = []
    def capture(total, optimizer, protected, **kwargs):
        if isinstance(optimizer, DirectionBlendOptimizer):
            observed.append((total.detach(), tuple(x.detach() for x in protected)))
        return original(total, optimizer, protected, **kwargs)
    monkeypatch.setattr(training, "constraint_geometry_backward", capture)
    batch = _batch()
    control.step(batch)
    result = candidate.step(batch)
    assert len(observed) == 1 and len(observed[0][1]) == 1
    assert torch.equal(observed[0][1][0], result["loss_gan"])
    assert observed[0][0] > result["loss_gan"]
    _equal(control.D.state_dict(), candidate.D.state_dict())
    _equal(control.state_dict()["streams"], candidate.state_dict()["streams"])
    assert any(not torch.equal(a, b) for a, b in zip(control.G.parameters(), candidate.G.parameters()))
    assert not torch.equal(control.prior.z, candidate.prior.z)


def test_inactive_public_trainer_calls_no_auxiliary_objective_or_protected_hook(monkeypatch):
    trainer = _trainer(active=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("inactive mechanism was called")
    monkeypatch.setattr(Recipe, "kinetic_transport_loss", forbidden)
    monkeypatch.setattr(Recipe, "kinetic_transport_local_loss", forbidden)
    assert not hasattr(trainer.opt_g, "bind_protected_losses")
    result = trainer.step(_batch())
    assert "kinetic_transport" not in result and "kinetic_transport_local" not in result
    assert torch.equal(result["loss_g"], result["loss_gan"])
    assert not {"constraint_geometry", "direction_blend"} & trainer.opt_g.state_dict().keys()
    parameter = nn.Parameter(torch.tensor([.3, .4]))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    # Ordinary optimizers have no protected-loss contract.
    constraint_geometry_backward(parameter.square().sum(), optimizer, (None,), protected_evaluator=forbidden)
    torch.testing.assert_close(parameter.grad, parameter.detach() * 2)


def test_absent_optional_recipe_fields_resume_inactive_but_cannot_disable_active_checkpoint():
    trainer = _trainer(active=False)
    trainer.step(_batch())
    historical = trainer.state_dict()
    defaults = dict(constraint_geometry_mode="none", kinetic_transport_weight=0.,
                    kinetic_transport_local_weight=0., kinetic_transport_projections=32)
    assert not defaults.keys() & historical["recipe"].keys()
    explicit = deepcopy(historical)
    explicit["recipe"].update(defaults)
    assert _normalized_recipe(explicit["recipe"]) == _normalized_recipe(historical["recipe"])
    restored = _trainer(active=False)
    restored.load_state_dict(explicit)
    trainer.step(_batch())
    restored.step(_batch())
    _equal(trainer.state_dict(), restored.state_dict())
    active = _trainer(active=True)
    before = active.state_dict()
    with pytest.raises(ValueError, match="recipe"):
        active.load_state_dict(historical)
    _equal(before, active.state_dict())


@pytest.mark.parametrize("corruption", ["missing_mode", "negative_counter", "nonfinite_pending"])
def test_active_checkpoint_corruption_rejects_before_any_live_mutation(corruption):
    trainer = _trainer()
    trainer.step(_batch())
    before = trainer.state_dict()
    broken = deepcopy(before)
    optimizer = broken["optimizers"][0]
    if corruption == "missing_mode":
        optimizer.pop("direction_blend")
    elif corruption == "negative_counter":
        optimizer["direction_blend"]["stats"]["conflict_steps"] = -1
    else:
        size = sum(p.numel() for group in trainer.opt_g.param_groups for p in group["params"])
        optimizer["constraint_geometry"]["pending"] = torch.full((1, size), float("nan"))
    with pytest.raises(ValueError, match="optimizer"):
        trainer.load_state_dict(broken)
    _equal(before, trainer.state_dict())


@pytest.mark.parametrize("options", [dict(kinetic_transport_weight=True),
                                    dict(kinetic_transport_local_weight=float("nan")),
                                    dict(kinetic_transport_projections=0),
                                    dict(constraint_geometry_mode="direction_blend", optimizer_momentum=.5)])
def test_invalid_or_incompatible_opt_in_mechanisms_fail_before_training(options):
    with pytest.raises(ValueError):
        get_recipe("bcap", **options)

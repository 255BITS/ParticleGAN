"""Finite-motion, thermal restoration and exact continuation software checks."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from particlegan.bispf import BISPFStepRecord, SBNPPriorRegularizer, _LAST_THERMAL_SCALE, _attach_critic_input_hook
from particlegan.optim.dualnorm import NormalizedOptimizer
from particlegan.settling_repair import thermal_optimizer_step, unscaled_generator_gradients


def _optimizer():
    matrix = torch.nn.Parameter(torch.tensor([[0.3, -0.4], [0.1, 0.2]], dtype=torch.float64))
    bias = torch.nn.Parameter(torch.tensor([0.2, -0.1], dtype=torch.float64))
    prior = torch.nn.Parameter(torch.arange(8, dtype=torch.float64).reshape(4, 2) / 10)
    optimizer = NormalizedOptimizer([
        {"params": [matrix, bias], "lr": 0.012, "role": "generator"},
        {"params": [prior], "lr": 0.03, "role": "prior"},
    ], family="dualnorm", smoothing=0.001)
    return optimizer, (matrix, bias, prior)


def _gradients(optimizer, parameters):
    matrix, bias, prior = parameters
    matrix.grad = torch.tensor([[2.0, 0.5], [-0.3, 1.4]], dtype=matrix.dtype)
    bias.grad = torch.tensor([3.0, 4.0], dtype=bias.dtype)
    prior.grad = torch.ones_like(prior)
    optimizer.set_sampled_rows(prior, torch.tensor([0, 2, 2]))


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def test_cooling_scales_realized_matrix_bias_and_sampled_prior_moves():
    control, control_params = _optimizer()
    cooled, cooled_params = _optimizer()
    originals = [p.detach().clone() for p in cooled_params]
    _gradients(control, control_params)
    _gradients(cooled, cooled_params)
    gradients = [p.grad.clone() for p in cooled_params]
    control.step()
    stats = thermal_optimizer_step(cooled, 0.125, collect_stats=True)
    for baseline, actual, original, gradient in zip(control_params, cooled_params, originals, gradients):
        torch.testing.assert_close(actual - original, 0.125 * (baseline - original), rtol=1e-12, atol=1e-16)
        assert torch.equal(actual.grad, gradient)
    assert torch.equal(cooled_params[2][[1, 3]], originals[2][[1, 3]])
    assert cooled.sampled_rows_for(cooled_params[2]) is None
    _assert_tree_equal(cooled.state_dict(), control.state_dict())
    assert stats["cooled"] is True
    assert stats["roles"]["generator"]["applied_to_base_l2"] == pytest.approx(0.125)
    assert stats["roles"]["prior"]["applied_to_base_l2"] == pytest.approx(0.125)


def test_gradient_cooling_alone_does_not_bound_a_normalized_move():
    def move(gradient_scale, displacement_scale):
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        optimizer = NormalizedOptimizer([{"params": [parameter], "role": "generator", "lr": 0.012}],
                                       family="dualnorm", smoothing=0.001)
        parameter.grad = torch.tensor([3.0, 4.0], dtype=parameter.dtype) * gradient_scale
        thermal_optimizer_step(optimizer, displacement_scale)
        return float(parameter.detach().norm())
    full = move(1.0, 1.0)
    assert move(0.1, 1.0) / full > 0.999
    assert move(1.0, 0.1) / full == pytest.approx(0.1)


def test_untouched_prior_rows_retain_signed_zero_bits():
    optimizer, parameters = _optimizer()
    with torch.no_grad():
        parameters[2][1].fill_(-0.0)
    original_bits = parameters[2].detach().view(torch.int64).clone()
    _gradients(optimizer, parameters)
    thermal_optimizer_step(optimizer, 0.2)
    assert torch.equal(parameters[2].detach().view(torch.int64)[[1, 3]], original_bits[[1, 3]])


@pytest.mark.parametrize("collect_stats", [False, True])
def test_uncooled_step_is_bit_exact_including_optimizer_state(collect_stats):
    control, a = _optimizer()
    candidate, b = _optimizer()
    _gradients(control, a)
    _gradients(candidate, b)
    control.step()
    thermal_optimizer_step(candidate, 1.0, collect_stats=collect_stats)
    for left, right in zip(a, b):
        assert torch.equal(left, right)
    _assert_tree_equal(control.state_dict(), candidate.state_dict())


def test_unscaled_context_restores_hook_and_cache_on_nested_exception():
    device = torch.device("cpu")
    record = BISPFStepRecord()
    critic = torch.nn.Linear(2, 1, bias=False)
    _attach_critic_input_hook(critic, record)
    with torch.no_grad():
        critic.weight.fill_(2)
    record._thermal_scale = 0.2
    _LAST_THERMAL_SCALE[device] = 0.3
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="probe"):
        with unscaled_generator_gradients(record, device) as scale:
            assert scale == 0.2
            assert record._thermal_scale == _LAST_THERMAL_SCALE[device] == 1.0
            x = torch.zeros(2, 2, requires_grad=True)
            critic(x).sum().backward()
            assert torch.equal(x.grad, torch.full_like(x, 2))
            with unscaled_generator_gradients(record, device) as nested:
                assert nested == 1.0
            assert record._thermal_scale == _LAST_THERMAL_SCALE[device] == 1.0
            raise RuntimeError("probe")
    assert record._thermal_scale == 0.2
    assert _LAST_THERMAL_SCALE.pop(device) == 0.3
    assert torch.equal(torch.get_rng_state(), rng)
    x = torch.zeros(2, 2, requires_grad=True)
    critic(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 0.4))


def test_unscaled_context_restores_absent_cache_entry():
    device = torch.device("cpu")
    previous = _LAST_THERMAL_SCALE.pop(device, None)
    try:
        record = SimpleNamespace(_thermal_scale=0.25)
        with unscaled_generator_gradients(record, device) as scale:
            assert scale == 0.25
            assert _LAST_THERMAL_SCALE[device] == 1.0
        assert device not in _LAST_THERMAL_SCALE
        assert record._thermal_scale == 0.25
    finally:
        if previous is not None:
            _LAST_THERMAL_SCALE[device] = previous


def test_context_unscales_sbnp_objective_gradient_without_changing_its_value():
    device = torch.device("cpu")
    previous = _LAST_THERMAL_SCALE.pop(device, None)
    rows = torch.tensor([[0.1, 0.2], [0.6, -0.3], [-0.4, 0.7], [0.3, 0.9],
                         [0.4, -0.2], [-0.3, 0.1], [0.2, 0.5], [-0.2, -0.4]],
                        dtype=torch.float64, requires_grad=True)
    regularizer = SBNPPriorRegularizer(k_nn=2, repel_weight=0.0)
    try:
        _LAST_THERMAL_SCALE[device] = 0.2
        cooled = regularizer(rows)
        cooled_gradient, = torch.autograd.grad(cooled, rows)
        with unscaled_generator_gradients(SimpleNamespace(_thermal_scale=0.2), device):
            unscaled = regularizer(rows)
            unscaled_gradient, = torch.autograd.grad(unscaled, rows)
        assert torch.equal(cooled.detach(), unscaled.detach())
        # SBNP also has local shape terms. Complete unscaling must restore its
        # actual objective derivative, not blindly multiply the whole gradient.
        _LAST_THERMAL_SCALE[device] = 1.0
        reference_gradient, = torch.autograd.grad(regularizer(rows), rows)
        assert torch.equal(unscaled_gradient, reference_gradient)
        assert not torch.equal(cooled_gradient, unscaled_gradient)
    finally:
        _LAST_THERMAL_SCALE.pop(device, None)
        if previous is not None:
            _LAST_THERMAL_SCALE[device] = previous


def test_checkpointed_observer_and_optimizer_suffice_for_exact_continuation():
    optimizer, parameters = _optimizer()
    record = BISPFStepRecord()
    record._thermal_scale = 0.2
    _gradients(optimizer, parameters)
    thermal_optimizer_step(optimizer, record._thermal_scale)
    state = deepcopy(optimizer.state_dict())
    observer = deepcopy(record.state_dict())
    tensors = [p.detach().clone() for p in parameters]

    restored, restored_parameters = _optimizer()
    restored.load_state_dict(state)
    restored_record = BISPFStepRecord()
    restored_record.load_state_dict(observer)
    with torch.no_grad():
        for parameter, value in zip(restored_parameters, tensors):
            parameter.copy_(value)
    for opt, params, rec in ((optimizer, parameters, record),
                            (restored, restored_parameters, restored_record)):
        _gradients(opt, params)
        with unscaled_generator_gradients(rec, "cpu") as scale:
            thermal_optimizer_step(opt, scale)
    for left, right in zip(parameters, restored_parameters):
        assert torch.equal(left, right)
    _assert_tree_equal(optimizer.state_dict(), restored.state_dict())
    _assert_tree_equal(record.state_dict(), restored_record.state_dict())


@pytest.mark.parametrize("scale", [-0.1, 1.1, float("nan"), float("inf"), True, None])
def test_invalid_scale_rejects_before_any_optimizer_mutation(scale):
    optimizer, parameters = _optimizer()
    _gradients(optimizer, parameters)
    state = deepcopy(optimizer.state_dict())
    originals = [p.detach().clone() for p in parameters]
    with pytest.raises(ValueError, match="scale"):
        thermal_optimizer_step(optimizer, scale)
    _assert_tree_equal(state, optimizer.state_dict())
    for actual, original in zip(parameters, originals):
        assert torch.equal(actual, original)


def test_critic_groups_reject_before_mutation():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD([{"params": [parameter], "role": "critic"}], lr=0.1)
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(ValueError, match="generator"):
        thermal_optimizer_step(optimizer, 0.5)
    assert torch.equal(parameter, torch.ones(2))


def _public_trainer(application="displacement"):
    from particlegan import GANTrainer, get_recipe, init

    recipe = get_recipe("bispf_optimized", num_particles=64, z_dim=2,
                        prior_kind="mog", sigma_rel=0.025, standardize=False,
                        batch_size=16, total_steps=8,
                        bispf_thermal_application=application)
    # This small fixture is a software integration check, not a scientific
    # task/architecture substitute. Isolate constructor, model and every
    # sampler stream while exercising the actual public initializer and API.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        generator = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.SiLU(), torch.nn.Linear(8, 2))
        critic = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.SiLU(), torch.nn.Linear(8, 1))
        prior = recipe.make_prior()
        for index, module in enumerate((generator, critic, prior)):
            init.deterministic_orthogonal_(module, seed=index)
        streams = {name: torch.Generator().manual_seed(index + 100)
                   for index, name in enumerate(("latent_generator", "penalty_generator",
                       "noise_generator", "eval_generator", "input_noise_generator",
                       "prior_noise_generator", "model_generator"))}
        trainer = GANTrainer(recipe, generator, critic, prior=prior, seed=0, **streams)
    return trainer


def _force_software_thermal_fixture(monkeypatch, trainer, scale):
    # Exercise the real penalty/controller path and its normal state writes;
    # freeze only the cooler query to make finite movement assertions exact.
    monkeypatch.setattr(trainer.penalty.regularizer.cooler, "scale_for",
                        lambda *args, **kwargs: scale)


def _public_batch():
    return (torch.arange(32, dtype=torch.float32).reshape(16, 2) - 16) / 16


def test_public_trainer_uncooled_candidate_matches_existing_trainer_exactly(monkeypatch):
    baseline = _public_trainer("gradient")
    candidate = _public_trainer()
    for trainer in (baseline, candidate):
        _force_software_thermal_fixture(monkeypatch, trainer, 1.0)
    global_rng = torch.get_rng_state().clone()
    for _ in range(3):
        old = baseline.step(_public_batch(), collect_stats=True)
        new = candidate.step(_public_batch(), collect_stats=True)
        for field in ("loss_d", "loss_g", "loss_gan", "prior_regularization", "penalty"):
            assert torch.equal(old[field], new[field])
        assert "thermal_displacement" not in old
        assert new["thermal_displacement"]["thermal_scale"] == 1.0
        a, b = baseline.state_dict(), candidate.state_dict()
        a.pop("recipe")
        b.pop("recipe")
        _assert_tree_equal(a, b)
    assert torch.equal(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("scale", [0.0, 0.2])
def test_public_trainer_realized_move_and_cache_restoration(monkeypatch, scale):
    trainer = _public_trainer()
    _force_software_thermal_fixture(monkeypatch, trainer, scale)
    originals = [p.detach().clone() for group in trainer.opt_g.param_groups for p in group["params"]]
    stream_before = {name: getattr(trainer, name).get_state().clone() for name in trainer._STREAMS}
    output = trainer.step(_public_batch(), collect_stats=True)
    stats = output["thermal_displacement"]
    assert stats["thermal_scale"] == scale
    assert set(stats["roles"]) == {"generator", "prior"}
    for role in stats["roles"].values():
        assert role["base_squared_l2"] > 0
        assert role["gradient_squared_l2"] > 0
        assert role["applied_to_base_l2"] == pytest.approx(scale, abs=3e-6)
    if scale == 0:
        parameters = [p for group in trainer.opt_g.param_groups for p in group["params"]]
        for current, original in zip(parameters, originals):
            assert torch.equal(current, original)
    assert trainer.completed_steps == 1
    assert trainer.penalty.regularizer.record._thermal_scale == scale
    assert _LAST_THERMAL_SCALE[trainer.device] == scale
    assert all(parameter.requires_grad for parameter in trainer.D.parameters())
    assert all(state["step"] == 1 for state in trainer.opt_g.state.values())
    assert torch.equal(trainer.eval_generator.get_state(), stream_before["eval_generator"])
    for name in ("latent_generator", "prior_noise_generator"):
        assert not torch.equal(getattr(trainer, name).get_state(), stream_before[name])
    assert set(trainer.state_dict()["streams"]) == set(trainer._STREAMS)


def test_public_trainer_complete_checkpoint_resume_is_bit_exact(monkeypatch):
    trainer = _public_trainer()
    _force_software_thermal_fixture(monkeypatch, trainer, 0.2)
    for _ in range(2):
        trainer.step(_public_batch(), collect_stats=True)
    checkpoint = deepcopy(trainer.state_dict())
    assert checkpoint["recipe"]["bispf_thermal_application"] == "displacement"
    first_output = trainer.step(_public_batch(), collect_stats=True)
    first_samples = trainer.sample(32)
    first_state = trainer.state_dict()

    resumed = _public_trainer()
    _force_software_thermal_fixture(monkeypatch, resumed, 0.2)
    resumed.load_state_dict(checkpoint)
    second_output = resumed.step(_public_batch(), collect_stats=True)
    second_samples = resumed.sample(32)
    _assert_tree_equal(first_output, second_output)
    assert torch.equal(first_samples, second_samples)
    _assert_tree_equal(first_state, resumed.state_dict())
    assert _LAST_THERMAL_SCALE[resumed.device] == 0.2


def test_public_trainer_generator_exception_restores_thermal_readers(monkeypatch):
    trainer = _public_trainer()
    _force_software_thermal_fixture(monkeypatch, trainer, 0.2)

    def failed_generator_batch():
        assert trainer.penalty.regularizer.record._thermal_scale == 1.0
        assert _LAST_THERMAL_SCALE[trainer.device] == 1.0
        raise RuntimeError("generator data probe")

    with pytest.raises(RuntimeError, match="generator data probe"):
        trainer.step(_public_batch(), generator_real=failed_generator_batch)
    assert trainer.penalty.regularizer.record._thermal_scale == 0.2
    assert _LAST_THERMAL_SCALE[trainer.device] == 0.2
    assert all(parameter.requires_grad for parameter in trainer.D.parameters())

"""Public literal initialization: fixed tiny fixtures, no experiment training."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import BatchDistanceDiscriminator, MoGParticlePrior, ParticlePrior, init


def stream(seed=17):
    return torch.Generator(device="cpu").manual_seed(seed)


def snapshot(module, generators):
    return ({name: value.detach().clone() for name, value in
             [*module.named_parameters(), *module.named_buffers()]},
            {name: g.get_state().clone() for name, g in generators.items() if isinstance(g, torch.Generator)},
            torch.get_rng_state().clone())


def unchanged(module, generators, before):
    tensors, states, global_state = before
    assert all(torch.equal(tensors[name], value) for name, value in
               [*module.named_parameters(), *module.named_buffers()])
    assert all(torch.equal(states[name], generators[name].get_state()) for name in states)
    assert torch.equal(global_state, torch.get_rng_state())


def test_xavier_matches_torch_and_advances_only_named_weight_streams():
    model = nn.Sequential(nn.Linear(3, 5), nn.Softplus(), nn.Linear(5, 2)).double()
    model.register_buffer("frequencies", torch.arange(3.))
    expected = deepcopy(model)
    generators = {name: stream(i + 11) for i, (name, _) in enumerate(model.named_parameters()) if name.endswith("weight")}
    reference = {name: torch.Generator().set_state(g.get_state()) for name, g in generators.items()}
    global_state = torch.get_rng_state().clone()
    for name, parameter in expected.named_parameters():
        if name.endswith("weight"):
            nn.init.xavier_uniform_(parameter, gain=1.2, generator=reference[name])
        else:
            nn.init.zeros_(parameter)
    assert init.initialize_(model, method="xavier_uniform_zero_bias_v1",
                            parameter_generators=generators, gain=1.2) is model
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), expected.parameters()))
    assert torch.equal(model.frequencies, torch.arange(3.))
    assert all(torch.equal(g.get_state(), reference[name].get_state()) for name, g in generators.items())
    assert torch.equal(global_state, torch.get_rng_state())


def test_identity_and_frozen_values_are_preserved_without_generators():
    model = nn.Linear(3, 3)
    assert init.initialize_(model, method="identity_linear_v1") is model
    assert torch.equal(model.weight, torch.eye(3)) and torch.count_nonzero(model.bias) == 0
    model.bias.detach().fill_(2)
    model.bias.requires_grad_(False)
    init.initialize_(model, method="identity_linear_v1")
    assert torch.equal(model.bias, torch.full((3,), 2.))


def test_literal_prior_uniform_preserves_explicit_mog_width_and_registry():
    prior = MoGParticlePrior(16, 2, sigma=.037, standardize=False, generator=stream())
    prior.z.detach().zero_()  # Explicit sampling must not mistake this for KEEP.
    before = {name: value.clone() for name, value in prior.named_buffers()}
    generator, reference = stream(), stream()
    expected = torch.empty_like(prior.z)
    nn.init.uniform_(expected, -5., 5., generator=reference)
    spec = init.declarations(prior)["z"]
    init.initialize_(prior, method="sample_distributions_v1", distributions={"z": init.Uniform(-5, 5)},
                     parameter_generators={"z": generator})
    assert torch.equal(prior.z, expected) and prior.z.requires_grad
    assert torch.equal(generator.get_state(), reference.get_state())
    assert all(torch.equal(before[name], value) for name, value in prior.named_buffers())
    assert init.declarations(prior)["z"] == spec and isinstance(spec, init.R2Normal)
    assert prior._noise_enabled and not prior.standardize


def test_literal_normal_and_keep_use_existing_registry(monkeypatch):
    class Custom(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.zeros(4, 3))
            self.fixed = nn.Parameter(torch.ones(2))
    monkeypatch.setattr(init, "_REGISTRY", dict(init._REGISTRY))
    init.register(Custom, {"value": init.Normal(2., .25), "fixed": init.KEEP})
    model = Custom()
    expected = torch.empty_like(model.value)
    nn.init.normal_(expected, 2., .25, generator=stream())
    init.initialize_(model, method="sample_distributions_v1", parameter_generators={"value": stream()})
    assert torch.equal(model.value, expected) and torch.equal(model.fixed, torch.ones(2))


@pytest.mark.parametrize("change", ["late_distribution", "missing_generator", "extra_generator", "wrong_generator",
                                  "shared_generator", "degenerate_uniform", "degenerate_normal", "nonfinite", "overflow"])
def test_all_input_errors_leave_tensors_and_rng_unchanged(change):
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    distributions = {name: init.Normal() for name, _ in model.named_parameters()}
    generators = {name: stream(i + 1) for i, name in enumerate(distributions)}
    last = "1.bias"
    if change == "late_distribution": distributions[last] = init.R2Normal()
    if change == "missing_generator": generators.pop(last)
    if change == "extra_generator": generators["extra"] = stream()
    if change == "wrong_generator": generators[last] = object()
    if change == "shared_generator": generators[last] = generators["0.weight"]
    if change == "degenerate_uniform": distributions[last] = init.Uniform(1., 1.)
    if change == "degenerate_normal": distributions[last] = init.Normal(1., 0.)
    if change == "nonfinite": distributions[last] = init.Normal(float("nan"), 1.)
    if change == "overflow": distributions[last] = init.Uniform(-1e100, 1e100)
    before = snapshot(model, generators)
    with pytest.raises((TypeError, ValueError)):
        init.initialize_(model, method="sample_distributions_v1", distributions=distributions,
                         parameter_generators=generators)
    unchanged(model, generators, before)


def test_non_cpu_generator_is_rejected_before_drawing(monkeypatch):
    # No CUDA initialization is needed to exercise the device validation branch.
    class OtherGenerator:
        device = torch.device("cuda:0")
    monkeypatch.setattr(torch, "Generator", OtherGenerator)
    model = nn.Linear(2, 2)
    original = model.weight.detach().clone()
    with pytest.raises(ValueError, match="CPU"):
        init.initialize_(model, method="xavier_uniform_zero_bias_v1", parameter_generators={"weight": OtherGenerator()})
    assert torch.equal(model.weight, original)


@pytest.mark.parametrize("kind", ["same_parameter", "storage", "buffer"])
def test_tied_names_and_shared_storage_are_rejected(kind):
    model = nn.Linear(2, 2)
    if kind == "same_parameter": model.alias = model.weight
    if kind == "storage": model.alias = nn.Parameter(model.weight.detach())
    if kind == "buffer": model.register_buffer("alias", model.weight.detach())
    generators = {"weight": stream()}
    before = snapshot(model, generators)
    with pytest.raises(ValueError, match="alias|storage"):
        init.initialize_(model, method="xavier_uniform_zero_bias_v1", parameter_generators=generators)
    unchanged(model, generators, before)


def test_linear_subclass_extra_parameter_is_not_silently_dropped():
    class Gated(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.gate = nn.Parameter(torch.ones(2))
    model, generators = nn.Sequential(nn.Linear(2, 2), Gated()), {"0.weight": stream(), "1.weight": stream(2)}
    before = snapshot(model, generators)
    with pytest.raises(ValueError, match="unsupported trainable"):
        init.initialize_(model, method="xavier_uniform_zero_bias_v1", parameter_generators=generators)
    unchanged(model, generators, before)


@pytest.mark.parametrize("effect", ["frozen", "buffer", "global_rng", "raise", "shape"])
def test_staged_finalizer_failures_cannot_mutate_caller_or_streams(monkeypatch, effect):
    class Finalized(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.bias.requires_grad_(False)
            self.register_buffer("marker", torch.ones(1))
    def finalize(model):
        if effect == "frozen": model.bias.add_(1)
        if effect == "buffer": model.marker.add_(1)
        if effect == "global_rng": torch.rand(2)
        if effect == "raise": raise ValueError("late finalizer refusal")
        if effect == "shape": model.weight = nn.Parameter(torch.zeros(3, 3))
    monkeypatch.setattr(init, "_REGISTRY", dict(init._REGISTRY))
    init.register(Finalized, finalize=finalize)
    model, generators = Finalized(), {"weight": stream()}
    before = snapshot(model, generators)
    with pytest.raises(ValueError):
        init.initialize_(model, method="xavier_uniform_zero_bias_v1", parameter_generators=generators)
    unchanged(model, generators, before)


def test_batch_feature_finalizer_runs_on_staged_trainable_weights():
    model = BatchDistanceDiscriminator(hidden_dim=4, n_hidden=1)
    generators = {name: stream(i + 1) for i, (name, _) in enumerate(model.named_parameters()) if name.endswith("weight")}
    init.initialize_(model, method="xavier_uniform_zero_bias_v1", parameter_generators=generators)
    assert torch.count_nonzero(model.head.weight[:, -len(model.scales):]) == 0


def test_calibrated_mog_and_unoverridden_r2_are_refused_without_mutation():
    prior = MoGParticlePrior(8, 2, sigma=.02, generator=stream())
    generators = {"z": stream()}
    before = snapshot(prior, generators)
    with pytest.raises(ValueError, match="R2Normal"):
        init.initialize_(prior, method="sample_distributions_v1", parameter_generators=generators)
    unchanged(prior, generators, before)
    prior.d0.fill_(1.)
    before = snapshot(prior, generators)
    with pytest.raises(ValueError, match="calibrated MoG"):
        init.initialize_(prior, method="sample_distributions_v1", distributions={"z": init.Uniform(-5, 5)},
                         parameter_generators=generators)
    unchanged(prior, generators, before)


def test_frozen_prior_is_a_noop_and_unrelated_parameters_do_not_shift_draws():
    prior = ParticlePrior(4, 2, learnable=False, generator=stream())
    saved = prior.z.clone()
    init.initialize_(prior, method="sample_distributions_v1")
    assert torch.equal(prior.z, saved)
    a = nn.ModuleDict({"shared": nn.Linear(2, 2)})
    b = nn.ModuleDict({"extra": nn.Linear(3, 4), "shared": nn.Linear(2, 2)})
    init.initialize_(a, method="xavier_uniform_zero_bias_v1", parameter_generators={"shared.weight": stream()})
    init.initialize_(b, method="xavier_uniform_zero_bias_v1",
                     parameter_generators={"extra.weight": stream(99), "shared.weight": stream()})
    assert torch.equal(a["shared"].weight, b["shared"].weight)


def test_late_linear_shape_refusal_preserves_earlier_parameters_and_streams():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    model[1].weight = nn.Parameter(torch.ones(3, 2))
    generators = {"0.weight": stream(), "1.weight": stream(19)}
    before = snapshot(model, generators)
    with pytest.raises(ValueError, match="shape"):
        init.initialize_(model, method="xavier_uniform_zero_bias_v1", parameter_generators=generators)
    unchanged(model, generators, before)


def test_sampling_strict_false_only_skips_undeclared_parameters():
    class Extra(nn.Module):
        def __init__(self):
            super().__init__()
            self.extra = nn.Parameter(torch.ones(2))
            self.layer = nn.Linear(2, 2)
    model = Extra()
    generators = {"layer.weight": stream(), "layer.bias": stream(19)}
    before = snapshot(model, generators)
    with pytest.raises(ValueError, match="undeclared"):
        init.initialize_(model, method="sample_distributions_v1", parameter_generators=generators)
    unchanged(model, generators, before)
    init.initialize_(model, method="sample_distributions_v1", parameter_generators=generators, strict=False)
    assert torch.equal(model.extra, torch.ones(2))
    assert not torch.equal(model.layer.weight, before[0]["layer.weight"])


@pytest.mark.parametrize("view", ["expanded", "transposed"])
def test_late_noncontiguous_parameter_is_rejected_before_any_commit(view):
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    value = torch.ones(1, 2).expand(2, 2) if view == "expanded" else torch.arange(4.).reshape(2, 2).t()
    model[1].weight = nn.Parameter(value)
    assert not model[1].weight.is_contiguous()
    generators = {"0.weight": stream(), "1.weight": stream(19)}
    before = snapshot(model, generators)
    with pytest.raises(ValueError, match="contiguous trainable"):
        init.initialize_(model, method="xavier_uniform_zero_bias_v1", parameter_generators=generators)
    unchanged(model, generators, before)


def test_explicit_global_generator_is_refused_without_advancing_any_stream():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    generators = {"0.weight": stream(), "1.weight": torch.default_generator}
    before = snapshot(model, generators)
    with pytest.raises(ValueError, match="global default generator"):
        init.initialize_(model, method="xavier_uniform_zero_bias_v1", parameter_generators=generators)
    unchanged(model, generators, before)

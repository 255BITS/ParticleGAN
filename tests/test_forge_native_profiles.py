"""Opt-in native construction/initialization contracts, never qualification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmarks.toy100.native_models import native_component
from experiments.forge.api import CapabilityError, FormulationContext
from experiments.forge.contracts import stable_hash
from experiments.forge.nativeprofiles import (PROFILE_SOURCE, build_native_models,
    native_host_initialization, native_profile_blockers, native_profile_source_files,
    resolve_host_initialization, resolve_native_spec, task_from_profile, validate_native_continuation)
from experiments.forge.state import require_same_formulation, state_digest
from lib.toy_models import SimpleMLPDiscriminator


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def card(problem="grid100", continuation=False):
    name = problem + "_affine_square_named_v1" + ("_14k" if continuation else "")
    return json.loads((ROOT / "configs/forge/tasks" / (name + ".json")).read_text())


def tiny_context(**kwargs):
    # A separate small software fixture; never presented as the 20k-row host.
    policies = native_host_initialization(card())["components"]
    host = {"schema_version": 1, "owner": "task", "profile_sha256": stable_hash({"fixture": "tiny_restart_v1"}),
            "components": policies}
    values = {"num_particles": 12, "z_dim": 2, "batch_size": 4, "total_steps": 4,
              "input_noise_std": 0., "output_noise_std": 0.}
    values.update(kwargs.pop("recipe_overrides", {}))
    return FormulationContext(recipe_overrides=values, host_initialization=kwargs.pop("host_initialization", host), **kwargs)


def tiny_trainer(context):
    g = context.construct(lambda: nn.Linear(2, 2), component="generator")
    d = context.construct(lambda: nn.Sequential(nn.Linear(2, 6), nn.LeakyReLU(.2), nn.Linear(6, 1)), component="discriminator")
    return context.build_trainer(g, d)


@pytest.mark.parametrize("problem", ["grid100", "rotated100", "staggered100"])
def test_profile_cards_preserve_gates_and_bind_matching_continuation(problem):
    parent, child = card(problem), card(problem, True)
    validate_native_continuation(parent, child)
    assert native_host_initialization(parent) == native_host_initialization(child)
    original = json.loads((ROOT / "configs/forge/tasks" / (problem + ".json")).read_text())
    assert parent["evaluation"] == original["evaluation"]
    assert parent["execution"]["steps"] == 7000 and child["execution"]["steps"] == 14000
    assert native_profile_source_files(parent) == [PROFILE_SOURCE]
    with pytest.raises(ValueError, match="matching profile"):
        validate_native_continuation(original, child)
    wrong = deepcopy(child)
    wrong["execution"]["continuation_of"] = "another_candidate_checkpoint"
    with pytest.raises(ValueError, match="own 7k parent"):
        validate_native_continuation(parent, wrong)


def test_exact_affine_host_construction_and_literal_named_initialization():
    task = card()
    spec = resolve_native_spec(task)
    assert spec["resources"] == {"z_dim": 2, "num_particles": 20000, "batch_size": 2048}
    global_rng = torch.get_rng_state().clone()
    context = FormulationContext(recipe_overrides={**spec["resources"], "total_steps": 7000, "reg_coeff": .7},
                                 host_initialization=native_host_initialization(task))
    before = context.streams.audit()
    g, d = build_native_models(context, spec)
    trainer = context.build_trainer(g, d)
    assert torch.equal(global_rng, torch.get_rng_state())
    assert type(g) is nn.Linear and g.weight.shape == (2, 2)
    assert sum(p.numel() for p in g.parameters()) == 6
    assert isinstance(d, SimpleMLPDiscriminator) and d.fourier == 3
    assert d.net[0].in_features == 14 and sum(p.numel() for p in d.parameters()) == 35073
    assert torch.equal(g.weight, torch.eye(2)) and torch.equal(g.bias, torch.zeros(2))
    for name, parameter in d.named_parameters():
        expected = torch.zeros_like(parameter)
        if name.endswith("weight"):
            rng = torch.Generator().manual_seed(context.streams.seed_for("init", component="discriminator", purpose=name))
            nn.init.xavier_uniform_(expected, gain=1., generator=rng)
        assert torch.equal(parameter, expected)
    expected = torch.empty_like(trainer.prior.z)
    rng = torch.Generator().manual_seed(context.streams.seed_for("init", component="prior", purpose="z"))
    nn.init.uniform_(expected, -5., 5., generator=rng)
    assert torch.equal(trainer.prior.z, expected)
    assert -5 <= trainer.prior.z.min() and trainer.prior.z.max() <= 5
    assert trainer.prior.z.requires_grad and trainer.prior.standardize is False
    assert torch.equal(trainer.prior.sigma, torch.tensor(.025))
    assert trainer.prior_mechanisms["prior"]["mixture_weights"] == "uniform"
    assert context.recipe.reg_coeff == .7 and context.recipe.total_steps == 7000
    assert context.streams.seed == 0  # historical declaration's 1234 is not imported
    after = context.streams.audit()
    bindings = context.streams.manifest()["bindings"]
    assert all(before[key] == after[key] for key in before if bindings[key]["family"] != "init")
    receipt = context.receipt()
    assert context.streams.audit() == after
    assert receipt["initialization"]["generator"]["owner"] == "task"
    assert "random_stream" not in receipt["initialization"]["generator"]["parameters"]["weight"]
    for component, model in (("generator", g), ("discriminator", d), ("prior", trainer.prior)):
        for name, value in model.named_parameters():
            row = receipt["initialization"][component]["parameters"][name]
            assert row["tensor_sha256"] == state_digest(value)
            if "random_stream" in row:
                assert row["random_stream"]["device"] == "cpu"
                assert row["random_stream"]["initial_state_sha256"] != row["random_stream"]["final_state_sha256"]


def test_shared_constructor_matches_original_classes_without_changing_rng_order():
    spec = resolve_native_spec(card())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        reference = (nn.Linear(2, 2), SimpleMLPDiscriminator(2, 128, 3, 3))
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(0)
        actual = tuple(native_component(spec[role], role=role) for role in ("generator", "discriminator"))
        assert torch.equal(expected_rng, torch.get_rng_state())
    for left, right in zip(reference, actual):
        assert state_digest(left.state_dict()) == state_digest(right.state_dict())


@pytest.mark.parametrize("prior", [
    {"kind": "particle_cloud", "sigma": 0, "standardize": False, "learnable": True},
    {"kind": "mog", "sigma": 0, "standardize": False, "learnable": True},
    {"kind": "mog", "sigma": .025, "standardize": True, "learnable": True},
    {"kind": "mog", "sigma": .025, "standardize": False, "learnable": False},
    {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True, "init_std": 1.},
])
def test_task_prior_conflicts_are_blocked_before_allocation(prior):
    value = card()
    value["execution"]["prior"] = prior
    assert native_profile_blockers(value, {})


def test_profile_source_and_fixed_fields_are_strict(tmp_path):
    candidate = {"prior": {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}}
    assert native_profile_blockers(card(), candidate) == []
    assert native_profile_blockers(card(), {**candidate, "initializer": "supplied"})
    assert native_profile_blockers(card(), {**candidate, "recipe_overrides": {"num_particles": 32}})
    for field, value in (("generator", {"kind": "linear", "in_features": 2, "out_features": 3, "bias": True}),
                         ("resources", {"z_dim": 2, "num_particles": 32, "batch_size": 2048})):
        changed = card()
        changed["execution"]["host_definition"][field] = value
        assert native_profile_blockers(changed, candidate)
    changed = card()
    changed["execution"]["host_definition"]["unused_override"] = True
    assert native_profile_blockers(changed, candidate)
    target = tmp_path / PROFILE_SOURCE
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / PROFILE_SOURCE).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source changed"):
        resolve_native_spec(card(), root=tmp_path)
    target.unlink()
    assert native_profile_blockers(card(), candidate, root=tmp_path)
    original = json.loads((ROOT / "configs/forge/tasks/grid100.json").read_text())
    assert resolve_native_spec(original) is None
    assert native_profile_source_files(original) == []
    original["execution"]["host_initialization"] = {}
    assert native_profile_blockers(original, candidate)
    stripped = card()
    del stripped["execution"]["native_profile"]
    assert native_profile_blockers(stripped, candidate)
    for field in ("resources", "initialization"):
        stripped["execution"].pop("host_definition", None)
        stripped["execution"][field] = {}
        assert native_profile_blockers(stripped, candidate)


def test_snapshot_profile_resolution_needs_no_local_forge_cards(tmp_path):
    target = tmp_path / PROFILE_SOURCE
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / PROFILE_SOURCE).read_bytes())
    assert not (tmp_path / "configs/forge/tasks").exists()
    task = card()
    assert resolve_native_spec(task, root=tmp_path) == resolve_native_spec(task)
    assert native_host_initialization(task, root=tmp_path) == native_host_initialization(task)


@pytest.mark.parametrize("change", [
    lambda p: p["components"]["generator"].update(private_callback="anything"),
    lambda p: p["components"]["prior"]["parameters"].update(missing={"kind": "uniform", "low": 0, "high": 1}),
    lambda p: p["components"]["prior"]["parameters"]["z"].update(high=float("inf")),
    lambda p: p["components"]["discriminator"].update(method="invented"),
])
def test_component_policy_rejects_unknown_or_invalid_fields_without_rng_changes(change):
    policy = native_host_initialization(card())
    change(policy)
    before = torch.get_rng_state().clone()
    with pytest.raises(CapabilityError):
        tiny_context(host_initialization=policy)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("critic_policy", ["pinned", "fallback"])
def test_cross_component_validation_precedes_live_mutation_and_stream_registration(critic_policy):
    policy = native_host_initialization(card())
    if critic_policy == "fallback":
        del policy["components"]["discriminator"]
    context = tiny_context(host_initialization=policy)
    g = context.construct(lambda: nn.Linear(2, 2), component="generator")
    class Unsupported(nn.Module):
        def __init__(self):
            super().__init__()
            self.extra = nn.Parameter(torch.ones(3))
    d = Unsupported()
    before, streams = g.weight.clone(), context.streams.audit()
    with pytest.raises((ValueError, TypeError)):
        context.build_trainer(g, d)
    assert torch.equal(g.weight, before) and context.streams.audit() == streams
    assert context.initialization == {} and context._trainer is None


def test_pinned_policies_cannot_be_bypassed_or_applied_twice():
    context = tiny_context()
    g = context.construct(lambda: nn.Linear(2, 2), component="generator")
    d = context.construct(lambda: nn.Linear(2, 1), component="discriminator")
    with pytest.raises(CapabilityError, match="initialize=False"):
        context.build_trainer(g, d, initialize=False)
    trainer = context.build_trainer(g, d)
    before, initial = context.streams.audit(), deepcopy(context.initialization)
    assert context.initialize(g, component="generator") is g
    assert context.build_prior() is trainer.prior
    assert context.streams.audit() == before and context.initialization == initial
    with pytest.raises(CapabilityError, match="another model"):
        context.initialize(nn.Linear(2, 2), component="generator")
    with pytest.raises(CapabilityError, match="conflicts"):
        tiny_context(initializer_requirements={"generator": {"method": "xavier_uniform_zero_bias_v1", "gain": 1.}})


def test_parameter_named_draws_ignore_unrelated_parameter_order():
    class Pair(nn.Module):
        def __init__(self, extra=False):
            super().__init__()
            if extra:
                self.unrelated = nn.Linear(2, 2)
            self.shared = nn.Linear(2, 3)
    a, b = tiny_context(), tiny_context()
    left = a.construct(lambda: Pair(False), component="discriminator")
    right = b.construct(lambda: Pair(True), component="discriminator")
    torch.randn(9, generator=b.streams.generator("noise", component="unrelated", purpose="scratch"))
    a.initialize(left, component="discriminator")
    b.initialize(right, component="discriminator")
    assert torch.equal(left.shared.weight, right.shared.weight)
    assert a.initialization["discriminator"]["parameters"]["shared.weight"] == b.initialization["discriminator"]["parameters"]["shared.weight"]


def test_tiny_own_state_restart_preserves_optimizers_ema_and_all_named_streams(tmp_path):
    a, b = tiny_context(), tiny_context()
    ta, tb = tiny_trainer(a), tiny_trainer(b)
    def step(context, trainer):
        data = context.streams.generator("data", component="target", purpose="training")
        trainer.step(torch.randn(4, 2, generator=data))
    for _ in range(2):
        step(a, ta)
        step(b, tb)
    path = tmp_path / "prefix.pt"
    torch.save(b.state_dict(), path)
    resumed = tiny_context()
    tc = tiny_trainer(resumed)
    resumed.load_state_dict(torch.load(path, weights_only=True))
    sigma = tc.prior.sigma.clone()
    static_init = deepcopy(resumed.initialization)
    for _ in range(2):
        step(a, ta)
        step(resumed, tc)
    assert state_digest(a.state_dict()) == state_digest(resumed.state_dict())
    assert resumed.initialization == static_init and torch.equal(tc.prior.sigma, sigma)
    altered = deepcopy(resumed.state_dict())
    altered["initialization"]["prior"]["profile_sha256"] = "0" * 64
    # Recomputing an outer digest cannot authorize a different static policy.
    state_digest(altered)
    with pytest.raises(ValueError, match="initialization"):
        resumed.load_state_dict(altered)
    with pytest.raises(ValueError, match="initialization"):
        require_same_formulation(resumed.state_dict(), altered)
    old = tiny_context(host_initialization=None)
    tiny_trainer(old)
    with pytest.raises(ValueError, match="initialization"):
        resumed.load_state_dict(old.state_dict())


def test_absent_policy_retains_schema_and_legacy_initialization_entries():
    omitted = FormulationContext(recipe_overrides={"num_particles": 12, "z_dim": 2, "batch_size": 4, "total_steps": 4})
    explicit = FormulationContext(recipe_overrides={"num_particles": 12, "z_dim": 2, "batch_size": 4, "total_steps": 4}, host_initialization=None)
    tiny_trainer(omitted)
    tiny_trainer(explicit)
    assert state_digest(omitted.state_dict()) == state_digest(explicit.state_dict())
    assert set(omitted.state_dict()) == {"schema", "api_version", "recipe", "prior", "extensions", "initializer", "initialization", "streams", "trainer"}
    assert all(set(row) == {"initializer", "parameter_seeds"} for row in omitted.initialization.values())
    assert not any(binding["purpose"] not in {"construction", "locations"} and binding["family"] == "init"
                   for binding in omitted.streams.manifest()["bindings"].values())

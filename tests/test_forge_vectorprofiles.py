"""Published vector host selection and constructor parity with zero updates."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.api import FormulationContext
from experiments.forge.vectorprofiles import (
    ARCHITECTURE_FIELDS, LEADING_SOURCE, PLAN_SOURCE, PROFILE_SOURCES,
    build_vector_models, profile_declaration, profile_source_files,
    resolve_vector_spec, task_from_profile, vector_profile_blockers,
)

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("vector_two_broad", "vector_unequal_mass", "vector_unequal_width",
         "vector_anisotropic", "vector_overlap", "vector_spiral")


def read_task(name):
    return json.loads((ROOT / "configs/forge/tasks" / f"{name}.json").read_text())


def context(task):
    spec = task["execution"]["host_definition"]
    return FormulationContext(seed=0, device="cpu", prior=task["execution"]["prior"],
                              recipe_overrides={"z_dim": spec["z_dim"], "num_particles": spec["particles"]})


def same_state(left, right):
    assert left.state_dict().keys() == right.state_dict().keys()
    for key, value in left.state_dict().items():
        other = right.state_dict()[key]
        assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other


@pytest.mark.parametrize("name", NAMES)
def test_materialization_preserves_current_prior_gates_data_resources_and_base(name):
    path = ROOT / "configs/forge/tasks" / f"{name}.json"
    original = path.read_bytes()
    raw = json.loads(original)
    before = deepcopy(raw)
    variant = task_from_profile(raw, name + "_published")
    assert variant == read_task(name + "_published")
    assert raw == before and path.read_bytes() == original
    assert profile_source_files(raw) == {}
    assert profile_source_files(variant) == PROFILE_SOURCES
    assert variant["execution"]["vector_profile"] == profile_declaration()
    raw_spec = raw["execution"]["host_definition"]
    variant_spec = variant["execution"]["host_definition"]
    assert {k: v for k, v in variant_spec.items() if k not in ARCHITECTURE_FIELDS} == {
        k: v for k, v in raw_spec.items() if k not in ARCHITECTURE_FIELDS}
    assert variant_spec["hidden"] == raw_spec["hidden"] == 64
    assert variant_spec["layers"] == raw_spec["layers"] == 2
    assert {k: v for k, v in variant.items() if k not in {"id", "execution"}} == {
        k: v for k, v in raw.items() if k not in {"id", "execution"}}
    assert {k: v for k, v in variant["execution"].items() if k not in {"vector_profile", "host_definition"}} == {
        k: v for k, v in raw["execution"].items() if k != "host_definition"}
    assert variant["execution"]["prior"] == {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}
    # Current named location initialization remains exactly the raw task's. No
    # archived finite prior or init_std=.5 is imported by the architecture profile.
    a, b = context(raw), context(variant)
    prior_a, prior_b = a.build_prior(), b.build_prior()
    same_state(prior_a, prior_b)
    assert prior_b.z.std().item() > .8
    assert float(prior_b.sigma) == pytest.approx(.025)
    assert a.streams.audit() == b.streams.audit()


@pytest.mark.parametrize("name", NAMES)
def test_raw_factory_exactly_preserves_existing_constructor_draws_and_global_rng(name):
    from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
    task = read_task(name)
    spec = resolve_vector_spec(task)
    assert spec == task["execution"]["host_definition"]
    a, b = context(task), context(task)
    before = torch.get_rng_state().clone()
    new_g, new_d = build_vector_models(a, spec)
    old_g = b.construct(lambda: SimpleMLPGenerator(b.recipe.z_dim, spec["hidden"], spec["layers"], 2), component="generator")
    old_d = b.construct(lambda: SimpleMLPDiscriminator(2, spec.get("d_hidden", spec["hidden"]),
                       spec.get("d_layers", spec["layers"]), spec["fourier"]), component="discriminator")
    same_state(new_g, old_g)
    same_state(new_d, old_d)
    assert torch.equal(before, torch.get_rng_state())
    assert a.streams.audit() == b.streams.audit()
    spec["hidden"] = 1
    assert task["execution"]["host_definition"]["hidden"] == 64


@pytest.mark.parametrize("name", NAMES)
def test_published_factory_matches_authoritative_layering_and_has_finite_forward(name):
    from benchmarks.transfer_suite.public_default_verification import declared_spec, vector_discriminator
    from lib.toy_models import SimpleMLPGenerator
    from benchmarks.gan_v3 import gan_v3_recipe
    task = read_task(name + "_published")
    spec = resolve_vector_spec(task)
    job = next(row for row in json.loads((ROOT / PLAN_SOURCE).read_text()) if row["spec"]["name"] == name)
    profile = json.loads((ROOT / LEADING_SOURCE).read_text())
    a, b = context(task), context(task)
    authoritative, card, _ = declared_spec(job, profile, gan_v3_recipe())
    assert {k: v for k, v in spec.items() if k in ARCHITECTURE_FIELDS} == {
        k: v for k, v in authoritative.items() if k in ARCHITECTURE_FIELDS}
    before = torch.get_rng_state().clone()
    g, d = build_vector_models(a, spec)
    expected_g = b.construct(lambda: SimpleMLPGenerator(b.recipe.z_dim, spec["hidden"], spec["layers"], 2), component="generator")
    expected_d = b.construct(lambda: vector_discriminator(authoritative, card), component="discriminator")
    same_state(g, expected_g)
    same_state(d, expected_d)
    assert torch.equal(before, torch.get_rng_state())
    assert a.streams.audit() == b.streams.audit()
    if name == "vector_unequal_mass":
        from particlegan import BatchDistanceDiscriminator
        assert isinstance(d, BatchDistanceDiscriminator)
    elif name in {"vector_anisotropic", "vector_overlap", "vector_unequal_width"}:
        from benchmarks.transfer_suite.shared_critic_research import SharedResearchCritic
        assert isinstance(d, SharedResearchCritic)
    else:
        from lib.toy_models import SimpleMLPDiscriminator
        assert isinstance(d, SimpleMLPDiscriminator)
    with torch.no_grad():
        generated = g(torch.zeros(8, spec["z_dim"]))
        score = d(generated)
    assert generated.shape == (8, 2) and score.shape == (8,)
    assert torch.isfinite(generated).all() and torch.isfinite(score).all()


@pytest.mark.parametrize("mutation", ["id", "revision", "scope", "prior_policy", "parity", "source_hash", "source_path",
                                     "unknown_profile_field", "hidden", "d_hidden", "fourier", "card_value",
                                     "card_field", "implementation", "unknown_host_field", "missing_profile", "host"])
def test_unknown_or_mismatching_declarations_block_before_construction(mutation):
    task = read_task("vector_unequal_mass_published")
    profile = task["execution"]["vector_profile"]
    spec = task["execution"]["host_definition"]
    if mutation == "id": profile["id"] = "made_up"
    elif mutation == "revision": profile["revision"] = 2
    elif mutation == "scope": profile["scope"] = "historical_replay"
    elif mutation == "prior_policy": profile["prior_initialization"] = "historical_std_05"
    elif mutation == "parity": profile["historical_parity"] = True
    elif mutation == "source_hash": profile["sources"][LEADING_SOURCE] = "0" * 64
    elif mutation == "source_path": profile["sources"]["different.json"] = profile["sources"].pop(PLAN_SOURCE)
    elif mutation == "unknown_profile_field": profile["fallback"] = "mlp"
    elif mutation == "hidden": spec["hidden"] = 32
    elif mutation == "d_hidden": spec["d_hidden"] = 12
    elif mutation == "fourier": spec["fourier"] = 3
    elif mutation == "card_value": spec["research_discriminator"]["softplus_beta"] = 1.
    elif mutation == "card_field": spec["research_discriminator"]["spectral_normalization"] = True
    elif mutation == "implementation": spec["research_discriminator"]["implementation"] = "invented"
    elif mutation == "unknown_host_field": spec["generator_activation"] = "tanh"
    elif mutation == "missing_profile": del task["execution"]["vector_profile"]
    else: task["execution"]["host"] = "unknown"
    assert vector_profile_blockers(task)
    with pytest.raises(ValueError):
        resolve_vector_spec(task)


@pytest.mark.parametrize("source", [PLAN_SOURCE, LEADING_SOURCE])
def test_both_source_pins_are_required_at_execution_root_without_local_fallback(tmp_path, source):
    task = read_task("vector_unequal_mass_published")
    for name in PROFILE_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    assert resolve_vector_spec(task, root=tmp_path) == task["execution"]["host_definition"]
    path = tmp_path / source
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source changed"):
        resolve_vector_spec(task, root=tmp_path)
    path.unlink()
    assert vector_profile_blockers(task, root=tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_vector_spec(task, root=tmp_path)


def test_factory_rejects_ignored_architecture_options_and_wrong_context_shape():
    task = read_task("vector_two_broad")
    spec = resolve_vector_spec(task)
    spec["activation"] = "invented"
    with pytest.raises(ValueError, match="unsupported fields"):
        build_vector_models(context(task), spec)
    spec = resolve_vector_spec(task)
    spec["z_dim"] = 7
    with pytest.raises(ValueError, match="latent dimension"):
        build_vector_models(context(task), spec)
    with pytest.raises(ValueError, match="distinct task id"):
        task_from_profile(task, task["id"])


def test_runtime_uses_public_batch_distance_critic_before_first_update(tmp_path, monkeypatch):
    from experiments.forge.adapters import run_task
    from particlegan import BatchDistanceDiscriminator, GANTrainer

    class FirstUpdateReached(Exception):
        pass

    task = read_task("vector_unequal_mass_published")
    captured = {}

    def stop(trainer, real, **kwargs):
        captured.update(public_critic=isinstance(trainer.D, BatchDistanceDiscriminator),
                        shape=list(real.shape), sigma=float(trainer.prior.sigma),
                        completed_steps=trainer.completed_steps)
        raise FirstUpdateReached

    monkeypatch.setattr(GANTrainer, "step", stop)
    request = {"candidate": {"prior": task["execution"]["prior"]},
               "protocol": {"seed": 0}, "tasks": {task["id"]: task}}
    with pytest.raises(FirstUpdateReached):
        run_task(request, {"task_id": task["id"]}, tmp_path, "cpu")
    assert captured == {"public_critic": True, "shape": [task["execution"]["host_definition"]["batch"], 2],
                        "sigma": pytest.approx(.025), "completed_steps": 0}


def test_adapter_preflight_checks_execution_root_and_unknown_options(tmp_path):
    from experiments.forge.adapters import adapter_preflight
    task = read_task("vector_unequal_mass_published")
    assert not adapter_preflight(task, {}, root=ROOT)
    assert adapter_preflight(task, {}, root=tmp_path)
    del task["execution"]["vector_profile"]
    assert "requires an explicit" in adapter_preflight(task, {}, root=ROOT)[0]

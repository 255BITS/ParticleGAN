"""Fresh-object exact continuation of explicitly supported manual BiGAN recipes."""
import pytest
import torch

from benchmarks.toy_audit.api_contract import UnsupportedRecipeOverrides
from benchmarks.toy_audit.api_ring16 import ALICESpiralBiGANFixture, Ring16BiGANFixture
from experiments.forge.state import state_digest
from particlegan import get_recipe
from tests.bigan_software_cohort import build_fixture


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("fixture_cls", [Ring16BiGANFixture, ALICESpiralBiGANFixture])
@pytest.mark.parametrize("family", ["k3p", "ka2", "bispf", "bcap"])
def test_fresh_object_exact_continuation_crosses_penalty_anchor_and_a2(fixture_cls, family):
    kwargs = {"fixture_cls": fixture_cls, "recipe_name": family, "max_steps": 8,
                  "d_guard_min_steps": 2, "reg_every": 2, "lr_anneal_start": 0.}
    first = build_fixture(**kwargs)
    for _ in range(4):
        first.step()
    # Evaluation mode and every consumed evaluation draw are part of the state.
    first.observe(n=32)
    first.sample_target(7, generator=first.eval_rng)
    state = first.state_dict()
    digest = state_digest(state)
    second = build_fixture(**kwargs)
    second.load_state_dict(state)
    assert state_digest(second.state_dict()) == digest
    for _ in range(4):
        assert first.step() == second.step()
        assert state_digest(first.state_dict()) == state_digest(second.state_dict())
    assert state_digest(state) == digest
    assert all(torch.equal(a, second._rng_streams._streams[name].get_state())
               for name, a in first.state_dict()["streams"].items())
    if family in ("k3p", "ka2"):
        assert first.opt_g.latent_damping.observed > 0
    if family == "k3p":
        assert first.opt_d.record.anchor_started
    if family == "bispf":
        assert first.loss.thermal_source is first.opt_d.record
        assert second.regularizer.thermal_source is second.opt_d.record
        assert first.opt_d.record.initialized


def test_ka2_fresh_resume_crosses_public_moment_readiness_without_changing_formula():
    kwargs = {"recipe_name": "ka2", "max_steps": 8, "betas": (0., .5)}
    first = build_fixture(**kwargs)
    first.step()
    # The explicit beta2=.5 software recipe crosses the public readiness
    # threshold on the next penalty probe. Never patch the readiness constant.
    assert not first.opt_d.record.anchor_started
    saved = first.state_dict()
    second = build_fixture(**kwargs)
    second.load_state_dict(saved)
    assert first.step() == second.step()
    assert first.opt_d.record.anchor_started
    assert state_digest(first.state_dict()) == state_digest(second.state_dict())


@pytest.mark.parametrize("regularizer", ["sigreg", "moment_tether", "initial_mmd", "coulomb"])
def test_regularizer_draws_and_initial_references_resume(regularizer):
    kwargs = {"recipe_name": "bcap", "max_steps": 5, "prior_regularizer": regularizer, "prior_reg": .001}
    first = build_fixture(**kwargs)
    first.step()
    saved = first.state_dict()
    second = build_fixture(**kwargs)
    second.load_state_dict(saved)
    for _ in range(3):
        assert first.step() == second.step()
    assert state_digest(first.state_dict()) == state_digest(second.state_dict())


def test_frozen_prior_policy_continues_without_reconstructing_trainable_rows():
    first = build_fixture(max_steps=4, prior_update="frozen")
    first.step()
    saved = first.state_dict()
    second = build_fixture(max_steps=4, prior_update="frozen")
    second.load_state_dict(saved)
    assert first.step() == second.step()
    assert not second.prior.z.requires_grad
    assert torch.equal(saved["prior"]["z"], second.prior.z)
    assert state_digest(first.state_dict()) == state_digest(second.state_dict())


@pytest.mark.parametrize("damage", ["legacy", "missing_model", "missing_rng", "rng", "shape", "dtype",
                                   "nonfinite", "recipe", "host", "step", "record", "ema", "moment",
                                   "cooler", "regularizer", "regularizer_copy", "missing_moments", "latent_history"])
def test_rejects_incomplete_or_conflicting_checkpoint_before_any_mutation(damage):
    kwargs = {"recipe_name": "bispf" if damage == "cooler" else "ka2", "max_steps": 8,
                  "prior_regularizer": "sigreg", "prior_reg": .001}
    first = build_fixture(**kwargs)
    first.step()
    saved = first.state_dict()
    victim = build_fixture(**kwargs)
    before = state_digest(victim.state_dict())
    global_rng = torch.get_rng_state().clone()
    if damage == "legacy":
        del saved["version"]
    elif damage == "missing_model":
        del saved["generator"]
    elif damage == "missing_rng":
        del saved["streams"]["data"]
    elif damage == "rng":
        saved["streams"]["latent"] = torch.tensor([1], dtype=torch.uint8)
    elif damage in ("shape", "dtype", "nonfinite"):
        key = next(iter(saved["generator"]))
        if damage == "shape":
            saved["generator"][key] = saved["generator"][key][:1]
        elif damage == "dtype":
            saved["generator"][key] = saved["generator"][key].double()
        else:
            saved["generator"][key].fill_(float("nan"))
    elif damage == "recipe":
        saved["recipe"]["lr"] *= 2
    elif damage == "host":
        saved["host"]["cycle_weight"] = 0.
    elif damage == "step":
        saved["completed_steps"] = -1
    elif damage == "record":
        saved["penalty"]["record"]["observed_steps"] += 1
    elif damage == "ema":
        key = next(iter(saved["ema_critic"]))
        saved["ema_critic"][key] = torch.zeros_like(saved["ema_critic"][key])
    elif damage == "moment":
        next(iter(saved["opt_g"]["state"].values()))["exp_avg"] = torch.ones(1)
    elif damage == "missing_moments":
        saved["opt_g"]["state"] = {}
    elif damage == "latent_history":
        saved["opt_g"]["regularizer"]["latent"]["history"].fill_(float("nan"))
    elif damage == "cooler":
        del saved["penalty"]["cooler"]["initialized"]
    elif damage == "regularizer":
        del saved["regularizer"]["projection_rng_state"]
    else:
        saved["prior_regularizer"]["projection_rng_state"] = torch.Generator().manual_seed(1).get_state()
    with pytest.raises(ValueError):
        victim.load_state_dict(saved)
    assert state_digest(victim.state_dict()) == before
    assert torch.equal(global_rng, torch.get_rng_state())


@pytest.mark.parametrize("family", ["k3p", "ka2", "atlas"])
def test_unsupported_intact_presets_reject_before_optimizers(family, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("optimizer constructed for an unsupported recipe")
    monkeypatch.setattr(type(get_recipe()), "make_optimizers", forbidden)
    with pytest.raises(UnsupportedRecipeOverrides, match="intact recipe"):
        Ring16BiGANFixture(recipe_name=family, max_steps=2)


def test_constructor_and_interrupted_update_own_rng_and_restore_boundary(monkeypatch):
    before = torch.get_rng_state().clone()
    fixture = build_fixture(max_steps=3)
    assert torch.equal(before, torch.get_rng_state())
    state = fixture.state_dict()
    original = fixture.G.forward
    def interrupted(*args, **kwargs):
        torch.rand(3)  # Owned by the fixture's model stream, even on failure.
        raise RuntimeError("interrupted")
    monkeypatch.setattr(fixture.G, "forward", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        fixture.step()
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(RuntimeError, match="completed update boundary"):
        fixture.state_dict()
    monkeypatch.setattr(fixture.G, "forward", original)
    fixture.load_state_dict(state)
    assert state_digest(fixture.state_dict()) == state_digest(state)
    assert fixture.step()["step"] == 1


def test_ring_task_prior_law_is_explicit_outside_retired_recipe_overrides():
    from benchmarks.toy_audit.api_ring16 import _legacy_cases, list_cases
    from benchmarks.toy_audit.api_vectors import VectorFixture
    legacy = _legacy_cases()[0]
    actual = VectorFixture(legacy, device="cpu", seed=0, recipe_name="k3p", max_steps=1)
    assert type(actual.trainer.prior).__name__ == "MoGParticlePrior"
    assert float(actual.trainer.prior.sigma) == pytest.approx(.025)
    assert actual.trainer.prior.standardize is False
    assert actual.trainer.prior.sigma_rel == 0.
    for case in (legacy, list_cases()[0]):
        assert case["prior_options"]["prior_kind"] == "mog"
        assert case["prior_options"]["standardize"] is False
        assert case["prior_options"]["sigma_rel"] == 0.
        assert not {"prior_kind", "sigma_rel", "standardize"} & case.get("recipe_overrides", {}).keys()

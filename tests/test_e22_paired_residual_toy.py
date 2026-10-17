"""Public lifecycle, algebra, solvability and exact-resume contracts for one toy."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch


@pytest.fixture(scope="module")
def toy():
    path = Path(__file__).resolve().parents[1]/"examples/e22_paired_residual_toy.py"
    spec = importlib.util.spec_from_file_location("paired_residual_toy_fixture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.autograd.set_multithreading_enabled(False):
        yield module
    torch.set_num_threads(threads)


def roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer); buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True)


def modes_and_gradients(policy):
    models = (policy.G, policy.encoder, policy.router, policy.D, policy.prior)
    return ([module.training for model in models for module in model.modules()],
            [None if value.grad is None else value.grad.clone()
             for model in models for value in model.parameters()])


def test_common_initialization_frozen_host_disjoint_pools_and_exactly_representable_teacher(toy):
    loops = [toy.make_loop(profile) for profile in toy.PROFILES]
    assert len({loop.initial_weights_sha256 for loop in loops}) == 1
    assert len({toy.digest((loop.data_rng.get_state(), loop.paired_rng.get_state())) for loop in loops}) == 1
    assert len({loop.initial_report_rmse for loop in loops}) == 1
    for loop in loops:
        p = loop.policy
        assert p.recipe.num_particles == 128 and p.recipe.z_dim == 4 and p.recipe.batch_size == 16
        assert p.recipe.total_steps is None and p.recipe.output_noise_mode == "learnable"
        assert p.recipe.row_evidence_gate and p.recipe.particle_birth_death
        assert p.recipe.birth_death_backend == "auto" and p.recipe.reopen_guard == "settled"
        assert p.recipe.reopen_signal == "optimizer" and p.recipe.reopen_anchor == "release"
        assert p.roles == [["generator", "encoder", "table", "noise"], ["critic"]]
        assert all(group["params"] for optimizer in p.optimizers for group in optimizer.param_groups)
        assert p.G.host.weight.dtype == torch.bfloat16 and not p.G.host.weight.requires_grad
        assert p.table.dtype == torch.float32 and p.table.requires_grad and p.prior.z is p.table
        assert tuple(p.opt_g.param_groups[0]["betas"]) == (0., .999)
        assert not tuple(p.router.parameters())
        cohorts = [{tuple(row) for row in getattr(loop, name+"_context").tolist()}
                   for name in ("fit", "guard", "report")]
        assert all(cohorts[i].isdisjoint(cohorts[j]) for i in range(3) for j in range(i+1, 3))
    loop = loops[0]
    before = toy.digest(toy.checkpoint(loop))
    exact = deepcopy(loop.policy.G)
    with torch.no_grad():
        exact.affine.weight.copy_(torch.tensor([[.28, .13, .11], [-.17, .25, -.09]]))
        exact.affine.bias.zero_(); exact.condition.weight.zero_(); exact.condition.bias.zero_()
        for name in ("fit", "guard", "report"):
            context, target = getattr(loop, name+"_context"), getattr(loop, name+"_targets")
            actual = exact(context, torch.zeros(len(context), 4))
            assert actual.shape == (len(context), 2, 1, 32)
            torch.testing.assert_close(actual, target, rtol=1e-6, atol=1e-6)
    assert toy.digest(toy.checkpoint(loop)) == before


@pytest.mark.parametrize("profile", ["current", "even_critic", "d_antithetic"])
def test_one_public_lifecycle_two_actual_DV12_calls_and_applied_rate_observation(toy, profile):
    loop = toy.make_loop(profile)
    p = loop.policy
    perturbations, ratios, order, payoff = [], [], [], []
    original_perturb = type(p.controller).perturb_latent
    tester = p.lr_settle.testers[0][0]
    original_observe = type(tester).observe
    def perturb(self, latent, *args, **kwargs):
        if self is p.controller: perturbations.append(tuple(latent.shape))
        return original_perturb(self, latent, *args, **kwargs)
    def observe(self, params, ratio, *args, **kwargs):
        if self is tester: ratios.append(ratio)
        return original_observe(self, params, ratio, *args, **kwargs)
    methods = ("begin_step", "observe_critic_pair", "before_critic_backward", "after_critic_step",
               "before_generator_backward", "after_generator_backward", "after_generator_step", "finish_step")
    def wrap(name, original):
        def method(self, *args, **kwargs):
            if self is p:
                order.append(name)
                if name == "after_generator_backward": payoff.append(float(kwargs["loss_critic"]))
            return original(self, *args, **kwargs)
        return method
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch.object(type(p.controller), "perturb_latent", perturb))
        stack.enter_context(patch.object(type(tester), "observe", observe))
        for name in methods:
            stack.enter_context(patch.object(type(p), name, wrap(name, getattr(type(p), name))))
        row = toy.update(loop)
    assert order == list(methods)
    assert perturbations == [(16, 4), (16, 4)]
    assert ratios == [.5] and row["g_applied_base_ratio"] == .5
    assert p.initial_lrs[0][0] == .000204 and p.opt_g.param_groups[0]["lr"] == .000102
    assert row["dense_gradient_rows"] == 128 and all(value > 0 for value in row["gradient_energy"].values())
    assert all(value > 0 for value in row["generator_step_delta_l2"].values())
    assert payoff == [row["d_gan"]]
    assert row["ka2_calls"] == 1 and row["ka2_phase"] == "a"
    assert p.penalty.regularizer.record is p.opt_d.record
    assert p.opt_d.continuous_controller is p.controller
    assert p.birth_death.diagnostics()["rows"]["counters"]["updates"] == 1
    assert row["g_gan"] == pytest.approx((row["g_positive"]+row["g_negative"])*.5)
    assert row["adversarial_d_score_forwards"] == (4 if profile == "d_antithetic" else 2)


def test_D_antithetic_removes_cross_noise_from_initial_free_energy_gradient(toy):
    loop = toy.make_loop("d_antithetic")
    p = loop.policy
    before = toy.digest(toy.checkpoint(loop))
    # One deterministic algebra witness, not a training-seed experiment.
    residual = torch.linspace(-.4, .6, 16*64).reshape(16, 2, 1, 32)
    epsilon = torch.cos(torch.arange(16*64).float()*.37).reshape_as(residual)
    target, sigma = torch.zeros_like(residual), torch.tensor(1.3)
    value, real, fake = toy.critic_adversarial(loop, residual, target, sigma, epsilon)
    actual = torch.autograd.grad(value, p.D.energy.weight)[0]
    feature = residual.square().mean((2, 3))*4
    with torch.no_grad():
        # Odd score differences are independent of the shared Gaussian.
        expected = (p.D(residual).sigmoid()[:, None]*feature).mean(0)/3**.5
    torch.testing.assert_close(actual.flatten(), expected, rtol=2e-6, atol=2e-7)
    assert bool(actual.gt(0).all())
    torch.testing.assert_close(real, sigma*epsilon, rtol=0, atol=0)
    torch.testing.assert_close(fake, real+residual, rtol=0, atol=0)
    assert toy.digest(toy.checkpoint(loop)) == before


def test_clean_reports_preserve_whole_owner_RNG_modes_gradients_and_exact_resume(toy):
    loop = toy.make_loop("even_critic")
    for _ in range(101): toy.update(loop)
    before = roundtrip(toy.checkpoint(loop))
    modes = modes_and_gradients(loop.policy)
    report = toy.evaluate(loop)
    assert report["all_owners_and_rng_unchanged"]
    assert toy.digest(toy.checkpoint(loop)) == toy.digest(before)
    assert toy.digest(modes_and_gradients(loop.policy)) == toy.digest(modes)
    resumed = toy.make_loop("even_critic")
    toy.restore(resumed, before)
    assert toy.digest(toy.checkpoint(resumed)) == toy.digest(before)
    assert toy.digest(toy.evaluate(resumed)) == toy.digest(report)
    for _ in range(3):
        assert toy.digest(toy.update(loop)) == toy.digest(toy.update(resumed))
    assert toy.digest(toy.checkpoint(loop)) == toy.digest(toy.checkpoint(resumed))
    wrong = toy.make_loop("current")
    with pytest.raises(ValueError, match="profile"):
        toy.restore(wrong, before)


def test_legal_KA2_799_to_800_blend_and_independent_owner_resume(toy):
    loop = toy.make_loop("current")
    for _ in range(799): last = toy.update(loop)
    assert last["ka2_calls"] == 799 and last["ka2_phase"] == "a"
    saved = roundtrip(toy.checkpoint(loop))
    resumed = toy.make_loop("current")
    toy.restore(resumed, saved)
    assert resumed.policy.opt_d.ema_critic is not resumed.policy.D
    assert resumed.policy.opt_d.record is resumed.policy.penalty.regularizer.record
    for expected_call in (800, 801, 802):
        left, right = toy.update(loop), toy.update(resumed)
        assert left["ka2_calls"] == expected_call and left["ka2_phase"] == "blend"
        assert toy.digest(left) == toy.digest(right)
        assert toy.digest(toy.checkpoint(loop)) == toy.digest(toy.checkpoint(resumed))
    assert toy.evaluate(resumed)["all_owners_and_rng_unchanged"]


def test_real_watchdog_context_can_be_entered_and_closed(toy):
    with toy.watchdog(1):
        assert 2+2 == 4


def test_constructor_failure_records_step_zero_without_unbound_loop(toy, tmp_path, monkeypatch):
    def fail(profile): raise TimeoutError("declared constructor timeout witness")
    monkeypatch.setattr(toy, "make_loop", fail)
    monkeypatch.setattr(sys, "argv", ["toy", "--steps", "1", "--profiles", "current",
                                    "--output", str(tmp_path)])
    with pytest.raises(TimeoutError, match="constructor timeout witness"):
        toy.main()
    failure = json.loads((tmp_path/"failure.json").read_text())
    assert failure["profile"] == "current" and failure["last_completed_step"] == 0
    assert failure["completed_profiles"] == [] and not failure["interrupted_update_qualified"]
    assert json.loads((tmp_path/"current/progress.json").read_text()) == failure
    assert not (tmp_path/"result.json").exists()


def test_completion_write_error_records_actual_completed_state(toy, tmp_path, monkeypatch):
    original = toy.write_json
    def fail_result(path, value):
        if path.name == "result.json": raise OSError("declared completion write failure witness")
        return original(path, value)
    monkeypatch.setattr(toy, "write_json", fail_result)
    monkeypatch.setattr(sys, "argv", ["toy", "--steps", "1", "--profiles", "current",
                                    "--output", str(tmp_path)])
    with pytest.raises(OSError, match="completion write failure witness"):
        toy.main()
    failure = json.loads((tmp_path/"failure.json").read_text())
    assert failure["profile"] == "current" and failure["last_completed_step"] == 1
    assert failure["completed_profiles"] == ["current"]
    assert json.loads((tmp_path/"current/progress.json").read_text()) == failure
    assert not (tmp_path/"result.json").exists()
    assert not list(tmp_path.glob("*.tmp"))

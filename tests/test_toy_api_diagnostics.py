"""Actual public native controls; gate witnesses are software, not qualification."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particlegan import Recipe
from benchmarks.toy_audit import api_diagnostics as api


@pytest.fixture(autouse=True)
def cpu_one():
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False): yield
    finally: torch.set_num_threads(old)


@pytest.mark.parametrize("case",api.list_cases(),ids=lambda c:c["id"])
def test_actual_source_api_step_and_pure_goal_observation(case):
    f=api.build_case(case["id"],recipe_name=case["default_recipe"],max_steps=2)
    baseline=api.build_case(case["id"],recipe_name=case["default_recipe"],max_steps=2)
    assert isinstance(f.recipe,Recipe)
    assert api._tree_equal(f.step(),baseline.step())
    state=f.state_dict();o=f.observe(n=16)
    assert api._tree_equal(state,f.state_dict()),case["id"]
    assert type(o["passed"]) is bool and bool(o["failed_bounds"])==(not o["passed"])
    assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in o["metrics"].values())
    assert o["views"] and all(v["samples"].numel() for v in o["views"])
    assert api._tree_equal(f.step(),baseline.step())
    assert api._tree_equal(f.state_dict(),baseline.state_dict())
    with pytest.raises(RuntimeError,match="cap"):f.step()


def test_full_stiff_native_counterexample_and_two_controls():
    results={}
    for name in ("native","cancel","safe"):
        f=api.build_case("api-stiff-"+name,recipe_name="e22")
        for _ in range(96):f.step()
        results[name]=f.observe()
    assert not results["native"]["passed"]
    assert results["native"]["failed_bounds"]==["peak_game_excess"]
    assert results["cancel"]["passed"] and results["safe"]["passed"]


def test_reflected_joint_control_and_live_penalty_are_independent_bounds():
    f=api.build_case("api-sign-lander-controls",recipe_name="ka2",max_steps=1)
    f.step()
    # Explicit exact-control witness after a real API update. Never a model receipt.
    with torch.no_grad():
        f.games["paired"].value.fill_(1.)
        f.games["collapsed"].value.fill_(-1.)
        f.games["collapsed"].beta.fill_(-1.)
    positive=f.observe();assert positive["passed"]
    f.games["paired"].applications=0
    assert f.observe()["failed_bounds"]==["applied_penalties"]
    f.games["paired"].applications=1
    with torch.no_grad():f.games["paired"].value.fill_(-1.)
    assert "paired_relative_mse" in f.observe()["failed_bounds"]


def test_safe_fast_full_denominators_and_live_adversary_control():
    f=api.build_case("api-safe-fast-controls",recipe_name="ka2",max_steps=1);f.step()
    with torch.no_grad():
        f.games["baseline"].value.fill_(-20.)
        q=(api.landing.QUICK_SINK-api.landing.SLOW)/api.landing.RANGE
        f.games["combined"].value.fill_(math.log(q/(1-q)))
    combined=f.games["combined"]
    combined.gan_gradient_sum=2.;combined.late_gradient_sum=.5;combined.late_count=1
    positive=f.observe();assert positive["passed"],positive
    combined.gan_gradient_sum=0.
    assert "gan_gradient_sum" in f.observe()["failed_bounds"]
    combined.gan_gradient_sum=2.
    with torch.no_grad():
        q=(api.landing.CRASH_SINK-api.landing.SLOW)/api.landing.RANGE
        combined.value.fill_(math.log(q/(1-q)))
    bad=f.observe()
    assert not bad["passed"] and ("combined_crash_rate" in bad["failed_bounds"] or "combined_landings" in bad["failed_bounds"])


def test_synthetic_ae_gate_needs_independent_prior_and_paired_reconstruction(monkeypatch):
    f=api.build_case("api-ae-anchor-hold",recipe_name="ka2",max_steps=1);f.step()
    target=f.draw(4096,api._rng(99124))
    class FixedPrior:
        def sample(self,n,generator=None):return target[:n],torch.arange(n)
    f.prior=FixedPrior();f.G=nn.Identity()
    monkeypatch.setattr(f,"reconstruct",lambda x:x)
    assert f.observe(4096)["passed"]
    monkeypatch.setattr(f,"reconstruct",lambda x:-x)
    assert "reconstruction_mse" in f.observe(4096)["failed_bounds"]
    monkeypatch.setattr(f,"reconstruct",lambda x:x)
    target[:,0]=target[:,0].abs()
    assert "quality_mass_tv" in f.observe(4096)["failed_bounds"]


@pytest.mark.parametrize("name",["api-film-original_native","api-film-shift_zero_native"])
def test_film_zero_recipient_witness_cannot_pass_distant_host_denominator(name,monkeypatch):
    f=api.build_case(name,recipe_name="e22_routed",max_steps=1);f.step()
    target=f.loop.test_targets
    monkeypatch.setattr(f.loop.policy,"served_model",lambda:SimpleNamespace(routed_forward=lambda c:target.clone()))
    assert f.observe()["passed"]
    monkeypatch.setattr(f.loop.policy,"served_model",lambda:SimpleNamespace(routed_forward=lambda c:torch.zeros_like(target)))
    result=f.observe()
    assert result["metrics"]["relative_recipient_mse"]==pytest.approx(1.)
    assert not result["passed"]


def test_code_probe_uses_signed_benefit_on_actual_common_noise_game():
    from examples import e22_routed_convergence as source
    class QuadraticJudge(nn.Module):
        def forward(self,error,condition):return -error.square().mean((1,2))
        def features(self,error,condition):return error
    context=torch.zeros(24,source.TOKENS,source.WIDTH+769)
    panel=.125*torch.randn(4,24,source.TOKENS,source.WIDTH,generator=api._rng(72))
    fitted=torch.zeros(24,source.TOKENS,source.WIDTH)
    harmful=torch.ones_like(fitted)*.25
    good=source.score_residual(QuadraticJudge(),context,fitted,panel)["paired_game"]
    bad=source.score_residual(QuadraticJudge(),context,harmful,panel)["paired_game"]
    assert bad-good>1e-6
    # Harmless-to-harmful removal cannot qualify merely because abs(delta)>0.
    assert not api._bound({"delta":bad-good},"delta",lo=1e-6)
    assert api._bound({"delta":good-bad},"delta",lo=1e-6)==["delta"]


def test_prefix_cannot_claim_unexecuted_target_orientations():
    assert api.CASES["api-routed-moving"]["terminal_observations"]==1
    f=api.build_case("api-routed-moving",recipe_name="e22_routed",max_steps=1);f.step()
    result=f.observe()
    assert result["metrics"]["completed_orientation_checks"]==0
    assert "completed_orientation_checks" in result["failed_bounds"]


def test_replay_runs_exact_baseline_and_rejects_corrupted_owner():
    f=api.build_case("api-routed-replay",recipe_name="e22_routed",max_steps=3)
    for _ in range(3): f.step()
    assert f.replay_equal
    assert api._tree_equal(f.api.checkpoint(f.loop),f.api.checkpoint(f.baseline))
    with torch.no_grad(): next(f.baseline.policy.G.parameters()).add_(.01)
    assert not api._tree_equal(f.api.checkpoint(f.loop),f.api.checkpoint(f.baseline))
    f.replay_equal=False
    assert "checkpoint_owner_mismatch" in f.observe()["failed_bounds"]


def test_native_observer_prefix_parity():
    left=api.build_case("api-critic-lag-even_critic",recipe_name="e22_routed",max_steps=2)
    right=api.build_case("api-critic-lag-even_critic",recipe_name="e22_routed",max_steps=2)
    assert api._tree_equal(left.step(),right.step())
    left.observe()
    assert api._tree_equal(left.step(),right.step())
    assert api._tree_equal(left.state_dict(),right.state_dict())


@pytest.mark.parametrize("arm",("current","even_critic","d_antithetic"))
def test_critic_lag_grayscale_view_preserves_both_actual_feature_blocks(arm):
    f=api.build_case("api-critic-lag-"+arm,recipe_name="e22_routed",max_steps=2)
    f.step();f.step()
    with torch.no_grad():
        actual=f.loop.policy.served_model().routed_forward(f.loop.report_context)[:8]
    target=f.loop.report_targets[:8]
    assert actual.shape==target.shape==(8,2,1,32)
    state=f.state_dict()
    record=f.observe()
    assert api._tree_equal(state,f.state_dict())
    panel=record["views"][0]
    assert panel["kind"]=="image"
    assert panel["samples"].shape==panel["target"].shape==(8,1,2,32)
    assert torch.equal(panel["samples"].reshape_as(actual),actual)
    assert torch.equal(panel["target"].reshape_as(target),target)
    assert "Two 32-coordinate residual feature blocks (64 total)" in panel["caption"]
    # Exercise the actual shared media conversion that rejected two channels.
    from benchmarks.toy_audit.api_run import _image_grid
    assert _image_grid(panel["samples"]).shape==(2,8*32+7)


def test_wrong_checkout_import_unknown_source_and_recipe_are_rejected(monkeypatch,tmp_path):
    monkeypatch.setitem(__import__("sys").modules,"e22_routed_fake",SimpleNamespace(__file__=str(tmp_path/"e22_routed_fake.py")))
    with pytest.raises(ValueError,match="foreign-checkout"):api._example("e22_routed_paired")
    with pytest.raises(ValueError,match="unknown"):api.build_case("api-missing")
    with pytest.raises(ValueError,match="declared recipe"):api.build_case("api-stiff-safe",recipe_name="atlas")

"""API execution, observer parity and mathematical gate counterexamples."""
from copy import deepcopy
import math

import pytest
import torch

from particlegan import Recipe, GANLoss
from benchmarks.toy_audit import api_conditionals as api
from benchmarks.toy_audit.api_diagnostics import _tree_equal


@pytest.fixture(autouse=True)
def cpu_one():
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False): yield
    finally: torch.set_num_threads(old)


def oracle(task):
    if isinstance(task,api.SparseTask):
        def predict(c,r):
            x,s,_=task.toy.sample_given_class(c.argmax(1),r)
            return torch.cat((x,torch.nn.functional.one_hot(s,task.toy.n_symbols).float()),1)
    elif isinstance(task,api.PosteriorTask):
        def predict(c,r):
            k=task.toy.classes
            return task.toy.oracle_clean(c[:,k:k+2],c[:,:k].argmax(1),float(c[0,-1]),r)
    elif isinstance(task,api.RouteTask):
        def predict(c,r):
            if task.transition: return task.toy.sample(c[:,:2].argmax(1),c[:,2:5],(63*c[:,5]).round().long(),r)
            return task.toy.sample(c[:,:2].argmax(1),c[:,2:5],r)[0].flatten(1)
    elif isinstance(task,api.MaskTask):
        def predict(c,r):
            k=len(task.ids);base=c[:k]
            return api.misgan.bayes_posterior(task.problem,base[:,:8],base[:,8:],draws=len(c)//k,generator=r)[1].reshape(-1,8)
    elif task.name.startswith("trajectory"):
        def predict(c,r): return task.fast[torch.cdist(c,task.slow).argmin(1)]
    else:
        def predict(c,r): return task.targets(c)
    return predict


@pytest.mark.parametrize("case",api.list_cases(),ids=lambda c:c["id"])
def test_actual_public_update_caps_and_read_only_observation(case):
    f=api.build_case(case["id"],recipe_name=case["default_recipe"],max_steps=2)
    baseline=api.build_case(case["id"],recipe_name=case["default_recipe"],max_steps=2)
    assert isinstance(f.recipe,Recipe) and isinstance(f.loss,GANLoss)
    assert f.recipe.total_steps==case["default_steps"]
    before={k:p.clone() for k,p in f.G.named_parameters()}
    row=f.step()
    assert _tree_equal(row,baseline.step())
    assert row["step"]==1 and f.opt_d.record.calls==1
    assert any(not torch.equal(before[k],p) for k,p in f.G.named_parameters())
    state=f.state_dict();o=f.observe(n=16)
    assert _tree_equal(state,f.state_dict()),case["id"]
    assert type(o["passed"]) is bool and bool(o["failed_bounds"])==(not o["passed"])
    assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in o["metrics"].values())
    assert o["views"] and all(v["target"].numel() and v["samples"].numel() for v in o["views"])
    assert _tree_equal(f.step(),baseline.step())
    assert _tree_equal(f.state_dict(),baseline.state_dict())
    with pytest.raises(RuntimeError,match="cap"):f.step()


@pytest.mark.parametrize("case",api.list_cases(),ids=lambda c:c["id"])
def test_attainable_declared_gate_on_exact_law_witness(case):
    task=api._task(case["id"].removeprefix("api-"))
    o=task.evaluate(oracle(task),case["eval_samples"],99123)
    assert o["passed"],(case["id"],o["failed_bounds"],o["metrics"])


@pytest.mark.parametrize("name",["sparse-identity","sparse-split"])
def test_sparse_collapsed_width_and_incoherent_symbols_are_rejected(name):
    task=api._task(name);positive=oracle(task)
    def centres(c,r):
        raw=positive(c,r);mode,_=task.toy.assign(raw[:,:24]);raw[:,:24]=task.toy.centers[mode];return raw
    assert "covariance_min" in task.evaluate(centres,1024,99123)["failed_bounds"]
    if name.endswith("split"):
        def one_symbol(c,r):
            raw=positive(c,r);raw[:,24:]=torch.nn.functional.one_hot(2*c.argmax(1),16);return raw
        bad=task.evaluate(one_symbol,1024,99123)
        assert not bad["passed"] and "max_symbol_tv" in bad["failed_bounds"]


@pytest.mark.parametrize("classes",[1,4])
def test_posterior_mean_is_not_a_distribution(classes):
    task=api.PosteriorTask(classes)
    def mean(c,r):
        weights,mu,_=task.toy.posterior(c[:,classes:classes+2],c[:,:classes].argmax(1),float(c[0,-1]))
        return (weights[:,:,None]*mu).sum(1)
    assert not task.evaluate(mean,1024,99123)["passed"]


def test_routes_have_two_mass_and_coefficient_width_controls():
    task=api.RouteTask("discrete");positive=oracle(task)
    def centres(c,r):
        samples=positive(c,r).reshape(-1,2,64);d=task.toy.diagnose(samples,c[:,2:5])
        return task.toy.templates(c[:,2:5])[torch.arange(len(c)),d["route"]].flatten(1)
    bad=task.evaluate(centres,1024,99123)
    assert "min_coefficient_variance" in bad["failed_bounds"]
    def lower(c,r):return task.toy.templates(c[:,2:5])[:,0].flatten(1)
    assert "max_quality_route_mass_tv" in task.evaluate(lower,1024,99123)["failed_bounds"]


def test_transition_gate_rejects_aligned_marginals_with_shuffled_actions():
    task=api.RouteTask("continuous",True);positive=oracle(task)
    def shuffled(c,r):
        x=positive(c,r);x[:,2:4]=x[:,2:4].roll(17,0);return x
    result=task.evaluate(shuffled,512,99123)
    assert not result["passed"] and "joint_consistency_rms" in result["failed_bounds"]


@pytest.mark.parametrize("name",["trajectory-edit","trajectory-residual"])
def test_trajectory_row_permutation_cannot_pass_correct_target_marginal(name):
    task=api.PairedTask(name);positive=oracle(task)
    assert not task.evaluate(lambda c,r:positive(c,r).roll(6,0),1024,99123)["passed"]


def test_neutral_unused_and_intermediate_identity_controls():
    task=api.PairedTask("unipolar-hold")
    assert task.evaluate(lambda c,r:task.targets(c),1024,99123)["passed"]
    bad=task.evaluate(lambda c,r:torch.tensor([[1.,0.,0.,0.]]).expand(len(c),4),1024,99123)
    assert "neutral_hold" in bad["failed_bounds"]
    task=api.PairedTask("unused-token-hold")
    def shared(c,r):return task.targets(c)+c*torch.tensor([[0.,1.,0.,0.]])
    assert "unused_hold" in task.evaluate(shared,1024,99123)["failed_bounds"]
    task=api.PairedTask("midscale-identity")
    def stranger(c,r):
        x=task.targets(c);x[c[:,0]==.5,1]=0;x[c[:,0]==.5,3]=.55;return x
    assert not task.evaluate(stranger,1024,99123)["passed"]


def test_circle_direction_and_recovery_are_not_radius_only():
    def backwards(c,r):return api.circle.expert_transition(c[:,:2],c[:,2:4],c[:,4],-c[:,5])[0]
    result=api.circle_observation(backwards)
    assert result["metrics"]["main_radial_rmse"]<.001
    assert "direction_agreement" in result["failed_bounds"] and "signed_speed_error" in result["failed_bounds"]
    def tangent_only(c,r):
        pos,center,radius,omega=c[:,:2],c[:,2:4],c[:,4],c[:,5]
        q=pos-center;co,si=omega.cos(),omega.sin()
        return torch.stack((co*q[:,0]-si*q[:,1],si*q[:,0]+co*q[:,1]),1)-q
    assert "recovery_radial_rmse" in api.circle_observation(tangent_only)["failed_bounds"]


def test_sprite_free_run_and_render_reject_persistence():
    result=api.sprite_observation(lambda c,r:c,99123)
    assert not result["passed"] and "ood_50_position_rmse" in result["failed_bounds"]
    assert "Fully observed" in result["views"][0]["caption"]


@pytest.mark.parametrize("mechanism",["block","mcar_p20","mcar_p50","mcar_p80"])
def test_mask_conditional_mean_and_projection_blind_noise_controls(mechanism):
    task=api.MaskTask(mechanism)
    positive=oracle(task)
    def mean(c,r):
        # A collapsed constant draw per query preserves observed entries.
        k=len(task.ids);base=c[:k];_,draws=api.misgan.bayes_posterior(task.problem,base[:,:8],base[:,8:],draws=4096,generator=r)
        return draws.mean(0).repeat(len(c)//k,1)
    result=task.evaluate(mean,4096,99123)
    assert result["metrics"]["observed_max_error"]<1e-6
    assert not result["passed"] and "min_missing_variance_ratio" in result["failed_bounds"]
    def orthogonal(c,r):
        x=positive(c,r);p=task.problem
        direction=torch.zeros(8,dtype=torch.float64);direction[0]=1
        direction-=p.A@(p.A.T@direction)
        # Keep observed coordinates, deliberately corrupt unobserved 8D noise.
        return x+(1-c[:,8:])*(.05*direction.float()/p.std.float())
    result=task.evaluate(orthogonal,4096,99123)
    assert not result["passed"] and "orthogonal_rms_ratio" in result["failed_bounds"]


def test_observation_does_not_change_next_training_trajectory():
    left=api.build_case("api-routes-continuous",recipe_name="ka2",max_steps=2)
    right=api.build_case("api-routes-continuous",recipe_name="ka2",max_steps=2)
    assert _tree_equal(left.step(),right.step())
    left.observe(256)
    assert _tree_equal(left.step(),right.step())
    assert _tree_equal(left.state_dict(),right.state_dict())


def test_unknown_ids_and_unbound_policy_refuse_execution():
    with pytest.raises(ValueError,match="unknown"):api.build_case("api-made-up")
    with pytest.raises(ValueError,match="Atlas"):api.build_case("api-routes-discrete",recipe_name="atlas")
    with pytest.raises(ValueError,match="CPU"):api.build_case("api-routes-discrete",device="cuda",recipe_name="ka2")

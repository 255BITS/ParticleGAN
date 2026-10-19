"""Public CPU numerical/API fixtures for explicit diagnostic v2 policy.

Fixed analytic matrices; no GAN training, actual payloads, CUDA queries or RNG.
Metric: FP64 SVD of identical stored FP32 input, strict original epsilon32 cutoff.
"""
from copy import deepcopy
import pytest
import torch
from particlegan.optim import NormalizedOptimizer, svd_fallback_polar
from particlegan.optim import gram_svd_fallback as policy
from particlegan.optim.dualnorm import polar_factor


def gradient(side):
    n=16; tau=torch.tensor(n*torch.finfo(torch.float32).eps)
    tail=tau if side=="tie" else torch.nextafter(tau,tau.new_tensor(1. if side=="above" else 0.))
    s=torch.zeros(n);s[0]=1.;s[1]=tail
    return torch.diag(s)


def oracle(matrix):
    u,s,vh=torch.linalg.svd(matrix.double(),full_matrices=False)
    keep=s>max(matrix.shape)*torch.finfo(torch.float32).eps*s[0]
    return ((u*(keep*s/torch.hypot(s,s.new_tensor(.001))))@vh).float()


def optimizer(parameter):
    return NormalizedOptimizer([parameter],lr=8e-5,smoothing=.001,
                               dense_profile="gram_fp64_svd_diagnostic_v2")


@pytest.mark.parametrize("side",["above","below","tie"])
def test_public_strict_computed_rank_decision_is_nonfatal_and_explicit(side):
    g=gradient(side);rng=torch.get_rng_state().clone();before=g.clone()
    counters=policy.Counters();actual=svd_fallback_polar(g,counters=counters)
    assert torch.equal(actual,oracle(g))
    # The declared same FP64 formula is public polar_factor with ORIGINAL FP32
    # epsilon restored in oracle; no false FP32-native bit-parity promise.
    assert bool(actual[1,1]) == (side=="above")
    assert counters.fallback_passes==counters.uncertified_svd_rank_decisions==1
    last=counters.last_fallback
    assert last["mask_ambiguous_count"]>0 and last["rank_certified"] is False
    assert last["rank_decision_policy"]=="public_strict_computed_svd"
    assert last["uncertified_svd_rank_decision"] is True
    assert torch.equal(g,before) and torch.equal(rng,torch.get_rng_state())
    parameter=torch.nn.Parameter(torch.zeros_like(g));parameter.grad=g.clone()
    opt=optimizer(parameter);opt.step()
    assert torch.equal(parameter,-8e-5*actual)
    assert opt.state[parameter]["step"]==1
    assert opt.dense_profile_stats["aggregate"]["uncertified_svd_rank_decisions"]==1


def test_new_profile_disk_restore_strict_identity_and_rejected_load_atomic(tmp_path):
    parameter=torch.nn.Parameter(torch.zeros(16,16));parameter.grad=gradient("tie")
    opt=optimizer(parameter);opt.step();state=deepcopy(opt.state_dict())
    assert state["dualnorm"]["dense_profile"]==policy.VERSION
    assert "uncertified_svd_rank_decisions" not in repr(state)
    path=tmp_path/"state.pt";torch.save(state,path)
    restored=torch.load(path,weights_only=True);before=parameter.clone()
    opt.load_state_dict(restored)
    assert torch.equal(parameter,before) and opt.dense_profile_stats["aggregate"]["calls"]==0
    opt.step();stats=deepcopy(opt.dense_profile_stats);clock=opt.state[parameter]["step"]
    bad=deepcopy(restored);bad["dualnorm"]["dense_profile"]="gram_fp64_svd_fallback_v1"
    before=parameter.clone()
    with pytest.raises(ValueError):opt.load_state_dict(bad)
    assert torch.equal(parameter,before) and opt.state[parameter]["step"]==clock
    assert opt.dense_profile_stats==stats


@pytest.mark.parametrize("error",[policy.GramGuardError("significantly negative Gram eigenvalue"),
                                 RuntimeError("technical failure")])
def test_other_primary_errors_remain_fatal_and_unreplaced(monkeypatch,error):
    def fail(*args,**kwargs):raise error
    monkeypatch.setattr(policy,"_original_guard",fail)
    with pytest.raises(type(error)) as got:svd_fallback_polar(gradient("tie"))
    assert got.value is error


def test_corrupted_svd_orthogonality_remains_fatal_before_write(monkeypatch):
    original=torch.linalg.svd
    def corrupted(*args,**kwargs):
        u,s,vh=original(*args,**kwargs);return u*2,s,vh
    monkeypatch.setattr(torch.linalg,"svd",corrupted)
    parameter=torch.nn.Parameter(torch.zeros(16,16));parameter.grad=gradient("tie")
    opt=optimizer(parameter)
    with pytest.raises(policy.SVDHoldError,match="orthogonality"):opt.step()
    assert torch.count_nonzero(parameter)==0


def test_recipe_explicit_version_and_old_profile_rejection():
    from particlegan import get_recipe
    recipe=get_recipe("bcap",optimizer_family="dualnorm",optimizer_smoothing=.001,
                      optimizer_dense_profile=policy.VERSION)
    assert recipe.to_dict()["optimizer_dense_profile"]==policy.VERSION
    with pytest.raises(ValueError):recipe.replace(optimizer_dense_profile="gram_fp64_svd_fallback_v1")


def test_unchanged_primary_pass_outputs_byte_identical():
    from particlegan.optim import guarded_gram_polar
    g=torch.diag(torch.tensor([1.,.001,0.]))
    assert torch.equal(svd_fallback_polar(g),guarded_gram_polar(g))



def test_finite_corrupted_singular_values_remain_fatal_before_write(monkeypatch):
    original=torch.linalg.svd
    def corrupted(*args,**kwargs):
        u,s,vh=original(*args,**kwargs);return u,s*.5,vh
    monkeypatch.setattr(torch.linalg,"svd",corrupted)
    parameter=torch.nn.Parameter(torch.zeros(16,16));parameter.grad=gradient("tie")
    opt=optimizer(parameter)
    with pytest.raises(policy.SVDHoldError,match="reconstruction"):opt.step()
    assert torch.count_nonzero(parameter)==0
    assert opt.dense_profile_stats["aggregate"]["uncertified_svd_rank_decisions"]==0



@pytest.mark.parametrize("target",["left","right","reciprocal_left"])
def test_one_percent_finite_factor_corruption_is_fatal_without_rank_relaxation(monkeypatch,target):
    original=torch.linalg.svd
    def corrupted(*args,**kwargs):
        u,s,vh=original(*args,**kwargs)
        if target=="left":u=u*1.01
        elif target=="right":vh=vh*1.01
        else:u,s=u*1.01,s/1.01  # Reconstruction stays accurate; orthogonality must fail.
        return u,s,vh
    monkeypatch.setattr(torch.linalg,"svd",corrupted)
    parameter=torch.nn.Parameter(torch.zeros(16,16));parameter.grad=gradient("tie")
    opt=optimizer(parameter)
    with pytest.raises(policy.SVDHoldError,match="orthogonality"):opt.step()
    assert torch.count_nonzero(parameter)==0
    assert opt.dense_profile_stats["aggregate"]["fallback_passes"]==0


@pytest.mark.parametrize("target",["negative","unsorted","nonfinite"])
def test_invalid_singular_factors_remain_fatal(monkeypatch,target):
    original=torch.linalg.svd
    def corrupted(*args,**kwargs):
        u,s,vh=original(*args,**kwargs)
        if target=="negative":s=-s
        elif target=="unsorted":s=s.flip(0)
        else:s=s.clone();s[0]=float("nan")
        return u,s,vh
    monkeypatch.setattr(torch.linalg,"svd",corrupted)
    with pytest.raises(policy.SVDHoldError,match="nonfinite/negative/unsorted"):
        svd_fallback_polar(gradient("tie"))


@pytest.mark.parametrize("shape",[(16,16),(24,16),(16,24)])
def test_valid_fp64_cpu_factors_satisfy_independent_validity_budgets(shape):
    generator=torch.Generator(device="cpu").manual_seed(0)
    z=torch.randn(shape,dtype=torch.float64,generator=generator)
    u,s,vh=torch.linalg.svd(z,full_matrices=False)
    diagnostics=policy._factor_guard(z,u,s,vh)
    assert diagnostics["left_orthogonality_frobenius"]<=diagnostics["left_orthogonality_guard_budget"]
    assert diagnostics["right_orthogonality_frobenius"]<=diagnostics["right_orthogonality_guard_budget"]
    assert diagnostics["reconstruction_frobenius"]<=diagnostics["reconstruction_guard_budget"]

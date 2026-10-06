"""NEW CPU software qualifications only; exactly four native updates total.

Root must authorize the single suite before execution. No prior tests imported.
"""
from copy import deepcopy
import math
import pytest
import torch
from torch import nn
from examples import e22_routed_terminal_cancellation as case


SMALL=case.base.Geometry(width=8,text=16,rank=2,tokens=4,length=1,heads=2,output=4,frequency=4)


def test_public_initialization_paired_column_oracle_and_owned_terminal_precision():
    entry=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):head=case.PairedTerminal(4,4,"FP32")
    case.base.initialize(head,"software_terminal");head.requires_grad_(False)
    before={n:p.detach().clone() for n,p in head.named_parameters()};ids={n:id(p) for n,p in head.named_parameters()}
    a=torch.tensor([[[1.,-.7,.3,2.]]]);u=torch.tensor([[[.01,.02,-.03,.04]]])
    actual=head(a,u);oracle=torch.nn.functional.linear(torch.cat((a+u,a-u),-1),torch.cat((head.matrix.weight,-head.matrix.weight),-1))
    assert torch.equal(actual,oracle)
    assert torch.allclose(actual,2*torch.nn.functional.linear(u,head.matrix.weight),rtol=1e-5,atol=1e-7)
    head.arithmetic="BF16"
    with torch.autocast("cpu",dtype=torch.bfloat16):expected=torch.nn.functional.linear(torch.cat((a+u,a-u),-1),torch.cat((head.matrix.weight,-head.matrix.weight),-1)).float()
    assert torch.equal(head(a,u),expected)
    assert {n:id(p) for n,p in head.named_parameters()}==ids
    assert all(torch.equal(p,before[n]) for n,p in head.named_parameters())
    assert torch.equal(entry,torch.get_rng_state())


def test_public_two_update_observed_restore_replay_and_registered_EMA(monkeypatch):
    entry=torch.get_rng_state().clone();data=case.make_data(SMALL);loop=case.make_loop("particle_FP32",data)
    assert isinstance(loop.policy.table,nn.Parameter)
    assert all(p is not q and p.data_ptr()!=q.data_ptr() for p,q in zip(loop.policy.G.parameters(),loop.policy.ema_G.parameters()))
    assert loop.policy.G.backbone.final.arithmetic==loop.policy.ema_G.backbone.final.arithmetic=="FP32"
    saved=case.snapshot(loop)
    a=[case.base.update(loop),case.base.update(loop)]  # exactly2
    final=case.snapshot(loop)
    fresh=case.make_loop("particle_FP32",data)
    monkeypatch.setattr(case.init,"initialize_",lambda *a,**k:(_ for _ in ()).throw(AssertionError("trained restore initialized")))
    case.restore(fresh,saved)
    b=[case.base.update(fresh),case.base.update(fresh)]  # exactly2; total4
    assert case.base.digest(a)==case.base.digest(b)
    assert case.base.digest(final)==case.base.digest(case.snapshot(fresh))
    before=case.snapshot(fresh);v=case.observe(fresh,data["test"]["context"][:4])
    assert v.shape==(4,4,4) and torch.isfinite(v).all()
    assert case.base.digest(before)==case.base.digest(case.snapshot(fresh))
    assert torch.equal(entry,torch.get_rng_state())


def test_fixed_science_gate_oracles_and_destructive_source_code_live_controls():
    def metric(x):return {"rmse":x,"by_source":{str(s):x for s in range(6)}}
    quality={"ordinary_FP32":metric(1.),"particle_FP32":metric(.99)};zero=metric(1.01)
    live={"bank":511,"query":511};norms={s:{"C":1.,"particle_up":1.} for s in case.SITES}
    assert case.scientific_gate(quality,zero,live,norms)["pass"]
    harmed=deepcopy(quality);harmed["particle_FP32"]["by_source"]["5"]=1.+2e-6
    assert not case.scientific_gate(harmed,zero,live,norms)["pass"]
    assert not case.scientific_gate(quality,metric(.99),live,norms)["pass"]
    assert not case.scientific_gate(quality,zero,{"bank":0,"query":511},norms)["pass"]
    bad=deepcopy(quality);bad["particle_FP32"]["rmse"]=math.nan
    with pytest.raises(ValueError):case.scientific_gate(bad,zero,live,norms)
    residual=torch.ones(48,4,4);labels=[s for s in range(6) for _ in range(8)]
    assert case.base.accuracy(residual,labels)["rmse"]==1.
    assert case.base.accuracy(torch.zeros_like(residual),labels)["rmse"]==0.
    with pytest.raises(ValueError):case.base.accuracy(residual*math.nan,labels)


def test_fixed_physical_secant_identity_and_precision_not_a_convergence_requirement():
    y0=torch.ones(48,4,4,dtype=torch.float64);y1=y0-0.1
    result=case.secant(y0,y1)
    assert abs(result["delta_MSE"]+.19)<1e-14
    assert abs(result["output_linear"]+.2)<1e-14 and abs(result["Q"]-.01)<1e-14
    raw={"BF16_before":y0,"BF16_after":y1,"FP32_before":y0,"FP32_after":y0-.05,
         "BF16_gradient":{"x":torch.tensor([2.],dtype=torch.float64)},"FP32_gradient":{"x":torch.tensor([2.],dtype=torch.float64)},
         "delta":{"x":torch.tensor([-.1],dtype=torch.float64)}}
    witness=case.precision_witness(raw)
    assert not witness["helpful_parameter_slope_and_finite_harm"]
    assert "precision_status" in witness and "pass" not in witness
    with pytest.raises(ValueError):case.secant(y0,y1*math.nan)

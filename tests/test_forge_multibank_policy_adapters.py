"""CPU structural checks, at most two updates per fixture; no qualification."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from particlegan import init
from experiments.forge import multibank_policy_contracts as declaration
from experiments.forge.multibank_policy_adapters import CoverMultibankFixture, polar_context, host
from experiments.forge.policy_adapters import typed_state_digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    before = torch.get_num_threads(); torch.set_num_threads(1)
    try: yield
    finally: torch.set_num_threads(before)


def definition():
    task = declaration.make_variant(ROOT)
    request = dict(candidate=dict(id="synthetic_multibank", recipe_preset="atlas", task_cohort=declaration.COHORT,
                                  recipe_overrides=deepcopy(declaration.SHARED_OVERRIDES)), protocol=dict(seed=0))
    return request, task


def fixture():
    return CoverMultibankFixture(*definition())


def test_original_degrees_initial_two_bank_population_and_actual_public_roles():
    old = init.declarations(host._Residual(4)); value = fixture()
    assert init.declarations(host._Residual(4)) == old
    assert type(value.G).__mro__[1] is host._Residual
    assert sum(p.numel() for p in value.G.parameters()) == 8 and torch.count_nonzero(value.G.w_odd) == 0
    assert value.prior.z.shape == (24, 4) and sum(p.numel() for p in value.prior.parameters()) == 96
    assert torch.equal(value.prior.z[:12], value.prior.z[12:])
    assert torch.equal(value.router.bank_ids, torch.arange(24)//12)
    assert value.policy.roles == [["generator", "table", "noise"], ["critic"]]
    assert value.opt_g.latent_damping is not None and value.opt_g.direct_response is None
    assert value.recipe.total_steps is None and value.max_steps == host.GATE_STEPS == 800
    assert value.recipe.prior_kind == "particles" and value.recipe.sigma_rel == 0. and not value.recipe.standardize
    assert value.recipe.lr * value.recipe.prior_lr_mult == value.opt_g.param_groups[1]["lr"]
    assert value.controls()["cohort"] == declaration.COHORT and not value.controls()["independent_atlas_qualification"]


def test_original_clean_function_and_all_objective_gradients_preserve_bank_scaling():
    value = fixture()
    with torch.no_grad():
        value.G.w_odd.copy_(torch.tensor([.2, .1, -.03, .04])); value.G.w_even.copy_(torch.tensor([.1, -.2, .05, .06]))
        value.prior.z.copy_(torch.linspace(-.07, .09, 96).reshape(24, 4))
    rows = torch.arange(16)%12; jitter = torch.arange(64).reshape(16,4).float()/10000
    context = torch.cat((polar_context(0,rows,jitter),polar_context(1,rows,-jitter)))
    actual = value.policy.routed_generate(context,sigma=0.,perturb=False)
    g = host._Residual(4); g.load_state_dict(value.G.state_dict())
    plus = torch.nn.Parameter(value.prior.z[:12].detach().clone()); minus = torch.nn.Parameter(value.prior.z[12:].detach().clone())
    old = torch.cat((value.neu+g.delta(1.)+plus[rows]+jitter, value.neu+g.delta(-1.)+minus[rows]-jitter))
    torch.testing.assert_close(actual,old,rtol=0,atol=0)
    real = torch.cat((value.poles_p.expand(16,-1), value.poles_m.expand(16,-1)))
    old_parts = torch.cat((plus,minus))
    original_aux = value.spread(old_parts)+.02*old_parts.square().mean()+1.5*((value.neu+g.delta(1.)-value.poles_p).square().mean()+(value.neu+g.delta(-1.)-value.poles_m).square().mean())
    old_loss = value.loss.g_loss(value.D(old),value.D(real).detach())+original_aux
    new_loss = value.loss.g_loss(value.D(actual),value.D(real).detach())+value.auxiliary_loss()
    torch.testing.assert_close(old_loss,new_loss,rtol=0,atol=0)
    ga = torch.autograd.grad(old_loss,(g.w_odd,g.w_even,plus,minus))
    gb = torch.autograd.grad(new_loss,(value.G.w_odd,value.G.w_even,value.prior.z))
    for a,b in zip(ga,(gb[0],gb[1],gb[2][:12],gb[2][12:])): torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_two_public_updates_observer_parity_and_complete_resume():
    a, b = fixture(), fixture()
    for _ in range(2):
        assert a.step() == b.step(); b.observe()
    assert typed_state_digest(a.state_dict()) == typed_state_digest(b.state_dict())
    assert b.guards()["optimizer_updates"] == {"generator":2,"prior":2,"discriminator":2}
    assert b.guards()["all_finite"] and b.guards()["hooks_exercised"]
    assert all(p["pure"] for p in b.purity)
    saved=b.state_dict(); pin=typed_state_digest(saved); restored=fixture(); restored.load_state_dict(saved)
    assert typed_state_digest(restored.state_dict()) == pin == typed_state_digest(saved)
    assert restored.observe() == a.observe()
    assert all(torch.equal(restored.last_views[k],a.last_views[k]) for k in a.last_views)
    assert restored.step() == a.step()
    assert typed_state_digest(restored.state_dict()) == typed_state_digest(a.state_dict())
    assert typed_state_digest(saved) == pin


def test_cross_bank_counterfactual_keeps_labels_and_cannot_hide_conditioned_harm():
    value=fixture()
    with torch.no_grad():
        value.G.w_odd.copy_(value.poles_p-value.neu)
        value.prior.z.zero_(); value.prior.z[12:,0]=-1.
        value.policy.ema_G.load_state_dict(value.G.state_dict()); value.policy.averaged_table.copy_(value.prior.z)
    control=value.policy.routed_control; base=control.candidate(copy=True)
    changed=control._split(base,0,12,torch.zeros(4))
    assert torch.equal(value.router.bank_ids,torch.arange(24)//12)
    before=typed_state_digest(value.state_dict())
    x,y=control._measure(value.guard_context,value.guard_targets,base,with_output_error=True)
    u,v=control._measure(value.guard_context,value.guard_targets,changed,with_output_error=True)
    assert float((v-y).max().detach())>0
    assert typed_state_digest(value.state_dict())==before
    # This is a real package candidate/guard calculation, not evidence that
    # the live fit pool selected this proposal or a move occurred.


@pytest.mark.parametrize("field",["gate","budget","source","bank","recipe","observation"])
def test_changed_law_source_or_scientific_bounds_rejected(field):
    request, task=definition()
    if field=="gate":task["evaluation"]["thresholds"][0][2]=0.
    elif field=="budget":task["execution"]["steps"]=2
    elif field=="source":task["execution"]["policy_contract"]["sources"][declaration.HOST_SOURCE]="0"*64
    elif field=="bank":task["execution"]["policy_contract"]["banks"]["plus"]=[0,24]
    elif field=="recipe":request["candidate"]["recipe_overrides"]["prior_lr_mult"]=1.
    else:task["evaluation"]["policy_observation"]["weight_selector"]="fast"
    with pytest.raises(ValueError):CoverMultibankFixture(request,task)


@pytest.mark.parametrize("field",["family","clock","nan","bank","streams","mechanism"])
def test_checkpoint_failure_preserves_all_current_owners(field):
    value=fixture(); value.step(); saved=value.state_dict(); restored=fixture(); before=typed_state_digest(restored.state_dict())
    if field=="family":saved["family"]="atlas"
    elif field=="clock":saved["caller_cursor"]=0
    elif field=="nan":saved["policy"]["models"]["prior"]["z"][0,0]=float("nan")
    elif field=="bank":saved["policy"]["models"]["router"]["bank_ids"].zero_()
    elif field=="streams":saved["policy"]["streams"]["noise_generator"]=saved["policy"]["streams"]["eval_generator"].clone()
    else:saved["mechanism_audit_state"]["a2"]["calls"]=500
    with pytest.raises(ValueError):restored.load_state_dict(saved)
    assert typed_state_digest(restored.state_dict())==before

"""NEW CPU prerequisites only; exactly2+2 tiny public API updates in one case."""
from copy import deepcopy
import json
import math
from types import SimpleNamespace

import pytest
import torch
from examples import e22_routed_caption_late_phase as api


@pytest.fixture(autouse=True)
def caller():
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]): yield
    finally: torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def data():
    with torch.random.fork_rng(devices=[]): prototype = api.base.Backbone(api.base.SMALL)
    entries = deepcopy(api.stats.read_profile()["entries"])
    for name, parameter in prototype.named_parameters(): entries[name]["shape"] = list(parameter.shape)
    with torch.random.fork_rng(devices=[]):
        return api.stats.make_data(api.base.SMALL, software_entries=entries)[0]


def metric(value):
    return {"rmse": value, "by_source": {str(i): value for i in range(6)}}


def live():
    return {"bank_updates":2047, "query_updates":2047,
            "C_norms":{s:.1 for s in api.base.SITES}, "particle_Up_norms":{s:.1 for s in api.base.SITES}}


def test_fixed_horizon_cadence_source_and_card_contract():
    assert api.STEPS == 2048 and api.SECONDS == 900
    assert api.ENDPOINTS == (512,768,800,1024,1536,2048)
    assert api.MEDIA_STEPS == (0,256,512,768,800,1024,1536,2048)
    card = json.loads(api.CARD.read_text()); api.validate_card(card,api.source_identity())
    assert card["accuracy_thresholds"]["live_denominator"] == 2047
    assert "execution_authorized" not in card and "expected_native_sha256" not in card
    assert card["intended_forward_counts"]["mandatory_paired_residual_total"] == 12618
    assert card["intended_forward_counts"]["mandatory_source_and_teacher_backbone_total"] == 25236
    bad = deepcopy(card); bad["steps"] = 512
    with pytest.raises(ValueError): api.validate_card(bad,api.source_identity())


def test_same_frozen_profile_flow_units_and_learned_role_names(data):
    pools, anchors = api.flow.context_data(api.base.SMALL)
    assert api.base.digest(anchors) == api.base.digest(data["flow"])
    assert all(torch.equal(data[k]["context"],pools[k]["context"]) for k,_,_ in api.flow.POOLS)
    assert torch.equal(data["scale"],data["fit_baseline"].flatten(0,1).std(0,correction=1))
    assert data["frozen_stats"]["profile_sha256"] == api.base.sha(api.stats.PROFILE)
    for arm in api.common.ARMS:
        loop = api.common.make_loop(arm,data)
        assert torch.equal(loop.policy.D.scale,data["scale"])
        assert torch.equal(loop.policy.opt_d.ema_critic.scale,data["scale"])
        assert all(not p.requires_grad for p in loop.policy.G.teacher_backbone.parameters())


def test_public_initial_ownership_output_restore_and_tangent(data):
    before = api.base.digest(torch.get_rng_state()); proof = api.common.preflight(data)
    assert proof["pass"] and proof["native_updates"] == 0 and proof["fresh_public_restore_exact"]
    assert proof["initial_tangent"]["main_Up_gradient_matches_ordinary_exact"]
    assert all(v>0 for v in proof["initial_tangent"]["particle_Up_gradient_norms"].values())
    assert before == api.base.digest(torch.get_rng_state())


def test_only_two_plus_two_public_updates_restore_and_readonly_actual_phase(data):
    original = api.common.make_loop(api.common.UNTIED,data); zero = api.base.checkpoint(original)
    rows = [api.base.update(original) for _ in range(2)]; two = api.base.checkpoint(original)
    before = api.base.digest(two); telemetry = api.phase_observation(original.policy)
    assert api.base.digest(api.base.checkpoint(original)) == before
    assert telemetry["critic_calls"] == getattr(original.policy.opt_d.record,"calls",None)
    assert telemetry["critic_observed_steps"] == getattr(original.policy.opt_d.record,"observed_steps",None)
    assert telemetry["penalty_last_stats"] == api.scalar_observation(original.policy.penalty.last_stats)
    assert math.isfinite(telemetry["output_sigma"]) and telemetry["output_sigma"] > 0
    fresh = api.common.make_loop(api.common.UNTIED,data); api.base.restore(fresh,zero)
    repeated = [api.base.update(fresh) for _ in range(2)]
    assert api.base.digest(rows) == api.base.digest(repeated)
    assert api.base.digest(api.base.checkpoint(fresh)) == before
    api.base.learned_finite(fresh.policy)


def test_scalar_telemetry_nonfinite_placeholders_are_explicit_not_advancing_calls():
    record = SimpleNamespace(calls=815,observed_steps=814)
    policy = SimpleNamespace(penalty=SimpleNamespace(last_stats={"phase":"blend","w":torch.tensor(.7),"undefined":float("nan")}),
        opt_d=SimpleNamespace(record=record,param_groups=[{"lr":.01}]),output_sigma=lambda:torch.tensor(.125))
    observed = api.phase_observation(policy)
    assert (record.calls,record.observed_steps)==(815,814)
    assert observed["penalty_last_stats"]["undefined"] is None
    assert observed["null_placeholder_paths"]==[".penalty_last_stats.undefined"]
    assert observed["output_sigma"]==.125 and observed["critic_group_lrs"]==[.01]
    json.dumps(observed,allow_nan=False)
    policy.penalty.last_stats=None
    unavailable=api.phase_observation(policy)
    assert unavailable["penalty_last_stats"]=={} and not unavailable["availability"]["penalty_last_stats"]
    assert ".penalty_last_stats.unavailable" in unavailable["null_placeholder_paths"]
    with pytest.raises(ValueError): api.scalar_observation(torch.ones(2))


def test_phase_summary_uses_observed_label_not_hardcoded800():
    summary={"phase_counts":{},"first_blend_step":None,"last_observation":None}
    for step,phase in ((1,"a"),(798,"a"),(799,"blend"),(800,"blend"),(801,None)):
        api.add_phase(summary,step,{"penalty_last_stats":{} if phase is None else {"phase":phase}})
    assert summary["first_blend_step"]==799
    assert summary["phase_counts"]=={"a":2,"blend":2,"unavailable":1}


def test_terminal_gate_and_full_physical_scorer_oracles():
    assert all(api.controls().values())
    assert api.scientific_gate(metric(1),metric(1),metric(.999),metric(1.01),**live())["pass"]
    zero=api.base.accuracy(torch.zeros(6,2,4),range(6));one=api.base.accuracy(torch.ones(6,2,4),range(6))
    assert zero["rmse"]==0 and one["rmse"]==1


@pytest.mark.parametrize("kind",["ordinary","shared","one_source","code","live","C","Up"])
def test_terminal_destructive_controls_reject_each_bound(kind):
    ordinary,shared,candidate,zero=(metric(v) for v in (1.,1.,.99,1.01)); kwargs=live()
    if kind=="ordinary":ordinary=metric(.98)
    elif kind=="shared":shared=metric(.98)
    elif kind=="one_source":candidate["by_source"]["3"]=1.000002
    elif kind=="code":zero=metric(.99)
    elif kind=="live":kwargs["query_updates"]=511
    elif kind=="C":kwargs["C_norms"][api.base.SITES[0]]=0.
    elif kind=="Up":kwargs["particle_Up_norms"][api.base.SITES[0]]=0.
    assert not api.scientific_gate(ordinary,shared,candidate,zero,**kwargs)["pass"]


def test_invalid_metrics_counts_and_exclusive_output_rejected(tmp_path):
    bad=metric(float("nan"))
    with pytest.raises(ValueError):api.scientific_gate(metric(1),metric(1),bad,metric(1.01),**live())
    with pytest.raises(ValueError):api.scientific_gate(metric(1),metric(1),metric(.99),metric(1.01),**{**live(),"bank_updates":2048})
    old=tmp_path/"retained";old.mkdir();(old/"completion.json").write_text("retain\n")
    with pytest.raises(ValueError):api.units.fresh_output(old)
    assert (old/"completion.json").read_text()=="retain\n"

"""NEW flow contracts: reduced CPU geometry, exactly2+2 API updates total."""
from copy import deepcopy
import json
import math
import pytest
import torch
from examples import e22_routed_caption_flow as api


@pytest.fixture(autouse=True)
def caller():
    threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):yield
    finally:torch.set_num_threads(threads)


def test_gaussian_marginal_and_pair_covariance_are_exact_algebra():
    for t in (.1,.18,.22,.35,.43,.6,.68,.72,.85,.93):
        a,b=api.coefficients(t)
        assert a*a+b*b==pytest.approx(1,abs=3e-16)
        c,d=api.coefficients(t+.1)
        assert a*c+b*d==pytest.approx(.992,abs=3e-16)
    a,b=api.coefficients(.35);c,d=api.coefficients(.6)
    assert a*c+b*d==pytest.approx(api.LAW["expected_grid_adjacent_cosine"],abs=3e-16)
    assert .949<api.LAW["expected_grid_adjacent_cosine"]<.951


def test_fixed_independent_anchors_follow_original_cpu102_addressing():
    before=api.base.digest(torch.get_rng_state());pools,flow=api.context_data(api.base.SMALL)
    rng=torch.Generator().manual_seed(102);all_addresses=[]
    for pool,times,draws in api.POOLS:
        iid={}
        for source in range(6):
            for t in times:
                for draw in draws:iid[source,t,draw]=torch.randn(4,4,generator=rng)
            for draw in draws:
                key=f"{pool}:{source}:{draw}";pair=flow["anchors"][pool][key]
                assert torch.equal(pair["a"],iid[source,times[0],draw])
                assert torch.equal(pair["b"],iid[source,times[1],draw])
                all_addresses.extend(api.base.digest(v) for v in pair.values())
        assert len(flow["anchors"][pool])==api.LAW["independent_trajectories"][pool]
        expected=[(s,t,d) for s in range(6) for t in times for d in draws]
        assert [(r["source"],r["time"],r["draw"]) for r in flow["rows"][pool]]==expected
        assert pools[pool]["source_ids"]==[s for s,t,d in expected]
        for index,record in enumerate(flow["rows"][pool]):
            pair=flow["anchors"][pool][record["anchor_key"]]
            assert torch.equal(pools[pool]["context"][index,:,:4],api.mix(pair["a"],pair["b"],record["time"]))
            assert record["a_sha256"]==api.base.digest(pair["a"])
            assert record["b_sha256"]==api.base.digest(pair["b"])
        assert not pools[pool]["targets"].count_nonzero()
    assert len(all_addresses)==60 and len(set(all_addresses))==60
    assert torch.equal(flow["CPU102_after_draws"],rng.get_state())
    assert api.base.digest(torch.get_rng_state())==before


def test_initial_fit_anchor_exact_and_tensor_unit_oracle():
    a=torch.arange(16,dtype=torch.float32).reshape(4,4);b=torch.flip(a,(0,))
    assert torch.equal(api.mix(a,b,.1),a)
    ca,cb=api.coefficients(.35)
    assert torch.equal(api.mix(torch.ones(4,4),torch.zeros(4,4),.35),torch.full((4,4),ca))
    assert torch.equal(api.mix(torch.zeros(4,4),torch.ones(4,4),.35),torch.full((4,4),cb))
    pools,flow=api.context_data(api.base.SMALL)
    for i,r in enumerate(flow["rows"]["fit"]):
        if r["time"]==.1:assert torch.equal(pools["fit"]["context"][i,:,:4],flow["anchors"]["fit"][r["anchor_key"]]["a"])


@pytest.mark.parametrize("bad",[float("nan"),float("inf")])
def test_nonfinite_anchors_rejected(bad):
    with pytest.raises(ValueError):api.mix(torch.full((4,4),bad),torch.zeros(4,4),.35)


@pytest.fixture(scope="module")
def data():
    with torch.random.fork_rng(devices=[]):return api.make_data(api.base.SMALL)[0]


def test_context_change_recomputes_untrained_fit_and_shared_units(data):
    assert data["digest"]==api.units.data_digest(data)
    assert torch.equal(data["scale"],data["fit_baseline"].flatten(0,1).std(0,correction=1))
    assert torch.isfinite(data["scale"]).all() and (data["scale"]>1e-8).all()
    for arm in api.common.ARMS:
        loop=api.common.make_loop(arm,data)
        assert loop.policy.completed_steps==0
        assert torch.equal(loop.policy.D.scale,data["scale"])
        assert torch.equal(loop.policy.opt_d.ema_critic.scale,data["scale"])
    ordinary=api.common.make_loop(api.common.ARMS[0],data)
    assert torch.equal(api.base.observe(ordinary,data["fit"]["context"]),data["fit_baseline"])


def test_public_fresh_initializer_owner_and_tangent_proof(data):
    before=api.base.digest(torch.get_rng_state());proof=api.common.preflight(data)
    assert proof["pass"] and proof["native_updates"]==0 and proof["fresh_public_restore_exact"]
    assert proof["initial_tangent"]["main_Up_gradient_matches_ordinary_exact"]
    assert all(v>0 for v in proof["initial_tangent"]["particle_Up_gradient_norms"].values())
    assert api.base.digest(torch.get_rng_state())==before


def test_two_original_plus_two_public_restore_updates_only(data):
    original=api.common.make_loop(api.common.UNTIED,data);zero=api.base.checkpoint(original)
    rows=[api.base.update(original) for _ in range(2)];two=api.base.checkpoint(original)
    replay=api.common.make_loop(api.common.UNTIED,data);api.base.restore(replay,zero)
    repeated=[api.base.update(replay) for _ in range(2)]
    assert api.base.digest(rows)==api.base.digest(repeated)
    assert api.base.digest(api.base.checkpoint(replay))==api.base.digest(two)
    assert rows[-1]["bank_live"] and rows[-1]["query_live"]
    api.base.learned_finite(replay.policy)
    before=api.base.digest(api.base.checkpoint(replay));value=api.base.observe(replay,data["test"]["context"][:4])
    api.base.restore(replay,two)
    assert api.base.digest(api.base.checkpoint(replay))==before
    assert torch.equal(value,api.base.observe(replay,data["test"]["context"][:4]))


def test_metric_and_source_code_destructive_controls_are_unchanged():
    assert all(api.common.controls().values())
    metric=lambda v:{"rmse":v,"by_source":{str(s):v for s in range(6)}}
    kwargs={"bank_updates":511,"query_updates":511,"C_norms":{s:.1 for s in api.base.SITES},
            "particle_Up_norms":{s:.1 for s in api.base.SITES}}
    assert api.common.scientific_gate(metric(1),metric(1),metric(.999),metric(1.01),**kwargs)["pass"]
    harmed=metric(.999);harmed["by_source"]["5"]=1.000002
    assert not api.common.scientific_gate(metric(1),metric(1),harmed,metric(1.01),**kwargs)["pass"]
    assert not api.common.scientific_gate(metric(1),metric(1),metric(.999),metric(.999),**kwargs)["pass"]


def test_fixed_card_has_no_historical_failure_or_native_allowlist():
    card=json.loads(api.CARD.read_text());api.validate_card(card,api.source_identity())
    assert card["flow"]["anchor_time"]==.1 and card["steps"]==512
    assert "execution_authorized" not in card and "native_python_sha256" not in card
    assert card["flow"]["independent_trajectories"]=={"fit":12,"guard":6,"test":12}


def test_renderer_raw_oracle_and_false_verdict_refusal():
    from examples import render_e22_routed_caption_flow as renderer
    zero=torch.zeros(6,4,2);value=torch.ones_like(zero)
    image=renderer.goal_frame(zero,value,value*.5,value*.25,step=512,vmax=1.,
                              terminal={"scientific_status":"FAIL","ordinary":1.,"shared":.5,"untied":.25})
    assert [image.getpixel((252,y)) for y in (171,345,519,693)]==[(255,255,255),(255,0,0),(255,127,127),(255,191,191)]
    assert renderer.frames.token_maps(torch.tensor([3.,4.]).expand(6,4,2))==pytest.approx(math.sqrt(12.5),rel=1e-14)
    with pytest.raises(ValueError):renderer.validate_observations({"task":"wrong"},{},{},{},{})


def test_exclusive_output_preserves_prior_bytes(tmp_path):
    run=tmp_path/"old";run.mkdir();(run/"completion.json").write_text("retained\n")
    with pytest.raises(ValueError):api.units.fresh_output(run)
    assert (run/"completion.json").read_text()=="retained\n"

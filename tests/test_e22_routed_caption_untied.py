"""Software only: four tiny native updates total, no fullhost/CUDA/quality."""
import json
import math
import sys
import pytest
import torch
from examples import e22_routed_caption_untied as api


@pytest.fixture(autouse=True)
def cpu_owner():
    threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):yield
    finally:torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def data():
    with torch.random.fork_rng(devices=[]):return api.base.make_data(api.base.SMALL,torch.device("cpu"))


def metric(v):return {"rmse":v,"by_source":{str(i):v for i in range(6)}}


def gate(ordinary=None,shared=None,untied=None,zero=None,**changes):
    return api.scientific_gate(*(metric(1) if v is None else v for v in (ordinary,shared)),metric(.99) if untied is None else untied,metric(1) if zero is None else zero,
        **{"bank_updates":511,"query_updates":511,"C_norms":{s:.1 for s in api.base.SITES},"particle_Up_norms":{s:.1 for s in api.base.SITES},**changes})


def test_scorer_positive_and_destructive_controls():
    assert all(api.controls().values()) and gate()["pass"]
    assert not gate(shared=metric(.98))["pass"]
    assert not gate(ordinary=metric(.98))["pass"]
    assert not gate(untied=metric(1))["pass"]


def test_both_controls_inclusive_bounds_and_strict_source_code_benefit():
    candidate=metric(.999);zero=metric(.999*1.001)
    assert gate(untied=candidate,zero=zero)["pass"]
    candidate=metric(.99);candidate["by_source"]["3"]=1+1e-6
    assert gate(untied=candidate,zero=metric(1.01))["pass"]
    candidate["by_source"]["3"]+=1e-12
    assert not gate(untied=candidate,zero=metric(1.01))["pass"]
    zero=metric(1);zero["by_source"]["4"] = .99
    assert not gate(zero=zero)["pass"]


@pytest.mark.parametrize("role",["bank","query"])
def test_live_denominator_is_511(role):
    assert gate(**{role+"_updates":460})["pass"]
    assert not gate(**{role+"_updates":459})["pass"]


@pytest.mark.parametrize("role",["C_norms","particle_Up_norms"])
@pytest.mark.parametrize("bad",[0.,float("nan"),float("inf")])
def test_every_conditioning_site_must_remain_live(role,bad):
    norms={s:.1 for s in api.base.SITES};norms[api.base.SITES[2]]=bad
    assert not gate(**{role:norms})["pass"]


def test_raw_f64_oracle_and_invalid_inputs():
    residual=torch.arange(1,13,dtype=torch.float64).reshape(6,2,1)/13
    result=api.base.accuracy(residual,torch.arange(6))
    assert result["rmse"]==pytest.approx(math.sqrt(sum(float(v)**2 for v in residual.flatten())/12),rel=1e-14)
    with pytest.raises(ValueError):api.base.accuracy(torch.full((6,2,1),float("nan")),torch.arange(6))
    invalid=metric(.99);invalid["by_source"].pop("5")
    with pytest.raises(ValueError):gate(untied=invalid)


def test_public_initial_tangent_and_fresh_restore_proof(data):
    before=api.base.digest(torch.get_rng_state());proof=api.preflight(data)
    assert proof["native_updates"]==0 and proof["initial_tangent"]["main_Up_gradient_matches_ordinary_exact"]
    assert all(v>0 for v in proof["initial_tangent"]["particle_Up_gradient_norms"].values())
    assert api.base.digest(torch.get_rng_state())==before
    assert api.counts(api.base.FULL)["untied_extra_Up"]==82944
    assert api.counts(api.base.FULL)["untied_particle_total"]==241400


def test_sampled_C_and_both_registered_heads_precede_EMA_and_optimizer(data):
    loop=api.make_loop(api.UNTIED,data);p=loop.policy
    optimized={id(v) for group in p.opt_g.param_groups for v in group["params"]}
    for fast,ema in zip(p.G.branches(),p.ema_G.branches()):
        assert isinstance(fast,api.UntiedAdapter) and isinstance(ema,api.UntiedAdapter)
        for name in ("up","particle_up"):
            a,b=getattr(fast,name).weight,getattr(ema,name).weight
            assert isinstance(a,torch.nn.Parameter) and id(a) in optimized
            assert a.data_ptr()!=b.data_ptr() and not a.count_nonzero() and torch.equal(a,b)
        assert fast.bridge.weight[:,data["geometry"].rank:].count_nonzero()
        assert not fast.bridge.weight[:,:data["geometry"].rank].count_nonzero() and not fast.bridge.bias.count_nonzero()
    x=data["test"]["context"][:4]
    with torch.no_grad():before=p.routed_generate(x,sigma=0,perturb=False,averaged=True).clone()
    with torch.no_grad():p.G.branches()[0].up.weight.fill_(.5)
    with torch.no_grad():
        assert torch.equal(before,p.routed_generate(x,sigma=0,perturb=False,averaged=True))
        assert not torch.equal(before,p.routed_generate(x,sigma=0,perturb=False))


def test_all_three_branch_matmuls_are_BF16(data):
    loop=api.make_loop(api.UNTIED,data);seen=[];hooks=[]
    try:
        for branch in loop.policy.G.branches():
            for layer in (branch.down,branch.up,branch.particle_up):
                hooks.append(layer.register_forward_hook(lambda m,args,out:seen.append(out.dtype)))
        api.base.observe(loop,data["test"]["context"][:4])
    finally:
        for hook in hooks:hook.remove()
    assert seen==[torch.bfloat16]*18


def test_zero_C_is_a_dead_conditional_destructive_control(data):
    loop=api.make_loop(api.UNTIED,data,software_C_zero=True);p=loop.policy
    x=data["fit"]["context"][:4];c=p.encoder.condition(x)
    panel=.125*torch.randn(4,data["geometry"].tokens,data["geometry"].output,generator=torch.Generator().manual_seed(72))
    residual=p.routed_generate(x,sigma=0,perturb=True)
    with torch.no_grad():real=p.D(panel,c)
    loss=p.recipe.make_loss().g_loss(p.D(panel+residual/p.D.scale,c),real)
    gradients=torch.autograd.grad(loss,[b.particle_up.weight for b in p.G.branches()]+[p.table],allow_unused=True)
    assert all(v is None or not v.count_nonzero() for v in gradients)
    assert p.completed_steps==0


def test_exact_four_update_public_restore_without_reinitializing(data,monkeypatch):
    """Two original plus two replay updates; the only software updates."""
    loop=api.make_loop(api.UNTIED,data);initial=api.base.checkpoint(loop)
    rows=[api.base.update(loop) for _ in range(2)];trained=api.base.checkpoint(loop)
    assert rows[-1]["bank_live"] and rows[-1]["query_live"]
    assert all(b.particle_up.weight.count_nonzero() for b in loop.policy.G.branches())
    replay=api.make_loop(api.UNTIED,data);api.base.restore(replay,initial)
    repeated=[api.base.update(replay) for _ in range(2)]
    assert api.base.digest(rows)==api.base.digest(repeated)
    assert api.base.digest(api.base.checkpoint(replay))==api.base.digest(trained)
    monkeypatch.setattr(api.init,"initialize_",lambda *a,**k:pytest.fail("trained restore reinitialized owners"))
    api.base.restore(replay,trained)
    assert all(isinstance(b,api.UntiedAdapter) for b in replay.policy.G.branches())
    assert api.base.digest(api.base.checkpoint(replay))==api.base.digest(trained)
    api.base.learned_finite(replay.policy)
    before=api.base.digest(api.base.checkpoint(replay))
    clean=api.base.observe(replay,data["test"]["context"][:4]);zero=api.base.observe(replay,data["test"]["context"][:4],zero_code=True)
    assert torch.isfinite(clean).all() and torch.isfinite(zero).all()
    assert api.base.digest(api.base.checkpoint(replay))==before


def test_science_output_refusal_preserves_prior_receipt(tmp_path,monkeypatch):
    out=tmp_path/"retained";out.mkdir();receipt=out/"completion.json";receipt.write_text("previous bytes\n")
    before=receipt.read_bytes();monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(api.base,"global_rng",lambda *a:pytest.fail("refusal initialized CUDA"))
    monkeypatch.setattr(sys,"argv",["untied","--run","--out",str(out)])
    assert api.main()==2 and receipt.read_bytes()==before


def test_four_row_goal_frame_uses_initial_scale_and_every_coordinate():
    from examples import render_e22_routed_caption_untied as media
    z=torch.zeros(6,4,2);a=torch.ones_like(z)
    frame=media.goal_frame(z,a,a*.5,a*.25,step=512,vmax=1.,terminal={"scientific_status":"FAIL","ordinary":1.,"shared":.5,"untied":.25})
    assert [frame.getpixel((252,y)) for y in (171,345,519,693)]==[(255,255,255),(255,0,0),(255,127,127),(255,191,191)]
    assert frame.size==(1230,860)
    maps=media.token_maps(torch.tensor([3.,4.]).expand(6,4,2))
    assert maps==pytest.approx(math.sqrt(12.5),rel=1e-14)


def test_media_refusal_leaves_previous_bytes_untouched(tmp_path,monkeypatch):
    from examples import render_e22_routed_caption_untied as media
    receipt=tmp_path/"media-completion.json";receipt.write_text("previous bytes\n")
    monkeypatch.setattr(sys,"argv",["render","--run-directory",str(tmp_path)])
    before=receipt.read_bytes()
    assert media.main()==2 and receipt.read_bytes()==before


def test_new_variant_card_is_explicit_and_future_api_reusable():
    card=json.loads(api.CARD.read_text())
    assert card["arms"]==list(api.ARMS) and card["steps"]==512 and card["seconds"]==300
    assert card["prior"]["sigma"]==0 and card["media_steps"]==list(api.base.MEDIA_STEPS)
    assert "native_python_sha256" not in card and "execution_authorized" not in card
    assert "82944" in card["changed_factors"][0]

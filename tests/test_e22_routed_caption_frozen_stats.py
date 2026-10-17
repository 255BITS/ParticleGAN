"""NEW reduced CPU prerequisites; exactly four tiny public native updates."""
from copy import deepcopy
import hashlib
import json
import math
import pytest
import torch
from torch import nn
from examples import e22_routed_caption_frozen_stats as api


@pytest.fixture(autouse=True)
def caller():
    threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):yield
    finally:torch.set_num_threads(threads)


def test_profile_all29_full_shapes_scalar_source_mapping_and_moments():
    profile=api.read_profile();entries=profile["entries"]
    assert len(entries)==29 and entries["pos_embed"]["shape"]==[1,256,576]
    assert entries["ctx_proj.weight"]["shape"]==[576,768]
    assert entries["block.cross_attn.kv.weight"]["shape"]==[1152,576]
    for name,item in entries.items():
        assert item["actual_source_key"].replace("blocks.0.","block.",1)==name
        assert item["population_std"]>0
        assert item["RMS_descriptive"]**2==pytest.approx(item["mean"]**2+item["population_std"]**2,rel=3e-15)
    assert profile["source_capsule_sha256"]=="fc18f90e1422a4a03287e368071cd71cab0d38c4e6cba1337c001b2d82598afc"


def test_distribution_theoretical_moments_and_families_without_sampling():
    entries=api.read_profile()["entries"];specs=api.distributions(entries)
    for name,item in entries.items():
        spec=specs[name]
        if name=="pos_embed":
            assert isinstance(spec,api.base.init.Normal)
            assert (spec.mean,spec.std)==(item["mean"],item["population_std"])
        else:
            assert isinstance(spec,api.base.init.Uniform)
            assert (spec.low+spec.high)/2==pytest.approx(item["mean"],abs=3e-17)
            assert (spec.high-spec.low)/math.sqrt(12)==pytest.approx(item["population_std"],rel=3e-16)


def tiny_entries():
    return {"weight":{"shape":[2,3],"mean":.04,"population_std":.13,"distribution":"Uniform"},
            "bias":{"shape":[2],"mean":-.1,"population_std":.07,"distribution":"Uniform"}}


def test_public_local_override_exact_streams_owners_and_freeze_not_registry():
    model=nn.Linear(3,2);reference=nn.Linear(3,2);entries=tiny_entries()
    ids={n:id(p) for n,p in model.named_parameters()};before=torch.get_rng_state().clone()
    registry=api.base.init.declarations(model)
    streams={n:torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(
        f"routed-caption-geometry-v1:frozen_backbone:{n}".encode()).digest()[:8],"little")%(2**63-1)) for n in entries}
    api.base.init.initialize_(reference,method="sample_distributions_v1",parameter_generators=streams,
                              distributions=api.distributions(entries))
    realized=api.initialize_frozen(model,entries)
    assert all(torch.equal(p,dict(reference.named_parameters())[n]) for n,p in model.named_parameters())
    assert ids=={n:id(p) for n,p in model.named_parameters()}
    assert all(not p.requires_grad for p in model.parameters()) and torch.equal(before,torch.get_rng_state())
    assert api.base.init.declarations(nn.Linear(3,2))==registry
    for name,p in model.named_parameters():
        x=p.double();assert realized[name]["mean"]==float(x.mean())
        assert realized[name]["population_std"]==float(x.std(correction=0))
    # Realized moments are observations, not forcibly corrected to requested moments.
    assert realized["bias"]["population_std"]!=entries["bias"]["population_std"]


@pytest.mark.parametrize("field,value",[("mean",float("nan")),("population_std",float("inf")),
                                        ("population_std",0.),("population_std",-.1)])
def test_bad_profile_rejected_before_any_parameter_mutation(field,value):
    model=nn.Linear(3,2);entries=tiny_entries();entries["weight"][field]=value
    before=api.base.digest(model.state_dict())
    with pytest.raises(ValueError):api.initialize_frozen(model,entries)
    assert api.base.digest(model.state_dict())==before and all(p.requires_grad for p in model.parameters())


def test_wrong_shape_frozen_owner_and_reduced_production_rejected():
    model=nn.Linear(3,2);entries=tiny_entries();entries["weight"]["shape"]=[3,2]
    before=api.base.digest(model.state_dict())
    with pytest.raises(ValueError):api.initialize_frozen(model,entries)
    model.requires_grad_(False)
    with pytest.raises(ValueError):api.initialize_frozen(model,tiny_entries())
    assert api.base.digest(model.state_dict())==before
    with pytest.raises(ValueError):api.make_data(api.base.SMALL)


@pytest.fixture(scope="module")
def data():
    # Scalar coefficients unchanged; only reduced shape metadata is software-only.
    with torch.random.fork_rng(devices=[]): prototype=api.base.Backbone(api.base.SMALL)
    entries=deepcopy(api.read_profile()["entries"])
    for name,p in prototype.named_parameters():entries[name]["shape"]=list(p.shape)
    with torch.random.fork_rng(devices=[]):return api.make_data(api.base.SMALL,software_entries=entries)[0]


def test_paired_host_public_init_skips_frozen_values_units_and_flow(data):
    c=data["frozen_stats"]
    assert c["software_only_reduced_shapes"] and c["fresh_initialize_before_freeze"]
    assert c["paired_fixed_owners_unchanged_during_initialization_and_FIT"]
    assert api.base.digest(data["frozen"])==c["frozen_state_digest"]
    pools,flow=api.flow.context_data(api.base.SMALL)
    assert api.base.digest(flow)==api.base.digest(data["flow"])
    assert all(torch.equal(data[p]["context"],pools[p]["context"]) for p,_,_ in api.flow.POOLS)
    assert data["digest"]==api.units.data_digest(data)
    assert torch.equal(data["scale"],data["fit_baseline"].flatten(0,1).std(0,correction=1))
    for arm in api.common.ARMS:
        loop=api.common.make_loop(arm,data)
        assert torch.equal(loop.policy.D.scale,data["scale"])
        assert torch.equal(loop.policy.opt_d.ema_critic.scale,data["scale"])
        assert all(not p.requires_grad for p in loop.policy.G.teacher_backbone.parameters())


def test_public_zero_update_ownership_restore_and_tangent(data):
    before=api.base.digest(torch.get_rng_state());proof=api.common.preflight(data)
    assert proof["pass"] and proof["native_updates"]==0 and proof["fresh_public_restore_exact"]
    assert proof["initial_tangent"]["main_Up_gradient_matches_ordinary_exact"]
    assert all(v>0 for v in proof["initial_tangent"]["particle_Up_gradient_norms"].values())
    assert before==api.base.digest(torch.get_rng_state())


def test_only_two_plus_two_public_restore_native_updates(data):
    original=api.common.make_loop(api.common.UNTIED,data);zero=api.base.checkpoint(original)
    rows=[api.base.update(original) for _ in range(2)];two=api.base.checkpoint(original)
    replay=api.common.make_loop(api.common.UNTIED,data);api.base.restore(replay,zero)
    repeated=[api.base.update(replay) for _ in range(2)]
    assert api.base.digest(rows)==api.base.digest(repeated)
    assert api.base.digest(api.base.checkpoint(replay))==api.base.digest(two)
    assert rows[-1]["bank_live"] and rows[-1]["query_live"]
    api.base.learned_finite(replay.policy)


def test_unchanged_metric_oracles_destructive_source_code_controls():
    assert all(api.common.controls().values())
    metric=lambda v:{"rmse":v,"by_source":{str(s):v for s in range(6)}}
    kwargs={"bank_updates":511,"query_updates":511,"C_norms":{s:.1 for s in api.base.SITES},
            "particle_Up_norms":{s:.1 for s in api.base.SITES}}
    assert api.common.scientific_gate(metric(1),metric(1),metric(.999),metric(1.01),**kwargs)["pass"]
    harmed=metric(.999);harmed["by_source"]["5"]=1.000002
    assert not api.common.scientific_gate(metric(1),metric(1),harmed,metric(1.01),**kwargs)["pass"]
    assert not api.common.scientific_gate(metric(1),metric(1),metric(.999),metric(.999),**kwargs)["pass"]


def test_fixed_card_profile_identity_gate_and_future_API_contract():
    card=json.loads(api.CARD.read_text());api.validate_card(card,api.source_identity())
    assert card["steps"]==512 and card["seconds"]==300
    assert card["profile_sha256"]==api.base.sha(api.PROFILE)
    assert "execution_authorized" not in card and "native_python_sha256" not in card
    bad=deepcopy(card);bad["profile_sha256"]="wrong"
    with pytest.raises(ValueError):api.validate_card(bad,api.source_identity())


def test_renderer_raw_oracle_false_identity_and_exclusive_outputs(tmp_path):
    from examples import render_e22_routed_caption_frozen_stats as renderer
    zero=torch.zeros(6,4,2);value=torch.ones_like(zero)
    image=renderer.goal_frame(zero,value,value*.5,value*.25,step=512,vmax=1.,
                             terminal={"scientific_status":"FAIL","ordinary":1.,"shared":.5,"untied":.25})
    assert [image.getpixel((252,y)) for y in (171,345,519,693)]==[(255,255,255),(255,0,0),(255,127,127),(255,191,191)]
    assert renderer.frames.token_maps(torch.tensor([3.,4.]).expand(6,4,2))==pytest.approx(math.sqrt(12.5),rel=1e-14)
    with pytest.raises(ValueError):renderer.validate_observations({"task":"wrong"},{},{},{},{})
    run=tmp_path/"old";run.mkdir();(run/"completion.json").write_text("retained\n")
    with pytest.raises(ValueError):api.units.fresh_output(run)
    assert (run/"completion.json").read_text()=="retained\n"

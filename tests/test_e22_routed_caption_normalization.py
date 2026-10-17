"""New normalization contracts only; CPU<=60s, at most2+2 tiny native updates."""
from dataclasses import asdict
import json
import math
import pytest
import torch
from examples import e22_routed_caption_normalization as api


@pytest.fixture(autouse=True)
def cpu_owner():
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]): yield
    finally: torch.set_num_threads(threads)


def synthetic_data(residual):
    g = api.base.SMALL
    data = {"geometry": g, "fit_baseline": residual,
            "fit": {"source_ids": list(range(len(residual))), "context": torch.zeros(*residual.shape[:2], g.output+2)},
            "scale": residual.flatten(0,1).std(0).clamp_min(.04), "unchanged": torch.tensor([.2,.3])}
    data["digest"] = api.data_digest(data)
    return data


@pytest.fixture(scope="module")
def data():
    with torch.random.fork_rng(devices=[]):
        original = api.base.make_data(api.base.SMALL,torch.device("cpu"))
        return api.normalize_data(original)[0]


def test_std_is_untrained_fit_only_and_other_fields_are_unchanged():
    residual = torch.arange(96,dtype=torch.float32).reshape(6,4,4)/10000
    original = synthetic_data(residual); before = api.base.digest(original["scale"])
    changed, evidence = api.normalize_data(original)
    assert torch.equal(changed["scale"],residual.flatten(0,1).std(0,correction=1))
    assert evidence["epsilon_floor_fraction"] == 0 and evidence["previous_floor_fraction"] == 1
    assert evidence["only_scale_and_derived_digest_changed"]
    assert changed["fit"] is original["fit"] and changed["fit_baseline"] is original["fit_baseline"]
    assert changed["unchanged"] is original["unchanged"]
    assert changed["digest"] != original["digest"] and original["digest"] == api.data_digest(original)
    assert before == api.base.digest(original["scale"])


@pytest.mark.parametrize("bad",[0.,1e-12,float("nan"),float("inf")])
def test_degenerate_or_nonfinite_coordinate_refused_before_science(bad):
    residual = torch.arange(96,dtype=torch.float32).reshape(6,4,4)/100
    residual[:,:,2] = bad
    with pytest.raises(ValueError): api.normalize_data(synthetic_data(residual))


def test_nonzero_but_near_zero_std_is_refused():
    residual = torch.arange(96,dtype=torch.float32).reshape(6,4,4)/100
    residual[:,:,2] = torch.arange(24).reshape(6,4)*1e-10
    with pytest.raises(ValueError,match="near-zero"): api.normalize_data(synthetic_data(residual))


def test_fixed_scale_provenance_rejects_silent_input_or_floor_change():
    original = synthetic_data(torch.arange(96,dtype=torch.float32).reshape(6,4,4)/100)
    original["scale"] = original["scale"] * 2
    with pytest.raises(ValueError,match="digest"): api.normalize_data(original)
    original["digest"] = api.data_digest(original)
    with pytest.raises(ValueError,match="floor law"): api.normalize_data(original)


def test_power_decomposition_uses_physical_coordinates_in_f64():
    value = torch.tensor([[[1.,3.],[3.,7.]],[[2.,4.],[4.,8.]]],dtype=torch.float64)
    result = api.power_decomposition(value)
    direct = sum(float(v)**2 for v in value.flatten())/value.numel()
    assert result["power"] == direct
    assert result["power"] == pytest.approx(result["token_mean_power"]+result["token_centered_power"],rel=1e-14)
    assert result["rms"] == math.sqrt(direct)


def test_frozen_scale_reaches_all_three_D_and_EMA_owners(data):
    for arm in api.common.ARMS:
        loop = api.common.make_loop(arm,data)
        assert torch.equal(loop.policy.D.scale,data["scale"])
        assert torch.equal(loop.policy.opt_d.ema_critic.scale,data["scale"])
        assert api.base.digest(loop.data["scale"]) == api.base.digest(data["scale"])
        assert loop.policy.completed_steps == 0


def test_new_scale_has_public_initial_owner_restore_and_tangent_proof(data):
    before = api.base.digest(torch.get_rng_state())
    proof = api.common.preflight(data)
    assert proof["pass"] and proof["native_updates"] == 0 and proof["fresh_public_restore_exact"]
    assert proof["initial_tangent"]["main_Up_gradient_matches_ordinary_exact"]
    assert all(v>0 for v in proof["initial_tangent"]["particle_Up_gradient_norms"].values())
    assert api.base.digest(torch.get_rng_state()) == before


def test_two_original_plus_two_fresh_public_replay_updates_only(data):
    original = api.common.make_loop(api.common.UNTIED,data)
    state0 = api.base.checkpoint(original)
    rows = [api.base.update(original) for _ in range(2)]
    state2 = api.base.checkpoint(original)
    replay = api.common.make_loop(api.common.UNTIED,data)
    api.base.restore(replay,state0)
    repeated = [api.base.update(replay) for _ in range(2)]
    assert api.base.digest(rows) == api.base.digest(repeated)
    assert api.base.digest(api.base.checkpoint(replay)) == api.base.digest(state2)
    assert rows[-1]["bank_live"] and rows[-1]["query_live"]
    api.base.learned_finite(replay.policy)
    before = api.base.digest(api.base.checkpoint(replay))
    prediction = api.base.observe(replay,data["test"]["context"][:4])
    api.base.restore(replay,state2)
    assert api.base.digest(api.base.checkpoint(replay)) == before
    assert torch.equal(prediction,api.base.observe(replay,data["test"]["context"][:4]))


def test_unchanged_both_control_metric_has_positive_and_destructive_oracles():
    assert all(api.common.controls().values())
    metric = lambda v: {"rmse":v,"by_source":{str(i):v for i in range(6)}}
    kwargs = {"bank_updates":511,"query_updates":511,"C_norms":{s:.1 for s in api.base.SITES},
              "particle_Up_norms":{s:.1 for s in api.base.SITES}}
    assert api.common.scientific_gate(metric(1),metric(1),metric(.999),metric(.999*1.001),**kwargs)["pass"]
    assert not api.common.scientific_gate(metric(1),metric(.998),metric(.999),metric(1.01),**kwargs)["pass"]
    assert not api.common.scientific_gate(metric(1),metric(1),metric(.999),metric(.999),**kwargs)["pass"]


def test_exclusive_output_preserves_prior_bytes(tmp_path):
    old = tmp_path/"old"; old.mkdir(); receipt = old/"completion.json"; receipt.write_text("retained bytes\n")
    with pytest.raises(ValueError): api.fresh_output(old)
    assert receipt.read_text() == "retained bytes\n"


def test_new_card_reuses_public_helpers_without_version_allowlist():
    card = json.loads(api.CARD.read_text())
    api.validate_card(card,api.source_identity())
    assert card["task"] != api.common.TASK and card["execution_helper_task"] == api.common.TASK
    assert card["normalization"]["epsilon"] == 1e-8
    assert "execution_authorized" not in card and "native_python_sha256" not in card
    assert "--run" in card["cli"] and card["steps"] == 512


def test_new_renderer_reuses_only_pure_frame_math_and_raw_coordinates():
    from examples import render_e22_routed_caption_normalization as media
    zero = torch.zeros(6,4,2); value = torch.ones_like(zero)
    frame = media.goal_frame(zero,value,value*.5,value*.25,step=512,vmax=1.,
                             terminal={"scientific_status":"FAIL","ordinary":1.,"shared":.5,"untied":.25})
    assert [frame.getpixel((252,y)) for y in (171,345,519,693)] == [(255,255,255),(255,0,0),(255,127,127),(255,191,191)]
    assert media.frames.token_maps(torch.tensor([3.,4.]).expand(6,4,2)) == pytest.approx(math.sqrt(12.5),rel=1e-14)


def test_new_renderer_rejects_false_verdict_and_invalid_terminal_label():
    from copy import deepcopy
    from examples import render_e22_routed_caption_normalization as renderer
    hashes = {"report_sha256":"report","protocol_sha256":"card","observed_media_sha256":"media"}
    card = {"task":api.TASK,"sources":{},"normalization":api.NORMALIZATION,"media_indices":list(api.base.MEDIA_INDICES)}
    report = {"task":api.TASK,"complete":True,"scientific_status":"PASS","gate":{"pass":True},
              "protocol_sha256":"card","observed_media_sha256":"media","source_identity":{},
              "normalization":{"law":api.NORMALIZATION,"epsilon_floor_fraction":0,"only_scale_and_derived_digest_changed":True},
              "accuracy":{arm:{"rmse":.1} for arm in api.common.ARMS}}
    receipt = {"task":api.TASK,"complete":True,"scientific_status":"PASS","protocol_sha256":"card",
               "report_sha256":"report","source_identity":{}}
    value = torch.ones(8,256,16)
    media = {"steps":api.base.MEDIA_STEPS,"indices":api.base.MEDIA_INDICES,"source_ids":[0,1,2,3,4,5,0,1],
             "target_residual":torch.zeros_like(value),"capture_native_state_rng_diagnostics_unchanged":True,
             "actual_residuals":{arm:{str(s):value for s in api.base.MEDIA_STEPS} for arm in api.common.ARMS}}
    renderer.validate_observations(report,receipt,media,hashes,card)
    wrong = deepcopy(report); wrong["gate"]["pass"] = False
    with pytest.raises(ValueError):renderer.validate_observations(wrong,receipt,media,hashes,card)
    wrong = deepcopy(report); wrong["accuracy"][api.common.UNTIED]["rmse"] = float("nan")
    with pytest.raises(ValueError):renderer.validate_observations(wrong,receipt,media,hashes,card)

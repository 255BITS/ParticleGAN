"""NEW raw-observation renderer controls; no model/API/native updates."""
from copy import deepcopy
import json
import math

import pytest
import torch
from examples import render_e22_routed_caption_late_phase as render


@pytest.fixture(autouse=True)
def cpu_threads():
    threads=torch.get_num_threads();torch.set_num_threads(1)
    render.STARTED = render.time.monotonic()
    try:yield
    finally:torch.set_num_threads(threads)


def artifacts():
    ids=[s for s in range(6) for _ in range(8)];indices=(0,8,16,24,32,40,1,9)
    values={a:torch.full((48,256,16),v,dtype=torch.float64) for a,v in zip(render.ARMS,(1.,.9,.8))}
    def metric(v):
        powers=v.square().flatten(1).mean(1);labels=torch.tensor(ids)
        return {"rmse":float(powers.mean().sqrt()),"by_source":{str(s):float(powers[labels==s].mean().sqrt()) for s in range(6)},
                "contexts":48,"coordinates_per_context":4096,"source_counts":{str(s):8 for s in range(6)}}
    media={"steps":render.STEPS,"indices":indices,"source_ids":[ids[i] for i in indices],
           "target_residual":torch.zeros(8,256,16),"capture_native_state_rng_diagnostics_unchanged":True,
           "actual_residuals":{a:{str(s):v[list(indices)] for s in render.STEPS} for a,v in values.items()}}
    curves={"steps":render.ENDPOINTS,"source_ids":ids,"physical_residuals":{a:{str(s):v for s in render.ENDPOINTS} for a,v in values.items()}}
    hashes={"report_sha256":"r","protocol_sha256":"p","observed_media_sha256":"m","curve_residual_sha256":"c"}
    report={"task":render.TASK,"complete":True,"scientific_status":"PASS","gate":{"pass":True},"quality_updates":6144,"replay_updates":0,
            "media_steps":render.STEPS,"endpoint_steps":render.ENDPOINTS,"live":{"denominator":2047},
            "source_identity":{"new":"sha"},"imported_package":{"sha":"observed"},"imported_package_unchanged":True,
            "protocol_sha256":"p","observed_media_sha256":"m","curve_residual_sha256":"c",
            "curves":{a:{str(s):metric(v) for s in render.ENDPOINTS} for a,v in values.items()},"accuracy":{a:metric(v) for a,v in values.items()}}
    receipt={"task":render.TASK,"complete":True,"scientific_status":"PASS","report_sha256":"r","protocol_sha256":"p",
             "source_identity":report["source_identity"],"imported_package":report["imported_package"]}
    card={"task":render.TASK,"steps":2048,"media_steps":render.STEPS,"endpoint_steps":render.ENDPOINTS,"media_indices":indices,
          "accuracy_thresholds":{"live_denominator":2047},"sources":report["source_identity"]}
    return report,receipt,media,curves,hashes,card


def test_full_retained_list_source_schema_and_destructive_status_or_camera():
    items=artifacts();render.validate_observations(*items)
    bad=deepcopy(items);bad[0]["scientific_status"]="FAIL"
    with pytest.raises(ValueError):render.validate_observations(*bad)
    bad=deepcopy(items);bad[2]["actual_residuals"][render.ARMS[0]]["800"][0].add_(.1)
    with pytest.raises(ValueError):render.validate_observations(*bad)
    bad=deepcopy(items);bad[0]["accuracy"][render.ARMS[0]]["rmse"]=float("nan")
    with pytest.raises(ValueError):render.validate_observations(*bad)


def test_raw_patch_RMS_and_color_oracle_without_plotting_import():
    zero=torch.zeros(6,4,2);one=torch.ones_like(zero)
    image=render.goal_frame(zero,one,one*.5,one*.25,step=2048,vmax=1.,
        terminal={"scientific_status":"FAIL","ordinary":1.,"shared":.5,"untied":.25})
    assert [image.getpixel((252,y)) for y in (171,345,519,693)]==[(255,255,255),(255,0,0),(255,127,127),(255,191,191)]
    actual=render.frames.token_maps(torch.tensor([3.,4.]).expand(6,4,2))
    assert actual==pytest.approx(math.sqrt(12.5),rel=1e-14)


def test_complete_trace_hash_phase_counts_use_actual_labels(tmp_path):
    report={"trace_sha256":{},"phase_summary":{}}
    for arm in render.ARMS:
        rows=[{"step":s,"loss_g":.5,"loss_d_game":.6,
               "phase_observation":{"penalty_last_stats":{"phase":"a" if s<37 else "blend"}}} for s in range(1,2049)]
        path=tmp_path/f"{arm}.jsonl";path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        report["trace_sha256"][arm]=render.frames.sha(path)
        report["phase_summary"][arm]={"phase_counts":{"a":36,"blend":2012},"first_blend_step":37,
             "actual_blend_observed":True,"last_observation":rows[-1]["phase_observation"]}
    records,_=render.read_traces(tmp_path,report)
    assert all(len(v)==2048 for v in records.values())
    bad=deepcopy(report);bad["phase_summary"][render.ARMS[0]]["first_blend_step"]=800
    with pytest.raises(ValueError):render.read_traces(tmp_path,bad)

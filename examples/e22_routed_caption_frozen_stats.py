"""Frozen scalar-moment variant; entirely asset-free public ParticleGAN execution.

CUDA_VISIBLE_DEVICES=0 python -m examples.e22_routed_caption_frozen_stats --run --out runs/caption-frozen-stats-v1
Exit0=completed PASS,1=completed FAIL,2=incomplete/error. No quality-based tuning.
"""
import time
STARTED = time.monotonic()
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from examples import e22_routed_caption_flow as flow

common, base, units = flow.common, flow.base, flow.units
TASK = "routed_caption_frozen_scalar_moments_v1"
SECONDS = 300
ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "docs/e22_routed_caption_frozen_stats_profile_v1.json"
CARD = ROOT / "docs/e22_routed_caption_frozen_stats_v1.json"
RULE = {"Linear": "Uniform(mean-sqrt(3)*population_std,mean+sqrt(3)*population_std)",
        "pos_embed": "Normal(mean,population_std)", "std_correction": 0,
        "draw_stream": "unchanged routed-caption-geometry-v1:frozen_backbone:{full_name}",
        "realized_moments": "descriptive only; no recentering/rescaling after sampling"}
SOURCES = ("examples/e22_routed_caption_frozen_stats.py", "examples/render_e22_routed_caption_frozen_stats.py",
           "tests/test_e22_routed_caption_frozen_stats.py", "docs/e22_routed_caption_frozen_stats_profile_v1.json",
           *flow.SOURCES)


def budget():
    if time.monotonic()-STARTED > SECONDS:
        raise TimeoutError("frozen-stats startup-through-final-write300s exceeded")


def source_identity():
    return {name: base.sha(ROOT/name) for name in SOURCES}


def read_profile():
    profile = json.loads(PROFILE.read_text()); entries = profile["entries"]
    if (profile["id"] != "supra_depth1_frozen_scalar_moments_v1"
        or profile["pure_frozen_tensor_count"] != 29 or len(entries) != 29
        or set(entries) != {"pos_embed", *(s+"."+local for s in
            ("x_embed","t_embed.mlp.0","t_embed.mlp.2","ctx_proj","block.self_attn.qkv",
             "block.self_attn.proj","block.cross_attn.q","block.cross_attn.kv","block.cross_attn.proj",
             "block.mlp.0","block.mlp.2","block.adaln.1","final.adaln.1","final.linear")
            for local in ("weight","bias"))}):
        raise ValueError("complete fixed29-key profile required")
    for name, item in entries.items():
        if (not math.isfinite(item["mean"]) or not math.isfinite(item["population_std"])
            or item["population_std"] <= 0 or not item["shape"]
            or any(type(d) is not int or d <= 0 for d in item["shape"])
            or item["distribution"] != ("Normal" if name == "pos_embed" else "Uniform")):
            raise ValueError("finite positive fixed scalar profile required")
    return profile


def distributions(entries):
    result = {}
    for name, item in entries.items():
        mean, std = item["mean"], item["population_std"]
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError("finite positive population std required before mutation")
        if item["distribution"] == "Normal": result[name] = base.init.Normal(mean, std)
        elif item["distribution"] == "Uniform":
            radius = math.sqrt(3)*std
            result[name] = base.init.Uniform(mean-radius, mean+radius)
        else: raise ValueError("only declared Normal/Uniform distributions allowed")
    return result


def initialize_frozen(module, entries):
    """Call-local calibration; never alters the public distribution registry."""
    parameters = dict(module.named_parameters())
    if (set(parameters) != set(entries) or any(not p.requires_grad or p.dtype != torch.float32
            or tuple(p.shape) != tuple(entries[n]["shape"]) for n,p in parameters.items())):
        raise ValueError("fresh trainable FP32 Parameters must exactly match calibration shapes")
    overrides = distributions(entries)
    streams = {name: torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(
        f"routed-caption-geometry-v1:frozen_backbone:{name}".encode()).digest()[:8],"little") % (2**63-1))
        for name in parameters}
    parameter_ids = {n:id(p) for n,p in parameters.items()}
    base.init.initialize_(module,method="sample_distributions_v1",
                          parameter_generators=streams,distributions=overrides)
    if parameter_ids != {n:id(p) for n,p in module.named_parameters()}:
        raise AssertionError("public initializer replaced Parameter owners")
    realized = {}
    for name,p in module.named_parameters():
        x=p.detach().double()
        if not torch.isfinite(x).all(): raise ValueError("calibrated samples must be finite")
        realized[name]={"shape":list(p.shape),"mean":float(x.mean()),
                        "population_std":float(x.std(correction=0)),"RMS":float(x.square().mean().sqrt())}
    module.requires_grad_(False)
    return realized


def paired_fixed_snapshot(host):
    return base.digest({"frozen_parameters":{n:p for n,p in host.named_parameters() if not p.requires_grad},
                        "buffers":dict(host.named_buffers()),
                        "flags":{n:p.requires_grad for n,p in host.named_parameters()}})


def make_data(g=base.FULL, device=torch.device("cpu"), *, software_entries=None):
    """Only fresh frozen scalar moments change; zero old-host observations.

    software_entries is only for explicit reduced CPU ownership/replay tests.
    The CLI always uses the shipped complete full-shape profile.
    """
    profile=read_profile()
    if software_entries is None and g != base.FULL:
        raise ValueError("production profile binds FULL geometry; reduced software entries must be explicit")
    entries=profile["entries"] if software_entries is None else software_entries
    captions,masks,text_law=base.caption_data(g)
    with torch.random.fork_rng(devices=[]): frozen=base.Backbone(g)
    realized=initialize_frozen(frozen,entries)
    pools,anchors=flow.context_data(g,device)
    calibration={"rule":RULE,"profile_id":profile["id"],"profile_sha256":base.sha(PROFILE),
                 "software_only_reduced_shapes":software_entries is not None,
                 "realized_frozen_moments":realized,"frozen_state_digest":base.digest(frozen.state_dict())}
    data={"geometry":g,"frozen":deepcopy(frozen.state_dict()),"captions":captions,"masks":masks,
          "text_law":text_law,**pools,"flow":anchors,"frozen_stats":calibration}
    with torch.random.fork_rng(devices=[]): host=base.Host(data,base.ARMS[0])
    if any(not torch.equal(v,host.teacher_backbone.state_dict()[n]) for n,v in data["frozen"].items()):
        raise AssertionError("live teacher must contain the same calibrated frozen tensors")
    fixed=paired_fixed_snapshot(host); base.initialize(host,"generator")
    if paired_fixed_snapshot(host)!=fixed: raise AssertionError("fresh learned initialization changed fixed owners")
    with torch.no_grad():
        for branch in host.branches(): branch.up.weight.zero_()
        host.to(device); fixed=paired_fixed_snapshot(host)
        data["fit_baseline"]=torch.cat([host(data["fit"]["context"][i:i+base.B])
                                       for i in range(0,len(data["fit"]["context"]),base.B)])
        if paired_fixed_snapshot(host)!=fixed: raise AssertionError("FIT pairing changed fixed owners")
        data["scale"]=data["fit_baseline"].flatten(0,1).std(0,correction=1).clamp_min(.04)
    calibration["paired_fixed_owners_unchanged_during_initialization_and_FIT"]=True
    calibration["fresh_initialize_before_freeze"]=True
    data["digest"]=units.data_digest(data);data,normalization=units.normalize_data(data)
    evidence={"rule":RULE,"profile_sha256":base.sha(PROFILE),"calibration":calibration,
              "normalization":normalization,"flow_law":flow.LAW,
              "latent_description":flow.latent_description(pools,anchors),"row_metadata":anchors["rows"],
              "data_digest":data["digest"],"unchanged_caption_flow_inputs_digest":base.digest(
                  {"captions":captions,"masks":masks,"text_law":text_law,"flow":anchors}),
              "scale_recomputed_from_new_untrained_fit":True,
              "limits":"Per-tensor scalar moments only; learned directions/covariance/positional structure are absent. Mean/centered powers and realized moments are descriptive, never tuned to actual42% or accuracy."}
    return data,evidence


def validate_card(card,sources):
    expected={"task":TASK,"execution_helper_task":common.TASK,"arms":list(common.ARMS),
              "geometry":asdict(base.FULL),"steps":common.STEPS,"seconds":SECONDS,
              "media_steps":list(base.MEDIA_STEPS),"media_indices":list(base.MEDIA_INDICES),
              "flow":flow.LAW,"normalization":units.NORMALIZATION,"accuracy_thresholds":units.THRESHOLDS,
              "frozen_calibration":RULE,"profile_sha256":base.sha(PROFILE),"sources":sources}
    if any(card.get(k)!=v for k,v in expected.items()):
        raise ValueError("fixed calibrated profile/input/geometry/gate/source protocol differs")
    read_profile()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true",required=True);parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,default=CARD);args=parser.parse_args()
    threads=torch.get_num_threads();torch.set_num_threads(1)
    sources=source_identity();native=base.package_identity()
    out=device=entry=report=error=card_sha=None;code=2
    try:
        card=json.loads(args.protocol.read_text());card_sha=base.sha(args.protocol);validate_card(card,sources)
        if args.out.exists():raise ValueError("fresh exclusive output required")
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or not torch.cuda.is_available():raise ValueError("physicalGPU0 requires CUDA_VISIBLE_DEVICES=0")
        if base.backend_flags()!=card["precision_backend"]:raise ValueError("declared precision backend differs")
        out=units.fresh_output(args.out);device=torch.device("cuda:0");entry=base.global_rng(device)
        common.controls();data,evidence=make_data(base.FULL,device);budget()
        inputs=out/"frozen-stats-inputs.pt"
        torch.save({"flow":data["flow"],"contexts":{k:data[k]["context"].cpu() for k,_,_ in flow.POOLS},
                    "frozen":data["frozen"],"fit_baseline":data["fit_baseline"].cpu(),
                    "scale":data["scale"].cpu(),"evidence":evidence},inputs)
        print(json.dumps({"task":TASK,"initial_frozen_stats":evidence},allow_nan=False),flush=True)
        initial=common.preflight(data);budget();executed=common.run(data,out);budget()
        if source_identity()!=sources or base.package_identity()!=native or base.sha(args.protocol)!=card_sha or base.backend_flags()!=card["precision_backend"]:
            raise AssertionError("source/profile/card/API/backend changed within run")
        report={**executed,"task":TASK,"execution_helper_task":executed["task"],"frozen_stats":evidence,
                "frozen_stats_input_sha256":base.sha(inputs),"initial_prerequisite":initial,"source_identity":sources,
                "protocol_sha256":card_sha,"imported_package":native,"imported_package_unchanged":True,
                "precision_backend":base.backend_flags(),
                "scope":"Only frozen per-tensor scalar-moment sampling changes; learned directions are not copied and FITstd values are derived. PASS rejects this calibration as sufficient to reproduce the transfer failure in one generated fixture, not actual/full-Supra qualification."}
        code=0 if report["gate"]["pass"] else 1
    except BaseException as caught:
        error={"type":type(caught).__name__,"message":str(caught)}
        import traceback;traceback.print_exc()
    finally:
        if entry is not None:
            base.set_global_rng(entry,device)
            if base.digest(base.global_rng(device))!=base.digest(entry):error={"type":"AssertionError","message":"caller CPU/CUDA RNG restoration differs"}
        torch.set_num_threads(threads);elapsed=time.monotonic()-STARTED
        if elapsed>SECONDS:error={"type":"TimeoutError","message":"startup/cleanup300s exceeded"}
        if error is not None:code=2
        completion={"task":TASK,"complete":error is None and report is not None,
                    "scientific_status":None if error or report is None else report["scientific_status"],"error":error,
                    "seconds":elapsed,"limit_seconds":SECONDS,"source_identity":sources,"protocol_sha256":card_sha,
                    "imported_package":native,"caller_CPU_CUDA_RNG_restored":entry is not None and base.digest(base.global_rng(device))==base.digest(entry)}
        if out is not None:
            if report is not None:
                report["seconds"]=elapsed;(out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
                completion["report_sha256"]=base.sha(out/"report.json")
            path=out/"completion.json";path.write_text(json.dumps(completion,indent=2,allow_nan=False)+"\n")
            if time.monotonic()-STARTED>SECONDS:
                completion.update(complete=False,error={"type":"TimeoutError","message":"final writes300s exceeded"},seconds=time.monotonic()-STARTED)
                path.write_text(json.dumps(completion,indent=2,allow_nan=False)+"\n");code=2
        print(json.dumps({"completion":completion,"scientific_gate":None if report is None else report["gate"]},allow_nan=False),flush=True)
    return code


if __name__=="__main__":raise SystemExit(main())

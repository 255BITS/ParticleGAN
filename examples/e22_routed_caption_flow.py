"""Marginal-preserving correlated-context variant of the public caption toy.

CUDA_VISIBLE_DEVICES=0 python -m examples.e22_routed_caption_flow --run --out runs/caption-flow-v1
Exit0=completed scientific PASS,1=completed FAIL,2=incomplete/error.
No output objective/guard, seed sweep, optimizer copy or permanent API allowlist.
"""
import time
STARTED = time.monotonic()
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import torch
from examples import e22_routed_caption_normalization as units

common, base = units.common, units.base
TASK = "routed_caption_correlated_context_v1"
SECONDS = 300
ANCHOR_TIME = .1
REFERENCE_COSINE = .992
REFERENCE_DT = .1
OMEGA = math.acos(REFERENCE_COSINE) / REFERENCE_DT
POOLS = (("fit", (.1, .35, .6, .85), (0, 1)),
         ("guard", (.22, .72), (2,)),
         ("test", (.18, .43, .68, .93), (3, 4)))
LAW = {"anchor_time": ANCHOR_TIME, "reference_cosine": REFERENCE_COSINE,
       "reference_time_delta": REFERENCE_DT, "angular_rate": OMEGA,
       "formula": "z(t)=cos(omega*(t-.1))*a+sin(omega*(t-.1))*b",
       "anchor_stream": "CPU102, original iid row addressing; first two time rows per pool/source/occurrence",
       "independent_trajectories": {"fit": 12, "guard": 6, "test": 12},
       "same_time_marginal": "N(0,I); independent a,b with no normalization or orthogonalization",
       "expected_grid_adjacent_cosine": math.cos(OMEGA*.25)}
CARD = Path(__file__).resolve().parents[1] / "docs/e22_routed_caption_flow_v1.json"
SOURCES = ("examples/e22_routed_caption_flow.py", "examples/render_e22_routed_caption_flow.py",
           "tests/test_e22_routed_caption_flow.py", "examples/e22_routed_caption_accuracy.py",
           "examples/e22_routed_caption_untied.py", "examples/e22_routed_caption_normalization.py",
           "examples/render_e22_routed_caption_untied.py")


def budget():
    if time.monotonic()-STARTED > SECONDS: raise TimeoutError("flow startup-through-final-write300s exceeded")


def source_identity():
    root = Path(__file__).resolve().parents[1]
    return {name: base.sha(root/name) for name in SOURCES}


def coefficients(t):
    theta = OMEGA*(float(t)-ANCHOR_TIME)
    return math.cos(theta), math.sin(theta)


def mix(a, b, t):
    if a.shape != b.shape or not a.is_floating_point() or a.dtype != b.dtype:
        raise ValueError("same-shape floating independent anchors required")
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("finite anchors required")
    ca, cb = coefficients(t)
    return a*ca+b*cb


def context_data(g, device=torch.device("cpu")):
    """Only the original private iid arrays' cross-time assignment changes.

    Consume all108 original arrays in the original pool/source/time/draw order.
    The first two time arrays for each occurrence are its independent a,b.
    Later original arrays are consumed but unused, keeping stream addressing.
    """
    rng = torch.Generator().manual_seed(102)
    pools, anchors, rows = {}, {}, {}
    for pool, times, draws in POOLS:
        iid, pool_anchors = {}, {}
        for source in range(6):
            for t in times:
                for draw in draws: iid[source,t,draw] = torch.randn(g.tokens,g.output,generator=rng)
            for draw in draws:
                key = f"{pool}:{source}:{draw}"
                pool_anchors[key] = {"a":iid[source,times[0],draw], "b":iid[source,times[1],draw]}
        contexts, ids, records = [], [], []
        for source in range(6):
            for t in times:
                for draw in draws:
                    key = f"{pool}:{source}:{draw}"; pair = pool_anchors[key]
                    latent = mix(pair["a"],pair["b"],t)
                    contexts.append(torch.cat((latent,torch.tensor([source,t]).expand(g.tokens,-1)),-1))
                    ids.append(source)
                    records.append({"pool":pool,"source":source,"draw":draw,"time":t,"anchor_key":key,
                                    "a_sha256":base.digest(pair["a"]),"b_sha256":base.digest(pair["b"]),
                                    "coefficients":list(coefficients(t))})
        pools[pool] = {"context":torch.stack(contexts).to(device),
                       "targets":torch.zeros(len(contexts),g.tokens,g.output,device=device),"source_ids":ids}
        anchors[pool], rows[pool] = pool_anchors, records
    return pools, {"law":LAW,"anchors":anchors,"rows":rows,"CPU102_after_draws":rng.get_state()}


def latent_description(pools, flow):
    result = {}
    for pool,times,draws in POOLS:
        x = pools[pool]["context"][:,:,:-2].detach().double()
        cosines = []
        for key in flow["anchors"][pool]:
            selected = [i for i,r in enumerate(flow["rows"][pool]) if r["anchor_key"]==key]
            for left,right in zip(selected,selected[1:]):
                a,b = x[left].flatten(),x[right].flatten(); denominator = float(a.norm()*b.norm())
                cosines.append(None if denominator==0 else float(a.dot(b))/denominator)
        result[pool] = {"contexts":len(x),"trajectories":len(flow["anchors"][pool]),
                        "latent_rms":float(x.square().mean().sqrt()),"realized_adjacent_cosines":cosines,
                        "expected_adjacent_cosines":[math.cos(OMEGA*(b-a)) for a,b in zip(times,times[1:])],
                        "context_digest":base.digest(pools[pool]["context"]),
                        "anchor_digest":base.digest(flow["anchors"][pool])}
    return result


def make_data(g=base.FULL, device=torch.device("cpu")):
    """Pure frozen host math/public named initialization; no native updates."""
    captions,masks,text_law = base.caption_data(g)
    with torch.random.fork_rng(devices=[]): frozen = base.Backbone(g)
    base.initialize(frozen,"frozen_backbone"); frozen.requires_grad_(False)
    pools,flow = context_data(g,device)
    data = {"geometry":g,"frozen":deepcopy(frozen.state_dict()),"captions":captions,"masks":masks,
            "text_law":text_law,**pools,"flow":flow}
    with torch.random.fork_rng(devices=[]): host = base.Host(data,base.ARMS[0])
    base.initialize(host,"generator")
    with torch.no_grad():
        for branch in host.branches(): branch.up.weight.zero_()
        host.to(device)
        data["fit_baseline"] = torch.cat([host(data["fit"]["context"][i:i+base.B])
                                          for i in range(0,len(data["fit"]["context"]),base.B)])
        data["scale"] = data["fit_baseline"].flatten(0,1).std(0,correction=1).clamp_min(.04)
    data["digest"] = units.data_digest(data)
    data,normalization = units.normalize_data(data)
    evidence = {"law":LAW,"normalization":normalization,"latent_description":latent_description(pools,flow),
                "row_metadata":flow["rows"],"data_digest":data["digest"],
                "unaffected_owner_inputs_digest":base.digest({k:data[k] for k in ("frozen","captions","masks","text_law")}),
                "scale_recomputed_from_new_untrained_fit":True,
                "limits":"Correlation only: theoretical marginalRMS1 (realized finite RMS descriptive), actual amplitude-vs-time/pretrained token directions are unmatched; .25 time spacing has expected correlation about.950, not.992."}
    return data,evidence


def validate_card(card,sources):
    expected = {"task":TASK,"execution_helper_task":common.TASK,"arms":list(common.ARMS),
                "geometry":asdict(base.FULL),"steps":common.STEPS,"seconds":SECONDS,
                "media_steps":list(base.MEDIA_STEPS),"media_indices":list(base.MEDIA_INDICES),
                "flow":LAW,"normalization":units.NORMALIZATION,"accuracy_thresholds":units.THRESHOLDS,"sources":sources}
    if any(card.get(k)!=v for k,v in expected.items()): raise ValueError("fixed flow/normalization/geometry/gate/source protocol differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        inputs=out/"flow-inputs.pt"
        torch.save({"flow":data["flow"],"contexts":{k:data[k]["context"].cpu() for k,_,_ in POOLS},
                    "fit_baseline":data["fit_baseline"].cpu(),"scale":data["scale"].cpu(),"evidence":evidence},inputs)
        print(json.dumps({"task":TASK,"initial_flow":evidence},allow_nan=False),flush=True)
        initial=common.preflight(data);budget()
        executed=common.run(data,out);budget()  # Immutable public-API loop, never patched/copied.
        if source_identity()!=sources or base.package_identity()!=native or base.sha(args.protocol)!=card_sha or base.backend_flags()!=card["precision_backend"]:
            raise AssertionError("source/card/API/backend changed within run")
        report={**executed,"task":TASK,"execution_helper_task":executed["task"],"flow":evidence,
                "flow_input_sha256":base.sha(inputs),"initial_prerequisite":initial,"source_identity":sources,
                "protocol_sha256":card_sha,"imported_package":native,"imported_package_unchanged":True,
                "precision_backend":base.backend_flags(),
                "scope":"Only cross-time latent correlation changes; FIT-std values are recomputed consequences. Terminal physical accuracy gate is unchanged. No actual-caption/full-Supra win or unique causal claim."}
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

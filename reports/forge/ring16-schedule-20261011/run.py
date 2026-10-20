"""Compare full 6400-update schedules with the saved 400-horizon baseline.

Run: .venv/bin/python -u reports/forge/ring16-schedule-20261011/run.py run
Tail: tail -F runs/api/ring16-schedule-20261011/progress.jsonl
Publish: .venv/bin/python reports/forge/ring16-schedule-20261011/publish.py
The archived public API and original source-bound baseline are explicit cohorts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = HERE / "protocol.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def declaration():
    protocol = json.loads(PROTOCOL.read_text())
    assert sha(ROOT / protocol["baseline_report"]) == protocol["baseline_report_sha256"]
    return protocol


def archived(row):
    p = declaration()
    archive = Path(p["parent_archive"])
    manifest = json.loads((archive/"certificates/manifest.json").read_text())["certificates"]
    documents = {}
    for kind, expected in row["parent_certificates"].items():
        entry = manifest[row["attempt_id"]][kind]
        path = archive / entry["path"]
        assert sha(path) == expected == entry["sha256"]
        documents[kind] = json.loads(path.read_text())
    request = documents["request"]["request"]
    assert request["source"]["digest"] == p["parent_source_digest"]
    for name, expected in request["source"]["files"].items():
        assert sha(Path(p["parent_snapshot"])/name) == expected, "changed source: " + name
    return request, documents["result"]["raw"]["result"]


def build(row, request):
    from experiments.forge.api import task_formulation_context
    from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
    candidate = json.loads((HERE/row["candidate_declaration"]).read_text())
    task = json.loads((HERE/row["task_declaration"]).read_text())
    assert sha(HERE/row["candidate_declaration"]) == row["candidate_sha256"]
    assert sha(HERE/row["task_declaration"]) == row["task_sha256"]
    spec = resolve_vector_spec(task)
    context = task_formulation_context(candidate, task, request["protocol"],
                                       device="cuda:0", root=declaration()["parent_snapshot"])
    trainer = context.build_trainer(*build_vector_models(context, spec), max_steps=1600)
    spec["thresholds"] = task["evaluation"]["thresholds"]
    return context, trainer, task, spec


def trial(row, output, preflight=False):
    p = declaration()
    sys.path.insert(0, p["parent_snapshot"])
    from benchmarks.toy_audit.reproducibility import reproducible_execution

    @reproducible_execution
    def execute(*, device="cuda:0", seed=0):
        import torch
        from benchmarks.toy_audit.api_vectors import _bounds
        from benchmarks.toy_audit.ring16_quality import score_samples
        from benchmarks.transfer_suite.vector_tasks import sample_target
        from experiments.forge.artifacts import verify_artifacts
        from experiments.forge.sources import runtime_manifest
        from experiments.forge.state import state_digest, require_same_formulation, require_optimizer_steps
        request, parent = archived(row)
        assert runtime_manifest() == request["runtime"]
        context, trainer, task, spec = build(row, request)
        data = context.streams.generator("data", component="target", purpose="training", device="cpu")
        evaluation = context.streams.generator("eval", component="live", purpose="samples")
        initial = context.state_dict()
        cp = parent["evidence"]["provenance_checkpoint"]
        verify_artifacts(cp["artifact_root"], cp["artifact_manifest"])
        parent_state = torch.load(Path(cp["artifact_root"])/cp["path"], map_location="cpu", weights_only=True)
        assert state_digest(parent_state) == cp["state_sha256"]
        baseline_state = torch.load(ROOT/row["baseline_raw"] / "state.pt", map_location="cpu", weights_only=True)
        assert sha(ROOT/row["baseline_raw"] / "state.pt") == row["baseline_final_checkpoint_sha256"]
        assert initial["initialization"] == parent_state["initialization"]
        for role, model in [("generator",trainer.G),("discriminator",trainer.D),("prior",trainer.prior)]:
            assert state_digest(model.state_dict()) == parent_state["initialization"][role]["initial_state_sha256"]
        assert initial["streams"]["manifest"] == parent_state["streams"]["manifest"]
        actual_delta = {k:[v,initial["recipe"].get(k)] for k,v in parent_state["recipe"].items()
                        if initial["recipe"].get(k) != v}
        expected_delta = {"total_steps":[400,6400]}
        if parent_state["recipe"]["network_lr_horizon_cap"] is not None:
            expected_delta["network_lr_horizon_cap"] = [1600,None]
        assert actual_delta == expected_delta
        assert initial["prior"] == parent_state["prior"]
        assert initial["prior_contract"] == parent_state["prior_contract"]
        data_key = next(k for k,v in initial["streams"]["manifest"]["bindings"].items()
                        if (v["family"],v["component"],v["purpose"],v["device"]) == ("data","target","training","cpu"))
        # Reconstruct the deterministic baseline law without training or changing its streams.
        replay = torch.Generator().manual_seed(initial["streams"]["manifest"]["bindings"][data_key]["seed"])
        prefix_digest, suffix_digest, full_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        for step in range(1,6401):
            values = sample_target(spec,128,replay,step-1).numpy().tobytes()
            (prefix_digest if step<=1600 else suffix_digest).update(values)
            full_digest.update(values)
            if step==1600:assert torch.equal(replay.get_state(),parent_state["streams"]["states"][data_key])
        assert torch.equal(replay.get_state(),baseline_state["streams"]["states"][data_key])
        baseline_receipt = json.loads((ROOT/row["baseline_raw"] / "receipt.json").read_text())
        assert suffix_digest.hexdigest() == baseline_receipt["new_data_sequence_sha256"]
        proof = dict(matched_initial_models=True,matched_initial_prior=True,matched_stream_bindings=True,
                     recipe_delta=actual_delta,prior_update_and_regularizer_unchanged=True,
                     baseline_prefix_data_verified_against_archived_stream=True,
                     baseline_suffix_data_verified_against_actual_digest=True,
                     expected_prefix_data_sha256=prefix_digest.hexdigest(),
                     expected_suffix_data_sha256=suffix_digest.hexdigest(),
                     expected_full_data_sha256=full_digest.hexdigest(),
                     initial_state_sha256=state_digest(initial), qualification_input=False)
        if preflight:
            print(json.dumps(dict(family=row["family"],**proof)),flush=True)
            return
        output.mkdir(parents=True,exist_ok=False)
        write(output/"initial-proof.json",proof)
        write(output/"recipe.json",context.recipe.to_dict())
        torch.save(initial,output/"initial-state.pt")
        records, curve = [], []
        consumed_prefix, consumed_suffix, consumed_all = hashlib.sha256(),hashlib.sha256(),hashlib.sha256()
        checks = {math.ceil(i*6400/384) for i in range(1,385)}
        started = time.monotonic()
        print(json.dumps(dict(event="start",family=row["family"],updates=6400)),flush=True)
        with (output/"metrics.jsonl").open("w") as stream:
            for step in range(1,6401):
                real = sample_target(spec,context.recipe.batch_size,data,step-1)
                payload = real.numpy().tobytes()
                (consumed_prefix if step<=1600 else consumed_suffix).update(payload)
                consumed_all.update(payload)
                trainer.step(real.to(device),collect_stats=True)
                if step in checks:
                    before = context.streams.audit()
                    samples = trainer.sample(4096,generator=evaluation,output_noise=False).detach().cpu()
                    metrics = score_samples(samples,spec,step)
                    allowed = [k for k,v in context.streams.manifest()["bindings"].items() if v["family"]=="eval"]
                    assert not context.streams.compare(before,context.streams.audit(),allowed=allowed)["unintended_rng_deviations"]
                    failed = _bounds(metrics,spec["thresholds"])
                    point = dict(step=step,metrics=metrics,passed=not failed,failed_bounds=failed)
                    curve.append(point);records.append(dict(step=step,samples=samples))
                    stream.write(json.dumps(point,allow_nan=False)+"\n");stream.flush()
                    if step%400==0:
                        print(json.dumps(dict(event="progress",family=row["family"],step=step,hq=metrics["hq"],
                            covariance=metrics["component_covariance_error"],min_eigen=metrics["component_min_eigen_ratio"],passed=not failed)),flush=True)
                if step==1600:
                    # Match the saved control's serialization/reconstruction boundary exactly.
                    prefix_state = context.state_dict()
                    require_optimizer_steps(prefix_state,1600)
                    torch.save(prefix_state,output/"prefix-state.pt")
                    context,trainer,_,_ = build(row,request)
                    context.load_state_dict(torch.load(output/"prefix-state.pt",map_location="cpu",weights_only=True))
                    restored = context.state_dict()
                    assert state_digest(restored)==state_digest(prefix_state)
                    trainer.extend_execution(6400)
                    extended = context.state_dict()
                    unextended = deepcopy(extended);unextended["trainer"]["max_steps"]=1600
                    assert state_digest(unextended)==state_digest(prefix_state)
                    write(output/"resume-proof.json",dict(exact_restore=True,only_cap_extended=True,
                        prefix_state_sha256=state_digest(prefix_state),restored_state_sha256=state_digest(restored),
                        matched_baseline_restart_update=1600,completed_steps_preserved=True))
                    data=context.streams.generator("data",component="target",purpose="training",device="cpu")
                    evaluation=context.streams.generator("eval",component="live",purpose="samples")
        torch.cuda.synchronize()
        elapsed=time.monotonic()-started
        final=context.state_dict()
        require_same_formulation(initial,final);require_optimizer_steps(final,6400)
        assert consumed_prefix.hexdigest()==proof["expected_prefix_data_sha256"]
        assert consumed_suffix.hexdigest()==proof["expected_suffix_data_sha256"]
        assert consumed_all.hexdigest()==proof["expected_full_data_sha256"]
        torch.save(final,output/"state.pt");torch.save(records,output/"observations.pt")
        write(output/"curve.json",curve)
        files=["initial-proof.json","recipe.json","initial-state.pt","prefix-state.pt","resume-proof.json","state.pt","observations.pt","curve.json"]
        write(output/"receipt.json",dict(schema_version=1,scope=p["scope"],qualification_input=False,
            family=row["family"],parent_attempt=row["attempt_id"],protocol_sha256=sha(PROTOCOL),
            runner_sha256=sha(Path(__file__)),parent_source_digest=p["parent_source_digest"],
            parent_source_commit=request["source"]["origin_commit"],runtime=runtime_manifest(),
            gpu=torch.cuda.get_device_name(),completed_updates=6400,training_loop_seconds=elapsed,
            matched_full_data_sha256=consumed_all.hexdigest(),initial_proof=proof,
            artifacts={name:sha(output/name) for name in files}))
        print(json.dumps(dict(event="completed",family=row["family"],seconds=elapsed)),flush=True)
    execute()


def launch(output,preflight=False):
    p=declaration()
    output.mkdir(parents=True,exist_ok=False)
    write(output/"frozen-protocol.json",p)
    slots=queue.Queue()
    for gpu in (0,1):slots.put(gpu)
    def job(row):
        gpu=slots.get();family=row["family"]
        log=output/(family+(".preflight.log" if preflight else ".log"))
        def event(**values):
            record=dict(time=time.time(),family=family,gpu=gpu,**values)
            with (output/"progress.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
            print(json.dumps(record),flush=True)
        try:
            event(event="start",log=str(log))
            command=[sys.executable,"-u",str(Path(__file__)),"trial","--family",family,"--output",str(output)]
            if preflight:command.append("--preflight")
            env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"CUBLAS_WORKSPACE_CONFIG":":4096:8",
                 "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
            with log.open("x") as stream:
                result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=row["timeout_seconds"])
            event(event="finished",returncode=result.returncode)
            if result.returncode:raise RuntimeError(f"{family} failed; see {log}")
        finally:slots.put(gpu)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(job,row) for row in p["runs"]]
        for future in futures:future.result()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["preflight","run","trial"])
    parser.add_argument("--family")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"runs/api/ring16-schedule-20261011")
    args=parser.parse_args()
    if args.action=="trial":
        row=next(r for r in declaration()["runs"] if r["family"]==args.family)
        trial(row,args.output/args.family,args.preflight)
    else:launch(args.output,args.action=="preflight")

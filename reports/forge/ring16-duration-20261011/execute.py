"""Bounded public-API duration diagnostic from exact archived ring checkpoints.

Run: .venv/bin/python -u reports/forge/ring16-duration-20261011/run.py run
Tail: tail -F runs/api/ring16-duration-20261011/progress.jsonl
Publish: .venv/bin/python reports/forge/ring16-duration-20261011/run.py publish
The archived API is imported explicitly; original failures are never regraded.
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


def load_protocol():
    return json.loads(PROTOCOL.read_text())


def event(path, **values):
    row = {"time": time.time(), **values}
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row, allow_nan=False), flush=True)


def setup(row):
    protocol = load_protocol()
    archive = Path(protocol["parent_archive"])
    manifest = json.loads((archive / "certificates/manifest.json").read_text())["certificates"]
    documents = {}
    for kind in ("request", "result", "evidence"):
        entry = manifest[row["attempt_id"]][kind]
        path = archive / entry["path"]
        assert sha(path) == entry["sha256"] == row["parent_certificates"][kind]
        documents[kind] = json.loads(path.read_text())
    request = documents["request"]["request"]
    snapshot = Path(request["source"]["snapshot_path"])
    assert request["source"]["digest"] == protocol["parent_source_digest"]
    for name, digest in request["source"]["files"].items():
        assert sha(snapshot / name) == digest, "archived source changed: " + name
    # Prevent the current checkout from replacing the original trainer contract.
    sys.path.insert(0, str(snapshot))
    import torch
    from experiments.forge.api import task_formulation_context
    from experiments.forge.artifacts import verify_artifacts
    from experiments.forge.state import state_digest, require_optimizer_steps
    from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
    task = request["tasks"]["ring16_acquisition"]
    spec = resolve_vector_spec(task)
    context = task_formulation_context(request["candidate"], task, request["protocol"],
                                       device="cuda:0", root=snapshot)
    trainer = context.build_trainer(*build_vector_models(context, spec), max_steps=1600)
    spec["thresholds"] = task["evaluation"]["thresholds"]
    result = documents["result"]["raw"]["result"]
    certificate = result["evidence"]["provenance_checkpoint"]
    assert certificate["purpose"] == "provenance_only" and not certificate["prerequisite_eligible"]
    verify_artifacts(certificate["artifact_root"], certificate["artifact_manifest"])
    checkpoint = Path(certificate["artifact_root"]) / certificate["path"]
    assert sha(checkpoint) == row["parent_checkpoint_sha256"] == certificate["sha256"]
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert state_digest(state) == certificate["state_sha256"]
    context.load_state_dict(state)
    restored = context.state_dict()
    assert state_digest(restored) == state_digest(state)
    require_optimizer_steps(restored, 1600)
    assert context.recipe.total_steps == 400
    return protocol, row, request, task, spec, context, trainer, state, result


def trial(row, output, preflight=False):
    protocol = load_protocol()
    sys.path.insert(0, str(protocol["parent_snapshot"]))
    from benchmarks.toy_audit.reproducibility import reproducible_execution

    @reproducible_execution
    def execute(*, device="cuda:0", seed=0):
        import torch
        from benchmarks.transfer_suite.vector_tasks import sample_target
        from benchmarks.toy_audit.ring16_quality import score_samples
        from benchmarks.toy_audit.api_vectors import _bounds
        from experiments.forge.state import state_digest, require_same_formulation, require_optimizer_steps
        from experiments.forge.sources import runtime_manifest
        protocol, row_, request, task, spec, context, trainer, state, parent = setup(row)
        assert runtime_manifest() == request["runtime"], "runtime differs from archived cohort"
        before = context.state_dict()
        trainer.extend_execution(protocol["max_total_updates"])
        extended = context.state_dict()
        unextended = deepcopy(extended)
        unextended["trainer"]["max_steps"] = before["trainer"]["max_steps"]
        assert state_digest(unextended) == state_digest(before)
        require_same_formulation(state, extended)
        proof = dict(state_restored_exactly=True, only_execution_cap_changed=True,
                     original_state_sha256=state_digest(state), restored_state_sha256=state_digest(before),
                     original_updates=1600, maximum_updates=protocol["max_total_updates"],
                     schedule_horizon=400, prior_contract=state["prior_contract"],
                     original_checkpoint_purpose="provenance_only", qualification_input=False)
        if preflight:
            print(json.dumps(dict(family=row["family"], **proof)), flush=True)
            return
        output.mkdir(parents=True, exist_ok=False)
        write(output / "resume-proof.json", proof)
        write(output / "recipe.json", context.recipe.to_dict())
        parent_dir = Path(protocol["parent_archive"]) / "queue/recipe-prior-refactor-selected-leaders-v1" / row["attempt_id"]
        saved = parent["evidence"]["saved_observer_outputs"]
        assert sha(parent_dir / saved["path"]) == saved["sha256"]
        records = torch.load(parent_dir / saved["path"], weights_only=True, map_location="cpu")
        curve = [dict(step=p["step"], metrics={k:v for k,v in p.items() if k != "step"},
                      passed=not _bounds(p, spec["thresholds"])) for p in parent["evidence"]["observations"]]
        data = context.streams.generator("data", component="target", purpose="training", device="cpu")
        evaluation = context.streams.generator("eval", component="live", purpose="samples")
        data_digest = hashlib.sha256()
        checks = {offset + math.ceil(i*1600/96) for offset in range(1600,6400,1600) for i in range(1,97)}
        started = time.monotonic()
        print(json.dumps(dict(event="start", family=row["family"], additional_updates=4800)), flush=True)
        with (output / "metrics.jsonl").open("w") as stream:
            for step in range(1601, 6401):
                real = sample_target(spec, context.recipe.batch_size, data, step-1)
                data_digest.update(real.numpy().tobytes())
                trainer.step(real.to(device), collect_stats=True)
                if step not in checks:
                    continue
                before_rng = context.streams.audit()
                samples = trainer.sample(4096, generator=evaluation, output_noise=False).detach().cpu()
                metrics = score_samples(samples, spec, step)
                after_rng = context.streams.audit()
                allowed = [k for k,v in context.streams.manifest()["bindings"].items() if v["family"] == "eval"]
                assert not context.streams.compare(before_rng, after_rng, allowed=allowed)["unintended_rng_deviations"]
                failed = _bounds(metrics, spec["thresholds"])
                point = dict(step=step, metrics=metrics, passed=not failed, failed_bounds=failed)
                curve.append(point)
                records.append(dict(step=step, samples=samples))
                stream.write(json.dumps(point, allow_nan=False)+"\n"); stream.flush()
                if step % 400 == 0:
                    print(json.dumps(dict(event="progress", family=row["family"], step=step,
                          hq=metrics["hq"], covariance=metrics["component_covariance_error"], passed=not failed)), flush=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        final = context.state_dict()
        require_same_formulation(state, final)
        require_optimizer_steps(final, 6400)
        assert final["prior_contract"] == state["prior_contract"]
        torch.save(final, output / "state.pt")
        torch.save(records, output / "observations.pt")
        write(output / "curve.json", curve)
        result = dict(schema_version=1, scope=protocol["scope"], qualification_input=False,
                      family=row["family"], parent_attempt=row["attempt_id"], parent_verdict="FAIL",
                      protocol_sha256=sha(PROTOCOL), runner_sha256=sha(Path(__file__)),
                      parent_source_digest=protocol["parent_source_digest"], parent_source_commit=request["source"]["origin_commit"],
                      parent_certificates=row["parent_certificates"], recipe_sha256=sha(output / "recipe.json"),
                      resume_proof=proof, runtime=runtime_manifest(), gpu=torch.cuda.get_device_name(),
                      completed_updates=6400, additional_updates=4800, continuation_loop_seconds=elapsed,
                      new_data_sequence_sha256=data_digest.hexdigest(), observations=len(curve),
                      artifacts={name:sha(output/name) for name in ["state.pt","observations.pt","curve.json","recipe.json","resume-proof.json"]})
        write(output / "receipt.json", result)
        print(json.dumps(dict(event="completed", family=row["family"], seconds=elapsed)), flush=True)
    execute()


def launch(output, preflight=False):
    protocol=load_protocol()
    output.mkdir(parents=True, exist_ok=True)
    write(output / "frozen-protocol.json", protocol)
    def job(index, row):
        family=row["family"]
        log=output/(family+(".preflight.log" if preflight else ".log"))
        event(output / "progress.jsonl", event="start", family=family, gpu=index%2, log=str(log))
        env={**os.environ,"CUDA_VISIBLE_DEVICES":str(index%2),"CUBLAS_WORKSPACE_CONFIG":":4096:8",
             "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
        command=[sys.executable,"-u",str(Path(__file__)),"trial","--family",family,"--output",str(output)]
        if preflight:command.append("--preflight")
        with log.open("x") as stream:
            result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                  timeout=row["timeout_seconds"])
        event(output / "progress.jsonl",event="finished",family=family,returncode=result.returncode)
        if result.returncode:raise RuntimeError(f"{family} failed; see {log}")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(job,i,row) for i,row in enumerate(protocol["runs"])]
        for future in futures:future.result()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["preflight","run","trial"])
    parser.add_argument("--family")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"runs/api/ring16-duration-20261011")
    args=parser.parse_args()
    if args.action=="trial":
        row=next(r for r in load_protocol()["runs"] if r["family"]==args.family)
        trial(row,args.output/args.family,args.preflight)
    else:launch(args.output,args.action=="preflight")


if __name__=="__main__":
    main()

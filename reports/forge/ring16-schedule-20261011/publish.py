"""Verify the schedule comparison and render saved training outputs; no updates."""
import argparse
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from run import archived, declaration, sha, write


def suffix(curve):
    count=0
    for point in reversed(curve):
        if not point["passed"]:break
        count+=1
    return count


def summary(curve, thresholds):
    first=None;longest=streak=0
    for p in curve:
        streak=streak+1 if p["passed"] else 0
        longest=max(longest,streak)
        if streak==5 and first is None:first=p["step"]
    cuts=[]
    for cap in (1600,3200,4800,6400):
        prefix=[p for p in curve if p["step"]<=cap]
        point=prefix[-1]
        metrics=point["metrics"]
        failures=[]
        for key,op,bound in thresholds:
            value=metrics.get(key)
            valid=isinstance(value,(int,float)) and math.isfinite(value)
            if not valid or (value<bound if op==">=" else value>bound):failures.append(f"{key} {op} {bound}")
        count=suffix(prefix)
        cuts.append(dict(updates=cap,terminal_passing_observations=count,verdict="PASS" if count>=5 else "FAIL",
                         failed_bounds=failures,metrics={k:v for k,v in metrics.items() if isinstance(v,(int,float))}))
    return dict(verdict=cuts[-1]["verdict"],budget_cuts=cuts,first_five_passing_checks_update=first,
                longest_passing_streak=longest,passing_observations=sum(p["passed"] for p in curve),
                final_failed_bounds=cuts[-1]["failed_bounds"])


def publish(raw,destination):
    p=declaration()
    assert json.loads((raw/"frozen-protocol.json").read_text())==p
    sys.path.insert(0,p["parent_snapshot"])
    import numpy as np
    import torch
    from benchmarks.toy_audit.api_run import render_gif
    from benchmarks.toy_audit.api_vectors import _bounds
    from benchmarks.toy_audit.ring16_quality import score_samples
    from benchmarks.transfer_suite.vector_tasks import sample_target
    from experiments.forge.state import state_digest,require_same_formulation,require_optimizer_steps
    torch.set_num_threads(1)
    events=[json.loads(line) for line in (raw/"progress.jsonl").read_text().splitlines()]
    baseline=json.loads((ROOT/p["baseline_report"]).read_text())
    destination.mkdir(parents=True,exist_ok=True)
    results=[]
    for row in p["runs"]:
        family=row["family"];directory=raw/family
        request,_=archived(row)
        receipt=json.loads((directory/"receipt.json").read_text())
        assert receipt["runtime"]==request["runtime"]
        assert receipt["protocol_sha256"]==sha(HERE/"protocol.json")
        assert receipt["runner_sha256"]==sha(HERE/"run.py")
        for name,digest in receipt["artifacts"].items():assert sha(directory/name)==digest
        assert sha(HERE/row["task_declaration"])==row["task_sha256"]
        assert sha(HERE/row["candidate_declaration"])==row["candidate_sha256"]
        task=json.loads((HERE/row["task_declaration"]).read_text())
        spec={**task["execution"]["host_definition"],"thresholds":task["evaluation"]["thresholds"]}
        assert spec["thresholds"]==p["evaluation"]["thresholds"]
        curve=json.loads((directory/"curve.json").read_text())
        records=torch.load(directory/"observations.pt",map_location="cpu",weights_only=True)
        control_curve=json.loads((ROOT/row["baseline_raw"] / "curve.json").read_text())
        control_records=torch.load(ROOT/row["baseline_raw"] / "observations.pt",map_location="cpu",weights_only=True)
        control=next(r for r in baseline["runs"] if r["family"]==family)
        control_directory=ROOT/row["baseline_raw"]
        assert sha(control_directory/"receipt.json")==control["raw_receipt_sha256"]
        for name,digest in control["artifacts"].items():assert sha(control_directory/name)==digest
        assert len(curve)==len(records)==len(control_curve)==len(control_records)==384
        steps=[math.ceil(i*6400/384) for i in range(1,385)]
        assert [r["step"] for r in records]==[v["step"] for v in curve]==[r["step"] for r in control_records]==steps
        assert all(point["passed"]==(not _bounds(point["metrics"],spec["thresholds"])) for point in curve)
        checks={1600,3200,4800,6400}|{point["step"] for point in curve[-5:]}
        recomputations=0
        for record,point in zip(records,curve):
            if point["step"] not in checks:continue
            metrics=score_samples(record["samples"],spec,point["step"])
            for key,_,_ in spec["thresholds"]:
                assert math.isclose(metrics[key],point["metrics"][key],rel_tol=0,abs_tol=1e-7)
            recomputations+=1
        initial=torch.load(directory/"initial-state.pt",map_location="cpu",weights_only=True)
        final=torch.load(directory/"state.pt",map_location="cpu",weights_only=True)
        prefix=torch.load(directory/"prefix-state.pt",map_location="cpu",weights_only=True)
        require_same_formulation(initial,final);require_optimizer_steps(final,6400)
        require_optimizer_steps(prefix,1600)
        proof=json.loads((directory/"initial-proof.json").read_text())
        resume=json.loads((directory/"resume-proof.json").read_text())
        assert state_digest(initial)==proof["initial_state_sha256"]
        assert state_digest(prefix)==resume["prefix_state_sha256"]==resume["restored_state_sha256"]
        assert resume["exact_restore"] and resume["only_cap_extended"]
        assert final["trainer"]["completed_steps"]==receipt["completed_updates"]==6400
        assert final["recipe"]["total_steps"]==6400
        assert final["trainer"].get("max_steps",final["recipe"]["total_steps"])==6400
        assert receipt["matched_full_data_sha256"]==proof["expected_full_data_sha256"]
        result=summary(curve,spec["thresholds"])
        baseline_summary=summary(control_curve,spec["thresholds"])
        assert baseline_summary["verdict"]==control["verdict"]
        assert baseline_summary["longest_passing_streak"]==control["longest_passing_streak"]
        target=sample_target(spec,4096,torch.Generator().manual_seed(78013),0)
        frames=[]
        for index in np.linspace(0,383,12).round().astype(int):
            observed,control_observed,point=records[index],control_records[index],curve[index]
            views=[]
            for horizon,record in [(400,control_observed),(6400,observed)]:
                views.append(dict(kind="scatter",title=f"Schedule horizon {horizon}",target=target,samples=record["samples"],
                                  xlim=[-4,4],ylim=[-4,4],xlabel="x",ylabel="y",
                                  caption="Saved clean live samples; same target, initialization, batches and restart boundary."))
            frames.append(dict(step=point["step"],views=views,passed=point["passed"],
                metrics={k:point["metrics"][k] for k in ["modes","hq","mass_tv","component_covariance_error","component_min_eigen_ratio"]}))
        gif=family+".gif"
        render_gif(dict(id=family+"-ring16-schedule",goal="Does a full6400-update schedule improve sustained ring density at a fixed6400-update budget?",default_steps=6400),
                   frames,destination/gif,full_budget=True,requested_steps=6400,final_verdict=result["verdict"])
        start=next(e for e in events if e["family"]==family and e["event"]=="start")
        finish=next(e for e in events if e["family"]==family and e["event"]=="finished")
        assert finish["returncode"]==0
        results.append({**receipt,**result,"baseline":baseline_summary,"process_wall_seconds":finish["time"]-start["time"],
            "raw_receipt_sha256":sha(directory/"receipt.json"),"saved_sample_gate_recomputations":recomputations,
            "final_state_sha256":state_digest(final),"matched_restart":resume,
            "candidate_declaration_sha256":row["candidate_sha256"],"task_declaration_sha256":row["task_sha256"],
            "gif":gif,"gif_sha256":sha(destination/gif)})
    assert len({r["matched_full_data_sha256"] for r in results})==1
    report=dict(schema_version=1,scope=p["scope"],qualification_input=False,baseline_reused_without_training=True,
        baseline_report=p["baseline_report"],baseline_report_sha256=p["baseline_report_sha256"],
        budget=p["budget"],new_runs=3,new_host_updates=19200,
        new_paid_process_seconds=sum(r["process_wall_seconds"] for r in results),
        training_loop_seconds=sum(r["training_loop_seconds"] for r in results),
        matched_initial_models_and_priors=True,matched_actual_full_data=True,matched_restart_boundary=True,
        protocol_sha256=sha(HERE/"protocol.json"),runner_sha256=sha(HERE/"run.py"),publisher_sha256=sha(Path(__file__)),
        raw_artifact_root=str(raw.resolve()),runs=results)
    write(destination/"results.json",report)
    print(json.dumps({"runs":[dict(family=r["family"],verdict=r["verdict"],first_five=r["first_five_passing_checks_update"],longest=r["longest_passing_streak"],
        final=r["budget_cuts"][-1],baseline_final=r["baseline"]["budget_cuts"][-1]) for r in results],"seconds":report["new_paid_process_seconds"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,default=ROOT/"runs/api/ring16-schedule-20261011")
    parser.add_argument("--output",type=Path,default=HERE)
    args=parser.parse_args()
    publish(args.raw,args.output)

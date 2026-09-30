"""CPU-only, completed-job audit of saved learned checkpoints and replay endpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import traceback

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
REVIEW = Path(__file__).resolve().parent
STUDY = REVIEW.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--validation",type=Path,default=STUDY/"validation")
parser.add_argument("--variant",default="CB64-RA2")
parser.add_argument("--output",type=Path,default=REVIEW/"learned-artifact-audit")
parser.add_argument("--watch",action="store_true")
args=parser.parse_args()
args.validation=args.validation.resolve();args.output=args.output.resolve()
assert args.output.is_relative_to(REVIEW)
args.output.mkdir(parents=True,exist_ok=True)
LEARNED=args.validation/"learned"
CHECKPOINTS=(0,100,250,500,750,1000,1250,1500,1750,2000)
LOSS_KEYS=("loss_d","loss_g","loss_gan","prior_regularization","penalty")
EXPECTED_GPU="GPU-72c1b506-891d-b8bc-b353-e020585e1c47"
captured_freeze=None


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+"\n")
def require(value,message):
    if not value:raise RuntimeError(message)


def verify_sources():
    global captured_freeze
    current=sha(args.validation/"source-freeze.json")
    require(captured_freeze is None or captured_freeze==current,"validation freeze changed after audit startup")
    frozen=read(args.validation/"source-freeze.json")
    for name,expected in frozen["local_sources"].items():require(sha(args.validation/name)==expected,"local source changed: "+name)
    for name,expected in frozen["external_sources"].items():require(sha(Path(name))==expected,"external input changed: "+name)
    captured_freeze=current
    return dict(status="VALID",source_freeze_sha256=current)


def load_cpu(path):
    # Returning the deserialized CPU storage preserves original device tags
    # for reproducing the frozen GPU typed fingerprints without CUDA loading.
    devices={}
    def mapping(storage,location):
        devices[storage._cdata]=location
        return storage
    value=torch.load(path,map_location=mapping,weights_only=False)
    return value,devices


def digest(value,devices):
    h=hashlib.sha256()
    def token(x):
        b=x if isinstance(x,bytes) else str(x).encode()
        h.update(str(len(b)).encode()+b":"+b)
    def add(x):
        if isinstance(x,torch.Tensor):
            token("tensor");token(tuple(x.shape));token(x.dtype)
            token(devices[x.untyped_storage()._cdata])
            token(x.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            token("dict");token(len(x))
            for key in sorted(x,key=lambda k:(type(k).__name__,repr(k))):add(key);add(x[key])
        elif isinstance(x,(tuple,list)):
            token(type(x).__name__);token(len(x))
            for item in x:add(item)
        elif isinstance(x,float):token("float64");token(struct.pack("!d",x))
        else:token(type(x).__name__);token(repr(x))
    add(value)
    return h.hexdigest()


def semantic(state):
    # Preserve original tensor storages/device tags; remove exactly the
    # original replay's one observational field, without cloning tensors.
    state=dict(state)
    if "birth_death" in state:
        state["birth_death"]=dict(state["birth_death"])
        state["birth_death"]["last"]=dict(state["birth_death"].get("last",{}))
        state["birth_death"]["last"].pop("eval_seconds",None)
    return state


def metadata(state):
    bd=state.get("birth_death") or {}
    policy=bd.get("population_policy") or {}
    return dict(backend=bd.get("backend","knn_beta"),backend_schema=bd.get("backend_schema"),
                actual_backend=policy.get("actual_backend",bd.get("backend","knn_beta")),
                matching_sampler=policy.get("matching_sampler","controller"),
                settings=bd.get("settings"),population_policy=policy,counters=bd.get("counters"),last=bd.get("last"),
                transient_snapshot_absent="snapshot" not in bd,
                transient_geometry_cache_absent="latent_geometry" not in bd)


def rng_placement(state,devices):
    fields={"cpu_rng":state["cpu_rng"],"cuda_rng":state["cuda_rng"],
            **{"streams."+name:value for name,value in state["streams"].items()}}
    if "birth_death" in state:fields["birth_death.stream"]=state["birth_death"]["stream"]
    result={}
    for name,value in fields.items():
        require(value.dtype==torch.uint8 and devices[value.untyped_storage()._cdata]=="cpu","RNG was not saved CPU uint8: "+name)
        result[name]=dict(original_device="cpu",loaded_device=str(value.device),dtype=str(value.dtype),numel=value.numel())
    return result


def verify_runtime(receipt,inputs):
    require(receipt["source_freeze_sha256"]==sha(LEARNED/"SOURCE-FREEZE.json"),"learned source receipt differs")
    for field,file in (("inputs_sha256","INPUTS.json"),("protocol_sha256","PROTOCOL.md"),
                       ("preparation_receipt_sha256","preparation-receipt.json")):
        require(receipt[field]==sha(LEARNED/file),field+" differs")
    require(receipt["local_source_sha256"]==read(LEARNED/"SOURCE-FREEZE.json")["local_source_sha256"],"local source map differs")
    require(receipt["read_only_file_sha256"]==inputs["read_only_file_sha256"],"data/evaluator map differs")
    runtime=receipt["runtime"]
    require(runtime["device"]=="cuda:0" and runtime["visible_devices"]=="0","CUDA device receipt differs")
    require(runtime["physical_gpu"]["uuid"]==EXPECTED_GPU and runtime["cuda_uuid_normalized"]==EXPECTED_GPU,"physical GPU UUID differs")
    require(runtime["cuda_memory_fraction"]==.2 and runtime["cpu_threads"]==2,"resource guards differ")
    require(runtime["deterministic_algorithms"] and not runtime["tf32_matmul"] and not runtime["tf32_cudnn"],"determinism receipt differs")


def audit_training(problem,event):
    out=LEARNED/"training"/problem/args.variant
    result=read(out/"result.json")
    require(sha(out/"result.json")==event["result_sha256"],"completed result hash differs")
    if result["status"]!="COMPLETE":return dict(problem=problem,primary_status=result["status"],evidence_status="UNVERIFIED",error=result)
    require(event["returncode"]==0,"completed training process failed")
    inputs=read(LEARNED/"INPUTS.json");receipt=read(out/"config.json")
    require(receipt==result["receipt"],"configuration receipt differs from result")
    verify_runtime(receipt,inputs)
    require(receipt["package"]==inputs["variants"][args.variant],"candidate config/source map differs")
    require(receipt["seed"]==314159 and receipt["serial_backward"],"seed or execution mode differs")
    require(result["device"]=="cuda:0" and result["steps"]==2000 and result["final"]["step"]==2000,"full training budget differs")
    for field,value in (("num_particles",1024),("z_dim",128),("batch_size",128)):
        require(receipt["recipe"][field]==value,"recipe resource differs: "+field)
    for field,value in inputs["expected_initial_hashes"][problem].items():require(receipt[field]==value,"matched initialization receipt differs: "+field)
    curves=[json.loads(line) for line in (out/"metrics.jsonl").read_text().splitlines() if line]
    require([row["step"] for row in curves]==list(CHECKPOINTS),"checkpoint metric schedule differs")
    checkpoints=[]
    for step in CHECKPOINTS:
        name=f"checkpoint-{step:04d}.pt";path=out/name
        require(sha(path)==result["checkpoint_sha256"][name],"checkpoint bytes differ: "+name)
        saved,devices=load_cpu(path);state=saved["trainer"]
        require(saved["receipt_sha256"]==sha(out/"config.json"),"checkpoint config digest differs")
        require(state["completed_steps"]==step and saved["data_position"]==2*step*128,"checkpoint real-stream cursor differs")
        require(state["device"]=="cuda:0" and state["serial_backward"],"checkpoint device/execution mode differs")
        if step==0:
            for role,field in (("G","initial_generator_sha256"),("D","initial_critic_sha256")):
                h=hashlib.sha256()
                for key,value in state["models"][role].items():h.update(key.encode());h.update(value.contiguous().numpy().tobytes())
                require(h.hexdigest()==receipt[field],"saved initial model bytes differ: "+role)
            require(hashlib.sha256(state["models"]["prior"]["z"].contiguous().numpy().tobytes()).hexdigest()==receipt["initial_prior_sha256"],"saved initial prior bytes differ")
        bd=metadata(state)
        require(bd["backend"]==curves[list(CHECKPOINTS).index(step)]["diagnostics"]["birth_death"].get("backend","knn_beta"),"diagnostic and actual backend differ")
        if args.validation==STUDY/"validation":
            ready=read(STUDY/"READY.json")
            require(bd["backend_schema"]==ready["backend_schema"],"actual backend schema differs from freeze")
            require(bd["settings"]["latent_kernel"]==ready["latent_kernel"] and bd["settings"]["mass_policy"]==ready["mass_policy"],"actual kernel/mass policy differs from freeze")
        checkpoints.append(dict(step=step,sha256=sha(path),backend=bd,rng=rng_placement(state,devices)))
    metrics=result["final"]["metrics"]
    gate=None
    if problem=="toy":
        checks=dict(precision=metrics["precision"]>=.9,coverage=metrics["coverage"]==25,mass_tv=metrics["mass_tv"]<=.1)
        gate=dict(status="PASS" if all(checks.values()) else "FAIL",checks=checks,
                  thresholds=dict(precision_min=.9,coverage=25,mass_tv_max=.1))
    return dict(problem=problem,primary_status="COMPLETE",evidence_status="VALID",final=result["final"],
                toy_quality_gate=gate,image_quality_gate=None,checkpoints=checkpoints,runtime=receipt["runtime"],
                training_seconds=result["training_seconds"],updates_per_second=result["updates_per_second"],
                peak_reserved_mib=result["peak_gpu_reserved_bytes"]/2**20,
                artifact_hashes={str(p.relative_to(args.validation)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()})


def audit_replay(problem,result):
    require(result["status"] in ("PASS","FAIL","ERROR"),"replay primary status is invalid")
    if result["status"]=="ERROR":return dict(problem=problem,primary_status="ERROR",evidence_status="UNVERIFIED",error=result)
    verify_runtime(result["receipt"],read(LEARNED/"INPUTS.json"))
    checkpoint=Path(result["checkpoint"])
    require(result["checkpoint_sha256"]==sha(checkpoint),"source checkpoint changed")
    require(result["run_config_sha256"]==sha(checkpoint.parent/"config.json"),"source checkpoint config changed")
    require(result["start_step"]==1000 and result["steps_replayed"]==10,"replay budget/cursor differs")
    require(result["excluded_observational_fields"]==["birth_death.last.eval_seconds"],"equality exclusions differ")
    require(result["loss_keys"]==list(LOSS_KEYS),"returned loss set differs")
    require(len(result["branches"])==2 and len(result["per_update_comparison"])==10,"branch/update evidence incomplete")
    branches=[]
    for row in result["branches"]:
        path=Path(row["endpoint"])
        require(sha(path)==row["endpoint_sha256"],"endpoint bytes differ")
        saved,devices=load_cpu(path);state=saved["trainer"]
        require(saved["source_checkpoint_sha256"]==sha(checkpoint),"endpoint source digest differs")
        require(saved["data_position"]==2*1010*128 and state["completed_steps"]==1010,"endpoint cursor differs")
        require(saved["update_fingerprints"]==row["update_fingerprints"],"per-update saved fingerprints differ")
        require([r["step"] for r in row["update_fingerprints"]]==list(range(1001,1011)),"replay update schedule differs")
        require(digest(state,devices)==row["whole_state_sha256"],"whole endpoint fingerprint differs")
        require(digest(semantic(state),devices)==row["semantic_state_sha256"],"semantic endpoint fingerprint differs")
        require(digest(saved["loss_tensors"],devices)==row["losses_sha256"],"returned loss fingerprint differs")
        for losses,fp in zip(saved["loss_tensors"],row["update_fingerprints"]):
            require(digest(losses,devices)==fp["loss_sha256"],"per-update loss bits differ")
        branches.append(dict(branch=row["branch"],endpoint_sha256=sha(path),backend=metadata(state),
                             rng=rng_placement(state,devices),verified_gpu_typed_fingerprints=True))
    required=all(result.get(key) is True for key in ("losses_bit_identical","semantic_state_bit_identical","restoration_semantic_bit_identical"))
    required=required and all(result["semantic_sections_bit_identical"].values()) and all(
        row["losses_bit_identical"] and row["semantic_state_bit_identical"] and all(row["semantic_sections_bit_identical"].values())
        for row in result["per_update_comparison"])
    require(result["status"]==("PASS" if required else "FAIL"),"original replay status differs from its exact comparisons")
    return dict(problem=problem,primary_status=result["status"],evidence_status="VALID",branches=branches,
                losses_bit_identical=result["losses_bit_identical"],semantic_state_bit_identical=result["semantic_state_bit_identical"],
                restoration_semantic_bit_identical=result["restoration_semantic_bit_identical"],
                per_update_comparison=result["per_update_comparison"],excluded_observational_fields=result["excluded_observational_fields"],
                peak_reserved_mib=result["peak_gpu_reserved_bytes"]/2**20)


def summarize(records,integrity):
    full={name:records.get(name,dict(primary_status="PENDING",evidence_status="PENDING"))
          for name in ("training-toy","training-mnist","replay-toy","replay-mnist")}
    complete=all(row["primary_status"]!="PENDING" for row in full.values())
    write(args.output/"summary.json",dict(complete=complete,records=full,source_integrity=integrity,
          cpu_only=True,cuda_initialized=torch.cuda.is_initialized(),scope=__doc__))
    lines=["# Corrected learned CUDA saved-artifact audit","",
           "Completed jobs only. All tensor loads use CPU storages; original device tags are retained for the frozen GPU fingerprints.","",
           "| Record | Primary | Evidence | Toy gate | Peak reserved MiB |","|---|---|---|---|---:|"]
    for name,row in full.items():lines.append(f"| {name} | {row['primary_status']} | {row['evidence_status']} | {(row.get('toy_quality_gate') or {}).get('status','—')} | {row.get('peak_reserved_mib','—')} |")
    lines+=["","Checkpoint/endpoint backend, matching sampler, kernel, mass policy, parent counters and RNG placement are retained in summary.json. MNIST has no newly introduced numerical gate.",""]
    (args.output/"REPORT.md").write_text("\n".join(lines))
    return complete


records={};done=set();integrity=verify_sources()
write(args.output/"AUDITOR-IDENTITY.json",dict(checker_sha256=sha(Path(__file__)),source_integrity=integrity,read_only=True))
while True:
    events=[]
    for line in (args.validation/"run.log").read_text().splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:continue
    changed=not records
    for event in events:
        name=event.get("name","")
        if event.get("event")!="job_complete" or name in done:continue
        if name not in (f"learned-toy-{args.variant}",f"learned-mnist-{args.variant}",f"replay-{args.variant}"):continue
        integrity=verify_sources()
        if name.startswith("learned-"):
            problem="toy" if name.startswith("learned-toy-") else "mnist"
            try:row=audit_training(problem,event)
            except Exception as error:row=dict(problem=problem,primary_status=event["status"],evidence_status="INVALID",error=str(error),traceback=traceback.format_exc())
            records["training-"+problem]=row
            print(json.dumps(dict(event="learned_completed_artifact_audit",problem=problem,primary_status=row["primary_status"],
                                 evidence_status=row["evidence_status"],toy_gate=row.get("toy_quality_gate"),error=row.get("error"))),flush=True)
        else:
            aggregate_path=LEARNED/f"replay-{args.variant}.json"
            require(sha(aggregate_path)==event["result_sha256"],"aggregate replay result digest differs")
            for problem,result in read(aggregate_path).items():
                try:row=audit_replay(problem,result)
                except Exception as error:row=dict(problem=problem,primary_status=result["status"],evidence_status="INVALID",error=str(error),traceback=traceback.format_exc())
                records["replay-"+problem]=row
                print(json.dumps(dict(event="replay_completed_artifact_audit",problem=problem,primary_status=row["primary_status"],
                                     evidence_status=row["evidence_status"],error=row.get("error"))),flush=True)
        done.add(name);changed=True
        require(not torch.cuda.is_initialized(),"saved-artifact audit created a CUDA context")
    if changed or not args.watch:
        complete=summarize(records,integrity)
        if complete:break
    if not args.watch or any(row.get("event")=="queue_aborted" for row in events):break
    time.sleep(10)

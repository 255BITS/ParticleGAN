"""One fresh canonical two_pole case using the maintained policy supervisor.

Import is inert. Preparation and certification consume saved declarations/bytes;
the only model call is run_first_case inside the supervised admitted child.
ROOT must wrap every parent operation in its single authorized parent ledger.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
from importlib.machinery import NamespaceLoader
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

MODULE = "experiments.forge.canonical_two_pole_case"
MEMBER = "experiments/forge/canonical_two_pole_case.py"
SCHEMA = "pg_canonical_two_pole_first_case_v1"
CASE = "canonical-two-pole-full-atlas-ember552-v1"
CONFIG = "configs/100gaussians/atlas.json"
TASK = "configs/forge/tasks/two_pole.json"
PROTOCOL = "configs/forge/protocols/screening.json"
HOST_MEMORY_MB = 2048


def _utils():
    from .contracts import atomic_json, file_hash, read_json, stable_hash
    return atomic_json, file_hash, read_json, stable_hash


def _base(request):
    return {k: deepcopy(v) for k, v in request.items()
            if k not in {"admission", "target", "command"}}


def validate_request(request):
    from .canonical_two_pole_repeat import validate_scientific_repeat
    _, _, _, digest = _utils()
    if request.get("schema") != SCHEMA or request.get("case_id") != CASE:
        raise ValueError("only the exact first canonical two-pole case is supported")
    repeat = validate_scientific_repeat(request)
    runtime = request["runtime"]
    if (runtime.get("device") != "cpu" or runtime.get("torch_threads") != 1
            or runtime.get("deterministic") is not True or runtime.get("tf32") is not False
            or runtime.get("dtype") != "float32" or runtime.get("gpus") != 0):
        raise ValueError("the declared CPU1/float32/no-GPU compute contract is required")
    if request["binding"]["source_contract"]["observation"]["observations"] != [math.ceil(i*80/24) for i in range(1,25)]:
        raise ValueError("ordinary 80-update/24-read observation protocol changed")
    source = request["source"]
    if digest(source["files"]) != source["digest"]:
        raise ValueError("invalid execution-source manifest")
    return repeat


def source_guard(request, root):
    from .sources import verify_snapshot
    _, file_hash, read_json, _ = _utils()
    validate_request(request)
    root = Path(root).resolve()
    manifest = {k: v for k, v in request["source"].items() if k != "snapshot_path"}
    if (root != Path(request["source"]["snapshot_path"]).resolve()
            or read_json(root / "forge-source.json") != manifest
            or Path(__file__).resolve() != root / MEMBER):
        raise ValueError("only the actual copied-source controller may execute")
    verify_snapshot(root, manifest)
    for name,module in tuple(sys.modules.items()):
        if name.split(".",1)[0] not in {"particlegan","benchmarks","experiments","lib"} or module is None:
            continue
        loaded=getattr(module,"__file__",None)
        namespaces=tuple(getattr(module,"__path__",()))
        if loaded is None and (not namespaces or not isinstance(getattr(getattr(module,"__spec__",None),"loader",None),NamespaceLoader)):
            raise ValueError("missing-file scientific module is not an owned namespace: "+name)
        if loaded is not None:
            member=Path(loaded).resolve()
            if not member.is_relative_to(root) or member.relative_to(root).as_posix() not in manifest["files"]:
                raise ValueError("foreign already-imported scientific module: "+name)
        for namespace in namespaces:
            if not Path(namespace).resolve().is_relative_to(root):
                raise ValueError("foreign scientific package namespace: "+name)


def preflight(request, root):
    from .canonical_two_pole_adapter import derive_host_function, resolve_binding
    _, _, _, digest = _utils()
    source_guard(request, root)
    binding = resolve_binding(root, request["candidate"], request["task"], request["protocol"])
    if binding != request["binding"]:
        raise ValueError("copied binding differs from the preregistered binding")
    compile(derive_host_function((Path(root)/"benchmarks/locked_shared/two_pole.py").read_text()),
            "<canonical-two-pole-source-only-overlay>", "exec")
    if any(n == "torch" or n.startswith("torch.") or n == "particlegan" or n.startswith("particlegan.")
           for n in sys.modules):
        raise ValueError("copied metadata preflight imported a scientific model package")
    return dict(schema="pg_canonical_two_pole_copied_preflight_v1", status="PASS",
        frozen_request_digest=digest(_base(request)), source_digest=request["source"]["digest"],
        repeat=deepcopy(request["protocol"]["scientific_repeat"]),
        source_contract_sha256=binding["source_contract_sha256"],
        model_constructors=0, forwards=0, updates=0, evaluation_draws=0)


def prepare(root, queue_root, output, private):
    """ROOT calls this only inside an active paid parent phase."""
    from .canonical_two_pole_adapter import resolve_binding
    from .canonical_two_pole_repeat import make_scientific_repeat
    from .queue import host_capacity
    from .sources import compute_profile, inspect_source, runtime_manifest, snapshot_source
    atomic_json, file_hash, read_json, digest = _utils()
    root, output, private = Path(root).resolve(), Path(output).resolve(), Path(private).resolve()
    if output.exists():
        raise ValueError("a fresh first-case output must not already exist; no reset")
    candidate = dict(schema_version=1, id="atlas-full-original-common26-ember552",
        trainer_family="atlas", recipe_preset="atlas", recipe_overrides=read_json(root/CONFIG),
        initializer="deterministic_orthogonal", extensions={})
    task, protocol = read_json(root/TASK), read_json(root/PROTOCOL)
    binding = resolve_binding(root, candidate, task, protocol)
    capacity = host_capacity()
    if capacity["cpu_threads"] < 1 or capacity["available_memory_mb"] < HOST_MEMORY_MB:
        raise ValueError("current CPU1/2048-MiB host fit unavailable before reservation")
    runtime = dict(**runtime_manifest(), device="cpu", gpus=0, torch_threads=1,
        deterministic=True, tf32=False, dtype="float32", compute_profile=compute_profile("cpu",threads=1))
    source = inspect_source(root, extra_paths=(TASK, PROTOCOL))
    origin_namespace = Path(queue_root)/"policy/source-origins"/digest(source["origin_commit"])
    snapshot = snapshot_source(root, origin_namespace, source)
    source = {**source,"snapshot_path":str(snapshot)}
    request = dict(schema=SCHEMA,case_id=CASE,task=task,candidate=candidate,protocol=protocol,
                   binding=binding,source=source,runtime=runtime)
    repeat = make_scientific_repeat(request)
    request["protocol"]["scientific_repeat"] = repeat
    validate_request(request)
    private.mkdir(parents=True,exist_ok=True)
    request_path=private/"frozen-request.json"
    atomic_json(request_path,request)
    environment={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONPATH":str(snapshot),
                 "PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    done=subprocess.run([sys.executable,"-u","-B","-m",MODULE,"--preflight",str(request_path)],
                        cwd=snapshot,env=environment,text=True,capture_output=True,timeout=30)
    (private/"copied-preflight.log").write_text(done.stdout+done.stderr)
    if done.returncode:
        raise ValueError("copied model-free preflight refused: "+done.stderr[-3000:])
    proof=json.loads(done.stdout)
    if proof.get("status")!="PASS" or proof.get("frozen_request_digest")!=digest(request):
        raise ValueError("copied-source preflight is not bound to the exact request")
    atomic_json(private/"copied-preflight.json",proof)
    spec=dict(id=CASE,representation_card={"sha256":binding["source_contract_sha256"]},
        export_grace_seconds=0,retries=0,frames=24,scientific_repeat=repeat,
        resources={"host_memory_mb":HOST_MEMORY_MB})
    packet=dict(schema=SCHEMA,spec=spec,spec_sha256=digest(spec),request=request,
        protocol=request["protocol"],scientific_repeat=repeat,
        execution_source=source,source={"commit":source["origin_commit"],"digest":source["digest"]},
        case_definitions={CASE:{"task":task,"candidate":candidate,"protocol":request["protocol"],
                                "binding":binding}},
        capacity_preflight={"capacity":capacity,"required_cpu_threads":1,"required_host_memory_mb":HOST_MEMORY_MB},
        runtime_contract=runtime,lane_runtime=runtime,family_paid_budget_seconds=300,
        rows=[dict(id=CASE,task_id="two_pole",timeout_seconds=300,allowance_seconds=300,status="NOT_RUN")],
        copied_preflight=proof,output=str(output),queue_root=str(Path(queue_root).resolve()),
        copied_preflight_pin={"path":str(private/"copied-preflight.json"),
            "sha256":file_hash(private/"copied-preflight.json"),"bytes":(private/"copied-preflight.json").stat().st_size},
        frozen_request_pin={"path":str(request_path),"sha256":file_hash(request_path),"bytes":request_path.stat().st_size})
    atomic_json(private/"prepared.json",packet)
    return packet


def admission_guard(request):
    from .queue import lease_held, process_identity
    _, _, read_json, _ = _utils()
    admission=request["admission"]
    directory=Path(admission["lease_paths"][-1]).parent
    supervisor=read_json(directory/"supervisor-request.json")
    child=read_json(directory/"child.json")
    if (supervisor.get("token")!=admission["token"] or child.get("token")!=admission["token"]
        or child.get("pid")!=os.getpid() or child.get("process_identity")!=process_identity(os.getpid())
        or supervisor.get("command")!=request["command"]
        or supervisor.get("source")!=request["source"]
        or supervisor.get("started_monotonic")!=admission["started_monotonic"]
        or supervisor.get("deadline_monotonic")!=admission["deadline_monotonic"]
        or child.get("deadline_monotonic")!=admission["deadline_monotonic"]
        or time.monotonic()>=admission["deadline_monotonic"]
        or admission["deadline_monotonic"]-admission["started_monotonic"]!=300):
        raise ValueError("actual maintained child/admission token/deadline does not match")
    for fd,path in zip(admission["lease_fds"],admission["lease_paths"],strict=True):
        inherited,declared=os.fstat(fd),os.stat(path)
        if (inherited.st_dev,inherited.st_ino)!=(declared.st_dev,declared.st_ino) or not lease_held(Path(path)):
            raise ValueError("the inherited physical execution lease is not owned")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("this canonical case must hide CUDA")


def _media(target,states,observations,grade):
    """Render only the saved ordinary observations, never another evaluator draw."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from PIL import Image
    frames=[]
    if [s["step"] for s in states]!=[o["step"] for o in observations]:
        raise ValueError("goal media requires its same24 retained observations")
    for i,(state,obs) in enumerate(zip(states,observations,strict=True)):
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        coordinates=[row[0] for row in state["particles"]]
        ax=axes[0]
        ax.axvline(-1,color="#2683a8",alpha=.6);ax.axvline(1,color="#2683a8",alpha=.6)
        ax.axvspan(-.3,.3,color="#f3d9ad",alpha=.5)
        ax.scatter(coordinates,list(range(12)),c="#6741b3",s=45)
        bound=max(1.3,max(abs(x) for x in coordinates)+.1)
        ax.set(xlim=(-bound,bound),ylim=(-1,12),xlabel="Live particle coordinate",ylabel="Particle row")
        ax.set_title("Travel from zero; target poles at ±1")
        ax=axes[1];curve=observations[:i+1];clocks=[o["step"] for o in curve]
        ax.plot(clocks,[o["mean_abs"] for o in curve],"o-",label="Mean |coordinate| (≥0.30)",color="#6741b3")
        ax.plot(clocks,[o["grad_med"] for o in curve],"o-",label="Critic median |slope| (≤1.00)",color="#159c85")
        ax.axhline(.3,color="#6741b3",linestyle=":");ax.axhline(1,color="#159c85",linestyle=":")
        ax.set(xlim=(0,80),ylim=(0,max(1.15,max(o["grad_med"] for o in observations)*1.1)),
               xlabel="Completed outer updates",ylabel="Original live metrics")
        ax.legend(loc="upper left",fontsize=8)
        gates=obs["mean_abs"]>=.3 and obs["grad_med"]<=1
        fig.suptitle(f"Full Atlas · canonical two_pole · update {obs['step']}/80 · observed gates {'PASS' if gates else 'FAIL'}")
        fig.tight_layout()
        buffer=io.BytesIO();fig.savefig(buffer,format="png",dpi=100);plt.close(fig)
        frame=Image.open(io.BytesIO(buffer.getvalue())).convert("RGB");frames.append(frame)
    frames[-1].save(target/"goal-final.png")
    frames[0].save(target/"goal.gif",save_all=True,append_images=frames[1:],duration=250,loop=0)


def child(path):
    from .canonical_two_pole_adapter import run_first_case
    from .canonical_two_pole_repeat import FreshRepeatGuard
    atomic_json,file_hash,read_json,digest=_utils()
    path=Path(path).resolve();raw_request=file_hash(path);request=read_json(path)
    root=Path(__file__).resolve().parents[2];target=Path(request["target"])
    def guard_source():
        if file_hash(path)!=raw_request:raise ValueError("admitted request changed")
        source_guard(request,root)
    guard_source();admission_guard(request)
    from .sources import runtime_manifest,compute_profile
    actual=runtime_manifest()
    if (any(request["runtime"].get(k)!=v for k,v in actual.items())
            or request["runtime"]["compute_profile"]!=compute_profile("cpu",threads=1)):
        raise ValueError("the actual child runtime differs from the frozen CPU cohort")
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    fresh=FreshRepeatGuard(request,source_guard=guard_source,admission_guard=lambda:admission_guard(request))
    original=fresh.construct
    def record_start(factory,reader):
        owner=original(factory,reader)
        atomic_json(target/"INITIALIZATION.json",fresh.require_owned(owner))
        atomic_json(target/"MODEL_STARTED.json",dict(schema="pg_canonical_two_pole_actual_start_v1",
            initialized_monotonic=time.monotonic(),frozen_request_digest=digest(_base(request)),
            initialization_sha256=file_hash(target/"INITIALIZATION.json"),models_constructed=True,
            completed_updates=0,device="cpu",gpus=0))
        print(json.dumps({"event":"actual_model_start","case":"two_pole","seed":0,"device":"cpu"}),flush=True)
        return owner
    fresh.construct=record_start
    result=run_first_case(root,request,source_guard=guard_source,fresh_repeat_guard=fresh)
    retained=result.pop("retained_goal_states");complete=result.pop("complete_state")
    states=[dict(step=s["step"],particles=s["particles"].tolist(),gradient=s["gradient"].tolist()) for s in retained]
    atomic_json(target/"raw-result.json",result);atomic_json(target/"goal-states.json",states)
    torch.save(complete,target/"state.pt")
    from .views import grade_result
    grade=grade_result(request["task"],result)
    atomic_json(target/"grade.json",grade)
    _media(target,states,result["evidence"]["observations"],grade)
    guard_source();admission_guard(request)
    files={name:{"sha256":file_hash(target/name),"bytes":(target/name).stat().st_size}
           for name in ("INITIALIZATION.json","MODEL_STARTED.json","raw-result.json","goal-states.json",
                        "state.pt","grade.json","goal.gif","goal-final.png")}
    attestation=dict(schema="pg_canonical_two_pole_child_attestation_v1",
        frozen_request_digest=digest(_base(request)),source_digest=request["source"]["digest"],
        repeat=request["protocol"]["scientific_repeat"],runtime_digest=digest(request["runtime"]),
        token_sha256=hashlib.sha256(request["admission"]["token"].encode()).hexdigest(),
        completed_updates=80,observation_steps=[s["step"] for s in states],grade=grade,files=files)
    atomic_json(target/"attestation.json",attestation)
    print(json.dumps({"event":"complete","case":"two_pole","status":grade["status"]}),flush=True)
    return 0 if grade["status"]=="PASS" else 1


def certify(packet,target,terminal,token):
    """Independent retained-byte grading only; no new model, draw or update."""
    from .views import grade_result
    _,file_hash,read_json,digest=_utils()
    target=Path(target);request=packet["request"]
    if terminal.get("token")!=token or terminal.get("attempt_status")!="completed":
        raise ValueError("matching completed maintained supervisor terminal is required")
    proof=read_json(target/"attestation.json")
    if (proof.get("schema")!="pg_canonical_two_pole_child_attestation_v1"
        or proof.get("frozen_request_digest")!=digest(request)
        or proof.get("source_digest")!=request["source"]["digest"]
        or proof.get("repeat")!=validate_request(request)
        or proof.get("runtime_digest")!=digest(request["runtime"])
        or proof.get("token_sha256")!=hashlib.sha256(token.encode()).hexdigest()
        or proof.get("completed_updates")!=80
        or proof.get("observation_steps")!=[math.ceil(i*80/24) for i in range(1,25)]):
        raise ValueError("child attestation is not this exact fresh canonical repeat")
    expected={"INITIALIZATION.json","MODEL_STARTED.json","raw-result.json","goal-states.json",
              "state.pt","grade.json","goal.gif","goal-final.png"}
    if set(proof["files"])!=expected:raise ValueError("incomplete child artifact manifest")
    for name,pin in proof["files"].items():
        if file_hash(target/name)!=pin["sha256"] or (target/name).stat().st_size!=pin["bytes"]:
            raise ValueError("retained child bytes changed: "+name)
    result=read_json(target/"raw-result.json");grade=grade_result(request["task"],result)
    if grade!=proof["grade"] or grade!=read_json(target/"grade.json"):
        raise ValueError("independent retained-byte grade differs")
    expected_exit=0 if grade["status"]=="PASS" else 1
    if type(terminal.get("child_returncode")) is not int or terminal["child_returncode"]!=expected_exit:
        raise ValueError("the actual child exit does not match its independently retained grade")
    initial=read_json(target/"INITIALIZATION.json")
    if initial.get("repeat")!=request["protocol"]["scientific_repeat"] or initial.get("factory_calls")!=1:
        raise ValueError("actual one-time initialization witness is missing")
    start=read_json(target/"MODEL_STARTED.json")
    if (start.get("initialization_sha256")!=proof["files"]["INITIALIZATION.json"]["sha256"]
        or start.get("models_constructed") is not True or start.get("completed_updates")!=0
        or not packet["started_monotonic"]<=start["initialized_monotonic"]<packet["deadline_monotonic"]):
        raise ValueError("model did not start inside its actual admitted deadline")
    return dict(grade=grade,actual_model_started=start,attestation_sha256=file_hash(target/"attestation.json"),
                result_sha256=proof["files"]["raw-result.json"]["sha256"],media="goal.gif",full_protocol_complete=grade["status"] in {"PASS","FAIL"})


def run(packet,output,*,parent,budget,predecessors):
    """A single physical attempt, no retries/reuse/next-task execution."""
    from .policy_execution import PolicyCoordinator
    from .sources import verify_snapshot
    atomic_json,file_hash,read_json,digest=_utils()
    output=Path(output).resolve()
    for field,expected in (("frozen_request_pin",packet["request"]),("copied_preflight_pin",packet["copied_preflight"])):
        pin=packet[field];path=Path(pin["path"])
        if file_hash(path)!=pin["sha256"] or path.stat().st_size!=pin["bytes"] or read_json(path)!=expected:
            raise ValueError("persisted copied-source prerequisite changed: "+field)
    proof=packet["copied_preflight"]
    if (proof.get("schema")!="pg_canonical_two_pole_copied_preflight_v1" or proof.get("status")!="PASS"
        or proof.get("frozen_request_digest")!=digest(packet["request"])
        or proof.get("source_digest")!=packet["execution_source"]["digest"]
        or proof.get("repeat")!=validate_request(packet["request"])
        or proof.get("source_contract_sha256")!=packet["request"]["binding"]["source_contract_sha256"]
        or any(type(proof.get(k)) is not int or proof[k]!=0 for k in
               ("model_constructors","forwards","updates","evaluation_draws"))):
        raise ValueError("exact zero-model copied-source preflight is missing before reservation")
    budget.require_first_case_fit(budget._maintained,parent.snapshot(),predecessors=predecessors)
    verify_snapshot(Path(packet["execution_source"]["snapshot_path"]),packet["execution_source"])
    validate_request(packet["request"])
    coordinator=PolicyCoordinator(packet["queue_root"])
    key,canonical=coordinator.register(packet,output,"atlas",packet["lane_runtime"])
    if canonical.resolve()!=output:raise ValueError("fresh case cannot attach to an older canonical study")
    row=packet["rows"][0];trial={"family":"atlas","recipe_overrides":packet["request"]["candidate"]["recipe_overrides"]}
    attempt=coordinator.attempt_key(packet,trial,row)
    if coordinator.retained(attempt) is not None:
        raise ValueError("recognized fresh repeat already has an attempt; no reuse or retry")
    with coordinator.study_lease(key) as study:
        if study is None:raise ValueError("first-case study lease is busy")
        with coordinator.admit(attempt,packet,row,"cpu") as (admitted,lease):
            if admitted["status"]!="running" or lease is None:
                raise ValueError("first-case physical admission refused: "+admitted.get("reason",admitted["status"]))
            packet.update(started_monotonic=admitted["started_monotonic"],deadline_monotonic=admitted["deadline_monotonic"])
            target=output/"two_pole"
            terminal_path=Path(admitted["lease_path"]).parent/"supervisor-terminal.json"
            previous=os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"]=""
            certificate=None
            try:
                target.mkdir()
                request=deepcopy(packet["request"])
                path=target/"request.json"
                command=[sys.executable,"-u","-B","-m",MODULE,"--child",str(path)]
                request.update(target=str(target),command=command,admission={
                    "token":admitted["token"],"started_monotonic":admitted["started_monotonic"],
                    "deadline_monotonic":admitted["deadline_monotonic"],
                    "lease_fds":[study.fileno(),lease.fileno()],"lease_paths":[str(study.name),str(lease.name)]})
                atomic_json(path,request)
                row.update(status="RUNNING",attempt_key=attempt)
                atomic_json(output/"study.json",packet)
                print(json.dumps({"event":"launch","case":"two_pole","allowance_seconds":300,
                                  "source_digest":packet["execution_source"]["digest"]}),flush=True)
                with parent.pause():
                    coordinator.launch(command,packet,target/"run.log",(study,lease),300)
                terminal=read_json(terminal_path)
                certificate=certify(packet,target,terminal,admitted["token"])
                status=certificate["grade"]["status"]
                reason=certificate["grade"].get("reason")
            except Exception as error:
                terminal=read_json(terminal_path) if terminal_path.exists() else None
                status="INCOMPLETE" if isinstance(error,subprocess.TimeoutExpired) else "INVALID"
                reason=type(error).__name__+": "+str(error)
            finally:
                if previous is None:os.environ.pop("CUDA_VISIBLE_DEVICES",None)
                else:os.environ["CUDA_VISIBLE_DEVICES"]=previous
            matched=terminal if terminal and terminal.get("token")==admitted["token"] else None
            can_certify=(certificate is not None and status in {"PASS","FAIL"}
                         and matched is not None and matched["paid_wall_seconds"]<=300)
            cost=budget.case_cost(budget._maintained,matched,expected_token=admitted["token"],certified=can_certify)
            if cost["overrun_seconds"]>0:
                status,reason="INCOMPLETE","actual case exceeded the unchanged300-second allowance"
                if certificate is not None:certificate["full_protocol_complete"]=False
            scientific=dict(status=status,reason=reason,certificate=certificate,**cost)
            coordinator.complete(attempt,scientific)
            row.update(scientific)
            packet["budget_accounting"]=budget.inclusive_accounting(budget._maintained,parent.snapshot(),
                predecessors=predecessors,case={"id":"two_pole",**scientific})
            packet["progression"]={"stopped":True,"reason":"first non-PASS" if status!="PASS" else
                "next canonical unused_token_hold has no reviewed full-policy adapter; no next-case authorization",
                "next_task":"unused_token_hold","next_status":"BLOCKED" if status=="PASS" else "NOT_RUN",
                "later_25_status":"NOT_RUN","whole26_complete":False,"default_adoption":False,"speed_ranking":False}
            atomic_json(output/"study.json",packet)
            return packet


def main():
    _,_,read_json,_=_utils()
    if len(sys.argv)!=3 or sys.argv[1] not in {"--preflight","--child"}:
        raise SystemExit("Use the ROOT-owned paid parent driver; worker accepts exact copied preflight or child only")
    if sys.argv[1]=="--preflight":
        print(json.dumps(preflight(read_json(sys.argv[2]),Path(__file__).resolve().parents[2]),sort_keys=True));return 0
    return child(sys.argv[2])


if __name__=="__main__":
    raise SystemExit(main())

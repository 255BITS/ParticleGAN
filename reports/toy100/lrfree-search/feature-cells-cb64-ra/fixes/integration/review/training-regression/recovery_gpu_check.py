"""Root-only GPU fixed-input count recovery contract; CPU is a planning check."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package-root",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--device",choices=("cpu","cuda:0"),default="cuda:0")
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
require_new=not (args.output/"result.json").exists()
if not require_new:raise SystemExit("result exists; choose a new output")
os.environ.update(CUDA_VISIBLE_DEVICES="0" if args.device=="cuda:0" else "",CUDA_DEVICE_ORDER="PCI_BUS_ID",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
for name in ("ABSENT","ABSENT_START","ABSENT_END","LRFREE_NATIVE_TEST_STEPS"):os.environ.pop(name,None)
sys.dont_write_bytecode=True
sys.path.insert(0,str(args.package_root.resolve()))
import torch
import particlegan.feature_cells as module
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
if Path(module.__file__).resolve().parent.parent!=args.package_root.resolve():raise RuntimeError("wrong selected package")
gpu=None
if args.device=="cuda:0":
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.2,0)
    props=torch.cuda.get_device_properties(0)
    uuid="GPU-"+str(props.uuid).removeprefix("GPU-").lower()
    if uuid!="GPU-72c1b506-891d-b8bc-b353-e020585e1c47":raise RuntimeError("wrong physical GPU0")
    gpu=dict(name=props.name,uuid=uuid,memory_fraction=.2,total_bytes=props.total_memory)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def plain(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value
def convert(value):
    if isinstance(value,torch.Tensor):return value.to(args.device)
    if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(convert(v) for v in value)
    return value


reference=HERE/"V3-FEATURE-CELLS.py"
if sha(reference)!="080dccbc404e1fe4112cd18275eaf94b8f68e6086996b88b5f9b5718770573c1":raise RuntimeError("v3 reference source changed")
tree=ast.parse(reference.read_text())
node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=="FeatureCellSnapshot"
          for n in c.body if isinstance(n,ast.FunctionDef) and n.name=="ordinary_transport")
namespace=dict(module.__dict__)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(reference),"exec"),namespace)
v3_method=namespace["ordinary_transport"]
original_path=ROOT/"stability/mass-gpu-inputs.pt"
if sha(original_path)!="959453b0c0b88508780c5389fecd0ddb6b341660e5f24aacd3805291be829859":raise RuntimeError("original fixed inputs changed")
original=torch.load(original_path,map_location="cpu",weights_only=False)
input_files=[original_path,HERE/"snapshot-1000.pt",HERE/"snapshot-2000.pt"]
if (HERE/"READY.json").exists():
    ready=json.loads((HERE/"READY.json").read_text())
    for path in input_files[1:]:
        if sha(path)!=ready["evidence_sha256"][path.name]:raise RuntimeError("saved reconstructed input changed: "+path.name)
sources_before={str(p.relative_to(args.package_root)):sha(p) for p in sorted(args.package_root.rglob("*.py"))}
records=[];failure=None


def plan(value,method,seed,flags=None,pvalues=None):
    snap=module.FeatureCellSnapshot.__new__(module.FeatureCellSnapshot)
    for key,item in convert(deepcopy(value["snapshot"])).items():setattr(snap,key,item)
    snap.device=torch.device(args.device)
    q=value["q"].to(args.device)
    flags=(value["flags"] if flags is None else flags).to(args.device)
    pvalues=(value["pvalues"] if pvalues is None else pvalues).to(args.device)
    stream=torch.Generator(device=args.device).manual_seed(seed)
    comparison=convert(deepcopy(value["comparison"]))
    child,parent,detail=method(snap,q,flags,comparison,generator=stream,pvalues=pvalues)
    iso_child,iso_parent,isolation=snap.select_parents(q,flags,ordinary_children=child,ordinary_parents=parent,
                generator=stream,pvalues=pvalues)
    return snap,flags,pvalues,child,parent,detail,iso_child,iso_parent,isolation,stream.get_state()


def check(name,original_value,reference_plan,fixed,extra=None):
    snap,flags,pvalues,child,parent,detail,iso_child,iso_parent,isolation,rng=fixed
    ids=snap.query_cell_ids
    supported=torch.bincount(ids[~flags],minlength=snap.cells)
    planned=supported-torch.bincount(ids[child[~flags[child]]],minlength=snap.cells)+torch.bincount(ids[parent],minlength=snap.cells)
    target=snap._group_counts(detail["target_counts"])
    previous=snap._group_counts(supported)
    checks=dict(ordinary_budget=len(child)<=int(.05*len(flags)),
        unique_children=len(torch.unique(torch.cat((child,iso_child))))==len(child)+len(iso_child),
        unique_parents=len(torch.unique(torch.cat((parent,iso_parent))))==len(parent)+len(iso_parent),
        ordinary_parent_not_deleted=not bool(torch.isin(parent,child).any()),
        isolation_parent_not_used_or_deleted=not bool(torch.isin(iso_parent,torch.cat((child,parent))).any()),
        ordinary_supported_parents=bool((~flags[parent]&(pvalues[parent]>.05)).all()),
        planned_supported_counts=torch.equal(planned,detail["planned_supported_counts"]),
        isolation_guard_unchanged=isolation["guard_passed"]==(0<int(flags.sum())<=.05*len(flags)))
    if int(flags.sum())>.05*len(flags):
        checks.update(flagged_only_deaths=bool(flags[child].all()),isolation_still_rejects=len(iso_child)==0,
                      actual_target_caps=bool((snap._group_counts(planned)<=torch.maximum(target,previous)).all()))
    else:
        checks.update(v3_actions_exact=all(torch.equal(fixed[i],reference_plan[i]) for i in (3,4,6,7)),
                      v3_rng_exact=torch.equal(rng,reference_plan[9]))
    if extra:checks.update(extra)
    row=dict(name=name,flags=int(flags.sum()),reference_ordinary=len(reference_plan[3]),ordinary_moves=len(child),
             flagged_deaths=int(flags[child].sum()),supported_deaths=int((~flags[child]).sum()),
             unique_parents=len(torch.unique(torch.cat((parent,iso_parent)))),isolation_moves=len(iso_child),
             group_targets=target,planned_supported=snap._group_counts(planned),checks=checks)
    records.append(plain(row));print(json.dumps(dict(event="case_before_assertions",row=plain(row))),flush=True)
    for key,value in checks.items():print(json.dumps(dict(event="named_check",case=name,name=key,passed=value)),flush=True)
    assert all(checks.values()),name+": "+", ".join(k for k,v in checks.items() if not v)


try:
    for name,value in original["scenarios"].items():
        previous=plan(value,v3_method,original["seed"])
        fixed=plan(value,module.FeatureCellSnapshot.ordinary_transport,original["seed"])
        check(name,value,previous,fixed,dict(original_46_repairs=len(fixed[6])==46))
    for step in (1000,2000):
        value=torch.load(HERE/f"snapshot-{step:04d}.pt",map_location="cpu",weights_only=False)
        previous=plan(value,v3_method,314159)
        fixed=plan(value,module.FeatureCellSnapshot.ordinary_transport,314159)
        check(f"saved_toy_{step}",value,previous,fixed,dict(count_recovery_acts=len(fixed[3])>len(previous[3])))
    zero=plan(value,module.FeatureCellSnapshot.ordinary_transport,314159,pvalues=torch.zeros_like(value["pvalues"]))
    check("no_eligible_parent",value,zero,zero,dict(no_actions=len(zero[3])==0 and len(zero[6])==0))
    value=deepcopy(original["scenarios"]["rare_hole"])
    proto=plan(value,module.FeatureCellSnapshot.ordinary_transport,original["seed"])
    snap,flags=proto[:2];ids=snap.query_cell_ids;groups=snap._mass_topology()
    clean=snap._group_counts(torch.bincount(ids[~flags],minlength=snap.cells));rare=int(clean.argmin())
    rare_rows=(~flags)&(groups[ids]==rare)
    extra=(~flags&~rare_rows).nonzero().flatten().cpu()
    for n in (51,52):
        flags=value["flags"].clone();flags[extra[:n-int(flags.sum())]]=True
        previous=plan(value,v3_method,original["seed"],flags=flags)
        fixed=plan(value,module.FeatureCellSnapshot.ordinary_transport,original["seed"],flags=flags)
        check(f"guard_boundary_{n}",value,previous,fixed,
              dict(rare_survivor_not_deleted=not bool(rare_rows[fixed[3]].any()),
                   rare_full_group_not_inflated=int((groups[ids[fixed[4]]]==rare).sum())==0))
except Exception as error:
    failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    print(json.dumps(dict(event="exception",**failure)),flush=True)
sources_after={str(p.relative_to(args.package_root)):sha(p) for p in sorted(args.package_root.rglob("*.py"))}
result=dict(status="PASS" if failure is None and sources_before==sources_after else "ERROR",error=failure,records=records,
    device=args.device,gpu=gpu,source_sha256_before=sources_before,source_sha256_after=sources_after,
    source_unchanged=sources_before==sources_after,input_sha256={str(p):sha(p) for p in input_files},
    peak_reserved_mib=None if args.device=="cpu" else torch.cuda.max_memory_reserved(0)/2**20,
    scope="Fixed-input recovery action contract; no training or quality verdict")
(args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("status","error","source_unchanged","device")}),flush=True)
raise SystemExit(0 if result["status"]=="PASS" else 1)

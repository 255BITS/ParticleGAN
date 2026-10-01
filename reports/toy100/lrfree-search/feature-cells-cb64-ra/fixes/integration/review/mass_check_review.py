"""Frozen-input CPU/CUDA mass diagnostic, with named checks and failure rows."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
STABILITY = ROOT / "stability"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package-root",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
args = parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
result_path = args.output / "result.json"
if result_path.exists():
    raise SystemExit("result exists; choose a new output")
os.environ.update(CUDA_VISIBLE_DEVICES="0" if args.device=="cuda:0" else "",
                  CUDA_DEVICE_ORDER="PCI_BUS_ID",CUBLAS_WORKSPACE_CONFIG=":4096:8",
                  OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
for key in ("ABSENT","ABSENT_START","ABSENT_END","LRFREE_NATIVE_TEST_STEPS"):
    os.environ.pop(key,None)
sys.dont_write_bytecode = True
sys.path.insert(0,str(args.package_root.resolve()))
import torch
import particlegan
from particlegan.feature_cells import FeatureCellSnapshot
if Path(particlegan.__file__).resolve().parent.parent != args.package_root.resolve():
    raise RuntimeError("wrong package import")
torch.set_num_threads(1); torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
properties = None
uuid = None
if args.device == "cuda:0":
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.2,0)
    properties = torch.cuda.get_device_properties(0)
    uuid = str(properties.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-"+uuid
    if uuid != "GPU-72c1b506-891d-b8bc-b353-e020585e1c47":
        raise RuntimeError("wrong physical GPU "+uuid)
inputs_path = STABILITY / "mass-gpu-inputs.pt"
ready = json.loads((STABILITY / "READY.json").read_text())
input_sha = hashlib.sha256(inputs_path.read_bytes()).hexdigest()
if input_sha != ready["evidence_sha256"]["mass-gpu-inputs.pt"]:
    raise RuntimeError("frozen inputs changed")
fixture_path = Path("/ml2/hypergan/gan-attempts/scaling-portability-20260929/scaling_a/shared_toy.py")
spec = importlib.util.spec_from_file_location("review_mass_evaluator_fixture",fixture_path)
fixture = importlib.util.module_from_spec(spec);sys.modules[spec.name] = fixture;spec.loader.exec_module(fixture)
data = torch.load(inputs_path,map_location="cpu",weights_only=False)
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources_before = {str(p.relative_to(args.package_root)):sha(p) for p in sorted(args.package_root.rglob("*.py"))}


def plain(item):
    if isinstance(item,torch.Tensor):return item.detach().cpu().tolist()
    if isinstance(item,dict):return {key:plain(value) for key,value in item.items()}
    if isinstance(item,(list,tuple)):return [plain(value) for value in item]
    return item


records = []
failure = None
try:
    for scenario,value in data["scenarios"].items():
        snapshot = FeatureCellSnapshot.__new__(FeatureCellSnapshot)
        for key,item in value["snapshot"].items():
            setattr(snapshot,key,item.to(args.device) if isinstance(item,torch.Tensor) else item)
        snapshot.device = torch.device(args.device)
        q,flags,pvalues = (value[key].to(args.device) for key in ("q","flags","pvalues"))
        comparison = {key:item.to(args.device) if isinstance(item,torch.Tensor) else item
                      for key,item in value["comparison"].items()}
        stream = torch.Generator(device=args.device).manual_seed(data["seed"])
        child,parent,ordinary = snapshot.ordinary_transport(q,flags,comparison,generator=stream,pvalues=pvalues)
        iso_child,iso_parent,detail = snapshot.select_parents(q,flags,ordinary_children=child,
                ordinary_parents=parent,generator=stream,pvalues=pvalues)
        all_child,all_parent = torch.cat((child,iso_child)).cpu(),torch.cat((parent,iso_parent)).cpu()
        after = value["z"].clone();after[all_child] = value["z"][all_parent]
        toy = fixture.Toy.make(scenario)
        _,labels = toy.oracle(after)
        mass = torch.bincount(torch.tensor(labels),minlength=9).double()/len(after)
        target = torch.bincount(value["real_labels"],minlength=9).double()/len(after)
        modes = torch.tensor(toy.oracle(value["z"])[1])
        planted = value["planted"][iso_child.cpu()]
        agreement = float((modes[iso_parent.cpu()][planted]==value["original_labels"][iso_child.cpu()][planted]).double().mean())
        ids,_ = snapshot.assign(q)
        groups = snapshot._mass_topology()
        row = dict(scenario=scenario,ordinary_moves=len(child),isolation_moves=len(iso_child),
                   unique_parents=len(torch.unique(all_parent)),total_parents=len(all_parent),
                   mass_tv=float((mass-target).abs().sum()/2),rare_ratio=float(mass[7]/target[7]),
                   repair_recall=float((torch.tensor(labels)[value["planted"]]<8).double().mean()),
                   intended_parent_agreement=agreement,groups=snapshot.mass_groups,
                   ordinary_between_group_moves=ordinary["between_group_moves"],
                   max_distance_rows=snapshot.work["max_distance_rows"],max_distance_columns=snapshot.work["max_distance_columns"],
                   masses=mass.tolist(),target_masses=target.tolist(),
                   ordinary_child_ids=child.cpu().tolist(),ordinary_parent_ids=parent.cpu().tolist(),
                   ordinary_child_modes=modes[child.cpu()].tolist(),ordinary_parent_modes=modes[parent.cpu()].tolist(),
                   group_table_counts=snapshot._group_counts(torch.bincount(ids,minlength=snapshot.cells)),
                   group_nonflagged_counts=snapshot._group_counts(torch.bincount(ids[~flags],minlength=snapshot.cells)),
                   group_eligible_counts=snapshot._group_counts(torch.bincount(ids[~flags&(pvalues>.05)],minlength=snapshot.cells)),
                   ordinary=ordinary,isolation={key:val for key,val in detail.items() if key not in
                    ("candidate_ids","candidate_mask","anchor_reference_rows","anchor_cell_ids","parent_cell_ids")})
        checks = dict(unique_children=len(torch.unique(all_child))==len(all_child),
                      unique_parents=row["unique_parents"]==row["total_parents"],
                      all_flagged_repaired=row["isolation_moves"]==int(flags.sum()),
                      mass_tv_at_most_point01=row["mass_tv"]<=.01,
                      rare_mass_exact_target=row["rare_ratio"]==1.,
                      repair_recall_one=row["repair_recall"]==1.,
                      zero_cross_group_ordinary=row["ordinary_between_group_moves"]==0,
                      eight_groups=row["groups"]==8)
        if scenario=="nominal":checks["nominal_agreement_at_least_point85"] = agreement>=.85
        row["checks"] = checks
        records.append(plain(row))
        print(json.dumps(dict(event="case_before_assertions",row=plain(row))),flush=True)
        for name,passed in checks.items():
            print(json.dumps(dict(event="named_check",scenario=scenario,name=name,passed=passed)),flush=True)
        assert all(checks.values()),"failed checks: "+", ".join(name for name,passed in checks.items() if not passed)
    if args.device=="cuda:0":torch.cuda.synchronize()
except Exception as error:
    failure = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    print(json.dumps(dict(event="exception",**failure)),flush=True)
sources_after = {str(p.relative_to(args.package_root)):sha(p) for p in sorted(args.package_root.rglob("*.py"))}
receipt = dict(status="PASS" if failure is None else "ERROR",error=failure,records=records,
               scope="mass action on original frozen CPU partition/score inputs; no quality verdict",
               device=args.device,seed=data["seed"],inputs_sha256=input_sha,
               source_sha256_before=sources_before,source_sha256_after=sources_after,
               source_unchanged=sources_before==sources_after,package_root=str(args.package_root.resolve()),
               gpu=None if properties is None else properties.name,uuid=uuid,
               peak_reserved_mib=None if args.device=="cpu" else round(torch.cuda.max_memory_reserved()/2**20,3))
result_path.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({key:receipt[key] for key in ("status","error","source_unchanged","device")}),flush=True)
raise SystemExit(0 if failure is None and receipt["source_unchanged"] else 1)

"""Root-only CUDA allocation check on frozen CPU feature/partition inputs.

No training, no fitting or support-score changes; no CUDA quality verdict.
The original full quality harness is a separate root-scheduled gate.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package-root",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args = parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
result_path = args.output / "result.json"
if result_path.exists():
    raise SystemExit("a diagnostic result already exists; choose a new output")
for key,value in dict(CUDA_VISIBLE_DEVICES="0",CUDA_DEVICE_ORDER="PCI_BUS_ID",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",
                      PYTHONDONTWRITEBYTECODE="1").items():
    os.environ[key] = value
for key in ("ABSENT","ABSENT_START","ABSENT_END","LRFREE_NATIVE_TEST_STEPS"):
    os.environ.pop(key,None)
sys.dont_write_bytecode = True
sys.path.insert(0,str(args.package_root.resolve()))
import torch
from particlegan.feature_cells import FeatureCellSnapshot
torch.set_num_threads(1); torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.2,0)
properties = torch.cuda.get_device_properties(0)
uuid = str(properties.uuid)
if not uuid.startswith("GPU-"):
    uuid = "GPU-"+uuid
expected_uuid = "GPU-72c1b506-891d-b8bc-b353-e020585e1c47"
if uuid != expected_uuid:
    raise RuntimeError(f"wrong physical GPU: {uuid}")
inputs_path = ROOT / "mass-gpu-inputs.pt"
ready = json.loads((ROOT / "READY.json").read_text())
if hashlib.sha256(inputs_path.read_bytes()).hexdigest() != ready["evidence_sha256"]["mass-gpu-inputs.pt"]:
    raise RuntimeError("frozen mass inputs changed")
fixture_path = Path("/ml2/hypergan/gan-attempts/scaling-portability-20260929/scaling_a/shared_toy.py")
spec = importlib.util.spec_from_file_location("mass_gpu_evaluator_fixture",fixture_path)
fixture = importlib.util.module_from_spec(spec); sys.modules[spec.name] = fixture
spec.loader.exec_module(fixture)
data = torch.load(inputs_path,map_location="cpu",weights_only=False)
records = []
failure = None
try:
    for scenario,value in data["scenarios"].items():
        snapshot = FeatureCellSnapshot.__new__(FeatureCellSnapshot)
        for key,item in value["snapshot"].items():
            setattr(snapshot,key,item.to("cuda:0") if isinstance(item,torch.Tensor) else item)
        snapshot.device = torch.device("cuda:0")
        q,flags,pvalues = (value[key].to("cuda:0") for key in ("q","flags","pvalues"))
        comparison = {key:item.to("cuda:0") if isinstance(item,torch.Tensor) else item
                      for key,item in value["comparison"].items()}
        stream = torch.Generator(device="cuda:0").manual_seed(data["seed"])
        child,parent,ordinary = snapshot.ordinary_transport(q,flags,comparison,generator=stream,pvalues=pvalues)
        iso_child,iso_parent,detail = snapshot.select_parents(q,flags,ordinary_children=child,
                ordinary_parents=parent,generator=stream,pvalues=pvalues)
        all_child,all_parent = torch.cat((child,iso_child)).cpu(),torch.cat((parent,iso_parent)).cpu()
        after = value["z"].clone(); after[all_child] = value["z"][all_parent]
        toy = fixture.Toy.make(scenario)
        _,labels = toy.oracle(after)
        mass = torch.bincount(torch.tensor(labels),minlength=9).double()/len(after)
        target = torch.bincount(value["real_labels"],minlength=9).double()/len(after)
        modes = torch.tensor(toy.oracle(value["z"])[1])
        planted = value["planted"][iso_child.cpu()]
        agreement = float((modes[iso_parent.cpu()][planted] == value["original_labels"][iso_child.cpu()][planted]).double().mean())
        row = dict(scenario=scenario,ordinary_moves=len(child),isolation_moves=len(iso_child),
                   unique_parents=len(torch.unique(all_parent)),total_parents=len(all_parent),
                   mass_tv=float((mass-target).abs().sum()/2),rare_ratio=float(mass[7]/target[7]),
                   repair_recall=float((torch.tensor(labels)[value["planted"]]<8).double().mean()),
                   intended_parent_agreement=agreement,groups=snapshot.mass_groups,
                   ordinary_between_group_moves=ordinary["between_group_moves"],
                   max_distance_rows=snapshot.work["max_distance_rows"],max_distance_columns=snapshot.work["max_distance_columns"])
        assert len(torch.unique(all_child)) == len(all_child)
        assert row["unique_parents"] == row["total_parents"]
        assert row["isolation_moves"] == int(flags.sum())
        assert row["mass_tv"] <= .01
        assert row["rare_ratio"] == 1.
        assert row["repair_recall"] == 1.
        assert row["ordinary_between_group_moves"] == 0
        assert row["groups"] == 8
        if scenario == "nominal":
            assert row["intended_parent_agreement"] >= .85
        records.append(row); print(json.dumps(row),flush=True)
    torch.cuda.synchronize()
except Exception as error:
    failure = dict(type=type(error).__name__,message=str(error))
package_hashes = {str(path.relative_to(args.package_root)):hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(args.package_root.rglob("*.py"))}
receipt = dict(status="PASS" if failure is None else "ERROR",error=failure,
               scope="CUDA mass action on frozen CPU partition/score inputs; no GPU quality claim",
               seed=data["seed"],records=records,gpu=properties.name,uuid=uuid,
               cuda_memory_fraction=.2,numeric_threads=1,package_root=str(args.package_root.resolve()),
               package_source_sha256=package_hashes,
               inputs_sha256=hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
               peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,3),
               peak_reserved_mib=round(torch.cuda.max_memory_reserved()/2**20,3))
result_path.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({key:receipt[key] for key in ("status","error","scope","peak_reserved_mib")}),flush=True)
raise SystemExit(0 if failure is None else 1)

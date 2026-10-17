"""Root-only CUDA joint3K contract; --device cpu verifies the same runner."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package-root",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--device",choices=("cpu","cuda:0"),default="cuda:0")
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
if (args.output/"result.json").exists():raise SystemExit("result exists; choose a new output")
os.environ.update(CUDA_VISIBLE_DEVICES="0" if args.device=="cuda:0" else "",CUDA_DEVICE_ORDER="PCI_BUS_ID",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
for key in ("ABSENT","ABSENT_START","ABSENT_END","LRFREE_NATIVE_TEST_STEPS"):os.environ.pop(key,None)
sys.dont_write_bytecode=True
sys.path.insert(0,str(args.package_root.resolve()))
import torch
import particlegan.feature_cells as module
from contract_cases import run_contracts
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


reference=HERE/"V4-FEATURE-CELLS.py"
if sha(reference)!="2f55fb6da21d2dc61645d4c9a2f0299f3f652ee7027de235661c8b2cce2f708a":raise RuntimeError("v4 method reference changed")
tree=ast.parse(reference.read_text())
node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=="FeatureCellSnapshot"
          for n in c.body if isinstance(n,ast.FunctionDef) and n.name=="ordinary_transport")
namespace=dict(module.__dict__)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(reference),"exec"),namespace)
v4_method=namespace["ordinary_transport"]
inputs=HERE/"inputs.pt"
if (HERE/"READY.json").exists():
    ready=json.loads((HERE/"READY.json").read_text())
    for name,digest in ready["local_source_sha256"].items():
        if sha(HERE/name)!=digest:raise RuntimeError("frozen runner source changed: "+name)
    if sha(inputs)!=ready["evidence_sha256"]["inputs.pt"]:raise RuntimeError("frozen inputs changed")
data=torch.load(inputs,map_location="cpu",weights_only=False)
before={str(p.relative_to(args.package_root)):sha(p) for p in sorted(args.package_root.rglob("*.py"))}
records=[];error=None
try:
    records=run_contracts(module,data,args.device,v4_method)
    if args.device=="cuda:0":torch.cuda.synchronize()
except Exception as failure:
    error=dict(type=type(failure).__name__,message=str(failure),traceback=traceback.format_exc())
    print(json.dumps(dict(event="exception",**error)),flush=True)
after={str(p.relative_to(args.package_root)):sha(p) for p in sorted(args.package_root.rglob("*.py"))}
result=dict(status="PASS" if error is None and before==after else "ERROR",error=error,records=records,
    device=args.device,gpu=gpu,cuda_initialized=torch.cuda.is_initialized(),source_sha256_before=before,
    source_sha256_after=after,source_unchanged=before==after,input_sha256=sha(inputs),
    runner_source_sha256={p.name:sha(p) for p in (Path(__file__),HERE/"contract_cases.py",reference)},
    generator_scope="existing saved CPU planning states" if args.device=="cpu" else "existing case seeds on current GPU generator; never CPU RNG bytes",
    scope="Fixed-input joint mass/support accounting and original mass regression; no quality verdict",
    peak_reserved_mib=None if args.device=="cpu" else torch.cuda.max_memory_reserved(0)/2**20)
(args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("status","error","source_unchanged","device","cuda_initialized")}),flush=True)
raise SystemExit(0 if result["status"]=="PASS" else 1)

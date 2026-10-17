"""Focused root-only CUDA birth contracts; CPU uses identical fixed inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
if (args.output/'result.json').exists():raise SystemExit('result exists; choose a new output')
os.environ.update(CUDA_VISIBLE_DEVICES='0' if args.device=='cuda:0' else '',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.dont_write_bytecode=True
sys.path.insert(0,str(args.package_root.resolve()))
import torch
import particlegan.feature_cells as module
import particlegan.birth_phase as birth
from birth_contract_cases import run_contracts
from birth_edge_contracts import run_edges
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
if Path(module.__file__).resolve().parent.parent!=args.package_root.resolve():raise RuntimeError('wrong selected package')
gpu=None
if args.device=='cuda:0':
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.2,0)
    props=torch.cuda.get_device_properties(0)
    gpu=dict(name=props.name,uuid=str(props.uuid),memory_fraction=.2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
GLOBAL=HERE.parent/'global-count'
sys.path.insert(0,str(GLOBAL))
from contract_cases import frozen_snapshot, planning_stream, convert
inputs=[GLOBAL/'inputs.pt',GLOBAL/'exhaustion-input.pt']
data=torch.load(inputs[0],map_location='cpu',weights_only=False)
data['cases'].update(torch.load(inputs[1],map_location='cpu',weights_only=False)['cases'])
before={str(p.relative_to(args.package_root)):sha(p) for p in sorted(args.package_root.rglob('*.py'))}
records=[];edges=[];error=None
try:
    records=run_contracts(module,birth,frozen_snapshot,planning_stream,convert,data,args.device)
    edges=run_edges(module,birth,frozen_snapshot,planning_stream,convert,data,args.device)
    if args.device=='cuda:0':torch.cuda.synchronize()
except Exception as failure:
    error=dict(type=type(failure).__name__,message=str(failure),traceback=traceback.format_exc())
    print(json.dumps(dict(event='exception',**error)),flush=True)
after={str(p.relative_to(args.package_root)):sha(p) for p in sorted(args.package_root.rglob('*.py'))}
result=dict(status='PASS' if error is None and before==after else 'ERROR',error=error,records=records,edge_records=edges,
    source_sha256_before=before,source_sha256_after=after,source_unchanged=before==after,
    device=args.device,gpu=gpu,cuda_initialized=torch.cuda.is_initialized(),
    input_sha256={str(p):sha(p) for p in inputs},
    runner_source_sha256={str(p):sha(p) for p in (Path(__file__),HERE/'birth_contract_cases.py',HERE/'birth_edge_contracts.py',GLOBAL/'contract_cases.py')},
    scope='Fixed-input novel birth/copy/isolation ledger and reset mechanics; no training or quality verdict')
(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(event='birth_contract_complete',status=result['status'],error=error,device=args.device)),flush=True)
raise SystemExit(0 if result['status']=='PASS' else 1)

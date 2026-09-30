"""Coordinator-only physical GPU0 lineage suite, accepts a combined package."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path
import subprocess
import torch
import lineage_checks

UUID = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'


def require(condition, message):
    if not condition: raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output already contains evidence')
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                               check=True, capture_output=True, text=True).stdout
    physical = dict(line.strip().split(', ', 1) for line in inventory.strip().splitlines())
    require(physical.get('0') == UUID, 'Physical GPU0 UUID mismatch')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One CUDA GPU required')
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    cuda_uuid = 'GPU-'+str(properties.uuid).removeprefix('GPU-').lower()
    require(cuda_uuid == UUID, 'Visible cuda0 is not frozen physical GPU0')
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    code = lineage_checks.run(args.package_root, args.output, 'cuda:0')
    receipt = json.loads(args.output.read_text())
    receipt['runtime'] = dict(physical_gpu=0, cuda_uuid=cuda_uuid, gpu=properties.name,
        torch=str(torch.__version__), cuda_runtime=torch.version.cuda, memory_fraction=.2,
        deterministic_algorithms=True, tf32=False, cpu_threads=1,
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    receipt['runner_sha256'] = lineage_checks.sha(Path(__file__))
    receipt['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
    receipt['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
    args.output.write_text(json.dumps(receipt, indent=2)+'\n')
    return code


if __name__ == '__main__': raise SystemExit(main())

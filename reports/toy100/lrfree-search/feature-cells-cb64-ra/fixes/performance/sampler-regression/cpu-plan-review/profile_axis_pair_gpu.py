"""Root-only paired cached/cold GPU0 profile, frozen RA3 versus exact axis cache."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import time
from types import ModuleType, SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
UUID='GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
INPUT=ROOT/'geometry/gpu-inputs.pt'
INPUT_SHA='07bfa15a806659459bd4f1bb634cf58969bff7a9f25ec69924e1fe8ba22fb263'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_map(path):
    return {str(p.relative_to(path)):sha(p) for p in sorted((path/'particlegan').rglob('*.py'))}


def module(name,path):
    holder=ModuleType(name);holder.__path__=[str(path/'particlegan')];sys.modules[name]=holder
    return importlib.import_module(name+'.feature_cells')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-package-root',type=Path,default=ROOT/'pkg-CB64-RA3')
    parser.add_argument('--package-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Output evidence already exists')
    assert sha(INPUT)==INPUT_SHA
    old_map,new_map=source_map(args.reference_package_root),source_map(args.package_root)
    inventory=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],
                             check=True,capture_output=True,text=True).stdout
    physical=dict(line.strip().split(', ',1) for line in inventory.strip().splitlines())
    assert physical.get('0')==UUID
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_device(0)
    props=torch.cuda.get_device_properties(0)
    assert 'GPU-'+str(props.uuid).removeprefix('GPU-').lower()==UUID
    torch.cuda.set_per_process_memory_fraction(.2,0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=module('axis_gpu_old',args.reference_package_root)
    new=module('axis_gpu_new',args.package_root)
    cases=torch.load(INPUT,map_location='cpu',weights_only=False)['cases']
    records=[]
    for name in ('native_cpu_saved_density','mnist_E22_0'):
        case=next(c for c in cases if c['name']==name)
        points,query,width,noise=[case[k].to('cuda:0') for k in ('prior','query','bandwidth','noise')]
        prior=SimpleNamespace(z=points)
        for count in ((2048,20000) if points.shape[1]==2 else (128,1024)):
            select=torch.arange(count,device=points.device)%len(query)
            q,draw=query[select],noise[select]
            kernels={'reference':old.BoundedLatentGeometry(), 'optimized':new.BoundedLatentGeometry()}
            a=kernels['reference'].displacement(q,prior,width,draw)
            b=kernels['optimized'].displacement(q,prior,width,draw)
            assert torch.equal(a,b),'cold displacement bits changed'
            assert all(type(axis) is int for axis,_,_ in kernels['optimized']._orders(points))
            row=dict(name=name,population=len(points),dimension=points.shape[1],query_rows=count,
                     query_policy='tile existing fixed rows/noise; no RNG draws',
                     cold_displacement_bit_identical=True,timings={},profiles={})
            for temperature in ('warm','cold'):
                endpoints={}
                for label,kernel in kernels.items():
                    def call():
                        if temperature=='cold':points.copy_(points)
                        return kernel.displacement(q,prior,width,draw)
                    call()
                    samples=[]
                    for _ in range(3):
                        torch.cuda.synchronize(0);start=time.perf_counter()
                        endpoints[label]=call()
                        torch.cuda.synchronize(0);samples.append((time.perf_counter()-start)*1000)
                    row['timings'][label+'_'+temperature]=dict(milliseconds=samples,median_ms=sorted(samples)[1])
                    call()  # Prime the current version before the observed warm call.
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
                        endpoints[label]=call();torch.cuda.synchronize(0)
                    events={e.key:e.count for e in profile.key_averages()}
                    row['profiles'][label+'_'+temperature]={key:events.get(key,0) for key in
                        ('aten::item','aten::_local_scalar_dense','aten::nonzero','aten::_to_copy','aten::copy_')}
                assert torch.equal(endpoints['reference'],endpoints['optimized']),temperature+' displacement bits changed'
                row[temperature+'_displacement_bit_identical']=True
            expected=((count+255)//256)*min(8,points.shape[1])
            assert row['profiles']['reference_warm']['aten::_local_scalar_dense']==expected
            assert row['profiles']['optimized_warm']['aten::_local_scalar_dense']==0
            row['expected_reference_axis_reads']=expected
            row['optimized_warm_axis_reads']=0
            records.append(row);print(json.dumps(row),flush=True)
    assert source_map(args.reference_package_root)==old_map and source_map(args.package_root)==new_map
    assert sha(INPUT)==INPUT_SHA
    result=dict(status='PASS',scope='paired fixed-input cache profile and bit parity; no training or quality verdict',
        gpu_uuid=UUID,cpu_threads=1,deterministic=True,tf32=False,new_seeds=0,
        reference_package_root=str(args.reference_package_root.resolve()),reference_sources=old_map,
        package_root=str(args.package_root.resolve()),package_sources=new_map,script_sha256=sha(__file__),
        input_sha256=INPUT_SHA,cases=records,peak_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',output=str(args.output))),flush=True)


if __name__=='__main__':main()

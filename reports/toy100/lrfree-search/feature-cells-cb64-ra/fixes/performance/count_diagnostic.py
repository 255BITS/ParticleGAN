"""Small frozen-reference conditional-count parity and CPU/CUDA profiling."""
import argparse
import os
import sys
sys.dont_write_bytecode=True
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
parser.add_argument('--output',required=True)
args=parser.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES='0' if args.device=='cuda' else '',
                  CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                  OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
                  PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import hashlib
import importlib
import json
import time
import types
import torch

ROOT=Path(__file__).resolve().parent
REFERENCE=Path('/ml2/hypergan/gan-attempts/feature-cells-config-20260929/pkg-CB64-RA')
FIX=ROOT/'pkg'
EXPECTED_REFERENCE='21040766f11d7d8c7a48d236a538fa5d309d28119817d57cb679db33d00a542a'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name,path):
    package=types.ModuleType(name);package.__path__=[str(path/'particlegan')]
    sys.modules[name]=package
    return importlib.import_module(name+'.feature_cells')


def log(event,**kw):print(json.dumps(dict(event=event,**kw)),flush=True)


def counts(rows,cells):
    result=torch.full((cells,),rows//cells,dtype=torch.long)
    result[:rows%cells]+=1
    return result


def test_case(real,fake,reference,fix,device):
    nr,nf=int(real.sum()),int(fake.sum());real,fake=real.to(device),fake.to(device)
    before={'count_test_terms':0};after={'count_test_terms':0}
    a=reference(real,fake,nr,nf,before);b=fix(real,fake,nr,nf,after)
    torch.testing.assert_close(a,b,rtol=5e-13,atol=5e-14)
    assert torch.equal(a<=.05/len(a),b<=.05/len(b)),'categorical rejection decision changed'
    assert before==after,'logical support enumeration receipt changed'
    assert bool(torch.isfinite(b).all()) and bool(((b>=0)&(b<=1)).all())
    return dict(max_abs_error=float((a-b).abs().max()),bit_exact=torch.equal(a,b),work=after)


def profile(function,real,fake,nr,nf,device,label):
    if device=='cuda':
        activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities) as prof:
            function(real,fake,nr,nf)
            torch.cuda.synchronize(0)
        prof.export_chrome_trace(str(ROOT/f'{label}-cuda-count-trace.json'))
    else:
        # Legacy CPU-only profiler does not ask Kineto to discover CUDA devices.
        with torch.autograd.profiler.profile(use_cuda=False,use_cpu=True,use_kineto=False) as prof:
            function(real,fake,nr,nf)
    rows={item.key:dict(count=item.count,self_cpu_us=item.self_cpu_time_total,
                       self_device_us=getattr(item,'self_device_time_total',0.)) for item in prof.key_averages()}
    return dict(events=rows,sync_candidate_calls={key:rows.get(key,{}).get('count',0)
        for key in ('aten::_local_scalar_dense','aten::item','aten::nonzero','cudaMemcpyAsync','cudaStreamSynchronize')})


def main():
    assert sha(REFERENCE/'particlegan/feature_cells.py')==EXPECTED_REFERENCE,'frozen reference source changed'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device='cpu'
    runtime=dict(torch=str(torch.__version__),cpu_threads=2,cuda_initialized_before=torch.cuda.is_initialized())
    if args.device=='cuda':
        assert torch.cuda.is_available(),'CUDA unavailable; no fallback'
        assert torch.cuda.device_count()==1,'GPU0 only'
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.2,0)
        raw=str(torch.cuda.get_device_properties(0).uuid)
        assert raw.removeprefix('GPU-').lower()=='72c1b506-891d-b8bc-b353-e020585e1c47'
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.cuda.reset_peak_memory_stats(0)
        runtime.update(uuid_raw=raw,memory_fraction=.2,cublas_workspace_config=':4096:8')
        device='cuda:0'
    else:assert not torch.cuda.is_initialized(),'CPU diagnostic may not initialize CUDA'
    original=load('perf_reference',REFERENCE).conditional_count_pvalues
    patched=load('perf_fix',FIX).conditional_count_pvalues
    parity=[]
    for nr in range(1,5):
        for nf in range(1,5):
            for observed_real in range(nr+1):
                for observed_fake in range(nf+1):
                    parity.append(test_case(torch.tensor([observed_real,nr-observed_real]),
                                            torch.tensor([observed_fake,nf-observed_fake]),original,patched,device))
    special=[('uniform64',counts(512,64),counts(1024,64)),
             ('rare64',counts(512,64),torch.cat((torch.tensor([1,0]),counts(1023,62)))),
             ('one-cell',torch.tensor([1]),torch.tensor([3])),
             ('empty-cells64',torch.cat((torch.tensor([512]),torch.zeros(63,dtype=torch.long))),
                              torch.cat((torch.tensor([0,1024]),torch.zeros(62,dtype=torch.long))))]
    named={}
    for name,real,fake in special:named[name]=test_case(real,fake,original,patched,device)
    invalid=[(torch.tensor([-1,2]),torch.tensor([1,1]),1,2),
             (torch.tensor([1,1]),torch.tensor([1,1]),3,2),
             (torch.tensor([[1,1]]),torch.tensor([[1,1]]),2,2),
             (torch.tensor([1.,1.]),torch.tensor([1,1]),2,2),
             (torch.tensor([1,1]),torch.tensor([1]),2,1),
             (torch.tensor([0,0]),torch.tensor([1,1]),0,2),
             (torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long),1,1)]
    for real,fake,nr,nf in invalid:
        for function in (original,patched):
            try:function(real.to(device),fake.to(device),nr,nf)
            except ValueError:pass
            else:raise AssertionError('malformed counts accepted')
    real,fake=counts(512,64).to(device),counts(1024,64).to(device)
    benchmarks={}
    for label,function in (('reference',original),('batched',patched)):
        for _ in range(3):function(real,fake,512,1024)
        if args.device=='cuda':torch.cuda.synchronize(0)
        started=time.perf_counter()
        for _ in range(20):function(real,fake,512,1024)
        if args.device=='cuda':torch.cuda.synchronize(0)
        elapsed=time.perf_counter()-started
        benchmarks[label]=dict(seconds_per_call=elapsed/20,repetitions=20,
                               profile=profile(function,real,fake,512,1024,args.device,label))
    if args.device=='cpu':assert not torch.cuda.is_initialized(),'CPU diagnostic initialized CUDA'
    receipt=dict(status='PASS',scope='conditional-count implementation microdiagnostic; no quality experiment',
                 device=device,runtime=runtime,exhaustive_tiny_tables=len(parity),malformed_tables=len(invalid),
                 max_abs_error=max([r['max_abs_error'] for r in parity]+[r['max_abs_error'] for r in named.values()]),
                 bit_exact_tiny_tables=sum(r['bit_exact'] for r in parity),special_cases=named,
                 rejection_decisions_identical=True,logical_enumeration_work_identical=True,
                 benchmarks=benchmarks,speed_ratio=benchmarks['reference']['seconds_per_call']/benchmarks['batched']['seconds_per_call'],
                 source_sha256={str(p):sha(p) for p in (Path(__file__),REFERENCE/'particlegan/feature_cells.py',FIX/'particlegan/feature_cells.py')},
                 command=[sys.executable,*sys.argv],cuda_initialized_at_completion=torch.cuda.is_initialized())
    if args.device=='cuda':receipt.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(0))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    assert not output.exists(),'refusing receipt overwrite'
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    log('count_diagnostic_complete',status='PASS',device=device,tiny_tables=len(parity),special_cases=named,
        max_abs_error=receipt['max_abs_error'],speed_ratio=receipt['speed_ratio'],
        sync_candidate_calls={label:value['profile']['sync_candidate_calls'] for label,value in benchmarks.items()},
        output=str(output),source_sha256=receipt['source_sha256'])


if __name__=='__main__':main()

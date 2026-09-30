"""Focused fit/pool/quota parity: stable row order, RNG streams, degenerate cells."""
import argparse
import os
import sys
sys.dont_write_bytecode=True
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
parser.add_argument('--output',required=True)
args=parser.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES='0' if args.device=='cuda' else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                  MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from unittest.mock import patch
import hashlib
import importlib
import json
import time
import types
import torch

ROOT=Path(__file__).resolve().parent
REFERENCE=Path('/ml2/hypergan/gan-attempts/feature-cells-config-20260929/pkg-CB64-RA')
DEVICE='cuda:0' if args.device=='cuda' else 'cpu'
SEED=314159


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name,path):
    package=types.ModuleType(name);package.__path__=[str(path/'particlegan')];sys.modules[name]=package
    return importlib.import_module(name+'.feature_cells')


def generator():return torch.Generator(device=DEVICE).manual_seed(SEED)


def compare(a,b,path='',errors=None):
    if errors is None:errors=[]
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and a.device==b.device,path
        if a.is_floating_point():
            torch.testing.assert_close(a,b,rtol=5e-13,atol=5e-14,msg=path)
            errors.append(dict(path=path,max_abs_error=float((a-b).abs().max()) if a.numel() else 0.,bit_exact=torch.equal(a,b)))
        else:assert torch.equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for key in a:compare(a[key],b[key],f'{path}.{key}',errors)
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b),path
        for index,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}[{index}]',errors)
    else:assert a==b,(path,a,b)
    return errors


def simple(module,cells):
    x=module.FeatureCellSnapshot();x.device=torch.device(DEVICE);x.width=x.rank=1;x.cells=cells;x.chunk=256
    x.mean=torch.zeros(1,dtype=torch.float64,device=DEVICE);x.scale=torch.ones_like(x.mean);x.basis=torch.ones((1,1),dtype=torch.float64,device=DEVICE)
    x.centers=torch.arange(cells,dtype=torch.float64,device=DEVICE)[:,None];x.calibration_rows=4*cells
    x.real_calibration_counts=torch.full((cells,),4,dtype=torch.long,device=DEVICE)
    x.work=dict(distance_cells=0,max_distance_rows=0,max_distance_columns=0,projection_products=0)
    return x


def pool_case(reference,fix,name,ids,eligible,ties=False):
    cells=max(1,int(ids.max())+1);features=ids.to(DEVICE).double()[:,None];eligible=eligible.to(DEVICE)
    a,b=simple(reference,cells),simple(fix,cells);ga,gb=generator(),generator()
    if ties:
        def priorities(*shape,**kw):return torch.zeros(*shape,device=kw['device'],dtype=kw['dtype'])
        with patch('torch.rand',side_effect=priorities):left=a._pool(features,eligible,ga);right=b._pool(features,eligible,gb)
    else:left=a._pool(features,eligible,ga);right=b._pool(features,eligible,gb)
    compare(left,right,name);assert torch.equal(ga.get_state(),gb.get_state()),f'{name}: pool RNG differs'
    return dict(name=name,rows=len(ids),cells=cells,tied_priorities=ties,bit_exact=True)


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if args.device=='cuda':
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.2,0)
        assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower()=='72c1b506-891d-b8bc-b353-e020585e1c47'
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    else:assert not torch.cuda.is_initialized()
    reference,fix=load('snapshot_reference',REFERENCE),load('snapshot_fix',ROOT/'pkg')
    source_hash=sha(ROOT/'pkg/particlegan/feature_cells.py')
    assert source_hash==json.loads((ROOT/'GPU-READY.json').read_text())['source_sha256'][str(ROOT/'pkg/particlegan/feature_cells.py')]
    cpu_generator=torch.Generator().manual_seed(SEED)
    inputs=[('learned128',torch.randn((1024,128),generator=cpu_generator),64,8),
            ('learned64',torch.randn((1024,64),generator=cpu_generator),64,8),
            ('constant',torch.ones((6,3)),64,8),
            ('tiny-odd',torch.arange(14,dtype=torch.float64).reshape(7,2),64,8),
            ('repeated-rows',torch.arange(32,dtype=torch.float64).reshape(4,8).repeat(32,1),64,8),
            ('rare-pattern',torch.cat((torch.zeros((120,4)),torch.eye(4).repeat(2,1))),16,4)]
    snapshots=[];floating=[]
    for name,real,cells,rank in inputs:
        real=real.to(DEVICE);ga,gb=generator(),generator()
        a=reference.FeatureCellSnapshot.fit(real,generator=ga,cells=cells,rank=rank)
        b=fix.FeatureCellSnapshot.fit(real,generator=gb,cells=cells,rank=rank)
        # Fit fields include centers, representatives, calibration/null scores,
        # duplicate guards and the original work/storage receipts.
        errors=compare(vars(a),vars(b),name);floating.extend(errors)
        assert torch.equal(ga.get_state(),gb.get_state()),f'{name}: fit RNG differs'
        query=real.flip(0)
        compare(a.assign(query),b.assign(query),name+'.assignment')
        compare(a.support(query),b.support(query),name+'.support')
        compare(a.cell_comparison(query),b.cell_comparison(query),name+'.comparison')
        snapshots.append(dict(name=name,rows=len(real),width=real.shape[1],cells=a.cells,rank=a.rank,
                              max_abs_error=max([e['max_abs_error'] for e in errors] or [0]),
                              all_float_fields_bit_exact=all(e['bit_exact'] for e in errors),rng_bit_exact=True))
    pool=[]
    fixtures=[('tiny',torch.tensor([0]),torch.tensor([True])),
              ('no-eligible',torch.arange(256)%4,torch.zeros(256,dtype=torch.bool)),
              ('reservoir-cap',torch.zeros(512,dtype=torch.long),torch.ones(512,dtype=torch.bool)),
              ('rare-cells',torch.cat((torch.zeros(70,dtype=torch.long),torch.full((130,),2))),torch.arange(200)%3!=0),
              ('many-cells',torch.arange(1024)%64,torch.arange(1024)%5!=0)]
    for name,ids,eligible in fixtures:
        pool.append(pool_case(reference,fix,name,ids,eligible))
        pool.append(pool_case(reference,fix,name+'-ties',ids,eligible,ties=True))
    transport=[]
    for name,inaccessible,flagged in (('ordinary',False,False),('inaccessible-birth-cell',True,False),('excluded-deaths',False,True)):
        a,b=simple(reference,4),simple(fix,4)
        real=torch.full((4,),64,dtype=torch.long,device=DEVICE);fake=torch.tensor([128,32,64,32],device=DEVICE)
        pa=reference.conditional_count_pvalues(real,fake,256,256);pb=fix.conditional_count_pvalues(real,fake,256,256)
        difference=fake.double()/256-real.double()/256
        def comparison(p):return dict(real_counts=real,fake_counts=fake,pvalues=p,difference=difference,
                                     excess=(p<=.05/4)&(difference>0),deficit=(p<=.05/4)&(difference<0))
        query=(torch.arange(256,device=DEVICE)%4).double()[:,None]
        flags=torch.zeros(256,dtype=torch.bool,device=DEVICE);pvalues=torch.ones(256,device=DEVICE)
        if inaccessible:pvalues[(query[:,0]==3)]=0.
        if flagged:flags[:16]=True
        ga,gb=generator(),generator()
        left=a.ordinary_transport(query,flags,comparison(pa),generator=ga,pvalues=pvalues)
        right=b.ordinary_transport(query,flags,comparison(pb),generator=gb,pvalues=pvalues)
        compare(left,right,name);assert torch.equal(ga.get_state(),gb.get_state()),f'{name}: transport RNG differs'
        transport.append(dict(name=name,moves=len(left[0]),bit_exact=True,rng_bit_exact=True))
    malformed=[torch.zeros((5,2)),torch.zeros((6,0)),torch.full((6,2),float('nan')),torch.ones((6,2),dtype=torch.long)]
    for real in malformed:
        for module in (reference,fix):
            try:module.FeatureCellSnapshot.fit(real.to(DEVICE),generator=generator())
            except ValueError:pass
            else:raise AssertionError('malformed feature input accepted')
    # Profile one normal-size fit without a GAN update or quality evaluation.
    fixture=inputs[0][1].to(DEVICE);profiles={}
    for label,module in (('reference',reference),('batched',fix)):
        started=time.perf_counter()
        if args.device=='cuda':torch.cuda.synchronize(0);started=time.perf_counter()
        module.FeatureCellSnapshot.fit(fixture,generator=generator(),cells=64,rank=8)
        if args.device=='cuda':torch.cuda.synchronize(0)
        elapsed=time.perf_counter()-started
        if args.device=='cpu':
            with torch.autograd.profiler.profile(use_cuda=False,use_cpu=True,use_kineto=False) as prof:
                module.FeatureCellSnapshot.fit(fixture,generator=generator(),cells=64,rank=8)
            events={item.key:item.count for item in prof.key_averages()}
        else:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                module.FeatureCellSnapshot.fit(fixture,generator=generator(),cells=64,rank=8);torch.cuda.synchronize(0)
            events={item.key:item.count for item in prof.key_averages()}
            prof.export_chrome_trace(str(ROOT/f'{label}-cuda-fit-trace.json'))
        profiles[label]=dict(seconds_per_fit=elapsed,events=events,
                            scalar_reads=events.get('aten::_local_scalar_dense',0),nonzero_calls=events.get('aten::nonzero',0))
    if args.device=='cpu':assert not torch.cuda.is_initialized()
    receipt=dict(status='PASS',scope='fit/pool/ordinary quota implementation parity; no learned training or quality test',
                 device=DEVICE,seed=SEED,snapshots=snapshots,pools=pool,transport=transport,malformed_features=len(malformed),
                 max_abs_error=max([r['max_abs_error'] for r in floating] or [0]),
                 all_float_fields_bit_exact=all(r['bit_exact'] for r in floating),profiles=profiles,
                 source_sha256={str(p):sha(p) for p in (Path(__file__),ROOT/'pkg/particlegan/feature_cells.py',REFERENCE/'particlegan/feature_cells.py')},
                 cuda_initialized_at_completion=torch.cuda.is_initialized(),command=[sys.executable,*sys.argv])
    output=Path(args.output);assert not output.exists(),'refusing receipt overwrite';output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(event='snapshot_parity_complete',status='PASS',device=DEVICE,
                         snapshots=snapshots,pool_cases=len(pool),transport=transport,
                         max_abs_error=receipt['max_abs_error'],all_float_fields_bit_exact=receipt['all_float_fields_bit_exact'],
                         profiles={k:{f:x[f] for f in ('seconds_per_fit','scalar_reads','nonzero_calls')} for k,x in profiles.items()},output=str(output))),flush=True)


if __name__=='__main__':main()

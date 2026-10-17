"""CPU read-only serialized training parity, with a narrow declared allowlist."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
STEPS=(0,100,250,500,750,1000,1250,1500,1750,2000)
POLICY='ema_support_inside_current_real_group_Q_v1'
EXPIRY='strictly_less_than_one_real_fifo_turnover_v1'


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as source:
        while block:=source.read(1024*1024):result.update(block)
    return result.hexdigest()


def tensor_bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def digest(value):
    result=hashlib.sha256()
    def walk(v):
        result.update(type(v).__name__.encode()+b'\0')
        if isinstance(v,torch.Tensor):
            result.update(str((v.dtype,tuple(v.shape))).encode()+b'\0'+tensor_bytes(v))
        elif isinstance(v,dict):
            for key in sorted(v,key=lambda x:(type(x).__name__,repr(x))):walk(key);walk(v[key])
        elif isinstance(v,(list,tuple)):
            for item in v:walk(item)
        elif isinstance(v,float):result.update(struct.pack('!d',v))
        else:result.update(repr(v).encode())
    walk(value)
    return result.hexdigest()


def describe(v):
    if isinstance(v,torch.Tensor):return dict(type='tensor',dtype=str(v.dtype),shape=list(v.shape),sha256=digest(v))
    if isinstance(v,(list,tuple,dict)):return dict(type=type(v).__name__,length=len(v),sha256=digest(v))
    return dict(type=type(v).__name__,value=v)


def differences(a,b,path='$'):
    result=[]
    def add(p,x,y,kind,**extra):result.append(dict(path=p,kind=kind,reference=describe(x),candidate=describe(y),**extra))
    def visit(x,y,p):
        if type(x) is not type(y):add(p,x,y,'type');return
        if isinstance(x,torch.Tensor):
            if x.dtype!=y.dtype or x.shape!=y.shape:add(p,x,y,'tensor_metadata');return
            if tensor_bytes(x)==tensor_bytes(y):return
            detail={}
            if x.is_floating_point():
                finite=torch.isfinite(x)&torch.isfinite(y)
                if bool(finite.any()):
                    delta=x[finite].double()-y[finite].double()
                    detail.update(finite_max_absolute_difference=float(delta.abs().max()),finite_RMS_difference=float(delta.square().mean().sqrt()))
                detail['nonfinite_pattern_equal']=bool(torch.equal(torch.isnan(x),torch.isnan(y)) and torch.equal(torch.isinf(x),torch.isinf(y)))
            detail['unequal_element_values']=int((x!=y).sum())
            add(p,x,y,'tensor_bytes',**detail);return
        if isinstance(x,dict):
            for key in sorted(set(x)|set(y),key=lambda k:(type(k).__name__,repr(k))):
                q=f'{p}/{key}'
                if key not in x:add(q,None,y[key],'candidate_only')
                elif key not in y:add(q,x[key],None,'reference_only')
                else:visit(x[key],y[key],q)
            return
        if isinstance(x,(list,tuple)):
            if len(x)!=len(y):add(p,x,y,'length');return
            for i,(xx,yy) in enumerate(zip(x,y)):visit(xx,yy,f'{p}/{i}')
            return
        equal=struct.pack('!d',x)==struct.pack('!d',y) if isinstance(x,float) else x==y
        if not equal:add(p,x,y,'scalar')
    visit(a,b,path)
    return result


def legacy_view(trainer,variant):
    """Copy and remove only the declared serving additions/performance leaves."""
    view=deepcopy(trainer);bd=view['birth_death'];last=bd['last'];removed={}
    assert variant in ('RA7','RA8')
    if variant=='RA8':
        assert type(bd['backend_schema']) is int and bd['backend_schema']==7
        removed['backend_schema']=bd['backend_schema'];bd['backend_schema']=6
        for key,expected in (('paired_average_policy',POLICY),('paired_average_expiry',EXPIRY)):
            assert bd['settings'].pop(key)==expected
            removed['settings/'+key]=expected
        stamp=bd.pop('paired_average')
        assert isinstance(stamp,dict) and stamp['schema']==1 and stamp['policy']==POLICY
        n=len(view['models']['prior']['z'])
        assert stamp['rows']==n and stamp['required']==n-math.floor(.05*n)
        assert type(stamp['eligible']) is bool and type(stamp['step']) is int
        assert stamp['step']<=view['completed_steps'] and stamp['snapshot']==bd['snapshot_serial']
        removed['paired_average']=stamp
        if bd['snapshot_serial']:
            assert last.pop('paired_average')==stamp
            assert last.pop('paired_average_forward_rows')==n
            assert last['step']==stamp['step'] and last['snapshot']==stamp['snapshot']
        else:assert stamp['step']==0 and not stamp['eligible'] and not last
    else:
        assert type(bd['backend_schema']) is int and bd['backend_schema']==6
        assert 'paired_average' not in bd and all(k not in bd['settings'] for k in ('paired_average_policy','paired_average_expiry'))
    for key in ('feature_distance_cells','projection_products'):
        assert type(bd['counters'][key]) is int
        removed['counters/'+key]=bd['counters'][key];bd['counters'][key]=0
    if 'eval_seconds' in last:
        removed['last/eval_seconds']=last['eval_seconds'];last['eval_seconds']=0.
    if 'work' in last:
        for key in ('distance_cells','projection_products'):
            assert type(last['work'][key]) is int
            removed['last/work/'+key]=last['work'][key];last['work'][key]=0
    return view,removed


def source_guard():
    manifest=json.loads((HERE/'SOURCE-FROZEN.json').read_text())
    failed=[name for name,expected in manifest['files'].items() if sha(name)!=expected]
    if failed:raise RuntimeError('Source/metadata integrity failure: '+repr(failed))
    return manifest


def compare(step,seal,output):
    output=Path(output)
    assert step in STEPS and not output.exists()
    manifest=source_guard();rng=torch.get_rng_state().clone()
    paths={v:ROOT/f'validation-cb64-{v.lower()}/learned/training/toy/CB64-{v}/checkpoint-{step:04d}.pt' for v in ('RA7','RA8')}
    before={v:sha(p) for v,p in paths.items()}
    reference=torch.load(paths['RA7'],map_location='cpu',weights_only=False)
    candidate=torch.load(paths['RA8'],map_location='cpu',weights_only=False)
    originals={v:digest(packet['trainer']) for v,packet in (('RA7',reference),('RA8',candidate))}
    for packet in (reference,candidate):
        assert packet['trainer']['completed_steps']==packet['record']['step']==step
        assert packet['data_position']==2*step*128
    a,ra=legacy_view(reference['trainer'],'RA7');b,rb=legacy_view(candidate['trainer'],'RA8')
    delta=differences(a,b);components={}
    for key in sorted(set(a)|set(b)):
        aa=a.get(key);bb=b.get(key)
        components[key]=dict(exact=digest(aa)==digest(bb),reference_sha256=digest(aa),candidate_sha256=digest(bb))
    immutable=all(originals[v]==digest(p['trainer']) for v,p in (('RA7',reference),('RA8',candidate)))
    assert immutable and before=={v:sha(p) for v,p in paths.items()}
    source_guard()
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    stamp=rb['paired_average']
    lease=stamp['eligible'] and stamp['snapshot']==candidate['trainer']['birth_death']['snapshot_serial'] and 0<=candidate['trainer']['birth_death']['rows_since_eval']<stamp['rows']
    receipt=dict(status='EXACT_LEGACY_TRAINING_PARITY' if not delta else 'TRAINING_DIVERGENCE',step=step,
        scope='descriptive serialized endpoint training neutrality, not quality acceptance',
        compared_all_serialized_trainer_leaves=True,legacy_view_reference_sha256=digest(a),legacy_view_candidate_sha256=digest(b),
        components=components,unexpected_difference_count=len(delta),unexpected_differences=delta,
        allowed_metadata_reference=ra,allowed_metadata_candidate=rb,paired_average_lease_live=bool(lease),
        checkpoint_sha256={str(paths[v]):value for v,value in before.items()},original_trainer_sha256=originals,
        data_position_exact=reference['data_position']==candidate['data_position'],
        original_checkpoint_objects_unchanged=immutable,all_sources_and_input_files_unchanged=True,
        helper_frozen_sha256=sha(HERE/'SOURCE-FROZEN.json'),source_frozen_utc=manifest['frozen_utc'],
        compared_utc=datetime.now(timezone.utc).isoformat(),post_save_seal=seal,
        CPU_only=True,cuda_initialized=False,global_rng_unchanged=True,new_emissions=0,new_training_steps=0,
        new_optimizer_steps=0,new_seed_experiments=0,quality_verdict=None,
        outer_log_scope='record metrics/diagnostics/timing and config receipt provenance are not training state and not compared as parity')
    with output.open('x') as target:target.write(json.dumps(receipt,indent=2,allow_nan=True)+'\n')
    return receipt

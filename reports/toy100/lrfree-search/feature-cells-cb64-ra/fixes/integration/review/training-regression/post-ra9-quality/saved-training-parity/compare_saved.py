"""CPU parse of frozen RA9/RA8 endpoints; no numeric replay or model execution."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import ast
from copy import deepcopy
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
ORIGINAL=ROOT/'integration/review/training-regression/post-ra8-quality/saved-training-parity'
STEPS=(0,100,250,500,750,1000,1250,1500,1750,2000)
POLICY='ema_support_inside_current_real_group_Q_v1'
EXPIRY='strictly_less_than_one_real_fifo_turnover_v1'
RESOLUTION='even_fit_average_rows_per_effective_rank_floor1_v1'
STAMP_KEYS={'schema','policy','snapshot','step','rows','required','cells','rank','groups','calibration_rows',
            'chart_valid','duplicate_ok','finite_rows','same_group_rows','ema_eligible_rows','coherent_rows','eligible'}
FC=ROOT/'pkg-CB64-RA9/particlegan/feature_cells.py'
FC_SHA='39558fb3839090eb9d933b8b37e4c73b7dfc3cf123b24c2ef58d818cc4052fcc'

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


def production_scalar_checker():
    """Compile only the frozen pure scalar checker/cap, without importing a model."""
    if sha(FC)!=FC_SHA:raise RuntimeError('RA9 feature-cell source changed')
    tree=ast.parse(FC.read_text());selected=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='_fit_cell_count':selected.append(node)
        if isinstance(node,ast.ClassDef) and node.name=='FeatureCellBirthDeath':
            selected.extend(item for item in node.body if isinstance(item,ast.FunctionDef)
                            and item.name=='_check_paired_average_state')
    assert len(selected)==2
    namespace=dict(math=math,Fraction=Fraction,Q=.05)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(FC),'exec'),namespace)
    return namespace['_check_paired_average_state']


def scalar_metadata(trainer,variant):
    """Recognize actual-K metadata using the reviewed production checker only."""
    assert variant in ('RA8','RA9')
    bd=trainer['birth_death'];settings=bd['settings'];stamp=bd['paired_average']
    n=len(trainer['models']['prior']['z']);requested=64 if variant=='RA8' else 128
    assert type(trainer['schema']) is int and trainer['schema']==5
    assert type(bd['backend_schema']) is int and bd['backend_schema']==(7 if variant=='RA8' else 8)
    assert type(settings['cells']) is int and settings['cells']==requested
    assert type(trainer['recipe']['birth_death_cells']) is int and trainer['recipe']['birth_death_cells']==requested
    assert settings['paired_average_policy']==POLICY and settings['paired_average_expiry']==EXPIRY
    if variant=='RA9':assert settings['resolution_policy']==RESOLUTION
    else:assert 'resolution_policy' not in settings
    assert type(bd['snapshot_serial']) is int and bd['snapshot_serial']>=0
    assert type(trainer['completed_steps']) is int and trainer['completed_steps']>=0
    assert type(bd['rows_since_eval']) is int and bd['rows_since_eval']>=0
    stub=SimpleNamespace(N=n,settings=settings,paired_average=dict.fromkeys(STAMP_KEYS))
    production_scalar_checker()(stub,bd)
    assert stamp['step']<=trainer['completed_steps']
    if not stamp['snapshot']:assert not bd['last']
    else:
        assert type(bd['last']['paired_average_forward_rows']) is int
        assert bd['last']['paired_average_forward_rows']==n
    lease=(stamp['eligible'] and bd['fill']==n and stamp['snapshot']==bd['snapshot_serial']
           and stamp['step']<=trainer['completed_steps'] and 0<=bd['rows_since_eval']<n)
    return dict(actual_cells=stamp['cells'],metric_rank=stamp['rank'],requested_cells=requested,
                coherent_rows=stamp['coherent_rows'],required=stamp['required'],lease_live=bool(lease),
                snapshot=stamp['snapshot'],step=stamp['step'],age_real_rows=bd['rows_since_eval'],
                multiplicity=bd['last'].get('count_multiplicity'),cutoff=bd['last'].get('count_cutoff'))


def legacy_view(trainer,variant):
    """Normalize only the declared prospective schema/settings and original timing."""
    metadata=scalar_metadata(trainer,variant)
    view=deepcopy(trainer);bd=view['birth_death'];removed={}
    if variant=='RA9':
        removed['backend_schema']=bd['backend_schema'];bd['backend_schema']=7
        removed['settings/resolution_policy']=bd['settings'].pop('resolution_policy')
        removed['settings/cells']=bd['settings']['cells'];bd['settings']['cells']=64
        removed['recipe/birth_death_cells']=view['recipe']['birth_death_cells']
        view['recipe']['birth_death_cells']=64
    if 'eval_seconds' in bd['last']:
        assert type(bd['last']['eval_seconds']) is float
        removed['last/eval_seconds']=bd['last']['eval_seconds'];bd['last']['eval_seconds']=0.
    return view,removed,metadata


def source_guard():
    manifest=json.loads((HERE/'SOURCE-FROZEN.json').read_text())
    failed=[name for name,expected in manifest['files'].items() if not Path(name).is_file() or sha(name)!=expected]
    if failed:raise RuntimeError('Source/metadata integrity failure: '+repr(failed))
    return manifest


def compare(step,seals,output):
    output=Path(output);assert step in STEPS and not output.exists()
    manifest=source_guard();rng=torch.get_rng_state().clone()
    paths={v:ROOT/f'validation-cb64-{v.lower()}/learned/training/toy/CB64-{v}/checkpoint-{step:04d}.pt'
           for v in ('RA8','RA9')}
    before={v:sha(p) for v,p in paths.items()}
    reference=torch.load(paths['RA8'],map_location='cpu',weights_only=False)
    candidate=torch.load(paths['RA9'],map_location='cpu',weights_only=False)
    originals={v:digest(packet['trainer']) for v,packet in (('RA8',reference),('RA9',candidate))}
    for packet in (reference,candidate):
        assert packet['trainer']['completed_steps']==packet['record']['step']==step
        assert packet['data_position']==2*step*128
    a,ra,ma=legacy_view(reference['trainer'],'RA8');b,rb,mb=legacy_view(candidate['trainer'],'RA9')
    delta=differences(a,b);components={}
    for key in sorted(set(a)|set(b)):
        aa=a.get(key);bb=b.get(key)
        components[key]=dict(exact=digest(aa)==digest(bb),reference_sha256=digest(aa),candidate_sha256=digest(bb))
    immutable=all(originals[v]==digest(p['trainer']) for v,p in (('RA8',reference),('RA9',candidate)))
    assert immutable and before=={v:sha(p) for v,p in paths.items()}
    source_guard();assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    receipt=dict(status='EXACT_LEGACY_TRAINING_PARITY' if not delta else 'TRAINING_DIVERGENCE',step=step,
        scope='CPU parse of serialized saved endpoints; not numerical replay, training, emissions or quality acceptance',
        compared_all_serialized_trainer_leaves=True,legacy_view_reference_sha256=digest(a),legacy_view_candidate_sha256=digest(b),
        components=components,unexpected_difference_count=len(delta),unexpected_differences=delta,
        allowed_metadata_reference=ra,allowed_metadata_candidate=rb,resolution_metadata_reference=ma,
        resolution_metadata_candidate=mb,paired_average_lease_live=mb['lease_live'],
        checkpoint_sha256={str(paths[v]):value for v,value in before.items()},original_trainer_sha256=originals,
        data_position_exact=reference['data_position']==candidate['data_position'],
        original_checkpoint_objects_unchanged=immutable,all_sources_and_input_files_unchanged=True,
        helper_frozen_sha256=sha(HERE/'SOURCE-FROZEN.json'),source_frozen_utc=manifest['frozen_utc'],
        compared_utc=datetime.now(timezone.utc).isoformat(),post_save_seals=seals,
        CPU_only=True,cuda_initialized=False,global_rng_unchanged=True,new_emissions=0,new_training_steps=0,
        new_optimizer_steps=0,new_seed_experiments=0,numerical_replay=False,quality_verdict=None,
        outer_log_scope='record metrics/diagnostics/timing and variant/config provenance are outside serialized training parity')
    with output.open('x') as target:target.write(json.dumps(receipt,indent=2,allow_nan=True)+'\n')
    return receipt

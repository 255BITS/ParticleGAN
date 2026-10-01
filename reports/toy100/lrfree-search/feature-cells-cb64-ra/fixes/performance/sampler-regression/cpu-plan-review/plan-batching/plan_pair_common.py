"""Fixed joint-count action/RNG/accounting comparisons, CPU or root GPU."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
COUNT=ROOT/'integration/review/training-regression/joint-count'
BASE=COUNT/'pkg-joint-count'
INPUT=COUNT/'inputs.pt'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(package):
    return {str(p.relative_to(package)):sha(p) for p in sorted((package/'particlegan').rglob('*.py'))}


def module(name,path):
    holder=ModuleType(name);holder.__path__=[str(path/'particlegan')];sys.modules[name]=holder
    return importlib.import_module(name+'.feature_cells')


def tools():
    spec=importlib.util.spec_from_file_location('paired_count_contract_tools',COUNT/'contract_cases.py')
    result=importlib.util.module_from_spec(spec);sys.modules[spec.name]=result;spec.loader.exec_module(result)
    return result


def same(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def prepare(module,value,device,*,flags=None,pvalues=None,max_moves=None,warm=False):
    contract=tools()
    snap=contract.frozen_snapshot(module,value,device)
    q=value['q'].to(device)
    flag_values=(value['flags'] if flags is None else flags).to(device).clone()
    parent_p=(value['pvalues'] if pvalues is None else pvalues).to(device).clone()
    comparison=snap.cell_comparison(value['fake_features'].to(device))
    generator=contract.planning_stream(value,device)
    if warm:snap._mass_topology()
    return snap,q,flag_values,parent_p,comparison,generator,max_moves


def execute(prepared):
    snap,q,flags,pvalues,comparison,generator,max_moves=prepared
    child,parent,detail=snap.ordinary_transport(q,flags,comparison,generator=generator,
                                             pvalues=pvalues,max_moves=max_moves)
    ordinary_rng=generator.get_state().clone()
    iso_child,iso_parent,isolation=snap.select_parents(q,flags,ordinary_children=child,
        ordinary_parents=parent,generator=generator,pvalues=pvalues)
    result=dict(child=child,parent=parent,detail=detail,iso_child=iso_child,iso_parent=iso_parent,
                isolation=isolation,ordinary_rng=ordinary_rng,final_rng=generator.get_state().clone(),
                comparison=comparison,work=deepcopy(snap.work),mass_topology=deepcopy(snap.mass_topology),
                mass_group_ids=snap.mass_group_ids.clone())
    return result


def variants(data):
    for name,value in data['cases'].items():
        yield name,value,{}
    value=data['cases']['saved_toy_1000']
    yield 'no_eligible_parent',value,dict(pvalues=torch.zeros_like(value['pvalues']))
    for budget in (0,3):yield 'budget'+str(budget),value,dict(max_moves=budget)
    # Existing contract boundary fixtures; no new feature or score values.
    proto=prepare(module('plan_variant_reference',BASE),value,'cpu')[0]
    categories=proto.count_categories(value['q'])
    available=(value['flags']&(categories.remainder(2)==1)).nonzero().flatten()
    for n in (51,52):
        flags=torch.zeros_like(value['flags']);flags[available[:n]]=True
        yield 'guard'+str(n),value,dict(flags=flags)


def paired(old,new,data,device):
    records=[]
    for name,value,kw in variants(data):
        a=execute(prepare(old,value,device,**kw))
        b=execute(prepare(new,value,device,**kw))
        equal={key:same(a[key],b[key]) for key in a}
        if not all(equal.values()):raise AssertionError(name+': '+', '.join(k for k,v in equal.items() if not v))
        record=dict(name=name,ordinary=len(a['child']),isolation=len(a['iso_child']),
                    mass_moves=a['detail']['mass_moves'],support_moves=a['detail']['support_moves'],
                    groups=a['mass_topology']['groups'],exact=equal)
        records.append(record);print(json.dumps(record),flush=True)
    return records


def profile(call,device):
    activities=[torch.profiler.ProfilerActivity.CPU]
    if str(device).startswith('cuda'):activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profiler:
        output=call()
        if str(device).startswith('cuda'):torch.cuda.synchronize(0)
    counts={event.key:event.count for event in profiler.key_averages()}
    return output,{key:counts.get(key,0) for key in
        ('aten::item','aten::_local_scalar_dense','aten::nonzero','aten::equal','aten::_to_copy','aten::copy_')}

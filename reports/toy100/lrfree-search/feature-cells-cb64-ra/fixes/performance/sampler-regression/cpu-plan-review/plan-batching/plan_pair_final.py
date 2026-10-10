"""Exact planner comparisons with a configurable, identical final count law."""
import ast
import hashlib
from pathlib import Path
import torch
import plan_pair_common as common

HERE=common.HERE
PERFORMANCE_METHODS={'_ordinary_mass_transport','_ordinary_support_transport','_ordinary_global_transport'}


def configure(reference,input_path=None,contract_root=None):
    common.BASE=Path(reference)
    if input_path is not None:common.INPUT=Path(input_path)
    if contract_root is not None:common.COUNT=Path(contract_root)


def ast_hash(node):
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def law_checks(reference,proposal,include_mst=False):
    old=ast.parse((Path(reference)/'particlegan/feature_cells.py').read_text())
    new=ast.parse((Path(proposal)/'particlegan/feature_cells.py').read_text())
    before=next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name=='FeatureCellSnapshot')
    after=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='FeatureCellSnapshot')
    original={n.name:n for n in before.body if isinstance(n,ast.FunctionDef)}
    changed={n.name:n for n in after.body if isinstance(n,ast.FunctionDef)}
    assert original.keys()==changed.keys(),'Count method interfaces differ'
    allowed=PERFORMANCE_METHODS|({'_mass_topology'} if include_mst else set())
    actual={name for name in original if ast_hash(original[name])!=ast_hash(changed[name])}
    assert actual<=allowed,'Unapproved count law changes: '+repr(actual-allowed)
    # Recovering the original methods must reproduce the full count class AST.
    after.body=[original[n.name] if isinstance(n,ast.FunctionDef) and n.name in allowed else n for n in after.body]
    assert ast_hash(before)==ast_hash(after),'Count state/statistical attributes differ'
    functions_old={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
    functions_new={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
    for name in ('_integer_allocate','conditional_count_pvalues'):
        assert ast_hash(functions_old[name])==ast_hash(functions_new[name]),name+' changed'
    for name in ('Q','PARENT_RESERVOIR','POWER_PASSES','LLOYD_PASSES'):
        a=next(n for n in old.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        b=next(n for n in new.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        assert ast_hash(a)==ast_hash(b),name+' changed'
    return dict(unchanged_count_class_except=sorted(actual),statistical_helpers_exact=True,
                original_method_ast_sha256={k:ast_hash(v) for k,v in original.items()},
                proposal_method_ast_sha256={k:ast_hash(v) for k,v in changed.items()})


def grouped_quota_checks(old,new,data,device):
    """Compare all four batched integer calls in each actually executed phase."""
    original=new._group_integer_allocate
    captures=[]
    def checked(capacity,groups,totals):
        result=original(capacity,groups,totals)
        expected=torch.zeros_like(capacity)
        for group in range(len(totals)):
            expected+=old._integer_allocate(capacity*(groups==group),int(totals[group]))
        assert torch.equal(expected,result),'Batched integer quotas changed'
        assert bool(((result>=0)&(result<=capacity)).all())
        captures.append(dict(groups=len(totals),cells=len(capacity),exact=True))
        return result
    new._group_integer_allocate=checked
    try:
        records=[]
        for name,value,kw in common.variants(data):
            start=len(captures)
            result=common.execute(common.prepare(new,value,device,**kw))
            records.append(dict(name=name,batched_calls=len(captures)-start,
                ordinary=len(result['child']),global_moves=result['detail'].get('global_moves',0)))
    finally:
        new._group_integer_allocate=original
    return dict(calls=len(captures),calls_exact=all(c['exact'] for c in captures),cases=records)


def actions(prepared):
    """The native planner methods only, without test state capture in profiles."""
    snap,q,flags,pvalues,comparison,generator,max_moves=prepared
    child,parent,detail=snap.ordinary_transport(q,flags,comparison,generator=generator,
                                             pvalues=pvalues,max_moves=max_moves)
    iso=snap.select_parents(q,flags,ordinary_children=child,ordinary_parents=parent,
                           generator=generator,pvalues=pvalues)
    return child,parent,detail,iso

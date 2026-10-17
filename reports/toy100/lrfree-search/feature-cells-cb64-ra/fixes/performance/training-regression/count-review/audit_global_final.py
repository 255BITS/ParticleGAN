"""Independent CPU review of final global certificates and exact planner splices."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode=True
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
from types import ModuleType,SimpleNamespace
import torch

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
COUNT=ROOT/"integration/review/training-regression/global-count"
JOINT=ROOT/"integration/review/training-regression/joint-count/pkg-joint-count"
PLAN=ROOT/"performance/sampler-regression/cpu-plan-review/plan-batching/pkg-PLAN-FINAL"
V4=ROOT/"integration/review/training-regression/pkg-count-recovery"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[Path(__file__),COUNT/"inputs.pt",COUNT/"prepare-inputs.json",COUNT/"PROTOCOL.md",
       JOINT/"particlegan/feature_cells.py",V4/"particlegan/feature_cells.py"]
paths += [p for root in (COUNT/"pkg-global-count",PLAN) for p in (root/"particlegan").glob("*.py")]
before={str(p):sha(p) for p in paths}
assert sha(COUNT/"pkg-global-count/particlegan/feature_cells.py")=="1d3c90aa3b72d086df0defcc8c356e48778310f005473e8c5c271afa44f89d76"
assert sha(PLAN/"particlegan/feature_cells.py")=="e6d2823d8e862dd8b29182cf0988325328523be0e77ae1fb2ea964707b34ad7f"
assert sha(COUNT/"inputs.pt")=="3e8deeb2fe534d28cf3731a4694c0543e280d2de270e0d9644731d9ec40057f2"


def package(name,root):
    holder=ModuleType(name);holder.__path__=[str(root/"particlegan")];sys.modules[name]=holder
    return importlib.import_module(name+".feature_cells")


count=package("independent_global_count",COUNT/"pkg-global-count")
plan=package("independent_global_plan",PLAN)
joint=package("independent_global_joint",JOINT)
v4=package("independent_global_v4",V4)
data=torch.load(COUNT/"inputs.pt",map_location="cpu",weights_only=False)
fallback=data["cases"]["saved_toy_1000"]["planning_rng"]
checks=[];plan_checks=[]
def checked(name,**kw):checks.append(dict(name=name,status="PASS",**kw))


def methods(path):
    module=ast.parse(Path(path).read_text())
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=="FeatureCellSnapshot")
    return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}


old_ast=methods(JOINT/"particlegan/feature_cells.py")
new_ast=methods(COUNT/"pkg-global-count/particlegan/feature_cells.py")
for name in ("fit","_fit_count_partition_metric","_count_categories_metric","support","_pool",
             "_mass_targets","_mass_topology","select_parents","_ordinary_mass_transport","_ordinary_support_transport"):
    assert old_ast[name]==new_ast[name],name
checked("even_fit_boundary_support_parent_law_and_both_previous_phase_sources_unchanged",
        count_boundary_fit_inputs="even references only",quality_gates_changed=False)


def exact_p(r,f,nr,nf):
    total=r+f;low,high=max(0,total-nf),min(total,nr)
    weights=[math.comb(nr,j)*math.comb(nf,total-j) for j in range(low,high+1)]
    return Fraction(sum(w for w in weights if w<=weights[r-low]),math.comb(nr+nf,total))


nulls=[];pooled=(6,6,4,0)
for nr,nf in ((8,8),(5,11)):
    total_prob,rejected=Fraction(0),Fraction(0);allocations=0;max_error=0.
    for allocation in itertools.product(*(range(n+1) for n in pooled)):
        if sum(allocation)!=nr:continue
        allocations+=1
        probability=Fraction(math.prod(math.comb(n,r) for n,r in zip(pooled,allocation)),math.comb(nr+nf,nr))
        total_prob+=probability
        r=torch.tensor(allocation);f=torch.tensor(pooled)-r
        families=((r.reshape(2,2).sum(1),f.reshape(2,2).sum(1)),(r,f),
                  (r.reshape(2,2).sum(0),f.reshape(2,2).sum(0)))
        got=torch.cat([count.conditional_count_pvalues(rc,fc,nr,nf) for rc,fc in families])
        exact=[exact_p(int(a),int(b),nr,nf) for rc,fc in families for a,b in zip(rc,fc)]
        max_error=max(max_error,max(abs(float(a)-float(b)) for a,b in zip(got,exact)))
        reject=any(p<=Fraction(1,160) for p in exact)
        assert bool((got<=.05/8).any())==reject
        if reject:rejected+=probability
        assert all(int(rc.sum())==nr and int(fc.sum())==nf for rc,fc in families)
    assert total_prob==1 and rejected<=Fraction(1,20) and max_error<2e-12
    nulls.append(dict(real_rows=nr,fake_rows=nf,allocations=allocations,hypotheses=8,
                      rejection_probability=float(rejected),rejection_probability_exact=str(rejected),max_pvalue_error=max_error))
checked("exact_overlapping_K_2K_and_global_2_null_actual_sample_sizes_and_empty_bin",nulls=nulls)


def restore(module,value):
    snap=module.FeatureCellSnapshot.__new__(module.FeatureCellSnapshot)
    snap.__dict__.update(deepcopy(value["snapshot"]))
    return snap


def stream(value):
    return torch.Generator(device="cpu").set_state(value["planning_rng"] if value.get("planning_rng") is not None else fallback)


def execute(module,value,max_moves=None,cold=False):
    snap=restore(module,value)
    if cold:
        for key in ("mass_group_ids","mass_groups","mass_topology"):snap.__dict__.pop(key,None)
    rng=stream(value);cmp=snap.cell_comparison(value["fake_features"])
    child,parent,detail=snap.ordinary_transport(value["q"],value["flags"],cmp,generator=rng,
                                               pvalues=value["pvalues"],max_moves=max_moves)
    ordinary_rng=rng.get_state().clone()
    ic,ip,iso=snap.select_parents(value["q"],value["flags"],ordinary_children=child,ordinary_parents=parent,
                                 generator=rng,pvalues=value["pvalues"])
    return snap,dict(child=child,parent=parent,detail=detail,ic=ic,ip=ip,iso=iso,comparison=cmp,
                     ordinary_rng=ordinary_rng,final_rng=rng.get_state(),work=snap.work,
                     groups=snap.mass_group_ids,topology=snap.mass_topology)


def inspect_plan(name,value,max_moves=None):
    snap,out=execute(count,value,max_moves)
    d,cmp=out["detail"],out["comparison"];n,k=len(value["q"]),snap.cells
    flags,p=value["flags"],value["pvalues"]
    child,parent,ic,ip=out["child"],out["parent"],out["ic"],out["ip"]
    ids,categories=d["query_cell_ids"],d["query_category_ids"]
    tally=lambda rows:torch.bincount(ids[rows],minlength=k)
    assert cmp["family_sizes"]==(k,2*k,2)
    assert len(cmp["pvalues"])==cmp["multiplicity"]==3*k+2 and cmp["cutoff"]==.05/(3*k+2)
    for family in ("mass","support","global_support"):
        c=cmp[family]
        assert int(c["real_counts"].sum())==snap.calibration_rows
        assert int(c["fake_counts"].sum())==len(value["fake_features"])
        difference=c["fake_counts"].double()/len(value["fake_features"])-c["real_counts"].double()/snap.calibration_rows
        want=count.conditional_count_pvalues(c["real_counts"],c["fake_counts"],snap.calibration_rows,len(value["fake_features"]))
        assert torch.equal(want,c["pvalues"]) and torch.equal(difference,c["difference"])
        assert torch.equal(c["excess"],(want<=cmp["cutoff"])&(difference>0))
        assert torch.equal(c["deficit"],(want<=cmp["cutoff"])&(difference<0))
    assert torch.equal(cmp["global_support"]["real_counts"],cmp["support"]["real_counts"].reshape(k,2).sum(0))
    assert torch.equal(cmp["global_support"]["fake_counts"],cmp["support"]["fake_counts"].reshape(k,2).sum(0))
    budget=math.floor(.05*n) if max_moves is None else max(0,min(max_moves,math.floor(.05*n)))
    assert len(child)==len(parent)<=budget and d["budget"]==budget
    all_child,all_parent=torch.cat((child,ic)),torch.cat((parent,ip))
    assert len(torch.unique(all_child))==len(all_child) and len(torch.unique(all_parent))==len(all_parent)
    assert not set(all_child.tolist())&set(all_parent.tolist())
    assert bool((~flags[all_parent]&(p[all_parent]>.05)).all())
    clean=tally((~flags).nonzero().flatten())
    mc,mp=d["mass_children"],d["mass_parents"]
    sc,sp=d["support_children"],d["support_parents"]
    gc,gp=d["global_children"],d["global_parents"]
    after_mass=clean-tally(mc[~flags[mc]])+tally(mp)
    after_local=after_mass+tally(sp)
    after_global=after_local+tally(gp)
    assert torch.equal(after_mass,d["supported_after_mass"])
    assert torch.equal(after_local,d["supported_after_local"])
    assert torch.equal(after_global,d["planned_supported_counts"])
    assert torch.equal(after_global,clean-tally(child[~flags[child]])+tally(parent))
    assert d["support_phase"]["budget"]==budget-len(mc)
    assert d["global_phase"]["budget"]==budget-len(mc)-len(sc)
    old,old_rng=restore(v4,value),stream(value)
    vc,vp,_=old.ordinary_transport(value["q"],flags,cmp["mass"],generator=old_rng,pvalues=p,max_moves=budget)
    assert torch.equal(vc,mc) and torch.equal(vp,mp)
    for c,pa,family in ((sc,sp,"support"),(gc,gp,"global_support")):
        assert bool(flags[c].all()) and bool((categories[c]%2==1).all())
        assert bool((categories[pa]%2==0).all())
        if family=="support":
            assert bool(cmp[family]["excess"][categories[c]].all()) and bool(cmp[family]["deficit"][categories[pa]].all())
        elif len(c):
            assert bool(cmp[family]["excess"][1]) and bool(cmp[family]["deficit"][0])
    for phase,previous_child,previous_parent,allocation in ((d["support_phase"],mc,mp,(sc,sp)),
            (d["global_phase"],torch.cat((mc,sc)),torch.cat((mp,sp)),(gc,gp))):
        if not phase["ran"]:continue
        if phase is d["support_phase"]:
            spent_death=torch.bincount(categories[previous_child],minlength=2*k).reshape(k,2)[:,1]
            spent_birth=torch.bincount(categories[previous_parent],minlength=2*k).reshape(k,2)[:,0]
            death_alloc,birth_alloc=tally(allocation[0]),tally(allocation[1])
            certified_difference=cmp["support"]["difference"].reshape(k,2)
            expected_raw_death=(n*certified_difference[:,1].clamp_min(0)+1e-10).floor().long()*cmp["support"]["excess"][1::2]
            expected_raw_birth=(n*(-certified_difference[:,0]).clamp_min(0)+1e-10).floor().long()*cmp["support"]["deficit"][0::2]
        else:
            spent_death=(categories[previous_child]%2==1).sum()
            spent_birth=(categories[previous_parent]%2==0).sum()
            death_alloc=birth_alloc=len(gc)
            assert torch.equal(phase["clean_counts"],after_local)
            certified_difference=cmp["global_support"]["difference"]
            expected_raw_death=(n*certified_difference[1].clamp_min(0)+1e-10).floor().long()*cmp["global_support"]["excess"][1]
            expected_raw_birth=(n*(-certified_difference[0]).clamp_min(0)+1e-10).floor().long()*cmp["global_support"]["deficit"][0]
        assert torch.equal(phase["raw_certified_death_capacity"],expected_raw_death)
        assert torch.equal(phase["raw_certified_birth_capacity"],expected_raw_birth)
        for direction,spent,allocated in (("death",spent_death,death_alloc),("birth",spent_birth,birth_alloc)):
            raw=phase[f"raw_certified_{direction}_capacity"]
            residual=(raw-spent).clamp_min(0)
            assert torch.equal(spent,phase[f"spent_certified_{direction}_capacity"])
            assert torch.equal(residual,phase[f"residual_certified_{direction}_capacity"])
            assert bool((allocated<=residual).all())
    counts=snap.reference_counts+snap.real_calibration_counts;denominator=int(counts.sum())
    targets=[n*int(c)//denominator for c in counts]
    order=sorted(range(k),key=lambda i:(-(n*int(counts[i])%denominator),i))
    for i in order[:n-sum(targets)]:targets[i]+=1
    target=torch.tensor(targets)
    assert torch.equal(target,d["target_counts"])
    assert bool((after_global<=torch.maximum(clean,target)).all())
    combined=after_global+tally(ip)
    assert bool((snap._group_counts(combined)<=torch.maximum(snap._group_counts(clean),snap._group_counts(target))).all())
    assert out["iso"]["guard_passed"]==(0<int(flags.sum())<=.05*n)
    if out["iso"]["guard_passed"]:
        assert torch.equal(out["iso"]["kept_counts"],after_global) and bool(flags[ic].all())
    else:assert len(ic)==0
    if name=="rare_hole":
        group_target=snap._group_counts(target);rare=int(group_target.argmin())
        legitimate=((~flags)&(snap.mass_group_ids[ids]==rare)).nonzero().flatten()
        assert len(legitimate)==2 and not set(legitimate.tolist())&set(all_child.tolist())
        assert int(snap._group_counts(combined)[rare])==2
    if name in ("global_no_signal","global_opposite_signal"):assert len(gc)==0
    return dict(name=name,mass=len(mc),local=len(sc),global_moves=len(gc),ordinary=len(child),isolation=len(ic),
                shared_budget=budget,shared_supported_ledger_exact=True,gross_certificates_reserved=True),snap,out


case_rows=[]
for name,value in data["cases"].items():
    row,_,_=inspect_plan(name,value);case_rows.append(row)
assert next(r for r in case_rows if r["name"]=="saved_toy_1000")["global_moves"]==24
assert next(r for r in case_rows if r["name"]=="saved_toy_2000")["global_moves"]==24
assert next(r for r in case_rows if r["name"]=="supported_mass_small_holes")["isolation"]==46
for budget in (0,49):
    row,_,_=inspect_plan("limited_budget",data["cases"]["saved_toy_1000"],budget)
    assert row["ordinary"]==budget
checked("all_saved_phases_fresh_certificates_budgets_exact_supported_targets_and_rare_survival",cases=case_rows,
        limited_budgets=[0,49])


def cloud(center,n,width):return torch.stack((torch.linspace(center-width,center+width,n,dtype=torch.float64),torch.zeros(n,dtype=torch.float64)),1)
even=torch.cat((cloud(-3.,154,.04),cloud(3.,358,.04)))
odd=torch.cat((cloud(-3.,26,.017),cloud(3.,486,.017)))
real=torch.empty((1024,2),dtype=torch.float64);real[0::2],real[1::2]=even,odd
rng=torch.Generator(device="cpu").set_state(fallback)
snap=count.FeatureCellSnapshot.fit(real,generator=rng,cells=2,rank=2,chunk=256)
left,right,outside=(torch.tensor([point],dtype=torch.float64) for point in ((-3.,0.),(3.,0.),(6.,0.)))
q=torch.cat((left.expand(64,-1),right.expand(256,-1),outside.expand(704,-1))).clone()
fake=torch.cat((left.expand(20,-1),right.expand(989,-1),outside.expand(15,-1))).clone()
flags,p,_=snap.support(q);assert int(flags.sum())==704
snap.cache_queries(q)
stress=dict(snapshot=deepcopy(vars(snap)),q=q,flags=flags,pvalues=p,fake_features=fake,planning_rng=rng.get_state())
row,snap,out=inspect_plan("global_certificate_exhausted",stress)
phase=out["detail"]["global_phase"]
assert phase["ran"] and row["mass"]==32 and row["global_moves"]==0
assert int(phase["raw_certified_birth_capacity"])==int(phase["raw_certified_death_capacity"])==15
assert int(phase["spent_certified_birth_capacity"])==int(phase["spent_certified_death_capacity"])==32
assert int(phase["residual_certified_birth_capacity"])==int(phase["residual_certified_death_capacity"])==0
assert int(phase["eligible_parent_counts"].sum())>0 and int(phase["clean_vacancies"].sum())>0
checked("global_certificate_exhaustion_blocks_reuse_despite_parents_vacancies_and_budget",case=row,
        raw_capacity=15,prior_gross_spending=32,residual_capacity=0,remaining_budget=phase["budget"])


constant=torch.ones((6,2),dtype=torch.float64)
degenerate=count.FeatureCellSnapshot.fit(constant,generator=stream({}),cells=4,rank=2,chunk=2)
cmp=degenerate.cell_comparison(constant[:2])
assert degenerate.cells==3 and cmp["multiplicity"]==11 and len(cmp["pvalues"])==11
assert bool((cmp["pvalues"]==1).all())
assert all(not bool((cmp[f]["excess"]|cmp[f]["deficit"]).any()) for f in ("mass","support","global_support"))
checked("degenerate_tied_cells_and_global_empty_bin_retain_actual_multiplicity",actual_K=3,hypotheses=11)


# Distinct count settings reject old semantic checkpoints before mutation.
recipe=SimpleNamespace(birth_death_space="critic",birth_death_isolation=True,birth_death_feature_scale="reference",
                       birth_death_cells=64,birth_death_metric_rank=8,birth_death_chunk=256,birth_death_parent_policy="real_anchor")
trainer=SimpleNamespace(prior=SimpleNamespace(z=torch.nn.Parameter(torch.arange(2048,dtype=torch.float64).reshape(1024,2))),
                        D=torch.nn.Linear(2,1,device="meta"),recipe=recipe,device=torch.device("cpu"),dtype=torch.float64,
                        controller=SimpleNamespace(latent_bandwidth=torch.ones(2)))
old_backend=joint.FeatureCellBirthDeath(trainer,data["cases"]["saved_toy_1000"]["seed"])
new_backend=count.FeatureCellBirthDeath(trainer,data["cases"]["saved_toy_1000"]["seed"])
before_state=deepcopy(new_backend.state_dict())
try:new_backend.load_state_dict(old_backend.state_dict())
except ValueError:pass
else:raise AssertionError("old count law checkpoint accepted")
assert new_backend.settings["count_family"]=="original_K_plus_support_2K_plus_global_2_common_Q_over_3K_plus_2"
assert torch.equal(new_backend.S,before_state["S"])
assert torch.equal(new_backend.stream.get_state(),before_state["stream"])
checked("new_count_settings_reject_previous_law_before_state_mutation")


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


base_files={p.name:sha(p) for p in (COUNT/"pkg-global-count/particlegan").glob("*.py")}
plan_files={p.name:sha(p) for p in (PLAN/"particlegan").glob("*.py")}
assert base_files.keys()==plan_files.keys()
assert {n for n in base_files if base_files[n]!=plan_files[n]}=={"feature_cells.py"}
optimized_ast=methods(PLAN/"particlegan/feature_cells.py")
changed_methods={name for name in new_ast if new_ast[name]!=optimized_ast[name]}
assert changed_methods=={"_ordinary_mass_transport","_ordinary_support_transport","_ordinary_global_transport"}
assert new_ast["_mass_topology"]==optimized_ast["_mass_topology"]
pair_rows=[]
for name,value in [(n,data["cases"][n]) for n in ("saved_toy_1000","saved_toy_2000","rare_hole","supported_mass_small_holes")]+[("global_certificate_exhausted",stress)]:
    a,oa=execute(count,value,cold=True);b,ob=execute(plan,value,cold=True)
    assert same(oa,ob),name
    pair_rows.append(dict(name=name,ordinary=len(oa["child"]),isolation=len(oa["ic"]),
                          all_phases_ledgers_certificates_rng_topology_and_work_bit_identical=True))
plan_checks.append(dict(name="exact_three_phase_batched_splices_only_original_MST_retained",status="PASS",cases=pair_rows))

assert before=={str(p):sha(p) for p in paths}
assert not torch.cuda.is_initialized()
common=dict(status="PASS",source_sha256=before,cuda_initialized=False,optimizer_updates=0,new_seeds=0,
            high_dimensional_support_law_qualified=False,learned_quality_qualified=False)
global_receipt=dict(common,checks=checks,scope="Fixed-partition overlapping3K+2 law and conservative planned actions only")
plan_receipt=dict(common,checks=plan_checks,scope="Exact CPU three-phase planner source/action/RNG parity; original MST retained; no GPU performance claim")
(HERE/"final-global-review.json").write_text(json.dumps(global_receipt,indent=2)+"\n")
(HERE/"final-plan-review.json").write_text(json.dumps(plan_receipt,indent=2)+"\n")
print(json.dumps(dict(status="PASS",global_checks=checks,plan_checks=plan_checks,cuda_initialized=False),indent=2),flush=True)

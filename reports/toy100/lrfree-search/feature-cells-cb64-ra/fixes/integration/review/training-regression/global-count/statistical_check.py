"""CPU exact overlapping-family null, even boundary and checkpoint contracts."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode=True
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
SNAPS=HERE.parent


def package(name,path):
    holder=ModuleType(name);holder.__path__=[str(path/"particlegan")];sys.modules[name]=holder
    return importlib.import_module(name+".feature_cells")


joint=package("global_stat_candidate",HERE/"pkg-global-count")
frozen=package("global_stat_frozen_3K",SNAPS/"joint-count/pkg-joint-count")
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=[Path(__file__),HERE/"pkg-global-count/particlegan/feature_cells.py",HERE/"inputs.pt",
    SNAPS/"joint-count/pkg-joint-count/particlegan/feature_cells.py",SNAPS/"joint-count/READY.json"]
before={str(p):sha(p) for p in sources}
data=torch.load(HERE/"inputs.pt",map_location="cpu",weights_only=False)
checks=[];nulls=[]


def exact_p(a,b,n,m):
    total=a+b
    probabilities={j:Fraction(math.comb(n,j)*math.comb(m,total-j),math.comb(n+m,total))
                   for j in range(max(0,total-m),min(n,total)+1)}
    return sum((x for x in probabilities.values() if x<=probabilities[a]),Fraction(0))


for n,m,pooled in ((8,8,(4,4,8,0)),(8,8,(6,6,4,0)),(8,12,(4,6,10,0))):
    outcomes=[];error=0.
    for real in itertools.product(*(range(min(c,n)+1) for c in pooled)):
        if sum(real)!=n:continue
        weight=Fraction(math.prod(math.comb(c,a) for c,a in zip(pooled,real)),math.comb(n+m,n))
        if not weight:continue
        fake=tuple(c-a for c,a in zip(pooled,real))
        mass_real=(real[0]+real[1],real[2]+real[3])
        mass_fake=(fake[0]+fake[1],fake[2]+fake[3])
        global_real=(real[0]+real[2],real[1]+real[3])
        global_fake=(fake[0]+fake[2],fake[1]+fake[3])
        p=torch.cat([joint.conditional_count_pvalues(torch.tensor(a),torch.tensor(b),n,m)
                     for a,b in ((mass_real,mass_fake),(real,fake),(global_real,global_fake))])
        oracle=[exact_p(a,b,n,m) for a,b in zip(mass_real+real+global_real,mass_fake+fake+global_fake)]
        error=max(error,float((p-torch.tensor([float(x) for x in oracle],dtype=torch.float64)).abs().max()))
        outcomes.append((weight,oracle))
    assert sum((w for w,_ in outcomes),Fraction(0))==1
    for index in range(8):
        for alpha in (Fraction(1,160),Fraction(1,20),Fraction(1,10),Fraction(1,2),Fraction(1)):
            rejected=sum((w for w,p in outcomes if p[index]<=alpha),Fraction(0))
            assert rejected<=alpha
    family=sum((w for w,p in outcomes if any(x<=Fraction(1,160) for x in p)),Fraction(0))
    assert family<=Fraction(1,20) and error<1e-12
    nulls.append(dict(real_rows=n,fake_rows=m,pooled_support=pooled,original_cells=2,refined_categories=4,
                     global_categories=2,actual_hypotheses=8,cutoff=.05/8,allocations=len(outcomes),family_probability_exact=str(family),
                     family_probability=float(family),max_pvalue_error=error))
checks.append("exact_overlapping_K_plus_2K_plus_global2_common_family_null")

value=data["cases"]["saved_toy_1000"]
real=value["real_features"][:64]
def stream():return torch.Generator().set_state(value["planning_rng"])
a_stream,b_stream,c_stream=stream(),stream(),stream()
a=joint.FeatureCellSnapshot.fit(real,generator=a_stream,cells=6,rank=8,chunk=16)
b=frozen.FeatureCellSnapshot.fit(real,generator=b_stream,cells=6,rank=8,chunk=16)
changed=real.clone();changed[1::2]+=100.
c=joint.FeatureCellSnapshot.fit(changed,generator=c_stream,cells=6,rank=8,chunk=16)
for name in ("mean","scale","basis","centers","cell_scale","count_boundary","reference_category_counts"):
    assert torch.equal(getattr(a,name),getattr(b,name)),name
    assert torch.equal(getattr(a,name),getattr(c,name)),name
assert torch.equal(a_stream.get_state(),b_stream.get_state()) and torch.equal(a_stream.get_state(),c_stream.get_state())
assert torch.equal(a.null_scores,b.null_scores)
for av,bv in zip(a.support(real),b.support(real)):assert torch.equal(av,bv)
comparison=a.cell_comparison(real)
assert comparison["multiplicity"]==20 and len(comparison["pvalues"])==20
before_boundary=a.count_boundary.clone();a.cell_comparison(real+100.);assert torch.equal(before_boundary,a.count_boundary)
checks.append("frozen_3K_geometry_score_boundary_rng_and_no_odd_fake_leakage")

for real in (torch.ones((6,3),dtype=torch.float64),torch.arange(18,dtype=torch.float64).reshape(6,3),
             torch.arange(16,dtype=torch.float64).remainder(2)[:,None].expand(-1,3).clone()):
    snap=joint.FeatureCellSnapshot.fit(real,generator=stream(),cells=8,rank=3,chunk=2)
    comparison=snap.cell_comparison(real+100.)
    assert len(comparison["pvalues"])==3*snap.cells+2
    assert bool(torch.isfinite(comparison["pvalues"]).all())
    if not snap.valid_metric:
        assert not any(bool((comparison[name]["excess"]|comparison[name]["deficit"]).any()) for name in ("mass","support","global_support"))
    scores=snap._scores_metric(snap.transform(real[0::2]));categories=snap.count_categories(real[0::2])
    assert bool((categories[scores==snap.count_boundary].remainder(2)==0).all())
checks.append("empty_tied_minimum_degenerate_bins_retained")

recipe=SimpleNamespace(birth_death_space="critic",birth_death_isolation=True,birth_death_feature_scale="std",
    birth_death_cells=64,birth_death_metric_rank=8,birth_death_chunk=256,birth_death_parent_policy="real_anchor")
trainer=SimpleNamespace(recipe=recipe,device=torch.device("cpu"),dtype=torch.float32,controller=None,
    prior=SimpleNamespace(z=torch.nn.Parameter(torch.zeros((1024,2)))),D=torch.nn.Linear(2,1))
controller=joint.FeatureCellBirthDeath(trainer,value["seed"])
state=controller.state_dict();restored=joint.FeatureCellBirthDeath(trainer,value["seed"])
restored.load_state_dict(state)
assert restored.settings["count_family"]=="original_K_plus_support_2K_plus_global_2_common_Q_over_3K_plus_2"
previous=deepcopy(state);previous["settings"]["count_family"]="original_K_plus_support_2K_common_Q_over_3K"
previous["settings"]["mass_policy"]="joint_mass_support_common_3K_unique_parents_v1"
try:restored.load_state_dict(previous)
except ValueError:pass
else:raise AssertionError("frozen3K checkpoint accepted by different count law")
assert torch.equal(state["stream"],restored.state_dict()["stream"])
checks.append("global_checkpoint_distinct_from_frozen_3K")


# Existing two action methods remain identical, including their draw order.
import ast
base_text=(SNAPS/"joint-count/pkg-joint-count/particlegan/feature_cells.py").read_text()
new_text=(HERE/"pkg-global-count/particlegan/feature_cells.py").read_text()
def methods(text):
    return {n.name:ast.dump(n,include_attributes=False) for c in ast.parse(text).body
        if isinstance(c,ast.ClassDef) and c.name=="FeatureCellSnapshot"
        for n in c.body if isinstance(n,ast.FunctionDef)}
old_methods,new_methods=methods(base_text),methods(new_text)
for name in ("_ordinary_mass_transport","_ordinary_support_transport","select_parents"):
    assert old_methods[name]==new_methods[name],name
checks.append("mass_local_support_and_isolation_method_bodies_unchanged")

assert before=={p:sha(Path(p)) for p in before} and not torch.cuda.is_initialized()
receipt=dict(status="PASS",checks=checks,exact_null_cases=nulls,source_sha256=before,sources_unchanged=True,
             cuda_initialized=False,new_seeds=0,optimizer_updates=0,
             scope="CPU finite conditional family, boundary and checkpoint contracts; no quality verdict")
(HERE/"statistical-check.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt),flush=True)

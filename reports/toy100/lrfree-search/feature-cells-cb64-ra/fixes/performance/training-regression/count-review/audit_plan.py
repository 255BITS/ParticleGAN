"""Independent finite CPU contracts for the separate batched planner prototype."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "integration/review/training-regression/joint-count/pkg-joint-count"
NEW = ROOT / "performance/sampler-regression/cpu-plan-review/plan-batching/pkg-PLAN"
INPUT = ROOT / "integration/review/training-regression/joint-count/inputs.pt"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths = [Path(__file__), INPUT] + [p for root in (BASE, NEW) for p in (root/"particlegan").glob("*.py")]
before = {str(p):sha(p) for p in paths}
assert sha(NEW/"particlegan/feature_cells.py") == "113a2e0553a45ef1898130af1f9c25703e8d03af25525636c3c3ecbb1fa5495e"


def package(name, root):
    holder = ModuleType(name)
    holder.__path__ = [str(root/"particlegan")]
    sys.modules[name] = holder
    return importlib.import_module(name+".feature_cells")


base, new = package("independent_plan_base", BASE), package("independent_plan_new", NEW)
checks = []


def checked(name, **kw):
    checks.append(dict(name=name,status="PASS",**kw))


# Independent rational allocation, including stable global-index tie order.
fixtures = [([2,0,7,1,0,3],[0,0,1,1,2,2],[1,3,9]),
            ([1,1,1,1,1,1],[0,2,0,1,2,1],[1,1,1]),
            ([64,64,1,1,51,0],[0,0,0,0,2,2],[51,20,1]),
            ([0,0,0,0],[0,2,0,2],[5,3,-1]),
            ([9,1,2,6],[1,0,1,0],[2,5])]
for cap, groups, totals in fixtures:
    want = [0]*len(cap)
    for group,total in enumerate(totals):
        members = [i for i,g in enumerate(groups) if g==group]
        available = sum(cap[i] for i in members)
        request = max(0,min(total,available))
        if available == 0:
            continue
        quotas = {i:Fraction(cap[i]*request,available) for i in members}
        for i in members:
            want[i] = quotas[i].numerator//quotas[i].denominator
        remainder = request-sum(want[i] for i in members)
        order = sorted(members,key=lambda i:(-(quotas[i]-want[i]),i))
        for i in order[:remainder]:
            want[i] += 1
    got = new._group_integer_allocate(torch.tensor(cap),torch.tensor(groups),torch.tensor(totals))
    assert got.tolist() == want, (cap,groups,totals,got.tolist(),want)
checked("group_allocation_matches_exact_rational_remainders_and_ties", fixtures=len(fixtures))


# Compare actual topology on finite, duplicate and tied real-centre tables.
tables = [[[0.,0.],[1.,0.],[1.,1.],[0.,1.]],
          [[0.,0.],[0.,0.],[1.,0.],[1.,0.],[9.,0.]],
          [[0.,0.],[.0001,0.],[10.,2.],[20.,0.]],
          [[0.,0.],[0.,0.],[0.,0.]],
          [[0.,0.],[3.,0.]]]
topology = []
for table in tables:
    snapshots = []
    for module in (base,new):
        snap = module.FeatureCellSnapshot.__new__(module.FeatureCellSnapshot)
        snap.centers = torch.tensor(table,dtype=torch.float64)
        snap.cells, snap.device = len(table),torch.device("cpu")
        snap.work = dict(distance_cells=0,max_distance_rows=0,max_distance_columns=0,retained_array_bytes=0)
        snap._mass_topology()
        snapshots.append(snap)
    a,b = snapshots
    assert torch.equal(a.mass_group_ids,b.mass_group_ids)
    assert a.mass_topology == b.mass_topology and a.work == b.work
    topology.append(dict(cells=a.cells,groups=a.mass_groups,group_ids=a.mass_group_ids.tolist(),
                         threshold_squared=a.mass_topology["threshold_squared"]))
checked("finite_tied_duplicate_Prim_groups_cut_and_work_match_frozen_source", cases=topology)


def same(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


data = torch.load(INPUT,map_location="cpu",weights_only=False)
fallback = data["cases"]["saved_toy_1000"]["planning_rng"]
plans = []
for name in ("saved_toy_1000","supported_mass_small_holes"):
    value = data["cases"][name]
    outputs = []
    for module in (base,new):
        snap = module.FeatureCellSnapshot.__new__(module.FeatureCellSnapshot)
        snap.__dict__.update(deepcopy(value["snapshot"]))
        for field in ("mass_group_ids","mass_groups","mass_topology"):
            snap.__dict__.pop(field,None)
        rng = torch.Generator(device="cpu").set_state(value["planning_rng"] if value.get("planning_rng") is not None else fallback)
        comparison = snap.cell_comparison(value["fake_features"])
        child,parent,detail = snap.ordinary_transport(value["q"],value["flags"],comparison,
                                                     generator=rng,pvalues=value["pvalues"])
        ordinary_rng = rng.get_state().clone()
        ic,ip,iso = snap.select_parents(value["q"],value["flags"],ordinary_children=child,
                                       ordinary_parents=parent,generator=rng,pvalues=value["pvalues"])
        outputs.append(dict(child=child,parent=parent,detail=detail,iso_child=ic,iso_parent=ip,iso=iso,
                            ordinary_rng=ordinary_rng,final_rng=rng.get_state(),comparison=comparison,
                            topology=snap.mass_topology,groups=snap.mass_group_ids,work=snap.work))
    assert same(*outputs), name
    a = outputs[0]
    plans.append(dict(name=name,ordinary=len(a["child"]),isolation=len(a["iso_child"]),
                      actions_all_details_comparison_topology_work_and_rng_bit_identical=True))
checked("cold_planner_broad_flags_and_supported_deaths_plus_small_isolation_exact", cases=plans)

assert before == {str(p):sha(p) for p in paths}
assert not torch.cuda.is_initialized()
receipt = dict(status="PASS",checks=checks,source_sha256=before,
               cuda_initialized=False,optimizer_updates=0,new_seeds=0,
               scope="Finite bounded planner allocation/topology and fixed action/RNG exact parity; no GPU speed or learned-quality claim")
(HERE/"plan-review.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(dict(status="PASS",checks=checks,cuda_initialized=False),indent=2),flush=True)

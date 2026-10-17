"""CPU-only fixed partition, exact null and combined action contracts."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
import torch

torch.set_num_threads(1); torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
HERE = Path(__file__).resolve().parent
SNAPS = HERE.parent
ROOT = SNAPS.parents[2]
SEED = 314159  # existing saved reconstruction seed, never varied


def package(name, path):
    holder = ModuleType(name); holder.__path__ = [str(path / "particlegan")]
    sys.modules[name] = holder
    return importlib.import_module(name + ".feature_cells")


v4 = package("support_count_v4", SNAPS / "pkg-count-recovery")
proposed = package("support_count_proposed", HERE / "pkg-support-count")
Q = proposed.Q
saved = {step: torch.load(SNAPS/f"snapshot-{step:04d}.pt", map_location="cpu", weights_only=False)
         for step in (1000, 2000)}
original = torch.load(ROOT/"stability/mass-gpu-inputs.pt", map_location="cpu", weights_only=False)
null_rows = []
case_rows = []


def stream(state=None):
    generator = torch.Generator().manual_seed(SEED)
    if state is not None: generator.set_state(state)
    return generator


def snapshot(module, value, *, refined=False):
    snap = module.FeatureCellSnapshot.__new__(module.FeatureCellSnapshot)
    snap.__dict__.update(deepcopy(value["snapshot"]))
    if refined:
        even = value["real_features"][0::2]
        metric = snap.transform(even)
        ids, _ = snap._assign_metric(metric)
        assert torch.equal(torch.bincount(ids,minlength=snap.cells),snap.reference_counts)
        snap._fit_count_partition_metric(metric, ids)
        cats = snap.count_categories(value["real_features"][1::2])
        snap.real_calibration_category_counts = torch.bincount(cats,minlength=2*snap.cells)
        snap._update_storage()
    return snap


def plan(module, value, *, refined=False, flags=None, pvalues=None, fake=None):
    snap = snapshot(module,value,refined=refined)
    flags = value["flags"].clone() if flags is None else flags.clone()
    pvalues = value["pvalues"].clone() if pvalues is None else pvalues.clone()
    generator = stream(value.get("planning_rng"))
    comparison = snap.cell_comparison(value["fake_features"] if fake is None else fake) if refined else deepcopy(value["comparison"])
    child,parent,detail = snap.ordinary_transport(value["q"],flags,comparison,generator=generator,pvalues=pvalues)
    iso_child,iso_parent,isolation = snap.select_parents(value["q"],flags,ordinary_children=child,
                    ordinary_parents=parent,generator=generator,pvalues=pvalues)
    return snap,flags,pvalues,comparison,child,parent,detail,iso_child,iso_parent,isolation


def plain(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [plain(v) for v in value]
    return value


def exact_count_pvalue(a, b, n, m):
    total=a+b
    weights={j:Fraction(math.comb(n,j)*math.comb(m,total-j), math.comb(n+m,total))
             for j in range(max(0,total-m),min(n,total)+1)}
    return sum((p for p in weights.values() if p<=weights[a]),Fraction(0))


class Contracts(unittest.TestCase):
    def check_plan(self, items):
        snap,flags,pvalues,comparison,child,parent,detail,iso_child,iso_parent,isolation=items
        ids=snap.query_cell_ids
        categories=detail["query_category_ids"]
        supported=torch.bincount(ids[~flags],minlength=snap.cells)
        planned=supported-torch.bincount(ids[child[~flags[child]]],minlength=snap.cells)+torch.bincount(ids[parent],minlength=snap.cells)
        combined=planned+torch.bincount(ids[iso_parent],minlength=snap.cells)
        self.assertEqual(len(child),len(parent));self.assertEqual(len(iso_child),len(iso_parent))
        self.assertLessEqual(len(child)+len(iso_child),math.floor(Q*len(flags)))
        self.assertEqual(len(torch.unique(torch.cat((child,iso_child)))),len(child)+len(iso_child))
        self.assertEqual(len(torch.unique(torch.cat((parent,iso_parent)))),len(parent)+len(iso_parent))
        self.assertFalse(bool(torch.isin(torch.cat((parent,iso_parent)),torch.cat((child,iso_child))).any()))
        self.assertTrue(bool(flags[child].all()))
        self.assertTrue(bool((categories[child].remainder(2)==1).all()))
        self.assertTrue(bool((categories[parent].remainder(2)==0).all()))
        self.assertTrue(bool((~flags[parent]&(pvalues[parent]>Q)).all()))
        self.assertTrue(bool(comparison["excess"][categories[child]].all()))
        self.assertTrue(bool(comparison["deficit"][categories[parent]].all()))
        self.assertTrue(torch.equal(planned,detail["planned_supported_counts"]))
        self.assertTrue(bool((planned<=torch.maximum(supported,detail["target_counts"])).all()))
        self.assertTrue(bool((snap._group_counts(combined)<=torch.maximum(snap._group_counts(supported),snap._group_counts(detail["target_counts"]))).all()))
        self.assertEqual(comparison["multiplicity"],2*snap.cells)
        self.assertEqual(comparison["cutoff"],Q/(2*snap.cells))
        if isolation["guard_passed"]:
            self.assertTrue(torch.equal(isolation["kept_counts"],planned))
        return combined

    def test_even_boundary_no_odd_or_fake_leakage_geometry_support_rng_exact(self):
        real=saved[1000]["real_features"]
        astream,bstream,cstream=stream(),stream(),stream()
        old=v4.FeatureCellSnapshot.fit(real,generator=astream,cells=64,rank=8,chunk=256)
        fitted=proposed.FeatureCellSnapshot.fit(real,generator=bstream,cells=64,rank=8,chunk=256)
        altered=real.clone();altered[1::2]=altered[1::2]+100.
        other=proposed.FeatureCellSnapshot.fit(altered,generator=cstream,cells=64,rank=8,chunk=256)
        for key in ("mean","scale","basis","centers","cell_scale","reference_counts","real_representatives","real_representative_rows"):
            self.assertTrue(torch.equal(getattr(old,key),getattr(fitted,key)),key)
            self.assertTrue(torch.equal(getattr(fitted,key),getattr(other,key)),key)
        self.assertTrue(torch.equal(astream.get_state(),bstream.get_state()))
        self.assertTrue(torch.equal(bstream.get_state(),cstream.get_state()))
        self.assertTrue(torch.equal(fitted.count_boundary,other.count_boundary))
        self.assertTrue(torch.equal(fitted.reference_category_counts,other.reference_category_counts))
        self.assertFalse(torch.equal(fitted.null_scores,other.null_scores))
        self.assertTrue(torch.equal(old.null_scores,fitted.null_scores))
        for a,b in zip(old.support(saved[1000]["q"]),fitted.support(saved[1000]["q"])):
            self.assertTrue(torch.equal(a,b))
        before=fitted.count_boundary.clone()
        fitted.cell_comparison(saved[1000]["fake_features"])
        fitted.cell_comparison(saved[1000]["fake_features"]+100.)
        self.assertTrue(torch.equal(before,fitted.count_boundary))
        with self.assertRaisesRegex(ValueError,"already frozen"):
            even=fitted.transform(real[0::2]);ids,_=fitted._assign_metric(even)
            fitted._fit_count_partition_metric(even,ids)

    def test_exact_conditional_null_and_actual_family_multiplicity(self):
        for n,m,pooled in ((8,8,(4,4,8,0)),(8,12,(4,6,10,0))):
            allocations=[]
            for counts in itertools.product(*(range(min(c,n)+1) for c in pooled)):
                if sum(counts)!=n:continue
                probability=Fraction(math.prod(math.comb(c,a) for c,a in zip(pooled,counts)),math.comb(n+m,n))
                if not probability:continue
                real=torch.tensor(counts,dtype=torch.long)
                fake=torch.tensor(pooled,dtype=torch.long)-real
                p=proposed.conditional_count_pvalues(real,fake,n,m)
                oracle=[exact_count_pvalue(a,int(b),n,m) for a,b in zip(counts,fake)]
                self.assertLess(float((p-torch.tensor([float(x) for x in oracle],dtype=torch.float64)).abs().max()),1e-12)
                allocations.append((probability,oracle))
            self.assertEqual(sum((w for w,_ in allocations),Fraction(0)),1)
            for index in range(len(pooled)):
                for alpha in (Fraction(1,80),Fraction(1,20),Fraction(1,10),Fraction(1,4),Fraction(1,2),Fraction(1)):
                    rejected=sum((w for w,p in allocations if p[index]<=alpha),Fraction(0))
                    self.assertLessEqual(rejected,alpha)
            family=sum((w for w,p in allocations if any(x<=Fraction(1,80) for x in p)),Fraction(0))
            self.assertLessEqual(family,Fraction(1,20))
            self.assertGreater(family,0)
            null_rows.append(dict(real_rows=n,fake_rows=m,pooled=pooled,possible_allocations=len(allocations),
                                  exact_family_rejection_probability=str(family),family_probability=float(family),cutoff=Q/4))

    def test_degenerate_empty_tied_minimum_and_refresh(self):
        cases=[torch.ones((16,3),dtype=torch.float64),
               torch.arange(16,dtype=torch.float64).remainder(2)[:,None].expand(-1,3).clone(),
               torch.arange(18,dtype=torch.float64).reshape(6,3)]
        for real in cases:
            snap=proposed.FeatureCellSnapshot.fit(real,generator=stream(),cells=8,rank=3,chunk=2)
            categories=snap.count_categories(real)
            self.assertEqual(int(snap.reference_category_counts.sum()),len(real[0::2]))
            self.assertEqual(int(snap.real_calibration_category_counts.sum()),len(real[1::2]))
            even_scores=snap._scores_metric(snap.transform(real[0::2]))
            inside=categories[0::2].remainder(2)==0
            self.assertTrue(bool(inside[even_scores==snap.count_boundary].all()))
            comparison=snap.cell_comparison(real+100.)
            self.assertEqual(len(comparison["pvalues"]),2*snap.cells)
            self.assertTrue(bool(torch.isfinite(comparison["pvalues"]).all()))
            if not snap.valid_metric:
                self.assertFalse(bool((comparison["excess"]|comparison["deficit"]).any()))
            snap.cache_queries(real)
            before=snap.count_boundary.clone();snap.refresh_rows(torch.tensor([0]),real[-1:].clone())
            self.assertTrue(torch.equal(before,snap.count_boundary))
            self.assertEqual(int(snap.query_counts.sum()),len(real))
        with self.assertRaises(ValueError):proposed.conditional_count_pvalues(torch.tensor([0]),torch.tensor([0]),1,1)

    def test_saved_snapshots_certificates_and_response(self):
        for step,value in saved.items():
            previous=plan(v4,value)
            fixed=plan(proposed,value,refined=True)
            self.check_plan(fixed)
            snap,flags,pvalues,comparison,child,parent,detail,iso_child,iso_parent,isolation=fixed
            after_flags,after_p,_=snap.support(value["q"])
            self.assertTrue(torch.equal(flags,after_flags));self.assertTrue(torch.equal(pvalues,after_p))
            self.assertGreater(len(child),len(previous[4]))
            rf=comparison["real_counts"].double()/snap.calibration_rows
            ff=comparison["fake_counts"].double()/len(value["fake_features"])
            row=dict(step=step,rows=len(value["q"]),flags=int(flags.sum()),v4_moves=len(previous[4]),
                     proposal_moves=len(child),isolation_moves=len(iso_child),count_boundary=float(snap.count_boundary),
                     discoveries=int((comparison["excess"]|comparison["deficit"]).sum()),
                     outside_excess=int(comparison["excess"].reshape(snap.cells,2)[:,1].sum()),
                     inside_deficit=int(comparison["deficit"].reshape(snap.cells,2)[:,0].sum()),
                     actual_inside_eligible=int(detail["eligible_parent_counts"].sum()),
                     real_inside_fraction=float(rf.reshape(snap.cells,2)[:,0].sum()),
                     emitted_inside_fraction=float(ff.reshape(snap.cells,2)[:,0].sum()),
                     refined_tv=float((rf-ff).abs().sum()*.5),
                     supported_deaths=int((~flags[child]).sum()),unique_parents=len(torch.unique(parent)),
                     child_categories=detail["child_category_ids"],parent_categories=detail["parent_category_ids"],
                     comparison=comparison,action_ledger=detail)
            case_rows.append(plain(row));print(json.dumps(dict(event="saved_case",**{k:plain(row[k]) for k in
                ("step","flags","v4_moves","proposal_moves","discoveries","real_inside_fraction","emitted_inside_fraction","supported_deaths")})),flush=True)

    def test_no_eligible_parent(self):
        value=saved[1000]
        fixed=plan(proposed,value,refined=True,pvalues=torch.zeros_like(value["pvalues"]))
        self.check_plan(fixed);self.assertEqual(len(fixed[4])+len(fixed[7]),0)

    def test_active_checkpoint_policy_and_derived_cache_reset(self):
        recipe=SimpleNamespace(birth_death_space="critic",birth_death_isolation=True,birth_death_feature_scale="std",
            birth_death_cells=64,birth_death_metric_rank=8,birth_death_chunk=256,birth_death_parent_policy="real_anchor")
        trainer=SimpleNamespace(recipe=recipe,device=torch.device("cpu"),dtype=torch.float32,controller=None,
            prior=SimpleNamespace(z=torch.nn.Parameter(torch.zeros((1024,2)))),D=torch.nn.Linear(2,1))
        controller=proposed.FeatureCellBirthDeath(trainer,SEED)
        state=controller.state_dict()
        self.assertEqual(state["settings"]["mass_policy"],"even_fit_support_categories_unique_parents_v1")
        restored=proposed.FeatureCellBirthDeath(trainer,SEED)
        restored.snapshot=snapshot(proposed,saved[1000],refined=True)
        restored.load_state_dict(state)
        self.assertIsNone(restored.snapshot)
        self.assertEqual(restored.latent_geometry.work["builds"],0)
        previous=deepcopy(state);previous["settings"].pop("count_partition")
        previous["settings"]["mass_policy"]="reference_topology_vacancies_unique_parents_v4"
        with self.assertRaisesRegex(ValueError,"backend/settings"):
            restored.load_state_dict(previous)
        self.assertTrue(torch.equal(state["stream"],restored.state_dict()["stream"]))

    def test_small_flag_shared_ledger_and_rare_group_preservation(self):
        value=deepcopy(saved[1000])
        source=snapshot(proposed,value,refined=True)
        ids=source.query_cell_ids;groups=source._mass_topology();categories=source.count_categories(value["q"])
        # Existing saved rows only. Keep precisely the smallest target group
        # intact and flag51 outside rows elsewhere, preserving its survivors.
        target=source._mass_targets(len(value["q"]));rare=int(source._group_counts(target).argmin())
        protected=groups[ids]==rare
        candidates=(value["flags"]&~protected&(categories.remainder(2)==1)).nonzero().flatten()
        flags=torch.zeros_like(value["flags"]);flags[candidates[:51]]=True
        fixed=plan(proposed,value,refined=True,flags=flags)
        combined=self.check_plan(fixed)
        self.assertTrue(fixed[9]["guard_passed"])
        self.assertGreater(len(fixed[4]),0)
        self.assertGreater(len(fixed[7]),0)
        self.assertFalse(bool(protected[fixed[4]].any()))
        snap_count=source._group_counts(combined)
        self.assertTrue(bool((snap_count<=torch.maximum(
            source._group_counts(torch.bincount(ids[~flags],minlength=source.cells)),source._group_counts(target))).all()))
        case_rows.append(plain(dict(name="small_flags_combined",flags=int(flags.sum()),ordinary=len(fixed[4]),
            isolation=len(fixed[7]),combined_moves=len(fixed[4])+len(fixed[7]),rare_group=rare,
            no_rare_deaths=True,kept_counts=fixed[9]["kept_counts"],combined_group_supported=snap_count,
            original_group_supported=source._group_counts(torch.bincount(ids[~flags],minlength=source.cells)),
            group_targets=source._group_counts(target))))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
sources=[Path(__file__),HERE/"PROTOCOL.md",SNAPS/"READY.json",SNAPS/"COUNT-RECOVERY.patch",
         SNAPS/"snapshot-1000.pt",SNAPS/"snapshot-2000.pt",ROOT/"stability/mass-gpu-inputs.pt"]
sources+=list(sorted((HERE/"pkg-support-count").rglob("*.py")))
sources+=list(sorted((SNAPS/"pkg-count-recovery").rglob("*.py")))
before={str(p):sha(p) for p in sources}
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
unchanged=before=={str(p):sha(p) for p in sources}
receipt=dict(status="PASS" if result.wasSuccessful() and unchanged else "FAIL",tests=result.testsRun,
             failures=len(result.failures),errors=len(result.errors),cuda_initialized=torch.cuda.is_initialized(),
             sources_unchanged=unchanged,source_sha256=before,exact_null_cases=null_rows,saved_cases=case_rows,
             scope="CPU fixed partition/null/action contracts; no training, new seeds or CUDA quality verdict")
(HERE/"cpu-check.json").write_text(json.dumps(plain(receipt),indent=2,allow_nan=False)+"\n")
print(json.dumps({k:receipt[k] for k in ("status","tests","failures","errors","cuda_initialized","sources_unchanged")}),flush=True)
assert not receipt["cuda_initialized"]
raise SystemExit(0 if receipt["status"]=="PASS" else 1)

"""CPU planning regressions; fixed saved partitions, scores and count evidence."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)


def package(name, path):
    holder = ModuleType(name); holder.__path__ = [str(path / "particlegan")]
    sys.modules[name] = holder
    return importlib.import_module(name + ".feature_cells")


current = package("recovery_current", ROOT / "pkg-CB64-RA2")
proposed = package("recovery_proposed", HERE / "pkg-count-recovery")
SEED = 314159
original = torch.load(ROOT / "stability/mass-gpu-inputs.pt", map_location="cpu", weights_only=False)
records = []


def snapshot(module, values):
    snap = module.FeatureCellSnapshot.__new__(module.FeatureCellSnapshot)
    for name, value in deepcopy(values).items():setattr(snap, name, value)
    return snap


def plan(module, value, *, flags=None, pvalues=None):
    snap = snapshot(module, value["snapshot"])
    flags = value["flags"].clone() if flags is None else flags.clone()
    pvalues = value["pvalues"].clone() if pvalues is None else pvalues.clone()
    generator = torch.Generator()
    if "planning_rng" in value:generator.set_state(value["planning_rng"])
    else:generator.manual_seed(original["seed"])
    comparison = deepcopy(value["comparison"])
    child, parent, detail = snap.ordinary_transport(value["q"], flags, comparison,
                                       generator=generator, pvalues=pvalues)
    iso_child, iso_parent, isolation = snap.select_parents(value["q"], flags, ordinary_children=child,
                    ordinary_parents=parent, generator=generator, pvalues=pvalues)
    return snap, flags, pvalues, child, parent, detail, iso_child, iso_parent, isolation, generator.get_state()


class CountRecovery(unittest.TestCase):
    def check_contract(self, snap, flags, pvalues, child, parent, detail, iso_child, iso_parent, isolation, rng):
        self.assertLessEqual(len(child), int(.05*len(flags)))
        self.assertEqual(len(child),len(parent))
        self.assertEqual(len(torch.unique(child)),len(child))
        self.assertEqual(len(torch.unique(torch.cat((parent,iso_parent)))),len(parent)+len(iso_parent))
        self.assertFalse(bool(torch.isin(parent,child).any()))
        self.assertFalse(bool(torch.isin(iso_parent,torch.cat((child,parent))).any()))
        self.assertTrue(bool((pvalues[parent]>.05).all()))
        self.assertFalse(bool(flags[parent].any()))
        comparison_ids = snap.query_cell_ids
        expected = torch.bincount(comparison_ids[~flags],minlength=snap.cells)
        expected -= torch.bincount(comparison_ids[child[~flags[child]]],minlength=snap.cells)
        expected += torch.bincount(comparison_ids[parent],minlength=snap.cells)
        self.assertTrue(torch.equal(detail["planned_supported_counts"],expected))
        if detail["isolation_guard_rejects"]:
            self.assertTrue(bool(flags[child].all()))
            self.assertEqual(len(iso_child),0)
            target = snap._group_counts(detail["target_counts"])
            initial = snap._group_counts(detail["clean_counts"])
            self.assertTrue(bool((snap._group_counts(expected)<=torch.maximum(target,initial)).all()))
        else:
            self.assertFalse(bool(flags[child].any()))

    def test_saved_broad_flags_act_with_unchanged_evidence_and_rng(self):
        for step in (1000,2000):
            value = torch.load(HERE / f"snapshot-{step:04d}.pt",map_location="cpu",weights_only=False)
            old = plan(current,value)
            fixed = plan(proposed,value)
            self.check_contract(*fixed)
            self.assertGreater(len(fixed[3]),len(old[3]),step)
            self.assertGreater(len(fixed[3]),0)
            self.assertTrue(fixed[5]["isolation_guard_rejects"])
            self.assertTrue(torch.equal(old[0].null_scores,fixed[0].null_scores))
            before = deepcopy(value["comparison"])
            fresh = fixed[0].cell_comparison(value["fake_features"])
            for key in before:self.assertTrue(torch.equal(before[key],fresh[key]),key)
            records.append(dict(step=step,flags=int(value["flags"].sum()),current_moves=len(old[3]),
                proposed_moves=len(fixed[3]),flagged_deaths=int(fixed[1][fixed[3]].sum()),
                supported_deaths=int((~fixed[1][fixed[3]]).sum()),unique_parents=len(torch.unique(fixed[4])),
                original_guard_passed=fixed[8]["guard_passed"],isolation_moves=len(fixed[6]),
                group_target=fixed[0]._group_counts(fixed[5]["target_counts"]).tolist(),
                planned_supported=fixed[0]._group_counts(fixed[5]["planned_supported_counts"]).tolist()))

    def test_original_nominal_and_rare_hole_inputs_remain_exact(self):
        for scenario,value in original["scenarios"].items():
            old = plan(current,value)
            fixed = plan(proposed,value)
            self.check_contract(*fixed)
            for index in (3,4,6,7,9):self.assertTrue(torch.equal(old[index],fixed[index]),(scenario,index))
            for key in old[5]:
                if isinstance(old[5][key],torch.Tensor):self.assertTrue(torch.equal(old[5][key],fixed[5][key]),key)
                else:self.assertEqual(old[5][key],fixed[5][key])
            self.assertEqual(len(fixed[6]),46)

    def test_no_eligible_parent_means_no_action(self):
        value = torch.load(HERE / "snapshot-1000.pt",map_location="cpu",weights_only=False)
        fixed = plan(proposed,value,pvalues=torch.zeros_like(value["pvalues"]))
        self.check_contract(*fixed)
        self.assertEqual(len(fixed[3]),0);self.assertEqual(len(fixed[6]),0)

    def test_guard_boundary_and_rare_survivors(self):
        value = deepcopy(original["scenarios"]["rare_hole"])
        base_flags = value["flags"].clone()
        snap = snapshot(proposed,value["snapshot"])
        ids,_ = snap.assign(value["q"])
        groups = snap._mass_topology()
        clean = snap._group_counts(torch.bincount(ids[~base_flags],minlength=snap.cells))
        rare_group = int(clean.argmin())
        legitimate_rare = (~base_flags)&(groups[ids]==rare_group)
        self.assertGreater(int(legitimate_rare.sum()),0)
        extra = (~base_flags & ~legitimate_rare).nonzero().flatten()
        for n in (51,52):
            flags = base_flags.clone(); flags[extra[:n-int(base_flags.sum())]] = True
            fixed = plan(proposed,value,flags=flags)
            self.check_contract(*fixed)
            self.assertEqual(fixed[5]["isolation_guard_rejects"],n>51)
            self.assertEqual(fixed[8]["guard_passed"],n<=51)
            self.assertFalse(bool(legitimate_rare[fixed[3]].any()))
            self.assertEqual(int((groups[ids[fixed[4]]]==rare_group).sum()),0)


result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CountRecovery))
receipt = dict(status="PASS" if result.wasSuccessful() else "FAIL",tests=result.testsRun,
               failures=len(result.failures),errors=len(result.errors),records=records,
               cuda_initialized=torch.cuda.is_initialized(),scope="CPU fixed-input action/ledger contracts; not CUDA quality",
               sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
                    ROOT/"pkg-CB64-RA2/particlegan/feature_cells.py",HERE/"pkg-count-recovery/particlegan/feature_cells.py"]})
(HERE/"recovery-regressions.json").write_text(json.dumps(receipt,indent=2)+"\n")
assert not receipt["cuda_initialized"]
raise SystemExit(0 if result.wasSuccessful() else 1)

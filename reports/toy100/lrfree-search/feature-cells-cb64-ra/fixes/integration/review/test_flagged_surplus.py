"""Regression for the exact frozen GPU-contract fixture and stream cursor."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import unittest

parser = argparse.ArgumentParser()
parser.add_argument("--package-root",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args = parser.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
sys.dont_write_bytecode = True
sys.path.insert(0,str(args.package_root.resolve()))
import torch
from particlegan.feature_cells import FeatureCellSnapshot
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
ROOT = Path(__file__).resolve().parent.parents[1]
data = torch.load(ROOT / "stability/mass-gpu-inputs.pt",map_location="cpu",weights_only=False)


def plan(scenario):
    value = data["scenarios"][scenario]
    snap = FeatureCellSnapshot.__new__(FeatureCellSnapshot)
    for name,item in copy.deepcopy(value["snapshot"]).items():setattr(snap,name,item)
    comparison = copy.deepcopy(value["comparison"])
    generator = torch.Generator().manual_seed(data["seed"])
    child,parent,ordinary = snap.ordinary_transport(value["q"],value["flags"],comparison,
                                   generator=generator,pvalues=value["pvalues"])
    iso_child,iso_parent,detail = snap.select_parents(value["q"],value["flags"],ordinary_children=child,
                                   ordinary_parents=parent,generator=generator,pvalues=value["pvalues"])
    return value,snap,child,parent,ordinary,iso_child,iso_parent,detail,comparison


class FlaggedSurplus(unittest.TestCase):
    def test_exact_original_frozen_input_and_fresh_stream(self):
        for scenario in ("nominal","rare_hole"):
            value,snap,child,parent,ordinary,iso_child,iso_parent,detail,comparison = plan(scenario)
            self.assertEqual(ordinary["between_group_moves"],0,scenario)
            self.assertEqual(len(iso_child),int(value["flags"].sum()),scenario)
            self.assertFalse(bool(torch.isin(parent,child).any()))
            self.assertFalse(bool(torch.isin(iso_parent,torch.cat((child,parent))).any()))
            self.assertEqual(len(torch.unique(torch.cat((parent,iso_parent)))),len(parent)+len(iso_parent))
            ids,_ = snap.assign(value["q"])
            supported = torch.bincount(ids[~value["flags"]],minlength=snap.cells)
            self.assertTrue(torch.equal(ordinary["clean_counts"],supported))
            kept = supported-torch.bincount(ids[child],minlength=snap.cells)+torch.bincount(ids[parent],minlength=snap.cells)
            self.assertTrue(torch.equal(detail["kept_counts"],kept))
            self.assertTrue(torch.equal(snap._group_counts(kept)+snap._group_counts(
                torch.bincount(ids[iso_parent],minlength=snap.cells)),snap._group_counts(ordinary["target_counts"])))
            for name,before in value["comparison"].items():
                if isinstance(before,torch.Tensor):
                    self.assertTrue(torch.equal(before,comparison[name]))

    def test_nonflagged_ineligible_survivors_reserve_their_mass(self):
        value = copy.deepcopy(data["scenarios"]["rare_hole"])
        # The frozen fixture includes legitimate nonflagged p<=Q rows.
        # Preserve all of them in mass accounting; do not equate parent
        # eligibility with whether a survivor reserves reference mass.
        self.assertGreater(int((~value["flags"]&(value["pvalues"]<=.05)).sum()),0)
        snap = FeatureCellSnapshot.__new__(FeatureCellSnapshot)
        for name,item in copy.deepcopy(value["snapshot"]).items():setattr(snap,name,item)
        child,parent,ordinary = snap.ordinary_transport(value["q"],value["flags"],value["comparison"],
            generator=torch.Generator().manual_seed(data["seed"]),pvalues=value["pvalues"])
        ids,_ = snap.assign(value["q"])
        counts = torch.bincount(ids[~value["flags"]],minlength=snap.cells)
        self.assertTrue(torch.equal(ordinary["clean_counts"],counts))
        self.assertEqual(ordinary["between_group_moves"],0)
        self.assertTrue(bool((value["pvalues"][parent]>.05).all()))


if __name__=="__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FlaggedSurplus))
    receipt = dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                   success=result.wasSuccessful(),package_root=str(args.package_root.resolve()),
                   scope="CPU regression on exact frozen GPU-contract inputs and fresh seed90229 stream")
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)
    raise SystemExit(0 if receipt["success"] else 1)

"""CPU-only one-seed saved-fixture mass/parent mechanism reproduction.

Oracle labels are used only after controller decisions. This is an exact-copy
diagnostic, not a CUDA quality screen or a new statistical qualification.
"""
import hashlib
from copy import deepcopy
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parent
OLD = Path("/ml2/hypergan/gan-attempts/feature-cells-config-20260929")
FIXTURE = Path("/ml2/hypergan/gan-attempts/scaling-portability-20260929/scaling_a/shared_toy.py")
GPU_INPUTS = {}


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def package(name, path):
    module = ModuleType(name)
    module.__path__ = [str(path / "particlegan")]
    sys.modules[name] = module
    return importlib.import_module(name + ".feature_cells"), importlib.import_module(name + ".recipes").Recipe


def scalar(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: scalar(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [scalar(item) for item in value]
    return value


@torch.no_grad()
def measure(name, module, recipe, scenario, cost, evaluator):
    shared = SimpleNamespace(cost=cost, config=json.loads((OLD / "configs/overrides-CB64-RA.json").read_text()),
                             cb_recipe=recipe, cb=module)
    ev = evaluator.Evaluator("cost", 1024, scenario, 8, "frozen_initialization", shared)
    trainer, bd = evaluator.make_trainer(ev, "cb64_ra", shared)
    q = bd._features(trainer, ev.G(ev.z))
    real = bd._features(trainer, ev.real_raw)
    snap = module.FeatureCellSnapshot.fit(real, generator=bd.stream)
    flags, pvalues, _ = snap.support(q)
    snap.cache_queries(q)
    pick = torch.randint(1024, (1024,), generator=bd.stream)
    fake = bd._capture_generated(trainer, ev.z[pick], sigma=.029, jitter=True)
    comparison = snap.cell_comparison(fake)
    if name == "baseline":
        GPU_INPUTS[scenario] = dict(snapshot=deepcopy(vars(snap)),q=q.clone(),flags=flags.clone(),
            pvalues=pvalues.clone(),comparison=deepcopy(comparison),z=ev.z.clone(),
            original_labels=torch.from_numpy(ev.oracle["original_labels"]).clone(),
            real_labels=torch.from_numpy(ev.oracle["real_labels"]).clone(),
            planted=torch.from_numpy(ev.oracle["planted"]).clone())
    ordinary_child, ordinary_parent, ordinary = snap.ordinary_transport(q, flags, comparison,
                                                generator=bd.stream, pvalues=pvalues)
    kwargs = dict(ordinary_children=ordinary_child, generator=bd.stream, pvalues=pvalues)
    if "ordinary_parents" in inspect.signature(snap.select_parents).parameters:
        kwargs["ordinary_parents"] = ordinary_parent
    child, parent, detail = snap.select_parents(q, flags, **kwargs)
    all_child, all_parent = torch.cat((ordinary_child, child)), torch.cat((ordinary_parent, parent))
    after = ev.z.clone()
    after[all_child] = ev.z[all_parent]
    counts_after = torch.bincount(snap.query_cell_ids, minlength=snap.cells)
    counts_after -= torch.bincount(snap.query_cell_ids[all_child], minlength=snap.cells)
    counts_after += torch.bincount(snap.query_cell_ids[all_parent], minlength=snap.cells)
    target = snap.reference_counts + snap.real_calibration_counts
    target = target.double() * (len(q) / int(target.sum()))
    before_l1 = float((snap.query_counts - target).abs().sum())
    after_l1 = float((counts_after - target).abs().sum())
    return dict(backend=name, scenario=scenario, seed=cost.SEED, population=1024,
                detector=ev.detector(flags), ordinary_moves=len(ordinary_child), isolation_moves=len(child),
                parents=len(all_parent), unique_parents=len(torch.unique(all_parent)),
                largest_parent_reuse=int(torch.bincount(all_parent, minlength=1024).max()),
                isolation_parents=ev.parents(ev.z, child, parent, detail), before=ev.evaluate(ev.z),
                after_exact_copy=ev.evaluate(after, all_child),
                clean_cell_count_l1_before=before_l1, clean_cell_count_l1_after=after_l1,
                ordinary=ordinary,
                isolation={key: value for key, value in detail.items() if key not in
                           ("candidate_ids", "candidate_mask", "anchor_reference_rows", "anchor_cell_ids", "parent_cell_ids")})


def main():
    cost = load_file("stability_cost_fixture", FIXTURE)
    evaluator = load_file("stability_old_evaluator", OLD / "geometry/run_validation.py")
    modules = {"baseline": package("stability_baseline", OLD / "pkg-CB64-RA"),
               "stability": package("stability_candidate", ROOT / "pkg")}
    records = []
    for scenario in ("nominal", "rare_hole"):
        for name, (module, recipe) in modules.items():
            row = measure(name, module, recipe, scenario, cost, evaluator)
            records.append(row)
            print(json.dumps(scalar({key: row[key] for key in ("backend", "scenario", "ordinary_moves", "isolation_moves", "unique_parents", "largest_parent_reuse", "isolation_parents", "after_exact_copy", "clean_cell_count_l1_before", "clean_cell_count_l1_after")})), flush=True)
    sources = [Path(__file__), FIXTURE, OLD / "geometry/run_validation.py",
               OLD / "pkg-CB64-RA/particlegan/feature_cells.py", ROOT / "pkg/particlegan/feature_cells.py"]
    result = dict(scope="CPU fixture mechanism, exact copies without jitter; no quality verdict",
                  sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, records=records)
    (ROOT / "mass-diagnostic.json").write_text(json.dumps(scalar(result), indent=2) + "\n")
    torch.save(dict(scope="frozen CPU partition/score inputs for root-only GPU mass planning check",
                    seed=cost.SEED,scenarios=GPU_INPUTS),ROOT / "mass-gpu-inputs.pt")


if __name__ == "__main__":
    main()

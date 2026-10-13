"""Compare supported update directions on certified saved beta1=0 gradients.

No modules, optimizers, updates, sampling or gate recomputation are executed.
These are conditional endpoint directions, not counterfactual trained outcomes.
"""
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
PARENT = "r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    denominator = float(a.norm() * b.norm())
    return None if not denominator else float((a @ b) / denominator)


def norm(a):
    return float(a.double().norm())


def comparison(gradient, direction):
    c = cosine(gradient, direction)
    return {"l2": norm(direction), "raw_gradient_cosine": c,
            "raw_gradient_angle_degrees": None if c is None else math.degrees(math.acos(max(-1., min(1., c)))),
            "orthogonal_to_raw_gradient_squared_fraction": None if c is None else max(0., 1. - c * c)}


def directions(saved, polar):
    groups = []
    if "trainer" in saved:
        optimizers = list(zip(("generator", "critic"), saved["trainer"]["optimizers"], strict=True))
    else:
        optimizers = [("direct_generator" if role == "generator" else "critic", optimizer)
                      for role, bundle in saved["optimizers"].items()
                      for optimizer in (bundle if isinstance(bundle, list) else [bundle])]
    for player, optimizer in optimizers:
        for index, group in enumerate(optimizer["param_groups"]):
            role = "prior" if player == "generator" and index == 1 else player
            assert group["betas"][0] == 0. and not group["amsgrad"]
            rate, eps, beta2 = group["lr"], group["eps"], group["betas"][1]
            rows, gradients, natives, adas, duals = [], [], [], [], []
            for parameter in group["params"]:
                state = optimizer["state"][parameter]
                # With beta1=0 the stored exp_avg is exactly the last parameter gradient.
                gradient = state["exp_avg"]
                second, step = state["exp_avg_sq"], int(state["step"])
                native = rate * gradient / ((second / (1. - beta2 ** step)).sqrt() + eps)
                ada = gradient / (gradient.norm() + eps) * native.norm()
                if role == "prior":
                    dual = rate * gradient / (gradient.norm(dim=1, keepdim=True) + eps)
                elif gradient.ndim == 2:
                    factor = math.sqrt(max(1., gradient.shape[0] / gradient.shape[1]))
                    dual = (torch.zeros_like(gradient) if bool(gradient.norm() < eps)
                            else rate * factor * polar(gradient))
                else:
                    dual = rate * gradient / (gradient.norm() + eps)
                row = {"parameter_id": parameter, "shape": list(gradient.shape),
                       "step": step, "raw_gradient_l2": norm(gradient),
                       "native_adam": comparison(gradient, native),
                       "ada_nsgda": comparison(gradient, ada),
                       "dualnorm": comparison(gradient, dual),
                       "ada_vs_native_norm_ratio": None if not norm(native) else norm(ada) / norm(native)}
                if role == "prior":
                    active = gradient.norm(dim=1) > 0
                    row["rows_with_nonzero_final_gradient"] = int(active.sum())
                    cosines = torch.tensor([cosine(a, b) for a, b in zip(gradient[active], native[active], strict=True)], dtype=torch.float64)
                    row["rowwise_native_gradient_cosine"] = {
                        "count": len(cosines), "minimum": float(cosines.min()),
                        "median": float(cosines.median()), "p90": float(cosines.quantile(.9)),
                        "maximum": float(cosines.max())}
                rows.append(row)
                gradients.append(gradient.flatten()); natives.append(native.flatten())
                adas.append(ada.flatten()); duals.append(dual.flatten())
            raw = torch.cat(gradients)
            groups.append({"role": role, "lr": rate, "betas": list(group["betas"]), "eps": eps,
                           "raw_gradient_l2": norm(raw),
                           "native_adam": comparison(raw, torch.cat(natives)),
                           "ada_nsgda": comparison(raw, torch.cat(adas)),
                           "dualnorm": comparison(raw, torch.cat(duals)), "parameters": rows})
    return groups


def main():
    helper = load(ROOT / "reports/forge/recipe-prior-refactor/workflow.py", "normalization_helpers")
    publisher = helper.load_publisher(ROOT)
    renderer = publisher._saved_renderer(ROOT)
    polar = load(ROOT / "particlegan/optim/dualnorm.py", "particlegan.optim.normalization_saved").polar_factor
    publication_path = ROOT / "reports/forge/r1r2-tier1-search/publication/results.json"
    publication = json.loads(publication_path.read_text())
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    results = []
    with renderer.forbid_live_execution(), torch.no_grad():
        for task in ("ring16_acquisition", "gaussian1d_smoke", "two_pole"):
            compact = next(r for r in publication["certified_results"] if r["candidate_id"] == PARENT and r["task_id"] == task)
            attempt = publisher.certified_attempt(ROOT, compact["attempt_id"])
            helper.saved_grading(attempt)
            saved, proof = publisher.checkpoint(attempt["result"]["task_results"][0])
            for path in ("particlegan/optim/dualnorm.py", "particlegan/recipes.py"):
                assert sha(ROOT / path) == attempt["request"]["source"]["files"][path]
            results.append({"task_id": task, "attempt_id": compact["attempt_id"],
                            "candidate_id": PARENT, "candidate_revision": compact["candidate_revision"],
                            "source_digest": compact["source_digest"], "runtime_cohort": {"runtime": attempt["request"]["runtime"], "compute_profiles": attempt["request"]["compute_profiles"]},
                            "checkpoint": proof, "recorded_gate_status": compact["status"],
                            "optimizer_groups": directions(saved, polar)})
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
    prior_art = []
    for family in ("ada-nsgda", "dualnorm-zero"):
        path = ROOT / f"reports/forge/configuration-search/bcap-optim-{family}-tier1-v1.json"
        report = json.loads(path.read_text())
        prior_art.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                          "source_digest": report["source_digest"], "scope": "different BCAP formulation and original task bindings; no R1/R2 credit or repeat",
                          "trials": [{"candidate_id": t["candidate_id"], "candidate_revision": t["candidate_revision"],
                                      "recipe_overrides": t["recipe_overrides"],
                                      "ordinary_statuses": {task: status for a in t["attempt_history"] for task, status in a["gate_statuses"].items() if task != "clockfree_audit_measurement_v1"}}
                                     for t in report["trials"]]})
    print(json.dumps({"schema_version": 1, "qualification_input": False,
                      "sampling_draws_added": 0, "optimizer_updates_added": 0,
                      "python_and_torch_rng_unchanged": True,
                      "publication_sha256": sha(publication_path),
                      "source_files": {p: sha(ROOT / p) for p in ("particlegan/optim/dualnorm.py", "particlegan/recipes.py")},
                      "results": results, "prior_art": prior_art,
                      "limits": "Conditional last-step direction comparison from certified native-Adam endpoint gradients and moments. No counterfactual trajectory, optimizer construction, training, draws or rescoring. CPU float32 SVD uses the actual polar helper; GPU rounding can differ. All directions are descent-direction magnitudes with sign suppressed equally. Native/Ada formulas approximate native float32 kernel rounding. Ada graft preserves tensor norm up to gradient-norm epsilon. Prior rownorm uses nonzero-gradient rows; native final sampled IDs are unavailable and no ownership receipt is synthesized."}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

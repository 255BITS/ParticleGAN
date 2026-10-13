"""Summarize saved Tier 1 evidence without training, sampling, or model construction.

Run with the registered project interpreter and redirect stdout to
``density-diagnosis.json``. Only source-bound archive artifacts are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch


CAMPAIGNS = {
    "r1r2-tier1-matched-baseline-v1": 1,
    "r1r2-tier1-constant-search-v1": 6,
    "r1r2-tier1-half-rate-b90-g01-v1": 1,
    "r1r2-tier1-half-rate-b99-g01-v1": 1,
    "r1r2-tier1-half-rate-b99-g03-v1": 1,
}
RING = "ring16_acquisition"
METRICS = (
    "step", "modes", "mass_tv", "hq", "component_covariance_error",
    "component_core_covariance_error", "component_min_eigen_ratio",
)
RECIPE_FIELDS = (
    "lr", "lr_schedule", "lr_floor", "network_lr_floor", "betas", "beta2_end",
    "reg_coeff", "reg_coeff_end", "prior_update", "prior_reg", "prior_l2",
)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def selected(metrics):
    return {key: metrics[key] for key in METRICS if key in metrics}


def satisfies(metrics, thresholds):
    operations = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
                  "==": lambda a, b: a == b}
    return all(key in metrics and metrics[key] is not None
               and math.isfinite(metrics[key]) and operations[op](metrics[key], bound)
               for key, op, bound in thresholds)


def geometry(attempt, request, endpoint, archive):
    """Functional saved MLP and first-order Jacobian; no module initialization."""
    spec = request["tasks"][RING]["execution"]["host_definition"]
    prior = request["tasks"][RING]["execution"]["prior"]
    assert (spec["layers"], spec["hidden"], spec["particles"], spec["z_dim"]) == (2, 64, 256, 4)
    assert prior == {"kind": "mog", "sigma": 0.1, "standardize": False}
    samples_path = attempt / "observed-samples.pt"
    state_path = attempt / "provenance/provenance-state.pt"
    saved_samples = torch.load(samples_path, map_location="cpu", weights_only=True)
    assert saved_samples[-1]["step"] == endpoint["step"] == 1600
    samples = saved_samples[-1]["samples"].double()
    models = torch.load(state_path, map_location="cpu", weights_only=False)["trainer"]["models"]
    w = lambda index: models["G"][f"net.{index}.weight"].double()
    b = lambda index: models["G"][f"net.{index}.bias"].double()
    locations = models["prior"]["z"].double()
    first = locations @ w(0).T + b(0)
    second = torch.nn.functional.leaky_relu(first, 0.2) @ w(2).T + b(2)
    centers = torch.nn.functional.leaky_relu(second, 0.2) @ w(4).T + b(4)
    jacobian = torch.einsum(
        "oi,bi,ij,bj,jk->bok", w(4), torch.where(second >= 0, 1.0, 0.2),
        w(2), torch.where(first >= 0, 1.0, 0.2), w(0),
    )
    local_covariance = prior["sigma"] ** 2 * (jacobian @ jacobian.transpose(1, 2))
    means = torch.tensor(spec["means"], dtype=torch.float64)
    target_covariance = torch.tensor(spec["covariances"], dtype=torch.float64)
    variance = float(target_covariance[0, 0, 0])
    assert torch.allclose(target_covariance, variance * torch.eye(2, dtype=torch.float64).expand(16, 2, 2))
    assignments = torch.cdist(samples, means).argmin(1)
    center_assignments = torch.cdist(centers, means).argmin(1)
    per_component, between, within, errors = [], [], [], []
    for index in range(16):
        members = samples[assignments == index]
        rows = centers[center_assignments == index]
        assert len(members) >= 10 and len(rows) >= 2
        centered = members - members.mean(0)
        row_centered = rows - rows.mean(0)
        covariance = centered.T @ centered / len(members)
        row_covariance = row_centered.T @ row_centered / len(rows)
        radial = means[index] / means[index].norm()
        tangent = torch.stack([-radial[1], radial[0]])
        per_component.append({
            "component": index, "sample_count": len(members), "prior_center_count": len(rows),
            "radial_variance_ratio": float(radial @ covariance @ radial / variance),
            "tangent_variance_ratio": float(tangent @ covariance @ tangent / variance),
        })
        errors.append(float((covariance - target_covariance[index]).norm() / target_covariance[index].norm()))
        between.append(float(torch.trace(row_covariance) / variance))
        within.append(float(torch.trace(local_covariance[center_assignments == index].mean(0)) / variance))
    reconstructed_error = sum(errors) / 16
    assert math.isclose(reconstructed_error, endpoint["component_covariance_error"], rel_tol=1e-5, abs_tol=1e-5)
    radial = [row["radial_variance_ratio"] for row in per_component]
    tangent = [row["tangent_variance_ratio"] for row in per_component]
    distance = (centers - means[center_assignments]).norm(dim=1)
    return {
        "attempt_archive_path": str(attempt.relative_to(archive)),
        "artifacts_sha256": {str(path.relative_to(attempt)): sha256(path) for path in (samples_path, state_path)},
        "method": "Nearest saved target center; population component covariance; functional two-hidden-layer LeakyReLU(0.2) G and Jacobian at saved learned prior locations.",
        "target_component_variance": variance,
        "target_total_trace_in_variance_units": 2.0,
        "prior_rows": len(locations),
        "prior_centers_outside_3sigma": int((distance > 3 * math.sqrt(variance)).sum()),
        "prior_centers_outside_4sigma": int((distance > 4 * math.sqrt(variance)).sum()),
        "radial_variance_ratio": {"min": min(radial), "mean": sum(radial) / 16, "max": max(radial)},
        "tangent_variance_ratio": {"min": min(tangent), "mean": sum(tangent) / 16, "max": max(tangent)},
        "unweighted_mean_between_prior_center_trace_ratio": sum(between) / 16,
        "unweighted_mean_local_linear_noise_trace_ratio": sum(within) / 16,
        "reconstructed_full_component_covariance_error": reconstructed_error,
        "components": per_component,
        "limitations": "The Jacobian estimate is first order at row centers, not an exact sample variance decomposition. Noise can cross activation boundaries or nearest-mode assignments; sampled row identities are unavailable. This diagnosis cannot attribute the defect to R1/R2 alone.",
    }


def diagnose(archive):
    rows, geometry_result = [], None
    for campaign, expected_recipes in CAMPAIGNS.items():
        candidates = {}
        for graded_path in sorted((archive / "queue" / campaign).glob("*/graded-result.json")):
            attempt = graded_path.parent
            request = read_json(attempt / "request.json")["request"]
            candidate = request["candidate"]["id"]
            record = candidates.setdefault(candidate, {"tasks": {}, "ring": None})
            grades = read_json(graded_path)
            for task, grade in grades["grades"].items():
                assert task not in record["tasks"], "Unexpected retry or duplicate task evidence"
                record["tasks"][task] = grade["gate_status"]
            if RING not in grades["grades"]:
                continue
            grade = grades["grades"][RING]
            raw = read_json(attempt / "raw-result.json")
            observations = raw["evidence"]["observations"]
            thresholds = request["tasks"][RING]["evaluation"]["thresholds"]
            covariance_bound = next(bound for key, op, bound in thresholds
                                    if key == "component_covariance_error" and op == "<=")
            passing = [satisfies(observation, thresholds) for observation in observations]
            suffix = next((index for index, passed in enumerate(reversed(passing)) if not passed), len(passing))
            convergence = grade["evaluator_result"]["convergence"]
            assert len(observations) == convergence["observations"] == 96
            assert sum(passing) == convergence["passing_observations"]
            assert suffix == convergence["passing_suffix"]
            assert observations[-1]["step"] == grade["metrics"]["step"] == 1600
            recipe = request["candidate"]["resolved_recipe"]
            record["ring"] = {
                "campaign": campaign, "candidate_id": candidate,
                "candidate_revision": request["candidate_revision"],
                "recipe_controls": {key: recipe[key] for key in RECIPE_FIELDS},
                "resolved_recipe_canonical_sha256": canonical_sha256(recipe),
                "runtime_canonical_sha256": canonical_sha256(request["runtime"]),
                "source_origin_commit": request["source"]["origin_commit"],
                "source_digest": request["source"]["digest"],
                "attempt_archive_path": str(attempt.relative_to(archive)),
                "artifacts_sha256": {name: sha256(attempt / name) for name in ("request.json", "raw-result.json", "graded-result.json")},
                "thresholds": thresholds, "gate_status": grade["gate_status"],
                "observations": len(observations),
                "covariance_bound_passing_observations": sum(observation["component_covariance_error"] <= covariance_bound for observation in observations),
                "full_passing_observations": sum(passing), "full_terminal_passing_suffix": suffix,
                "minimum_stable_checks": convergence["minimum_stable_checks"],
                "best_covariance_observation": selected(min(observations, key=lambda observation: observation["component_covariance_error"])),
                "endpoint": selected(grade["metrics"]),
            }
            if campaign == "r1r2-tier1-half-rate-b99-g01-v1":
                geometry_result = geometry(attempt, request, grade["metrics"], archive)
        assert len(candidates) == expected_recipes
        for candidate in sorted(candidates):
            record = candidates[candidate]
            assert record["ring"] is not None and len(record["tasks"]) == 7, "Full round readout must complete first"
            record["ring"]["whole_recipe_task_gates"] = record["tasks"]
            rows.append(record["ring"])
    assert len(rows) == 10 and geometry_result is not None
    return {
        "schema_version": 1,
        "evidence_type": "saved_observation_diagnosis_not_new_experiment",
        "archive_root": str(archive),
        "scope": "Ten unique global recipes: fresh scheduled baseline, six round1 constants, three round2 LR-halved constants. Cached baseline reuse adds no recipe or new evidence.",
        "recipe_count": len(rows), "new_optimizer_updates": 0, "new_sampling_draws": 0,
        "canonical_hash_method": "sha256 of UTF-8 json.dumps(value, sort_keys=True, separators=(',', ':'))",
        "rings": rows, "half_rate_b99_g01_endpoint_geometry": geometry_result,
        "conclusion": "No recipe passes sustained Ring16. No saved observation among the ten recipes meets the full covariance bound. LR halving improves the beta2=.99/gamma=.1 density endpoint and word acquisition, but retains tangential stretch with radial compression and loses sustained pole transport. Stop this bounded search without adopting a new default.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    result = diagnose(args.archive)
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    result["diagnosis_rng_states_unchanged"] = True
    result["reproduction_source_sha256"] = sha256(Path(__file__))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

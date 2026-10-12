"""Read saved five-track Ring16 evidence without training or sampling draws.

Run with the registered project interpreter and redirect stdout to
``density-readout.json``. Every required gate metric is reconstructed from saved
samples; the reference-target/SW1 branch of the original scorer is not invoked.
Missing component geometry stays explicitly unavailable and never fills a gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
RING = "ring16_acquisition"
METRICS = ("sample_count", "modes", "mass_tv", "hq",
           "component_covariance_error", "component_min_eigen_ratio")
CONTROLS = ("lr", "d_lr_mult", "prior_lr_mult", "betas", "prior_betas", "d_betas",
            "amsgrad", "kinetic_transport_local_weight", "kinetic_transport_weight",
            "lr_schedule", "beta2_end", "reg_arm", "reg_coeff", "reg_coeff_end",
            "reg_every", "prior_update", "prior_reg", "prior_l2")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare(value, op, bound):
    return finite(value) and {"<=": lambda: value <= bound, ">=": lambda: value >= bound,
                             "==": lambda: value == bound}[op]()


def passing(metrics, thresholds):
    return all(compare(metrics.get(key), op, bound) for key, op, bound in thresholds)


def selected(metrics):
    return {key: metrics.get(key) for key in ("step",) + METRICS}


@torch.no_grad()
def gate_metrics(points, spec):
    """Exact float32 gate-only equations from the pinned public toy scorers.

    Like vector_tasks.score_samples, unresolved (<10 sample) components receive
    covariance error1 and minimum eigen ratio0. Nonfinite panels receive None
    for every gate metric. No deterministic reference or projection draws occur.
    """
    if not torch.isfinite(points).all():
        return {key: None for key in METRICS}
    assert points.device.type == "cpu" and points.dtype == torch.float32
    means = points.new_tensor(spec["means"])
    covariances = points.new_tensor(spec["covariances"])
    target_mass = points.new_tensor(spec["masses"])
    labels = torch.cdist(points, means).argmin(1)
    counts = torch.bincount(labels, minlength=len(means))
    delta = points - means[labels]
    mahal = torch.einsum("ni,nij,nj->n", delta, torch.linalg.inv(covariances)[labels], delta)
    hq_counts = torch.bincount(labels[mahal <= 9], minlength=len(means))
    hq_mass_ratio = hq_counts / len(points) / target_mass
    errors, eigen_ratios = [], []
    for index in range(len(means)):
        members = points[labels == index]
        if len(members) < 10:
            errors.append(1.0)
            eigen_ratios.append(0.0)
            continue
        centered = members - members.mean(0)
        empirical = centered.T @ centered / len(members)
        inverse = torch.linalg.inv(torch.linalg.cholesky(covariances[index]))
        errors.append(float((empirical - covariances[index]).norm() / covariances[index].norm()))
        eigen_ratios.append(float(torch.linalg.eigvalsh(inverse @ empirical @ inverse.T).min()))
    return {"sample_count": len(points), "modes": int((hq_mass_ratio >= .25).sum()),
            "mass_tv": float((counts / len(points) - target_mass).abs().sum() / 2),
            "hq": float((mahal <= 9).float().mean()),
            "component_covariance_error": sum(errors) / len(errors),
            "component_min_eigen_ratio": min(eigen_ratios)}


def aggregate(values):
    return ({"count": len(values), "min": min(values), "mean": sum(values) / len(values),
             "max": max(values)} if values else
            {"count": 0, "min": None, "mean": None, "max": None})


@torch.no_grad()
def geometry(points, models, spec, prior):
    """Functional saved G and Jacobian; no randomly initialized modules."""
    if not torch.isfinite(points).all() or any(
            not torch.isfinite(value).all() for group in (models["G"], models["prior"])
            for value in group.values() if isinstance(value, torch.Tensor)):
        return {"status": "UNAVAILABLE", "reason": "Saved sample/model/prior state is nonfinite."}
    w = lambda index: models["G"][f"net.{index}.weight"].double()
    b = lambda index: models["G"][f"net.{index}.bias"].double()
    locations = models["prior"]["z"].double()
    first = locations @ w(0).T + b(0)
    second = torch.nn.functional.leaky_relu(first, .2) @ w(2).T + b(2)
    centers = torch.nn.functional.leaky_relu(second, .2) @ w(4).T + b(4)
    jacobian = torch.einsum("oi,bi,ij,bj,jk->bok", w(4), torch.where(second >= 0, 1., .2),
                            w(2), torch.where(first >= 0, 1., .2), w(0))
    local_covariance = prior["sigma"] ** 2 * (jacobian @ jacobian.transpose(1, 2))
    # Preserve the actual float32 scorer's sample assignments; diagnostic row
    # centers come from the independent float64 functional saved-model forward.
    labels = torch.cdist(points, points.new_tensor(spec["means"])).argmin(1)
    points = points.double()
    means, covariance = torch.tensor(spec["means"], dtype=torch.float64), torch.tensor(spec["covariances"], dtype=torch.float64)
    variance = float(covariance[0, 0, 0])
    assert torch.allclose(covariance, variance * torch.eye(2, dtype=torch.float64).expand(16, 2, 2))
    row_labels = torch.cdist(centers, means).argmin(1)
    rows, radial, tangent, between, local, errors = [], [], [], [], [], []
    for index in range(16):
        members, output_centers = points[labels == index], centers[row_labels == index]
        u = means[index] / means[index].norm()
        v = torch.stack([-u[1], u[0]])
        row = {"component": index, "sample_count": len(members), "prior_center_count": len(output_centers),
               "shape_status": "AVAILABLE" if len(members) >= 10 else "UNAVAILABLE",
               "center_geometry_status": "AVAILABLE" if len(output_centers) >= 2 else "UNAVAILABLE",
               "radial_variance_ratio": None, "tangent_variance_ratio": None,
               "between_center_trace_ratio": None, "local_linear_noise_trace_ratio": None}
        if len(members) >= 10:
            centered = members - members.mean(0)
            empirical = centered.T @ centered / len(members)
            row["radial_variance_ratio"] = float(u @ empirical @ u / variance)
            row["tangent_variance_ratio"] = float(v @ empirical @ v / variance)
            radial.append(row["radial_variance_ratio"])
            tangent.append(row["tangent_variance_ratio"])
            errors.append(float((empirical - covariance[index]).norm() / covariance[index].norm()))
        else:
            row["shape_unavailable_reason"] = "Fewer than10 assigned saved samples; original scorer uses error1/eigen0."
            errors.append(1.0)
        if len(output_centers) >= 2:
            centered = output_centers - output_centers.mean(0)
            center_covariance = centered.T @ centered / len(output_centers)
            row["between_center_trace_ratio"] = float(torch.trace(center_covariance) / variance)
            row["local_linear_noise_trace_ratio"] = float(torch.trace(local_covariance[row_labels == index].mean(0)) / variance)
            between.append(row["between_center_trace_ratio"])
            local.append(row["local_linear_noise_trace_ratio"])
        else:
            row["center_geometry_unavailable_reason"] = "Fewer than2 mapped prior means; report no component decomposition."
        rows.append(row)
    distance = (centers - means[row_labels]).norm(dim=1)
    return {"status": "AVAILABLE" if len(radial) == len(between) == 16 else "PARTIAL",
            "method": "Original float32 scorer's sample assignments; float64 population moments and mapped-center assignments; functional saved two-hidden-layer LeakyReLU(0.2) forward/Jacobian.",
            "target_component_variance": variance, "target_trace_in_variance_units": 2.,
            "prior_rows": len(locations), "prior_centers_outside_3sigma": int((distance > 3 * math.sqrt(variance)).sum()),
            "prior_centers_outside_4sigma": int((distance > 4 * math.sqrt(variance)).sum()),
            "radial_variance_ratio": aggregate(radial), "tangent_variance_ratio": aggregate(tangent),
            "unweighted_between_center_trace_ratio": aggregate(between),
            "unweighted_local_linear_noise_trace_ratio": aggregate(local),
            "reconstructed_float64_full_component_covariance_error": sum(errors) / 16,
            "components": rows,
            "limitations": "Jacobians are first order at row centers, not an exact sample variance partition. Noise can cross activations or mode assignments. Unavailable component geometry is excluded with its denominator declared, never substituted into a gate. No causal attribution to prior motion, R1/R2 alone or a unique repair follows."}


def stable_windows(flags, observations):
    windows, start = [], None
    for index, value in enumerate(flags + [False]):
        if value and start is None:
            start = index
        if not value and start is not None:
            windows.append({"first_step": observations[start]["step"],
                            "last_step": observations[index - 1]["step"], "observations": index - start})
            start = None
    return windows


def diagnose(archive):
    manifest = read(ROOT / "reports/forge/r1r2-five-track-repair/manifest.json")
    registration = ROOT / "reports/forge/r1r2-five-track-repair/registration.json"
    result = []
    for item in manifest["candidates"]:
        if item["role"] == manifest["baseline_role"]:
            continue
        attempts = []
        for grade_path in sorted((archive / "queue" / item["study"]).glob("*/graded-result.json")):
            grades = read(grade_path)["grades"]
            if RING in grades:
                attempts.append((grade_path.parent, grades[RING]))
        if len(attempts) != 1:
            raise ValueError(f"Need exactly one completed saved Ring16 attempt for {item['role']}; found {len(attempts)}")
        attempt, grade = attempts[0]
        request = read(attempt / "request.json")["request"]
        raw = read(attempt / "raw-result.json")
        task = request["tasks"][RING]
        spec, prior = task["execution"]["host_definition"], task["execution"]["prior"]
        assert (spec["layers"], spec["hidden"], spec["particles"], spec["z_dim"]) == (2, 64, 256, 4)
        assert prior == {"kind": "mog", "sigma": .1, "standardize": False}
        assert request["candidate"]["id"] == item["candidate"]
        assert request["source"]["origin_commit"] == "9ad31157b8327044f9364b5fc458659cced0c8f0"
        source_hashes = task["evaluation"]["sources"]
        assert all(sha(ROOT / path) == expected for path, expected in source_hashes.items())
        thresholds = task["evaluation"]["thresholds"]
        assert {key for key, _, _ in thresholds} == set(METRICS)
        saved = torch.load(attempt / "observed-samples.pt", map_location="cpu", weights_only=True)
        observations = raw["evidence"]["observations"]
        assert len(saved) == len(observations) == 96
        recreated, worst = [], {key: 0. for key in METRICS}
        for panel, original in zip(saved, observations, strict=True):
            assert panel["step"] == original["step"]
            metrics = {"step": panel["step"], **gate_metrics(panel["samples"], spec)}
            for key in METRICS:
                a, b = metrics[key], original[key]
                if a is None or b is None:
                    assert a is b
                else:
                    assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6), (item["role"], panel["step"], key, a, b)
                    worst[key] = max(worst[key], abs(a - b))
            assert passing(metrics, thresholds) == passing(original, thresholds)
            recreated.append(metrics)
        flags = [passing(row, thresholds) for row in recreated]
        windows = stable_windows(flags, recreated)
        suffix = next((index for index, value in enumerate(reversed(flags)) if not value), len(flags))
        convergence = grade["evaluator_result"]["convergence"]
        assert sum(flags) == convergence["passing_observations"] and suffix == convergence["passing_suffix"]
        assert recreated[-1]["step"] == grade["metrics"]["step"] == 1600
        required = convergence["minimum_stable_checks"]
        assert grade["gate_status"] == ("PASS" if suffix >= required else "FAIL")
        recipe = request["candidate"]["resolved_recipe"]
        state = torch.load(attempt / "provenance/provenance-state.pt", map_location="cpu", weights_only=False)
        shape = geometry(saved[-1]["samples"], state["trainer"]["models"], spec, prior)
        if shape["status"] != "UNAVAILABLE":
            assert math.isclose(shape["reconstructed_float64_full_component_covariance_error"], recreated[-1]["component_covariance_error"], rel_tol=1e-5, abs_tol=1e-5)
        covariance_bound = next(bound for key, op, bound in thresholds if key == "component_covariance_error" and op == "<=")
        finite_rows = [row for row in recreated if finite(row["component_covariance_error"])]
        endpoint = selected(recreated[-1])
        result.append({"role": item["role"], "study": item["study"], "candidate_id": item["candidate"],
                       "candidate_revision": request["candidate_revision"], "source_origin_commit": request["source"]["origin_commit"],
                       "source_digest": request["source"]["digest"], "runtime_canonical_sha256": canonical_sha(request["runtime"]),
                       "resolved_recipe_canonical_sha256": canonical_sha(recipe),
                       "recipe_controls": {key: recipe[key] for key in CONTROLS},
                       "task_evaluation_sources_sha256": source_hashes,
                       "attempt_archive_path": str(attempt.relative_to(archive)),
                       "artifacts_sha256": {name: sha(attempt / name) for name in ("request.json", "raw-result.json", "graded-result.json", "observed-samples.pt", "provenance/provenance-state.pt")},
                       "thresholds": thresholds, "gate_status": grade["gate_status"],
                       "recomputed_observations": len(recreated), "max_absolute_gate_metric_difference": worst,
                       "full_passing_observations": sum(flags), "full_terminal_passing_suffix": suffix,
                       "minimum_stable_checks": required, "full_passing_windows": windows,
                       "covariance_bound_passing_observations": sum(compare(row["component_covariance_error"], "<=", covariance_bound) for row in recreated),
                       "failures_by_bound": {key: sum(not compare(row[key], op, bound) for row in recreated) for key, op, bound in thresholds},
                       "endpoint_failed_bounds": [key for key, op, bound in thresholds if not compare(endpoint[key], op, bound)],
                       "endpoint": endpoint,
                       "best_covariance_observation": selected(min(finite_rows, key=lambda row: row["component_covariance_error"])) if finite_rows else None,
                       "milestones": [selected(row) for row in recreated if row["step"] in (400, 800, 1200, 1600)],
                       "last_five_observations": [selected(row) for row in recreated[-5:]],
                       "endpoint_geometry": shape})
    assert len(result) == 5 and len({row["source_digest"] for row in result}) == len({row["runtime_canonical_sha256"] for row in result}) == 1
    return {"schema_version": 1, "evidence_type": "saved_observation_diagnosis_not_new_experiment",
            "scope": "Five new global R1/R2 recipes; all96 saved Ring16 panels per recipe; endpoint saved-state geometry.",
            "archive_root": str(archive), "execution_commit": "9ad31157b8327044f9364b5fc458659cced0c8f0",
            "manifest_sha256": sha(ROOT / "reports/forge/r1r2-five-track-repair/manifest.json"),
            "registration_sha256": sha(registration), "recipe_count": len(result),
            "recomputed_saved_observations": sum(row["recomputed_observations"] for row in result),
            "new_optimizer_updates": 0, "new_sampling_draws": 0,
            "recomputed_metrics": list(METRICS),
            "reference_draws_avoided": "Only the pinned six gate equations are evaluated; target reference/global-moment/SW1 branches are excluded.",
            "rings": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("/mnt/ml7tb/ParticleGAN-forge/r1r2-five-track-repair-20261012"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    result = diagnose(args.archive)
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    result["diagnosis_rng_states_unchanged"] = True
    result["reproduction_source_sha256"] = sha(Path(__file__))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

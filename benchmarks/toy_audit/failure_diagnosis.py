"""Explain frozen toy failures from retained evidence, without training.

Historical gates and verdicts are immutable inputs. Saved-cloud decompositions,
finite-law bounds and terminal-state probes explain those inputs; none creates
a replacement qualification result. Bulk tensors remain in the artifact root.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = "toy-failure-diagnosis-v1"
NATIVE_BOUNDS = [
    ("modes", ">=", 100), ("precision", ">=", .97),
    ("min_hq_mode_mass", ">=", .005), ("mass_tv", "<=", .10),
    ("max_mode_mass", "<=", .02), ("min_cov_eig_ratio", ">=", .40),
    ("max_cov_eig_ratio", "<=", 1.70),
    ("min_radial_median_ratio", ">=", .65),
    ("max_radial_median_ratio", "<=", 1.40),
]
ACCURACY_BOUNDS = [
    ("mass_tv", "<=", .06), ("center_rms_sigma", "<=", .20),
    ("abs_cov_trace_bias", "<=", .10), ("radial_ks", "<=", .04),
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identity(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha(path)}


def check_receipt(summary, episode, expected_result=None, expected_capture=None):
    """Never silently diagnose changed files under a frozen receipt identity."""
    result_hash = sha(episode / "result.json")
    if result_hash != summary["result_sha256"] or (expected_result and result_hash != expected_result):
        raise ValueError("result artifact differs from frozen receipt")
    capture = episode / "observations.npz"
    if capture.exists():
        capture_hash = sha(capture)
        if capture_hash != summary["capture_sha256"] or (expected_capture and capture_hash != expected_capture):
            raise ValueError("capture artifact differs from frozen receipt")


def requirements(spec):
    thresholds = spec["thresholds"]
    if isinstance(thresholds, dict):
        return [("modes", ">=", thresholds["modes"]),
                ("hq", ">=", thresholds["hq_min"])]
    return thresholds


def gate_cells(metrics, bounds):
    cells = []
    for key, op, bound in bounds:
        value = metrics.get(key)
        valid = type(value) in (int, float) and math.isfinite(value)
        passed = valid and (value >= bound if op == ">=" else value <= bound)
        cells.append({"metric": key, "value": value, "op": op,
                      "bound": bound, "passed": bool(passed)})
    return cells


def passes(metrics, bounds):
    return all(cell["passed"] for cell in gate_cells(metrics, bounds))


def trajectory(rows, bounds, minimum=5):
    """Compact first/best/terminal receipt; a best sample never replaces final."""
    keys = [key for key, _, _ in bounds]

    def compact(row):
        return {"step": row["step"], **{key: row.get(key) for key in keys}}

    def deficit(row):
        cells = gate_cells(row, bounds)
        return sum(2. if type(c["value"]) not in (int, float)
                   else min(2., max(0., (c["bound"] - c["value"])
                                      if c["op"] == ">="
                                      else (c["value"] - c["bound"]))
                            / (abs(c["bound"]) or 1.)) for c in cells)

    ok = [passes(row, bounds) for row in rows]
    start = len(ok)
    while start and ok[start - 1]:
        start -= 1
    return {"first_observed": compact(rows[0]),
            "best_by_frozen_gate_shortfall": compact(min(rows, key=deficit)),
            "terminal": compact(rows[-1]),
            "passing_observations": sum(ok), "observations": len(rows),
            "passing_suffix": len(ok) - start, "minimum_stable_checks": minimum,
            "terminal_stability_met": len(ok) - start >= minimum,
            "first_passing_step": next((r["step"] for r, p in zip(rows, ok) if p), None),
            "last_failed_step": next((r["step"] for r, p in zip(reversed(rows), reversed(ok)) if not p), None),
            "best_checkpoint_is_not_qualification": True}


def covariance(points):
    if not len(points):
        return np.zeros((2, 2))
    centered = points - points.mean(0)
    return centered.T @ centered / len(points)


def noisy_gaussian_fit_witness(target_sigma, output_sigma):
    """A perfect noisy Gaussian fit need not pass the clean density gate."""
    if not math.isfinite(target_sigma) or target_sigma <= 0 or not math.isfinite(output_sigma) or output_sigma < 0:
        raise ValueError("finite positive target sigma and nonnegative noise required")
    feasible = output_sigma <= target_sigma
    ratio = 1. - output_sigma ** 2 / target_sigma ** 2 if feasible else None
    return {"target_sigma": target_sigma, "independent_output_sigma": output_sigma,
            "exact_gaussian_convolution_fit_possible": feasible,
            "exact_fit_clean_sigma": math.sqrt(max(0., target_sigma ** 2 - output_sigma ** 2)) if feasible else None,
            "exact_fit_clean_variance_over_target": ratio,
            "population_clean_covariance_gate_would_pass": ratio is not None and ratio >= .40,
            "scope": "Analytic independent-Gaussian convolution witness; not a fitted description or qualification of the observed generator. Apply to each mixture component with unchanged centers/masses."}


def shape(points, target):
    """Match the historical scorer's <10-draw sentinel, then decompose."""
    if len(points) < 10:
        return {"covariance_error": 1., "min_eigen_ratio": 0., "scorer_sentinel": True}
    empirical = covariance(points)
    inverse = np.linalg.inv(np.linalg.cholesky(target))
    return {"covariance_error": float(np.linalg.norm(empirical - target) / np.linalg.norm(target)),
            "min_eigen_ratio": float(np.linalg.eigvalsh(inverse @ empirical @ inverse.T).min()),
            "scorer_sentinel": False}


def vector_components(points, spec, step):
    """Attribute shape error to in-core shape, spills and between-group offsets."""
    points = np.asarray(points, dtype=np.float64)
    if spec.get("scale_end") is not None:
        fraction = min(1., step / (spec["steps"] * spec["scale_ramp_end"]))
        units = spec["scale_start"] + fraction * (spec["scale_end"] - spec["scale_start"])
    else:
        units = 1.
    means = np.asarray(spec["means"], dtype=np.float64) * units
    targets = np.asarray(spec["covariances"], dtype=np.float64) * units ** 2
    assignment = np.square(points[:, None] - means).sum(-1).argmin(1)
    result = []
    for k, target in enumerate(targets):
        members = points[assignment == k]
        delta = members - means[k]
        mahal = np.einsum("ni,ij,nj->n", delta, np.linalg.inv(target), delta)
        core, spill = members[mahal <= 16.], members[mahal > 16.]
        full_shape, core_shape = shape(members, target), shape(core, target)
        n, nc, ns = len(members), len(core), len(spill)
        core_cov, spill_cov = covariance(core), covariance(spill)
        fc, fs = (nc / n, ns / n) if n else (0., 0.)
        separation = (core.mean(0) - spill.mean(0)) if nc and ns else np.zeros(2)
        within_core = fc * core_cov
        within_spill = fs * spill_cov
        between = fc * fs * np.outer(separation, separation)
        total = covariance(members)
        trace = float(np.trace(total))
        variance_parts = {"core": float(np.trace(within_core)),
                          "spill": float(np.trace(within_spill)),
                          "between_core_and_spill": float(np.trace(between))}
        result.append({
            "component": k, "sample_count": n,
            "distinct_sample_coordinates": len(np.unique(members, axis=0)),
            "mass": n / len(points), "target_mass": spec["masses"][k],
            "declared_expected_particles": spec.get("particles", 0) * spec["masses"][k],
            "below_v5_particle_floor": spec.get("particles", 0) * spec["masses"][k] < 32,
            "hq_fraction_within_component": float(np.mean(mahal <= 9.)) if n else 0.,
            "spill_fraction_beyond_3sigma": float(np.mean(mahal > 9.)) if n else 1.,
            "core_sample_count_4sigma": nc, "far_spill_count_beyond_4sigma": ns,
            "full_shape": full_shape, "core_shape": core_shape,
            "population_variance_trace_parts": variance_parts,
            "spill_and_between_share_of_variance": ((variance_parts["spill"] + variance_parts["between_core_and_spill"]) / trace if trace else None),
            "covariance_decomposition_max_error": float(np.max(np.abs(total - within_core - within_spill - between))),
            "mean_mahalanobis_distance": float(np.sqrt(np.mean(mahal))) if n else None,
        })
    return result


def image_attribution(points, templates, thresholds):
    images = np.asarray(points, dtype=np.float64).reshape(len(points), -1)
    centers = np.asarray(templates, dtype=np.float64).reshape(len(templates), -1)
    distances = np.sqrt(np.square(images[:, None] - centers).mean(-1))
    assignment = distances.argmin(1)
    best = distances[np.arange(len(images)), assignment]
    good = best <= thresholds["quality_rmse"]
    n = len(images)
    counts = np.bincount(assignment, minlength=len(centers))
    hq_counts = np.bincount(assignment[good], minlength=len(centers))
    modes = int((hq_counts / n >= thresholds["min_mode_fraction"]).sum())
    squared_error = np.square(images - centers[assignment])
    discriminative = np.ptp(centers, axis=0) > 1e-12
    total_error = float(squared_error.sum())
    # Quantiles of distances are diagnostics, not relaxed gate thresholds.
    per_template = [{"template": k, "nearest_count": int(counts[k]),
                     "quality_count": int(hq_counts[k]), "nearest_fraction": float(counts[k] / n),
                     "quality_fraction": float(hq_counts[k] / n),
                     "assigned_mean_rmse": float(best[assignment == k].mean()) if counts[k] else None}
                    for k in range(len(centers))]
    return {"atoms": n, "quality_atoms": int(good.sum()), "bad_atoms": int((~good).sum()),
            "needed_quality_atoms": math.ceil(thresholds["hq_min"] * n),
            "needed_quality_atoms_per_template": math.ceil(thresholds["min_mode_fraction"] * n),
            "modes": modes, "hq": float(good.mean()),
            "nearest_template_rmse_quantiles": dict(zip(["min", "p50", "p90", "max"], map(float, np.quantile(best, [0., .5, .9, 1.])))),
            "per_template": per_template,
            "mean_generated_spatial_std": float(images.std(1).mean()),
            "mean_target_spatial_std": float(centers.std(1).mean()),
            "pixel_mse_share_on_template_distinguishing_pixels": (float(squared_error[:, discriminative].sum()) / total_error if total_error else 0.),
            "pixel_mse_share_on_shared_pixels": (float(squared_error[:, ~discriminative].sum()) / total_error if total_error else 0.),
            "uniform_generator_best_possible_rmse": list(map(float, np.sqrt(centers.var(1)))),
            "target_template_means": list(map(float, centers.mean(1))),
            "nearest_assignment_is_not_a_quality_hit": True}


def cause_record(category, established, confidence="observed", *, unresolved=None, next_action=None):
    return {"failure_class": category, "established": established, "confidence": confidence,
            "unresolved_optimization_mechanism": unresolved,
            "next_action": next_action}


def ordinary_diagnosis(case, summary, result, episode, arm):
    spec = summary["spec"]
    bounds = requirements(spec)
    curve = result.get("observations", result.get("curve"))
    failed = [cell for cell in gate_cells(summary["live"], bounds) if not cell["passed"]]
    diagnostic = {
        "catalog_id": case["id"], "arm": arm, "name": case["name"],
        "original_status": summary["verdict"]["status"],
        "source": {"base_sha": "4b16312e56328a679b92da69a287c0c9490259d9",
                   "proposal_head_sha": case.get("head_sha"),
                   "sampling": summary["sampling"], "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()},
        "evidence": [identity(episode / "summary.json"), identity(episode / "result.json")],
        "failed_final_gates": failed, "trajectory": trajectory(curve, bounds),
    }
    npz = episode / "observations.npz"
    unresolved = "The capture has outputs and metrics, but no G/D/prior weights, optimizer moments or per-role gradients. It cannot identify the training mechanism."
    next_action = ("On a future substantive fixed-protocol run, retain G/D/prior, optimizer and controller state at the first observed, best and terminal steps; audit paired critic gradients at those states before proposing a training change.")
    if npz.exists():
        diagnostic["evidence"].append(identity(npz))
        with np.load(npz) as data:
            final = data["live"][-1]
            if summary["kind"] == "image":
                attribution = image_attribution(final, data["templates"], spec["thresholds"])
                diagnostic["saved_cloud_analysis"] = attribution
                if attribution["modes"] != summary["live"]["modes"] or not np.isclose(attribution["hq"], summary["live"]["hq"], atol=1e-9):
                    raise ValueError("saved image cloud and original metric disagree")
                if not failed:
                    count = diagnostic["trajectory"]["passing_suffix"]
                    reason = cause_record("insufficient_terminal_stability", f"Final image gates pass, but only {count}/5 consecutive terminal observations pass. Earlier good snapshots are transient or acquired too late.", next_action="Retain the frozen verdict. Test an independent earlier-acquisition or stability hypothesis; extending or changing the gate produces new evidence.", unresolved=unresolved)
                elif spec.get("architecture") == "uniform_generator":
                    minimum = min(attribution["uniform_generator_best_possible_rmse"])
                    reason = cause_record("representation_impossible", f"G can emit only a constant image. Its best possible RMSE to any target is {minimum:.6g}, above the quality limit {spec['thresholds']['quality_rmse']}. No optimizer can make this architecture pass.", "proven structural", next_action="Keep as a negative representation control; exclude it from a required solvable-positive denominator.")
                elif spec.get("architecture") == "mean_discriminator":
                    means = attribution["target_template_means"]
                    reason = cause_record("critic_nonidentifiability", f"D receives only each image's mean. All {len(means)} target locations have the same mean {means[0]:.6g}; it cannot distinguish their spatial arrangement.", "proven structural", next_action="Keep as a critic-information negative control and pair with a spatially discriminating critic under the same law.")
                else:
                    per = attribution["per_template"]
                    balance = all(x["nearest_fraction"] >= spec["thresholds"]["min_mode_fraction"] for x in per)
                    category = "template_fidelity" if balance else "template_fidelity_and_occupancy"
                    reason = cause_record(category, f"Only {attribution['quality_atoms']}/{attribution['atoms']} atoms are within the frozen template RMSE, versus {attribution['needed_quality_atoms']} needed. Nearest-template labels {'cover' if balance else 'underrepresent'} the required groups, but bad pixels never count as genuine mode hits.", unresolved=unresolved, next_action=next_action)
            elif spec.get("identifiable"):
                components = vector_components(final, spec, int(data["steps"][-1]))
                diagnostic["saved_cloud_analysis"] = {"components": components,
                    "core_shape_is_diagnostic_and_does_not_replace_historical_gates": True}
                names = {cell["metric"] for cell in failed}
                rare = [x for x in components if x["below_v5_particle_floor"]]
                if case["id"] == "develop-vector_unequal_mass":
                    culprit = min(components, key=lambda x: x["full_shape"]["min_eigen_ratio"])
                    reason = cause_record("finite_particle_evaluator_mismatch", f"Only the full-component minimum eigenvalue gate fails. Component {culprit['component']} has expected {culprit['declared_expected_particles']:.4g} particles and {culprit['distinct_sample_coordinates']} distinct saved output coordinates. Its full/core eigen ratio is {culprit['full_shape']['min_eigen_ratio']:.6g}. The frozen gate scores that under-resolved component despite the spec's v5 floor; mass/HQ and resolved-core diagnostics pass.", "proven evaluator mismatch; optimizer cause unresolved", unresolved=unresolved, next_action="Preserve this historical FAIL. Bind any new resolution-aware test to a new protocol and its finite-atom oracle; keep rare-mass coverage gated. Do not relabel the old receipt.")
                    diagnostic["saved_cloud_analysis"]["underresolved_components"] = [x["component"] for x in rare]
                else:
                    full = float(np.mean([x["full_shape"]["covariance_error"] for x in components]))
                    core = float(np.mean([x["core_shape"]["covariance_error"] for x in components]))
                    missing = [x["component"] for x in components if not x["sample_count"]]
                    if "component_covariance_error" in names:
                        category = "far_spill_dominates_covariance" if full > 2 * core and core <= .85 else "spread_and_spill_failure"
                    elif "mass_tv" in names:
                        category = "mode_mass_loss" if missing else "mode_mass_imbalance"
                    else:
                        category = "density_or_shape_failure"
                    reason = cause_record(category, f"Saved components have full covariance error {full:.6g} versus 4-sigma-core error {core:.6g}; missing nearest-center components {missing}. Per-component variance decomposition separates core spread from distant spill/between-group displacement. Frozen mass/HQ/shape gates remain unchanged.", unresolved=unresolved, next_action=next_action)
                    culprit = max(components, key=lambda x: x["full_shape"]["covariance_error"])
                    diagnostic["saved_cloud_analysis"]["largest_full_covariance_error_component"] = culprit["component"]
                    diagnostic["saved_cloud_analysis"]["missing_nearest_center_components"] = missing
                    reason["next_action"] += f" Prioritize component {culprit['component']} (full covariance error {culprit['full_shape']['covariance_error']:.5g}, 3-sigma spill {culprit['spill_fraction_beyond_3sigma']:.5g}) and missing components {missing}."
            else:
                points = np.asarray(final, dtype=np.float64)
                target_mean = np.asarray(spec["masses"]) @ np.asarray(spec["means"])
                diagnostic["saved_cloud_analysis"] = {"sample_mean": points.mean(0).tolist(),
                    "analytic_target_mean": target_mean.tolist(),
                    "mean_error_norm_against_analytic_law": float(np.linalg.norm(points.mean(0) - target_mean)),
                    "component_labels_deliberately_not_scored": True}
                reason = cause_record("late_observable_mean_drift", "The overlapping law is not component-identifiable. Its final normalized observable mean error exceeds .15; its earlier passing observations do not survive the terminal window.", unresolved=unresolved, next_action=next_action)
    elif case["id"] == "develop-residual_student":
        raw = result.get("raw", {})
        diagnostic["saved_cloud_analysis"] = {"training_rows": raw.get("both_land_rows"),
                    "wrong_pad_fraction": summary["live"]["wrong_pad_rate"],
                    "mean_endpoint_distance": curve[-1].get("endpoint_l2"),
                    "identity_mse_passes": summary["live"]["identity_mse"] <= .02}
        reason = cause_record("late_paired_identity_regression", "Mean identity MSE passes, but 1/12 endpoints lands on a wrong pad at the terminal update. Twenty earlier observations pass; aggregate MSE hides that final row error.", unresolved=unresolved, next_action="Retain predicted endpoints plus G/D/prior/optimizer states at the last passing and failing updates. Measure that row's paired critic gradient and residual update; the current receipt does not identify its row ID.")
    else:
        last = curve[-1]
        diagnostic["saved_cloud_analysis"] = {key: last.get(key) for key in
             ["missing_modes", "hq_counts", "nearest_counts", "closest_distance_per_mode", "hq_radius"]}
        diagnostic["saved_cloud_analysis"]["exact_12_atom_support"] = last.get("support")
        reason = cause_record("support_misplacement", "The 12-atom live support has only 4/8 genuine 3-sigma ring hits. Nearest-center occupancy in all cells is insufficient: several atoms are displaced beyond the quality radius.", unresolved=unresolved, next_action="Save the critic, generator, table and moments for these 12 labelled support rows; compare center-directed and tangential gradients for missing and hit modes.")
    diagnostic.update(reason)
    diagnostic["missing_artifacts_for_causal_attribution"] = ([str(episode / "checkpoint-at-first-best-terminal.pt")] if reason["unresolved_optimization_mechanism"] else [])
    return diagnostic


def checkpoint_native_probe(directory, arrays):
    """One frozen terminal critic probe; no optimizer step or resampling study."""
    import torch
    from lib.toy_models import SimpleMLPDiscriminator
    from particlegan.gan_loss import GANLoss

    state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
    source = state["policy"]["served_source"]
    if source not in ("averaged", "fast"):
        raise ValueError("unknown native served source")
    family = "ema_" if source == "averaged" else ""
    g, table = state["models"][family + "G"], state["models"][family + "prior"]["z"]
    census = (table @ g["weight"].T + g["bias"]).numpy()
    sigma = float(state["policy"]["last_output_sigma"])
    # A clean sample still includes feature-cell/DV12 latent perturbation.
    # The raw table census below deliberately omits that perturbation and is
    # labelled separately; it must not be asserted equal to the served cloud.
    clean = arrays["clean"][-1]
    theta = float(arrays["angles"][-1])
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    centers = arrays["centers"] @ rotation.T
    ids = np.square(clean[:, None] - centers).sum(-1).argmin(1)
    means = np.stack([clean[ids == k].mean(0) if (ids == k).any() else centers[k] for k in range(100)])
    width_direction = clean - means[ids]
    center_direction = centers[ids] - means[ids]
    jitter = arrays["live"][-1] - clean
    synthetic_real = centers[ids] + jitter * (.03 / sigma)
    critic = SimpleMLPDiscriminator(2, 128, 3, 3).double().eval().requires_grad_(False)
    critic.load_state_dict(state["models"]["D"])
    real = torch.tensor(synthetic_real, dtype=torch.float64)
    observations = {}
    for law, values in [("clean_diagnostic", clean), ("noisy_with_captured_jitter", arrays["live"][-1])]:
        fake = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        loss = GANLoss().g_loss(critic(fake), critic(real))
        gradient, = torch.autograd.grad(loss, fake)
        w = torch.tensor(width_direction, dtype=torch.float64)
        c = torch.tensor(center_direction, dtype=torch.float64)
        derivative = (gradient * w).sum(1).detach().numpy()
        per_mode = [float(derivative[ids == k].sum()) for k in range(100) if (ids == k).any()]
        eps = .01
        with torch.no_grad():
            plus = GANLoss().g_loss(critic(fake + eps * w), critic(real))
            minus = GANLoss().g_loss(critic(fake - eps * w), critic(real))
        observations[law] = {"paired_generator_game": float(loss.detach()),
            "width_expansion_directional_derivative": float(derivative.sum()),
            "width_expansion_centered_difference_eps_0_01": float((plus - minus) / (2 * eps)),
            "modes_locally_rewarding_expansion": sum(v < 0 for v in per_mode),
            "modes_locally_penalizing_expansion": sum(v > 0 for v in per_mode),
            "center_correction_directional_derivative": float((gradient * c).sum()),
            "negative_derivative_means_direction_lowers_generator_game": True}
    spec = {"means": centers.tolist(), "covariances": [[[.03 ** 2, 0.], [0., .03 ** 2]]] * 100,
            "masses": [.01] * 100, "particles": len(census)}
    components = vector_components(census, spec, state["completed_steps"])
    return {"checkpoint": identity(directory / "checkpoint.pt"),
            "served_source": source, "completed_steps": state["completed_steps"],
            "raw_table_census_particles": len(census),
            "raw_table_census_omits_dv12_and_is_not_served_clean_law": True,
            "sampling_backend": state["backend_selection"]["sampling_backend"],
            "latent_bandwidth": np.asarray(state["controller"]["latent_bandwidth"]).tolist(),
            "output_sigma": sigma, "noise_variance_over_target_variance": sigma ** 2 / .03 ** 2,
            "raw_table_census_components_below_min_eigen_gate": sum(x["full_shape"]["min_eigen_ratio"] < .40 for x in components),
            "raw_table_census_mean_covariance_error": float(np.mean([x["full_shape"]["covariance_error"] for x in components])),
            "critic_probe": observations,
            "probe_scope": "Terminal trained D, served G/table, frozen paired 4096-row diagnostic clouds. Synthetic real uses each assigned target center plus the recorded jitter scaled to target sigma. FP64 CPU derivatives of the saved FP32 weights; no optimizer step. This is not the original training pairing, KA2-preconditioned update, a checkpoint replay, or a causal explanation of the training trajectory."}


def native_diagnoses(case, directory, probes):
    summary, gates = read(directory / "summary.json"), read(directory / "gates.json")
    with np.load(directory / "observations.npz") as data:
        arrays = {key: data[key] for key in data.files}
    noisy, clean = arrays["live"][-1], arrays["clean"][-1]
    sigma = read(directory / "observations.json")[-1]["sigma"]
    detail = {"captured_diagnostic_rows": len(clean),
              "paired_added_jitter_coordinate_std": (noisy - clean).std(0).tolist(),
              "recorded_output_sigma": sigma,
              "noise_variance_over_target_variance": sigma ** 2 / .03 ** 2,
              "diagnostic_cloud_is_not_the_20k_gate_draw": True,
              "clean_law_includes_feature_cell_dv12_latent_perturbation": True,
              "analytic_population_law_witness": noisy_gaussian_fit_witness(.03, sigma)}
    theta = float(arrays["angles"][-1])
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    centers = arrays["centers"] @ rotation.T
    component_spec = {"means": centers.tolist(), "covariances": [[[.03 ** 2, 0.], [0., .03 ** 2]]] * 100,
                      "masses": [.01] * 100, "particles": 20000}
    clean_modes = vector_components(clean, component_spec, summary["steps"])
    noisy_modes = vector_components(noisy, component_spec, summary["steps"])
    detail["final_diagnostic_components_4sigma_core_not_native_3sigma_gate"] = [
        {"component": c["component"], "clean_count": c["sample_count"],
         "clean_full_min_eigen_ratio": c["full_shape"]["min_eigen_ratio"],
         "clean_core_min_eigen_ratio": c["core_shape"]["min_eigen_ratio"],
         "noisy_count": n["sample_count"], "noisy_core_min_eigen_ratio": n["core_shape"]["min_eigen_ratio"],
         "noisy_3sigma_spill_fraction": n["spill_fraction_beyond_3sigma"]}
        for c, n in zip(clean_modes, noisy_modes)]
    if probes:
        detail["terminal_checkpoint_probe"] = checkpoint_native_probe(directory, arrays)
    output = []
    for law, status, bounds in [("clean", summary["native_clean_status"], NATIVE_BOUNDS),
                               ("noisy", summary["native_noisy_status"], NATIVE_BOUNDS),
                               ("accuracy", summary["accuracy_status"], ACCURACY_BOUNDS)]:
        if status != "FAIL":
            continue
        rows = [{"step": row["step"], **row[law]} for row in gates]
        failed = [x for x in gate_cells(rows[-1], bounds) if not x["passed"]]
        moving = "moving_original_status" in summary
        record = {"catalog_id": case["id"], "arm": law, "name": case["name"],
            "original_status": status,
            "source": {"base_sha": summary["base_sha"], "sampling": summary["sampling"],
                       "source_sha256": summary["source_sha256"]},
            "evidence": [identity(directory / p) for p in ["summary.json", "gates.json", "observations.npz"]],
            "failed_final_gates": failed, "trajectory": trajectory(rows, bounds),
            "saved_cloud_analysis": detail,
            "missing_artifacts_for_causal_attribution": [str(directory / "checkpoint-at-width-acquisition-and-contraction.pt"), "original caller data-stream cursor at those checkpoints"]}
        if moving:
            record.update(cause_record("reacquisition_is_not_full_density_recovery", "The original relative reacquisition criterion permits 95/100 modes and 90% of the pre-shift HQ. Final strict scoring has only 97 genuine modes, missing low-mass components and overly broad residual populations; accuracy moments are undefined when a component lacks enough in-radius samples. Only two gate observations occur after the last target jump, below a five-check final hold even if both were good.", "proven gate mismatch; optimizer cause unresolved", unresolved="The endpoint lacks full coverage; the terminal checkpoint probe can describe local directions but does not explain the shift history.", next_action="Retain the original moving PASS. Define an independent new moving protocol with absolute density gates and a separately budgeted stationary terminal hold; inspect saved endpoint gradients before new training."))
            record["original_moving_status"] = summary["moving_original_status"]
        else:
            record.update(cause_record("clean_spread_contraction_under_noisy_training_law", "All 100 centers and mass gates pass, but clean within-mode covariance and radial spread contract below their frozen lower bounds. The paired served-law observations differ only by recorded Gaussian output jitter; its variance is approximately 93% of the target component variance. An exact noisy Gaussian fit at sigma_out=.029 needs clean sigma=.007681 and variance ratio .06556, which itself fails the clean .40 covariance floor. The optimized noisy law and the clean full-width gate ask different population questions.", "proven population-law mismatch; optimizer cause unresolved", unresolved="The Gaussian-convolution witness explains why optimizing the declared noisy law need not recover the full-width clean law. It does not establish why the actual KA2/Adam/critic trajectory produced its particular contraction. One terminal critic probe cannot establish that path mechanism.", next_action="Bind the training and served laws to the intended question; preserve both existing noisy PASS and clean FAIL. Use terminal gradients and acquisition/contraction checkpoints before proposing a training-noise change, which must remain a new cohort."))
        output.append(record)
    return output


def load_misgan_problem(source, mechanism):
    import torch
    name = "toy_failure_misgan_source"
    spec = importlib.util.spec_from_file_location(name, source / "lib/misgan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.Problem(mechanism, 20000, 10000, 0, torch.device("cpu"))


def misgan_diagnosis(case, directory, oracle, probes):
    summary = read(directory / "run/summary.json")
    final, mechanism = summary["final"], summary["config"]["mechanism"]
    reference = oracle[mechanism]
    record = {"catalog_id": case["id"], "arm": "generation_and_conditional_imputation", "name": case["name"],
        "original_status": "Measured; no declared binary gate",
        "source": {"proposal_head_sha": case["head_sha"], "sampling": case["sampling"],
                   "source_sha256": read(directory / "replay.json")["source_sha256"]},
        "evidence": [identity(directory / p) for p in ["run/summary.json", "observations.npz", "replay.json"]],
        "failed_final_gates": [], "no_original_binary_gate": True,
        "posterior_reference": reference,
        "measured_gaps_not_posthoc_binary_verdicts": {
            "posterior_tv_above_finite_draw_bayes": final["itv"] - reference["bayes"]["itv"],
            "ambiguous_accuracy_minus_bayes": final["acc_lo"] - reference["bayes"]["acc_lo"],
            "missing_coordinate_std_over_bayes": final["istd"] / reference["bayes"]["istd"],
            "rmse_minus_bayes": final["rmse"] - reference["bayes"]["rmse"]},
        "first_observed": {k: summary["history"][0][k] for k in ["step", "modes", "hq", "off", "acc_lo", "itv", "istd"]},
        "terminal": {k: final[k] for k in ["step", "modes", "hq", "off", "acc_lo", "itv", "istd"]},
        "missing_artifacts_for_causal_attribution": [str(directory / "run/critic-x-mask-imputer-and-optimizer-state.pt"), "imputer acquisition/near-deterministic transition checkpoints; caller RNG/data cursors"]}
    with np.load(directory / "observations.npz") as data:
        points, ids = data["imputed"][-1], data["ids"][-1]
        centers = np.array([(x, y) for x in np.arange(10) - 4.5 for y in np.arange(10) - 4.5])
        nearest = np.square(points[..., None, :] - centers).sum(-1).argmin(-1)
        diversity = [len(np.unique(nearest[:, i])) for i in range(len(ids))]
        detail = {"fixed_first_ambiguous_row_ids": list(map(int, ids)),
                  "captured_16_draw_unique_mode_counts_per_row": diversity,
                  "captured_2d_std_per_row": np.linalg.norm(points.std(0), axis=-1).tolist(),
                  "capture_subset_not_an_independent_random_reference": True,
                  "projection_metric_is_blind_to_6d_orthogonal_error": True}
    if probes:
        import torch
        from lib.toy_models import SimpleMLPGenerator
        module, problem = load_misgan_problem(directory / "proposal", mechanism)
        analytic_off = problem.off_plane(problem.x_ref)
        detail["generated_off_plane_over_true_data_reference"] = final["off"] / analytic_off
        detail["true_data_off_plane_mean"] = analytic_off
        state = torch.load(directory / "run/ckpt.pt", map_location="cpu", weights_only=False)
        imputer = SimpleMLPGenerator(z_dim=24, out_dim=8).eval().requires_grad_(False)
        imputer.load_state_dict(state["imputer"])
        table = state["imputer_prior"]["z"]
        chosen = table[torch.linspace(0, len(table) - 1, 64).long()]
        row_ids = torch.as_tensor(ids, dtype=torch.long)
        x, m = problem.x_test[row_ids], problem.m_test[row_ids]
        with torch.no_grad():
            conditioning = torch.cat([x * m, m], -1)
            inputs = torch.cat([conditioning[None].expand(64, -1, -1),
                                chosen[:, None].expand(-1, len(ids), -1)], -1)
            output = m[None] * x[None] + (1 - m[None]) * imputer(inputs.reshape(-1, 24)).reshape(64, len(ids), 8)
            projected = problem.to2d(output.reshape(-1, 8)).reshape(64, len(ids), 2)
            modes = torch.cdist(projected.reshape(-1, 2), module.grid_centers()).argmin(1).reshape(64, len(ids))
            posterior, _ = module.bayes_posterior(problem, x, m)
            posterior_rows = []
            for i in range(len(ids)):
                empirical = torch.bincount(modes[:, i], minlength=100).float() / 64
                posterior_rows.append(float((empirical - posterior[i]).abs().sum() / 2))
        detail["terminal_checkpoint_noise_probe"] = {
            "checkpoint": identity(directory / "run/ckpt.pt"),
            "scope": "Inference only: first saved ambiguous contexts crossed with 64 evenly spaced saved noise-table rows; fixed source/data seed, no training or random-seed study. This is a diagnostic enumeration, not the original CUDA 16-draw cohort.",
            "noise_table_mean_coordinate_std": float(table.std(0, correction=0).mean()),
            "probe_noise_mean_coordinate_std": float(chosen.std(0, correction=0).mean()),
            "projected_output_std_norm_per_row": projected.std(0, correction=0).norm(dim=-1).tolist(),
            "unique_output_modes_per_row": [len(torch.unique(modes[:, i])) for i in range(len(ids))],
            "per_row_posterior_tv_on_fixed_probe": posterior_rows,
            "observed_coordinate_preservation_max_error": float(((output - x[None]) * m[None]).abs().max())}
    record["saved_cloud_analysis"] = detail
    statements = {
        "mcar_p20": "Projected generation is good, but full-8D off-plane distance exceeds the true data scale and conditional posterior TV exceeds the finite-draw oracle. Only one ambiguous test row exists, so its perfect acc_lo is weak evidence of posterior recovery.",
        "mcar_p50": "Generation reaches 99 projected modes yet only 38.18% 3-sigma fidelity. The imputer is almost deterministic despite a broad learned noise table: missing-coordinate std is about 6% of Bayes, and ambiguous-row accuracy .112 is far below .555. This is loss of useful conditional noise response, not a collapsed input-noise bank.",
        "mcar_p80": "Severe data fidelity and conditional failures coexist: 32 modes, 8.51% HQ, posterior TV .510 versus .179 finite-draw Bayes. Diversity magnitude alone looks plausible (.325 versus .334) but lands in the wrong posterior modes.",
        "block": "Projected generation covers all 100 modes with 99.82% HQ, while the 10% single-sensor rows have ambiguous accuracy .097 versus .777 Bayes. Excess diversity (.152 versus .062) and posterior TV show that good marginal generation does not imply correct conditioning.",
    }
    record.update(cause_record("conditional_posterior_or_manifold_mismatch", statements[mechanism], "observed; structural scorer limitation proven",
        unresolved="The checkpoint saves only EMA G_x/imputer and their tables; no trained D_x/D_m/D_i, critic optimizer moments or acquisition history. It cannot attribute the mismatch to critic gradients, masking, KA2 or particular updates.",
        next_action="Freeze posterior/diversity and full-8D manifold tolerances relative to analytic finite-draw references. Preserve marginal and conditional metrics separately. Inspect a future complete checkpoint's D_i sensitivity to context/noise before proposing a bounded training hypothesis."))
    return record


def build(catalog_path, artifacts, probes=True):
    catalog = read(catalog_path)
    output, coverage, errors = [], [], []
    if probes:
        import torch
        torch.set_num_threads(1)
    for case in catalog["cases"]:
        joined = []
        identifier = case["id"]
        if identifier.startswith("develop-"):
            directory = artifacts / "develop-frozen-reference" / case["name"]
            episode = directory / "episode-00" if (directory / "episode-00").exists() else directory
            summary, result = read(episode / "summary.json"), read(episode / "result.json")
            check_receipt(summary, episode, case["source_result_sha256"])
            if summary["verdict"]["status"] == "FAIL":
                output.append(ordinary_diagnosis(case, summary, result, episode, "frozen_reference"))
                joined.append("frozen_reference")
        elif case.get("arms") and case.get("pr") not in (226, 227):
            directory = artifacts / identifier
            contexts = []
            for index, arm in enumerate(case["arms"]):
                episode = directory / f"episode-{index:02d}"
                summary = read(episode / "summary.json")
                check_receipt(summary, episode, arm["result_sha256"], arm["capture_sha256"])
                label = "proposal_arm" if index == 0 else "positive_control"
                contexts.append({"arm": label, "original_status": arm["status"],
                                 "spec_sha256": hashlib.sha256(json.dumps(summary["spec"], sort_keys=True).encode()).hexdigest()})
                if arm["status"] == "FAIL":
                    row = ordinary_diagnosis(case, summary, read(episode / "result.json"), episode, label)
                    row["original_contrast"] = case["status"]
                    row["control_claim_limit"] = "The original control is a specific protocol/architecture/init composite. Its PASS establishes solvability of that composite; it does not isolate each changed factor or qualify current Atlas/KA2."
                    output.append(row)
                    joined.append(label)
            for row in output:
                if row["catalog_id"] == identifier:
                    row["arm_context"] = contexts
                    row["replay_identity"] = identity(directory / "replay.json")
            if case.get("adapter"):
                raw = artifacts / f"pr{case['pr']}" / "replay.json"
                errors.append({"catalog_id": identifier, "pr": case["pr"], "original_current_api_error": read(raw)["error"].strip().splitlines()[-1],
                               "evidence": identity(raw), "adapted_replay_identity": identity(directory / "replay.json"),
                               "adaptation_is_not_current_ka2_qualification": True})
        elif identifier.startswith("atlas-"):
            name = "native-moving-rotated100" if "moving" in case["name"] else "native-" + case["name"]
            if case["name"] == "grid100":
                name += "-v2"
            rows = native_diagnoses(case, artifacts / name, probes)
            output.extend(rows)
            joined.extend(row["arm"] for row in rows)
        elif case.get("pr") == 196:
            oracle = read(catalog_path.parent / "misgan-oracles.json")["controls"]
            output.append(misgan_diagnosis(case, artifacts / identifier, oracle, probes))
            joined.append("generation_and_conditional_imputation")
        elif case.get("pr") in (22, 153):
            receipt = artifacts / f"pr{case['pr']}-current" / "replay.json"
            raw = read(receipt)
            row = {"catalog_id": identifier, "arm": "current_api_host", "name": case["name"],
                   "original_status": case["status"], "source": {"proposal_head_sha": case["head_sha"], "source_sha256": raw["source_sha256"]},
                   "evidence": [identity(receipt)], "failed_final_gates": [],
                   "error": raw["error"].strip().splitlines()[-1], "missing_artifacts_for_causal_attribution": [],
                   **cause_record("execution_blocker", "The host raises the retained public-API exception before any update. This is a host/API incompatibility, not a scientific training failure.", "proven execution cause",
                        next_action="Add a separately scoped current-API-compatible test adapter and verify its loss/update semantics; retain this original error. Only then capture actual training and closed-loop checkpoints.")}
            output.append(row)
            joined.append(row["arm"])
        elif case.get("pr") == 224:
            directory = artifacts / "pr224-v2"
            rows = read(directory / "observations.json")["native"]
            row = {"catalog_id": identifier, "arm": "native_release", "name": case["name"],
                "original_status": "Counterexample reproduced; both controls stable",
                "source": {"proposal_head_sha": case["head_sha"], "source_sha256": read(directory / "replay.json")["source_sha256"]},
                "evidence": [identity(directory / p) for p in ["replay.json", "observations.json"]],
                "failed_final_gates": [{"metric": "bounded_stationary_paired_game", "value": case["final"]["native"]["final_game"], "reference": math.log(2)}],
                "observed_release_step": case["final"]["native"]["releases"][0],
                "peak_game": case["final"]["native"]["peak_game"],
                "first_update_after_release": {k: rows[48][k] for k in ["step", "applied_lr", "game_after"]},
                "missing_artifacts_for_causal_attribution": [],
                **cause_record("stiff_coordinate_lr_release_instability", "The predeclared native release doubles the stiff-coordinate spectral step factor from 1.6 (stable) to 3.2 (unstable). Cancelling only that release keeps log(2); the safe geometry still releases safely. This is a causally isolated constructed controller fixture, not a trained-dataset failure.", "proven within constructed unit fixture",
                     next_action="Keep both unsafe and safe-release controls. A production change needs separate moving-target and real learned-critic evidence; this fixture alone does not justify disabling every release.")}
            output.append(row)
            joined.append(row["arm"])
        coverage.append({"catalog_id": identifier, "original_status": case["status"],
                         "diagnosed_arms": joined,
                         "scope": "diagnosed" if joined else "no recorded failing arm" if case.get("media") else "source-reviewed only; no executed outcome"})
        print(json.dumps({"event": "FAILURE_DIAGNOSIS_CASE", "catalog_id": identifier, "diagnosed": joined}), flush=True)
    result = {"schema": SCHEMA, "source_catalog": identity(catalog_path),
              "diagnosis_implementation": identity(Path(__file__).resolve()),
              "scope": "Every recorded failure in catalog109, plus conditional/manifold deficiencies in all four ungated MisGAN laws and the constructed PR224 failure. No training, seed studies, qualification relabels, or production edits.",
              "artifact_root": str(artifacts), "terminal_state_probes_executed": probes,
              "case_coverage": coverage, "diagnoses": output,
              "historical_api_errors_before_explicit_archived_adaptation": errors,
              "counts": {"catalog_cases": len(coverage), "diagnosed_cases": sum(bool(row["diagnosed_arms"]) for row in coverage),
                         "diagnosed_arms": len(output), "frozen_reference_failures": sum(row["arm"] == "frozen_reference" for row in output),
                         "proposal_failing_arms": sum(row["arm"] in ("proposal_arm", "positive_control") for row in output),
                         "historical_api_errors": len(errors),
                         "by_failure_class": dict(sorted(Counter(row["failure_class"] for row in output).items()))}}
    validate_coverage(catalog, result)
    return result


def validate_coverage(catalog, report):
    """Fail closed if any recorded failing arm is omitted or duplicated."""
    actual = [(row["catalog_id"], row["arm"]) for row in report["diagnoses"]]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate failure diagnosis")
    expected = set()
    for case in catalog["cases"]:
        identifier = case["id"]
        if identifier.startswith("develop-") and case["status"] == "FAIL":
            expected.add((identifier, "frozen_reference"))
        if case.get("pr") not in (226, 227):
            expected.update((identifier, "proposal_arm" if index == 0 else "positive_control")
                            for index, arm in enumerate(case.get("arms", [])) if arm.get("status") == "FAIL")
        if identifier.startswith("atlas-"):
            expected.add((identifier, "clean"))
            if "moving" in case["name"]:
                expected.update([(identifier, "noisy"), (identifier, "accuracy")])
        if case.get("pr") == 196:
            expected.add((identifier, "generation_and_conditional_imputation"))
        if case.get("pr") in (22, 153):
            expected.add((identifier, "current_api_host"))
        if case.get("pr") == 224:
            expected.add((identifier, "native_release"))
    if expected != set(actual):
        raise ValueError(f"failure coverage differs: missing={expected - set(actual)}, extra={set(actual) - expected}")
    covered_ids = [row["catalog_id"] for row in report["case_coverage"]]
    if len(covered_ids) != len(set(covered_ids)):
        raise ValueError("duplicate catalog case coverage")
    if set(covered_ids) != {case["id"] for case in catalog["cases"]}:
        raise ValueError("case coverage differs from source catalog")


def markdown(report):
    counts = report["counts"]
    lines = ["# Diagnosing the recorded toy failures", "",
        f"This zero-training audit joins **all {counts['catalog_cases']} catalog entries** and explains **{counts['diagnosed_arms']} failing or deficient arms**: 17 failed frozen references, 43 failed proposal arms, six clean/strict native results, four ungated MisGAN laws, two execution blockers and PR224's constructed unsafe release. The [machine-readable map](failure-diagnosis.json) binds every case/arm to exact source and artifact hashes. Historical outcomes, thresholds, budgets and clean/noisy cohorts remain unchanged.", "",
        "## What the evidence establishes", "",
        "- **Three late image acquisitions fail the hold:** PR62's original arm has a final passing suffix of 3, PR71 has 2, and PR63's positive control has 4; the declared gate requires 5. Their final samples pass, but the scientific verdict remains FAIL. PR154's control ends at HQ .875, below .9, so that failure is still endpoint fidelity.",
        "- **Rare-mass shape is under-resolved:** the historical unequal-mass gate fails only the 2% component's minimum covariance eigenvalue (.0812 versus .15). A 256-particle table allocates about 5.12 atoms to it; 4,096 evaluator draws do not create new support atoms. Mass, HQ and resolved-core measurements pass. This is a frozen evaluator/resolution mismatch, preserved as the original FAIL.",
        "- **Ring covariance errors often come from spills:** full covariance error can be many times the in-core error. The decomposition attributes the exact population covariance to core variation, far-spill variation and the offset between those groups. A healthy core does not excuse spill or occupancy failures; no historical gate is substituted.",
        "- **Mean-only D and uniform G are structurally limited:** equal-mean blobs are indistinguishable to the former; the latter's best possible stripe RMSE is .433, above .1. These are useful negative controls, not demonstrated solvable positives. Width2/latent1 is an observed capacity/optimization stress, not a proven representation impossibility.",
        "- **Atlas optimizes a different population law from the clean gate:** output sigma about .029 supplies about 93% of a sigma-.03 target's component variance. Analytically, a clean Gaussian with sigma **.007681** convolved with sigma-.029 jitter exactly recovers the target, yet its clean variance ratio **.06556** fails the native **.40** floor. Clean served spread contracts while paired noisy samples satisfy spread gates. **Clean sampling still includes feature-cell/DV12 latent perturbation**; the saved raw-table census deliberately omits it and is not the clean serving law. The terminal critic probe examines expansion and centering directions without a training update; it does not establish the actual optimizer path.",
        "- **Moving recovery is weaker than density recovery:** the original gate accepts 95 modes and 90% of baseline HQ. Its endpoint has 97 modes and about .926 precision, failing the absolute density requirements. The last shift leaves only two terminal gate observations, so the stationary five-check hold is not established either.",
        "- **MisGAN marginals do not establish posterior recovery:** p50's output is almost deterministic despite a broad saved noise table; p80 has roughly plausible diversity magnitude but wrong posterior modes; block has excellent projected generation and poor single-sensor conditioning. The 2D projection cannot observe six orthogonal dimensions. All remain measured ungated results, not post hoc binary FAILs.",
        "- **Circle/sprite never train on this snapshot:** their retained exceptions name the missing host import/API method. All 13 archived vector scripts' initial public-API errors are retained separately; their adapted experiments are not current KA2/Atlas qualifications.", "",
        "**Confidence is scoped.** Structural impossibility, loss of critic information, sampling-law spread and exact gate shortfalls are established. Output trajectories do not reveal an Adam/KA2/critic causal mechanism. Each unresolved record names the missing weights, optimizer states or caller cursor and a bounded next diagnostic. Passing composite controls establish solvability under their exact architecture/init recipe; they do not isolate each changed factor.", "",
        "The [separate revised vector controls](NON_IMAGE_QUALITY.md) strengthen narrow distribution gates and retain exact initialization confounds. Any failures exposed only by those revised gates belong to that new diagnostic cohort; this report explains the frozen original results.", "",
        "## Every failing frozen reference and proposal arm", "",
        "`original` and `control` denote the proposal's frozen arm ordering. First/best/terminal trajectories and per-mode/per-template decompositions are in the JSON; best snapshots never replace terminal evidence.", "",
        "| Catalog ID / arm | Frozen failing gate or hold | Diagnosis |", "|---|---|---|"]
    for row in report["diagnoses"]:
        if row["arm"] not in ("frozen_reference", "proposal_arm", "positive_control"):
            continue
        cells = row["failed_final_gates"]
        failure = "; ".join(f"{x['metric']}={x['value']:.5g} {x['op']} {x['bound']:g}" for x in cells)
        if not failure:
            failure = f"Final gates pass; suffix {row['trajectory']['passing_suffix']}/5"
        detail = row["saved_cloud_analysis"]
        if "components" in detail:
            components = detail["components"]
            full = np.mean([x["full_shape"]["covariance_error"] for x in components])
            core = np.mean([x["core_shape"]["covariance_error"] for x in components])
            note = f"; full/core covariance error {full:.3g}/{core:.3g}"
        elif "quality_atoms" in detail:
            note = f"; {detail['quality_atoms']}/{detail['atoms']} quality atoms"
        else:
            note = ""
        lines.append(f"| `{row['catalog_id']}` / {row['arm']} | {failure} | {row['failure_class'].replace('_', ' ')}{note} |")
    if report["terminal_state_probes_executed"]:
        lines.extend(["", "## Frozen terminal native critic probe", "",
            "The probe increases within-mode clean width while keeping each empirical mode mean fixed, using the terminal trained critic and captured jitter. Synthetic paired reals use the assigned target center plus that fixed jitter scaled to target sigma. A positive derivative means expansion locally increases the generator game. This FP64 CPU input-gradient probe does not reproduce the training pairing, controller-preconditioned parameter update or earlier trajectory.", "",
            "| Saved endpoint | Width expansion derivative under captured noisy law | Modes penalizing expansion | Center correction derivative |", "|---|---:|---:|---:|"])
        for row in report["diagnoses"]:
            if row["arm"] != "clean" or not row["catalog_id"].startswith("atlas-"):
                continue
            probe = row["saved_cloud_analysis"]["terminal_checkpoint_probe"]["critic_probe"]["noisy_with_captured_jitter"]
            lines.append(f"| `{row['catalog_id']}` | {probe['width_expansion_directional_derivative']:.6g} | {probe['modes_locally_penalizing_expansion']} | {probe['center_correction_directional_derivative']:.6g} |")
        lines.extend(["", "All three stationary endpoints locally penalize expansion on this declared probe while rewarding center correction. This supports a terminal critic contraction signature. Acquisition/contraction checkpoints and actual caller pairings are still needed to explain the training mechanism."])
    lines.extend(["", "## Native and conditional endpoint evidence", "",
        "| Catalog ID / law | Observed failure or deficiency | Scope |", "|---|---|---|"])
    for row in report["diagnoses"]:
        if row["arm"] in ("frozen_reference", "proposal_arm", "positive_control"):
            continue
        text = row.get("error") or row["established"]
        lines.append(f"| `{row['catalog_id']}` / {row['arm']} | {text.replace('|', '/')} | {row['confidence']} |")
    lines.extend(["", "## Reproduce without training", "", "```bash",
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\",
        "  python -u -m benchmarks.toy_audit.failure_diagnosis \\",
        "  --artifacts /ml2/hypergan/toy-audit-artifacts-20261001 \\",
        "  --output /tmp/toy-failure-diagnosis",
        "```", "", "The default reads terminal checkpoints for a fixed native critic direction probe and a MisGAN noise-response enumeration. `--no-state-probes` produces a separately labelled cloud/metric-only diagnostic. No training loop, checkpoint mutation, continuation, RNG-seed search, default/config repair or qualification promotion runs.", "",
        "The generated JSON is compact: hashes, aggregate decompositions, first/best/terminal measurements and exact next actions. Raw arrays, checkpoint weights, per-update streams and execution logs remain outside Git.", ""])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=Path("reports/toy_audit/catalog.json"))
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--no-state-probes", action="store_true")
    args = ap.parse_args()
    report = build(args.catalog, args.artifacts, probes=not args.no_state_probes)
    write(args.output / "failure-diagnosis.json", report)
    (args.output / "FAILURE_DIAGNOSIS.md").write_text(markdown(report))
    print(json.dumps({"event": "FAILURE_DIAGNOSIS_DONE", **report["counts"]}), flush=True)


if __name__ == "__main__":
    main()

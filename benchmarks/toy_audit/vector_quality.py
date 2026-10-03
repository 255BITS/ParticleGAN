"""Calibrate every poor-rated vector definition and audit comparison factors.

This is an evaluator/source audit. It runs no training and retains the original
trained gate. Initialization comes from actual arm receipts, not only the spec:
the historical proposal specs omit the patched ParticlePrior initializer.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
from statistics import NormalDist

import numpy as np
import torch

from benchmarks.transfer_suite import vector_tasks
from benchmarks.transfer_suite.protocol import requirements
from .definition_quality import terminal_window

VERSION = "toy-vector-definition-quality-v1"
LAW_KEYS = ("kind", "means", "covariances", "masses", "scale_start", "scale_end", "scale_ramp_end")
ARCHITECTURE_KEYS = {"d_hidden", "d_layers", "fourier", "research_discriminator"}
PROJECTION_KS_MAX = .06


@torch.no_grad()
def projection_ks(points, spec, step):
    """32 fixed projected empirical CDFs against the analytic mixture law.

    This gate needs no observable mixture labels and is invariant to joint
    rescaling of samples and their target law. It remains a finite projection
    test, not an assertion of equality of the entire two-dimensional density.
    """
    points = torch.as_tensor(points, dtype=torch.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not torch.isfinite(points).all():
        raise ValueError("projection evaluation needs finite two-dimensional points")
    scale = vector_tasks.target_scale(spec, step)
    means = torch.tensor(spec["means"], dtype=torch.float64) * scale
    covariances = torch.tensor(spec["covariances"], dtype=torch.float64) * scale ** 2
    masses = torch.tensor(spec["masses"], dtype=torch.float64)
    n = len(points)
    rank_hi, rank_lo = torch.arange(1, n + 1) / n, torch.arange(n) / n
    values = []
    for theta in np.arange(32) * np.pi / 32:
        direction = torch.tensor([np.cos(theta), np.sin(theta)], dtype=torch.float64)
        ordered = torch.sort(points @ direction).values
        projected_mean = means @ direction
        projected_sd = torch.sqrt(torch.einsum("i,kij,j->k", direction, covariances, direction))
        standardized = (ordered[:, None] - projected_mean) / projected_sd
        cdf = (.5 * (1 + torch.erf(standardized / np.sqrt(2)))) @ masses
        values.append(float(torch.maximum((rank_hi - cdf).max(), (cdf - rank_lo).max())))
    return max(values)


def finite_cloud_witness(spec):
    """A deterministic, equal-atom approximation with the declared particle count.

    Stratified Gaussian quantiles supply each component, then its exact mean
    and covariance are restored. This is a constructive evaluator control,
    not a trained generator or proof of its optimizer's representational reach.
    """
    n = spec["particles"]
    expected = np.asarray(spec["masses"]) * n
    counts = np.floor(expected).astype(int)
    counts[np.argsort(-(expected - counts))[:n - counts.sum()]] += 1
    points = []
    normal = NormalDist()
    for count, mean, covariance in zip(counts, spec["means"], spec["covariances"]):
        if count < 4:
            raise ValueError("the finite-cloud oracle needs at least four atoms per mode")
        bits = int(np.ceil(np.log2(count)))
        reversed_indices = [int(f"{index:0{bits}b}"[::-1], 2) for index in range(count)]
        uniform = np.stack([(np.arange(count) + .5) / count,
                            (np.asarray(reversed_indices) + .5) / 2 ** bits], 1)
        z = np.array([[normal.inv_cdf(float(value)) for value in row] for row in uniform])
        z -= z.mean(0)
        values, vectors = np.linalg.eigh(z.T @ z / count)
        z = z @ (vectors @ np.diag(1 / np.sqrt(values)) @ vectors.T)
        points.append(np.asarray(mean) + z @ np.linalg.cholesky(covariance).T)
    cloud = np.concatenate(points) * vector_tasks.target_scale(spec, spec["steps"])
    # Isolated evaluator sampling from the actual finite population. No best
    # draw/seed selection, learned kernel or artificial output noise is added.
    indices = np.random.default_rng(1931).integers(0, n, size=4096)
    return torch.tensor(cloud[indices], dtype=torch.float32)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def comparison_factors(arms, initializer_receipts):
    if len(arms) != 2 or len(initializer_receipts) != 2:
        raise ValueError("a comparison requires two arms and both initialization receipts")
    left, right = [arm["spec"] for arm in arms]
    changes = {key: [left.get(key), right.get(key)] for key in sorted(set(left) | set(right))
               if left.get(key) != right.get(key)}
    inits = [receipt.get("init_std") for receipt in initializer_receipts]
    missing = any(value is None for value in inits)
    prior_changed = not missing and inits[0] != inits[1]
    return dict(spec_changes=changes, prior_init_std=inits, initializer_unknown=missing,
                prior_initialization_changed=prior_changed,
                critic_components_changed=sorted(ARCHITECTURE_KEYS & changes.keys()),
                law_matched=all(left.get(key) == right.get(key) for key in LAW_KEYS),
                critic_only_identified=bool(not missing and not prior_changed
                                            and changes and set(changes) <= ARCHITECTURE_KEYS),
                individual_critic_mechanism_identified=False)


def score_control(points, spec):
    result = vector_tasks.score_samples(points, spec, spec["steps"])
    required = requirements(spec)
    ks = projection_ks(points, spec, spec["steps"])
    historical = vector_tasks.passes(result, required)
    return dict(passed=historical, projection_ks=ks,
                revised_passed=bool(historical and ks <= PROJECTION_KS_MAX),
                failed_bounds=[dict(metric=key, operator=op, bound=bound, value=result.get(key))
                               for key, op, bound in required
                               if not vector_tasks.passes(result, [(key, op, bound)])],
                metrics={key: result.get(key) for key in sorted({key for key, _, _ in required}
                         | {"min_mass_ratio", "component_mass", "component_resolved",
                            "resolved_core_covariance_error", "resolved_core_min_eigen_ratio",
                            "resolved_max_component_spill"})})


def initializer_receipts(directory):
    receipts = []
    for path in sorted((directory / "proposal").rglob("*.json.gz")):
        payload = json.loads(gzip.decompress(path.read_bytes()))
        if "init_std" in payload and "kind" in payload:
            receipts.append(dict(arm=path.name.removesuffix(".json.gz"), kind=payload["kind"],
                                 init_std=payload["init_std"], recipe_sha256=digest(payload["recipe"]),
                                 source=str(path), artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    # The capture runs the published arm before its MLP control. Name the
    # binding explicitly rather than trusting alphabetical file order.
    receipts.sort(key=lambda row: (row["kind"] == "mlp", row["arm"]))
    return receipts


def rescore_retained(case, records, archive):
    rows = []
    expected = case.get("arms") or [dict(status=case["status"], spec=case["spec"])]
    if len(records) != len(expected):
        raise ValueError(f"{case['id']}: incomplete retained arms")
    for index, (record, original) in enumerate(zip(records, expected)):
        spec = record["spec"]
        if digest(spec) != digest(original["spec"]):
            raise ValueError(f"{case['id']}: retained spec identity differs")
        directory = Path(record["artifact"])
        clouds_path, metrics_path = directory / "observations.npz", directory / "result.json"
        result = json.loads(metrics_path.read_text())
        clouds = np.load(clouds_path, allow_pickle=False)
        observations = result["observations"]
        if len(observations) != len(clouds["live"]) or len(observations) != len(clouds["steps"]):
            raise ValueError("captured clouds and numerical observations must have identical cadence")
        old_passed, revised_passed, curve = [], [], []
        for observation, points, step in zip(observations, clouds["live"], clouds["steps"]):
            if int(step) != observation["step"]:
                raise ValueError("captured cloud step differs from the numerical receipt")
            passed = vector_tasks.passes(observation, requirements(spec))
            ks = projection_ks(points, spec, int(step))
            old_passed.append(passed)
            revised_passed.append(bool(passed and ks <= PROJECTION_KS_MAX))
            curve.append(dict(step=int(step), original_passed=passed, projection_ks=ks,
                              revised_passed=revised_passed[-1]))
        historical = terminal_window(old_passed)
        revised = terminal_window(revised_passed)
        if ("PASS" if historical["passed"] else "FAIL") != original["status"]:
            raise ValueError(f"{case['id']}: historical status does not reproduce")
        curve_path = archive / f"{case['id']}-arm-{index + 1}.json"
        curve_bytes = (json.dumps(curve, indent=2, sort_keys=True) + "\n").encode()
        if curve_path.exists() and curve_path.read_bytes() != curve_bytes:
            raise ValueError(f"refusing to overwrite a different rescore curve: {curve_path}")
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        curve_path.write_bytes(curve_bytes)
        rows.append(dict(arm=index + 1, historical_status=original["status"],
                         revised_status="PASS" if revised["passed"] else "FAIL", convergence=revised,
                         final=curve[-1], observation_count=len(curve),
                         rescore_curve=dict(path=str(curve_path), sha256=hashlib.sha256(curve_bytes).hexdigest()),
                         artifacts={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (metrics_path, clouds_path)}))
    return rows


def build_report(catalog, artifacts, archive):
    cases = []
    references = {row["name"]: row for row in json.loads((artifacts / "develop-frozen-reference/index.json").read_text())}
    for case in catalog["cases"]:
        spec = case.get("spec", {})
        if case["rating"] > 3 or spec.get("kind") != "gaussian_mixture":
            continue
        n = 4096
        target = vector_tasks.sample_target(spec, n, torch.Generator().manual_seed(1931), spec["steps"])
        centers = torch.tensor(spec["means"], dtype=torch.float32) * vector_tasks.target_scale(spec, spec["steps"])
        ids = torch.multinomial(torch.tensor(spec["masses"]), n, True, generator=torch.Generator().manual_seed(991))
        controls = {
            "independent_target_draw": score_control(target, spec),
            "declared_finite_cloud_witness": score_control(finite_cloud_witness(spec), spec),
            "global_mean_point": score_control(target.mean(0, keepdim=True).expand_as(target), spec),
            "correct_centers_and_mass_no_width": score_control(centers[ids], spec),
        }
        if spec["identifiable"]:
            assignment = torch.cdist(target, centers).argmin(1)
            first = target[assignment == 0]
            controls["one_valid_component_only"] = score_control(first[torch.arange(n) % len(first)], spec)
        else:
            # Overlapping-component IDs are latent and cannot be inferred
            # accurately from nearest centers. Its gate is distributional.
            controls["contract"] = "Do not assign observable component labels to overlapping mixtures."
        row = dict(id=case["id"], name=case["name"], rating=case["rating"], historical_status=case["status"],
                   law_sha256=digest({key: spec.get(key) for key in LAW_KEYS}),
                   spec_sha256=digest(spec), required_bounds=requirements(spec), controls=controls,
                   improvement="Per-law oracle and collapse/width/coverage controls, exact factor binding and separate data-law denominator.",
                   source_of_convergence="Unchanged catalog training receipts; oracle controls confer no new convergence credit.")
        if len(case.get("arms", [])) == 2:
            receipts = initializer_receipts(artifacts / case["id"])
            if len(receipts) != 2:
                raise ValueError(f"{case['id']}: missing exact initializer receipts")
            row["comparison"] = comparison_factors(case["arms"], receipts)
            row["initializer_receipts"] = receipts
            row["verifies"] = ("The complete published host fails while its complete MLP/initialization control passes on this law. "
                               "The contrast does not isolate the critic kernel, width, Fourier features or initialization as its cause.")
            records = json.loads((artifacts / case["id"] / "capture-index.json").read_text())
        elif spec["runner"] == "stress":
            row["verifies"] = ("Distribution quality under the declared optimizer/capacity/cadence intervention; "
                               "a robustness diagnostic on an existing ring law, not an independent new data family.")
            records = [references[case["name"]]]
        else:
            row["verifies"] = ("Sensitivity to the declared narrow component scale. This nonblocking diagnostic "
                               "does not establish that a fixed two-band Fourier host can solve the law within its budget.")
            records = [references[case["name"]]]
        row["arms"] = rescore_retained(case, records, archive)
        row["improvement"] += " A separate analytic projection-CDF gate now detects missing within-mode width without component labels."
        cases.append(row)
    return dict(version=VERSION, scope="definition and evaluator calibration; zero training updates",
                historical_verdicts_unchanged=True, cases=cases,
                counts=dict(cases=len(cases), target_laws=len({row["law_sha256"] for row in cases}),
                            controls=sum(sum(isinstance(value, dict) for value in row["controls"].values()) for row in cases),
                            confounded_contrasts=sum(row.get("comparison", {}).get("prior_initialization_changed", False) for row in cases),
                            retained_arms=sum(len(row["arms"]) for row in cases),
                            historical_passes=sum(arm["historical_status"] == "PASS" for row in cases for arm in row["arms"]),
                            revised_passes=sum(arm["revised_status"] == "PASS" for row in cases for arm in row["arms"])),
                added_gate=dict(metric="max_32_projection_ks", bound=PROJECTION_KS_MAX,
                                target="analytic Gaussian-mixture CDF", minimum_stable_checks=5,
                                historical_bounds_retained=True),
                catalog_sha256=digest(catalog), runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("reports/toy_audit/catalog.json"))
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curves", type=Path, required=True, help="External archive for per-checkpoint rescoring streams")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = build_report(json.loads(args.catalog.read_text()), args.artifacts, args.curves)
    root = Path(__file__).resolve().parents[2]
    result["source_sha256"] = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                               for path in ("benchmarks/toy_audit/vector_quality.py", "benchmarks/transfer_suite/vector_tasks.py",
                                            "benchmarks/transfer_suite/protocol.py", "benchmarks/toy_audit/definition_quality.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result["counts"]), flush=True)


if __name__ == "__main__":
    main()

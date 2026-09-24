"""Conditional anchor-region bounds and read-only saved-receipt assessment.

Known host means are used ONLY for this offline assessment. Nothing here is
imported by a trainer or chooses an update. The future-noise/grouping premises
are not established by observing a finite archive.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def region_bound(*, n, k, separation, radius, epsilon, delta, tolerance=0., objective_error=0.):
    """Sufficient bound for finite exact-cost arithmetic (or error <= eta).

    A non-rest selected native proposal is at most tau above the eligible
    fitted proposal in computed cost. If each computed cost differs from its
    mathematical value by at most eta, its true loss is <=2*delta²+tau+2*eta.
    """
    values = (separation, radius, epsilon, delta, tolerance, objective_error)
    if not (type(n) is int and type(k) is int and n > k >= 1
            and all(math.isfinite(x) and x >= 0 for x in values) and separation > 0):
        raise ValueError("finite nonnegative bounds, positive separation and N>K>=1 required")
    matching_margin = separation - (1 + math.sqrt(k)) * (radius + epsilon)
    loss = 2 * delta * delta + tolerance + 2 * objective_error
    precision_radius = epsilon + math.sqrt(n * loss)
    coverage_radius = epsilon + math.sqrt(k * loss)
    return dict(matching_margin=matching_margin, matching_certified=matching_margin > 0,
                moving_loss_bound=loss, precision_radius=precision_radius,
                distinct_anchor_radius=coverage_radius, fitted_radius=epsilon + delta,
                invariant_if_rest_on_ineligible_fit=matching_margin > 0 and precision_radius <= radius,
                all_time_noise_bound_established=False)


def population_hq_lower_bound(radius, *, hq_radius=.21, output_sigma=.029):
    """2-D triangle inequality plus isotropic Gaussian radial tail."""
    if not (0 <= radius < hq_radius and output_sigma > 0):
        return 0.
    return 1 - math.exp(-(hq_radius-radius)**2 / (2 * output_sigma**2))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def assess(summary_path, diagnosis_path):
    import torch
    from scipy.optimize import linear_sum_assignment
    from benchmarks.locked_shared import mode_hold

    summary_raw, diagnosis_raw = summary_path.read_bytes(), diagnosis_path.read_bytes()
    summary, diagnosis = json.loads(summary_raw), json.loads(diagnosis_raw)
    if summary["declaration"]["method"] != "pr84_sample_group_distinct_anchor_joint_output_fit":
        raise ValueError("not the declared sample-anchor saved-state experiment")
    host_source = Path(mode_hold.__file__).read_bytes()
    if _sha(host_source) != summary["declaration"]["sources"]["benchmarks/locked_shared/mode_hold.py"]:
        raise RuntimeError("posthoc host geometry differs from the source-bound experiment")
    if (diagnosis["status"] != "EXACT_REFERENCE_PARITY"
            or diagnosis["selected_states_sha256"] != summary["declaration"]["states_sha256"]):
        raise RuntimeError("initial cloud receipt is not the same captured experiment")
    means = mode_hold.ring_means().double()
    separations = torch.cdist(means, means)
    separations.fill_diagonal_(float("inf"))
    separation = float(separations.min())
    original_pre = {row["step"]: row["stages"]["pre_step"] for row in diagnosis["rows"]}
    initial, rows = [], []
    for branch in summary["branches"]:
        candidate = branch["variants"]["reallocation"]
        points = {row["step"]: row for row in candidate["points"]}
        previous = torch.tensor(original_pre[branch["start"]], dtype=torch.float64)
        initial.append(dict(step=branch["start"], support=previous.tolist(),
                            max_distance=float(torch.cdist(previous, means).min(1).values.max())))
        for correction in candidate["dynamics"]["corrections"]:
            centers = torch.tensor(correction["centers"], dtype=torch.float64)
            a, b = linear_sum_assignment(torch.cdist(centers, means).numpy())
            if len(a) != len(means) or len(centers) != len(means):
                raise RuntimeError("sample groups do not bijectively match the host modes")
            mapping = torch.tensor(b)
            epsilon = float((centers[a] - means[b]).norm(dim=1).max())
            distances = torch.cdist(previous, means)
            nearest = distances.argmin(1)
            pre_radius = float(distances.min(1).values.max())
            covered = len(set(nearest.tolist())) == len(means)
            target = torch.tensor(correction["mm"]["target"], dtype=torch.float64)
            # Convert true labels back to current sample-group indices.
            inverse = torch.empty_like(mapping)
            inverse[mapping] = torch.arange(len(mapping))
            expected_target = centers[inverse[nearest]]
            target_center_error = float((target - expected_target).norm(dim=1).max())
            rounding = float((target.float().double() - target).norm(dim=1).max())
            delta = correction["fit"]["absolute_threshold"] + rounding + target_center_error
            tolerance = 64 * torch.finfo(torch.float64).eps * max(1., correction["pre_cost"])
            current = torch.tensor(points[correction["step"]]["support"], dtype=torch.float64)
            after_distance = torch.cdist(current, means)
            bound = region_bound(n=len(current), k=len(means), separation=separation,
                                 radius=pre_radius, epsilon=epsilon, delta=delta, tolerance=tolerance)
            rows.append(dict(step=correction["step"], epsilon=epsilon, pre_radius=pre_radius,
                pre_all_modes_represented=covered, target_center_error=target_center_error,
                float32_target_rounding=rounding, declared_delta_with_rounding=delta,
                observed_fit_error=correction["fit"]["final_max_row_error"],
                fit_status=correction["fit"]["status"], selected=correction["selected"],
                actual_post_radius=float(after_distance.min(1).values.max()),
                post_all_modes_represented=len(set(after_distance.argmin(1).tolist())) == len(means),
                comparison_tolerance=tolerance, matching_margin=bound["matching_margin"],
                moving_radius_bound=bound["precision_radius"], final_loss=correction["final_cost"]))
            previous = current
    epsilon = max(r["epsilon"] for r in rows)
    delta = max(r["declared_delta_with_rounding"] for r in rows)
    radius = max(r["max_distance"] for r in initial)
    tolerance = max(r["comparison_tolerance"] for r in rows)
    n = len(initial[0]["support"])
    uniform = region_bound(n=n, k=len(means), separation=separation, radius=radius,
                           epsilon=epsilon, delta=delta, tolerance=tolerance)
    return dict(scope="posthoc conditional geometry only; no training/controller modification",
        shared_gate_eligible=False, means_used_only_for_offline_assessment=True,
        summary_sha256=_sha(summary_raw), diagnosis_sha256=_sha(diagnosis_raw),
        host_source_sha256=_sha(host_source), source_sha256=_sha(Path(__file__).read_bytes()),
        n=n, k=len(means), separation=separation, radius=radius, epsilon=epsilon, delta=delta,
        tolerance=tolerance, initial_clouds=initial, uniform_conditional_bound=uniform,
        observed=dict(checks=len(rows), all_fit_converged=all(r["fit_status"] == "CONVERGED" for r in rows),
            selections=sorted(set(r["selected"] for r in rows)),
            all_matching_margins_positive=all(r["matching_margin"] > 0 for r in rows),
            max_target_center_error=max(r["target_center_error"] for r in rows),
            max_post_radius=max(r["actual_post_radius"] for r in rows),
            all_pre_covered=all(r["pre_all_modes_represented"] for r in rows),
            all_post_covered=all(r["post_all_modes_represented"] for r in rows)),
        conditional_population_hq_lower_bound=population_hq_lower_bound(uniform["precision_radius"]),
        finite_noisy_evaluation_guaranteed=False, future_group_or_centroid_bound_proven=False,
        software_rounding_scope="target rounding included; no verified global bound on all objective arithmetic error",
        rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = assess(args.summary, args.diagnosis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in value.items() if k not in ("rows", "initial_clouds")}), flush=True)


if __name__ == "__main__":
    main()

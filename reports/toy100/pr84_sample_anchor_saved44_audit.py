"""Independent sample-anchor selection, field and noisy-grade audit of saved44."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--saved-filter", type=Path, required=True)
    parser.add_argument("--generic-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from benchmarks.locked_shared import mode_hold
    from reports.toy100.sample_group_anchor import field
    from reports.toy100.coverage_fixed_eval import fixed_draw, score_support

    torch.set_num_threads(1)
    summary_bytes = (args.saved_filter / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    generic = json.loads(args.generic_audit.read_text())
    if (summary["status"] != "PASS" or generic["passing"] != 44
            or generic["checks"] != 44 or not generic["all_pass"]):
        raise RuntimeError("complete generic saved44 parity is required")
    for path, expected in summary["declaration"]["sources"].items():
        if sha((args.saved_filter / "source" / path).read_bytes()) != expected:
            raise RuntimeError(f"frozen source differs: {path}")
    means = mode_hold.ring_means()
    rows = []
    for branch in summary["branches"]:
        arm = branch["variants"]["reallocation"]
        dynamics = arm["dynamics"]
        count = branch["end"] - branch["start"] + 1
        if (len(dynamics["corrections"]) != count
                or dynamics["native_batch_checks"] != 3 * count
                or dynamics["correction_rng_checks"] != count
                or dynamics["correction_owner_checks"] != count
                or dynamics["rng_replay_verified"] != 2 * count):
            raise RuntimeError("sampler, owner or replay receipt count differs")
        for point, correction in zip(arm["points"], dynamics["corrections"]):
            step = point["step"]
            centers = torch.tensor(correction["centers"], dtype=torch.float64)
            if len(centers) != 8 or centers.shape[1] != 2:
                raise RuntimeError(f"unexpected inferred groups at {step}")
            grouping = correction["grouping"]
            members = grouping["member_indices"]
            if (grouping["n_groups"] != 8 or len(members) != 8
                    or sorted(index for group in members for index in group) != list(range(128))
                    or grouping["member_sizes"] != [len(group) for group in members]
                    or not grouping["largest_within_edge"] < grouping["cut_threshold"]
                        < grouping["smallest_between_edge"]):
                raise RuntimeError(f"native real128 grouping receipt invalid at {step}")
            target = torch.tensor(correction["mm"]["target"], dtype=torch.float64)
            selected = torch.tensor(point["support"], dtype=torch.float32)
            target_value = field(target, centers)["total"]
            actual_value = field(selected, centers)["total"]
            if (abs(target_value - correction["target_cost"]) > 1e-10
                    or abs(actual_value - correction["final_cost"]) > 1e-10
                    or correction["selected"] != "joint_fit"
                    or correction["fit"]["status"] != "CONVERGED"
                    or correction["final_cost"] > min(correction["pre_cost"],
                                                       correction["native_cost"]) + 1e-10):
                raise RuntimeError(f"selected whole-map loss or fit differs at {step}")
            indices, noise = fixed_draw(step, selected)
            grade = score_support(selected, indices, noise, means)
            if grade != point["grade"] or grade["modes"] != 8 or grade["hq"] < .9:
                raise RuntimeError(f"fixed evaluation grade differs at {step}")
            rows.append(dict(step=step, hq=grade["hq"],
                             n_groups=len(centers), selected=correction["selected"],
                             target_cost=target_value, final_cost=actual_value))
    audit = dict(scope="read-only saved44 selection/field/grade audit",
                 frozen_summary_sha256=sha(summary_bytes),
                 original_control_verified_by_generic_audit=True,
                 native_phase_batch_checks=132,
                 complete_correction_owner_rng_checks=44,
                 inferred_group_count_eight=44,
                 all_joint_fits_selected_and_converged=44,
                 independently_recomputed_target_and_final_anchor_costs=44,
                 independently_recomputed_fixed_noisy_grades=44,
                 min_hq=min(row["hq"] for row in rows),
                 all_moded_eight=True,
                 rows=rows)
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="SAMPLE_ANCHOR_SAVED44_AUDIT", checked=len(rows),
                          min_hq=audit["min_hq"])), flush=True)


if __name__ == "__main__":
    main()

"""Read-only objective/outlier analysis of the exact failed update-1391 sidecar."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.locked_shared import mode_hold
from reports.toy100.chamfer_discrete_reallocation import greedy_real_reallocate, _cost
from reports.toy100.chamfer_pullback import chamfer_targets
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support


def equal(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(
            equal(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return isinstance(second, type(first)) and len(first) == len(second) and all(
            equal(a, b) for a, b in zip(first, second))
    return first == second


def grade(points, means):
    indices, noise = fixed_draw(1391, points.float())
    return score_support(points.float(), indices, noise, means)


def costs(real, points):
    return dict(full128=_cost(real, points), proposal_A64=_cost(real[:64], points),
                validation_B64=_cost(real[64:], points))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    receipt = json.loads((args.capture / "result.json").read_text())
    raw = gzip.decompress((args.capture / "update-1391.pt.gz").read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != receipt["capture_state_raw_sha256"] or receipt["status"] != "EXACT_PREFIX_REPLAY_AND_PASSIVE_CAPTURE":
        raise RuntimeError("exact failed-step capture or source receipt differs")
    saved = torch.load(io.BytesIO(raw), weights_only=True)
    real, means = saved["real"], mode_hold.ring_means()
    if real.shape != (128, 2):
        raise RuntimeError("expected the native D real128 batch")
    stage_names = ("pre_points", "native_points", "allocated_points",
                   "target_points", "selected_points")
    stages = {name: dict(cost=costs(real, saved[name]),
                         fixed_grade=grade(saved[name], means),
                         centers_posthoc=[float(v) for v in torch.cdist(
                             saved[name].float(), means).min(dim=1).values],
                         points=saved[name].tolist())
              for name in stage_names}
    actual = saved["selected_points"].double()
    target = saved["target_points"].double()
    max_target_error = float((actual - target).norm(dim=1).max())
    if abs(stages["selected_points"]["cost"]["full128"] - receipt["objective"]["final"]) > 1e-10:
        raise RuntimeError("actual selected cost does not replay")
    if stages["selected_points"]["fixed_grade"] != receipt["grade"]:
        raise RuntimeError("actual selected noisy grade does not replay")
    allocated_assignment = saved["target_assignment"]
    assigned = {}
    for particle in (3, 5):
        indices = torch.where(allocated_assignment == particle)[0].tolist()
        nearest = int(saved["target_nearest_real"][particle])
        row = dict(real_indices=indices, real_points=real[indices].tolist(),
                   nearest_real_index=nearest, nearest_real_point=real[nearest].tolist(),
                   real_distances_to_nearest_diagnostic_center=[
                       float(torch.cdist(real[i : i + 1], means).min())
                       for i in indices],
                   target_assigned_count=int(saved["target_assigned_counts"][particle]))
        assigned[str(particle)] = row
    # A-only free-output proposal is diagnostic, not an actual neural update.
    train = real[:64]
    allocated_a, allocation_a = greedy_real_reallocate(train, saved["pre_points"])
    target_a, counts_a, _, _ = chamfer_targets(train.double(), allocated_a.double())
    split = dict(scope="free-output A64 proposal only; B64 never enters its target",
                 pre_cost=costs(real, saved["pre_points"]),
                 allocated_A_cost=costs(real, allocated_a),
                 target_A_cost=costs(real, target_a),
                 A_only_relocations=len(allocation_a["moves"]),
                 target_A_counts=counts_a.tolist(),
                 target_A_fixed_grade=grade(target_a, means),
                 strict_A_improves=costs(real, target_a)["proposal_A64"] < costs(real, saved["pre_points"])["proposal_A64"],
                 strict_B_improves=costs(real, target_a)["validation_B64"] < costs(real, saved["pre_points"])["validation_B64"])
    native, selected = saved["post_native"], saved["post_selected"]
    unchanged = {key: equal(native[key], selected[key])
                 for key in ("critic", "optimizer_d", "optimizer_g", "ema_g", "ema_z", "rng", "noise")}
    if not all(unchanged.values()):
        raise RuntimeError("passive capture finds a non-owner state changed by correction")
    result = dict(scope="exact update-1391 read-only attribution; no method training",
                  capture_raw_sha256=digest, native_D_batch=128,
                  fixed_eval_step=1391, stages=stages,
                  max_actual_to_target_output_error=max_target_error,
                  low_count_bad_particle_assignments=assigned,
                  full128_proposal_B64_is_not_held_out=True,
                  split_A_only_free_output=split,
                  post_native_to_selected_unchanged=unchanged,
                  interpretation="full128 fit minimizes sampled C+Q but selected real outliers with one/two assignments; A-only target worsens independent B64 cost, so a strict B check would rest at this frozen step")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="REALLOCATION_1391_ANALYZED",
                          selected_hq=stages["selected_points"]["fixed_grade"]["hq"],
                          target_error=max_target_error,
                          strict_B_improves=split["strict_B_improves"])), flush=True)


if __name__ == "__main__":
    main()

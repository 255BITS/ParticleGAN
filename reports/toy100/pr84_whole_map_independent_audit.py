"""Read-only replay and geometry audit of the frozen round-6 whole-map filter.

The optional relocation calculation is an offline finite search over the same
sampled real minibatch. It changes no saved model, optimizer, or training run.
Target centers are used only for the post-hoc grade and attribution.
"""

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
from reports.toy100.chamfer_discrete_reallocation import greedy_real_reallocate
from reports.toy100.chamfer_pullback import chamfer_targets, chamfer_terms
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
from reports.toy100.pr84_early_geometry import _model


def sha(value):
    return hashlib.sha256(value).hexdigest()


def objective(real, points):
    return float(sum(chamfer_terms(real, points)))


def attribution(real, points, means):
    assignment = torch.cdist(real, points).argmin(dim=1)
    real_mode = torch.cdist(real, means).argmin(dim=1)  # Post hoc only.
    nearest = torch.cdist(points, means)
    rows = []
    for index, point in enumerate(points):
        assigned = real_mode[assignment == index]
        distance, mode = nearest[index].min(dim=0)
        rows.append(dict(particle=index, assigned_real=int(len(assigned)),
                         assigned_real_nearest_mode_counts=torch.bincount(
                             assigned, minlength=len(means)).tolist(),
                         point=point.tolist(), nearest_mode=int(mode),
                         distance_to_nearest_mode=float(distance)))
    return rows


def replay_case(record, state, means):
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        points = _model(state, "g")(state["prior"]["z"]).detach()
    if not torch.equal(points, torch.tensor(record["initial"], dtype=points.dtype)):
        raise RuntimeError(f"{record['label']}: initial support differs")
    stream = torch.Generator().set_state(state["rng"]["data"])
    real = None
    for index, stored in enumerate(record["records"], 1):
        real = mode_hold.sample_ring(means, 128, mode_hold.SIGMA, stream)
        before = objective(real, points)
        target, counts, _, _ = chamfer_targets(real, points)
        points = target.to(points.dtype)
        after = objective(real, points)
        fixed_index, noise = fixed_draw(240 + index, points)
        grade = score_support(points, fixed_index, noise, means)
        if (not torch.equal(points, torch.tensor(stored["points"], dtype=points.dtype))
                or counts.tolist() != stored["counts"]
                or before != stored["before"] or after != stored["after"]
                or grade != stored["grade"]):
            raise RuntimeError(f"{record['label']}: replay differs at update {index}")
    assert real is not None
    c, q = chamfer_terms(real, points)
    result = dict(label=record["label"], exact_real_batches_targets_and_grades=100,
                  final_modes=record["final"]["modes"],
                  final_hq=record["final"]["hq"],
                  final_sampled_c=float(c), final_sampled_q=float(q),
                  assignment_rows=attribution(real, points, means),
                  data_stream_after100_sha256=sha(stream.get_state().numpy().tobytes()))
    if record["label"].startswith("cold"):
        moves = []
        proposal, receipt = greedy_real_reallocate(real, points)
        for move in receipt["moves"]:
            donor, sample = move["donor"], move["real_sample"]
            trial = points.clone()
            trial[donor] = real[sample]
            actual = objective(real, trial)
            if abs(move["objective_after"] - actual) > 1e-5:
                raise RuntimeError("selected relocation does not match direct sampled C+Q")
            fixed_index, noise = fixed_draw(340, trial)
            grade = score_support(trial, fixed_index, noise, means)
            moves.append(dict(move=move["move"], donor=donor, real_sample=sample,
                              real_sample_mode_posthoc=int(torch.cdist(
                                  real[sample : sample + 1], means).argmin(dim=1)),
                              sampled_c_plus_q=actual, decrease=move["decrease"],
                              noisy_modes=grade["modes"], noisy_hq=grade["hq"]))
            points = trial
        if not torch.equal(points, proposal):
            raise RuntimeError("recorded relocation path does not match proposal")
        result["sampled_data_relocations"] = moves
        result["first_noisy_eight_mode_move"] = next(
            (row["move"] for row in moves if row["noisy_modes"] == 8), None)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whole-result", type=Path, required=True)
    parser.add_argument("--cold-capture", type=Path, required=True)
    parser.add_argument("--warm-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result_bytes = args.whole_result.read_bytes()
    frozen = json.loads(result_bytes)
    declaration = frozen["declaration"]
    for path, expected in declaration["source"].items():
        source = ROOT / path
        if not source.exists():
            source = args.whole_result.parent / "source" / path
        if sha(source.read_bytes()) != expected:
            raise RuntimeError(f"source SHA differs: {path}")
    cold_bytes = gzip.decompress(args.cold_capture.read_bytes())
    warm_bytes = args.warm_capture.read_bytes()
    if sha(cold_bytes) != declaration["cold_states_sha256"]:
        raise RuntimeError("cold sidecar SHA differs")
    if sha(warm_bytes) != declaration["warm_states_sha256"]:
        raise RuntimeError("warm sidecar SHA differs")
    cold = torch.load(io.BytesIO(cold_bytes), weights_only=True)["selected"]
    warm = torch.load(io.BytesIO(warm_bytes), weights_only=True)
    states = {"cold1": cold[1]["pre_step"],
              "cold100": cold[100]["post_bounded_g"],
              "warm1324": warm[1324]["pre_step"]}
    means = mode_hold.ring_means()
    cases = [replay_case(case, states[case["label"]], means)
             for case in frozen["results"]]
    audit = dict(scope="read-only sampled C+Q geometry; no neural update",
                 exact_frozen_result_sha256=sha(result_bytes),
                 cold_states_sha256=sha(cold_bytes),
                 warm_states_sha256=sha(warm_bytes),
                 no_oracle_selection=True,
                 relocation_selection="all 12 donors x 128 observed real positions, exact sampled C+Q; strict improvement; no coefficient",
                 caveat="same minibatch offline relocation; no host continuation or latent/network realizability evidence",
                 cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="WHOLE_MAP_INDEPENDENT_AUDIT", cases=[
        dict(label=x["label"], exact=x["exact_real_batches_targets_and_grades"],
             modes=x["final_modes"], hq=x["final_hq"],
             first_eight_mode_move=x.get("first_noisy_eight_mode_move"))
        for x in cases])), flush=True)


if __name__ == "__main__":
    main()

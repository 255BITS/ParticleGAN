"""Read-only independent regrade of the fixed 1200-step split-output runs."""

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
from reports.toy100.chamfer_discrete_reallocation import _cost
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
from reports.toy100.pr84_early_geometry import _model


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cold-capture", type=Path, required=True)
    parser.add_argument("--warm-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    compressed = (args.run / "result.json.gz").read_bytes()
    raw = gzip.decompress(compressed)
    result = json.loads(raw)
    summary = json.loads((args.run / "summary.json").read_text())
    declaration = result["declaration"]
    if (sha(raw) != summary["result_raw_sha256"]
            or summary["case"] != declaration["case"]
            or len(result["outcome"]["records"]) != 1200):
        raise RuntimeError("incomplete result or raw SHA mismatch")
    for path, expected in declaration["source"].items():
        if sha((args.run / "source" / path).read_bytes()) != expected:
            raise RuntimeError(f"frozen source differs: {path}")
    cold_raw = gzip.decompress(args.cold_capture.read_bytes())
    warm_raw = args.warm_capture.read_bytes()
    if (sha(cold_raw) != declaration["cold_states_sha256"]
            or sha(warm_raw) != declaration["warm_states_sha256"]):
        raise RuntimeError("input sidecar SHA differs")
    cold = torch.load(io.BytesIO(cold_raw), weights_only=True)["selected"]
    warm = torch.load(io.BytesIO(warm_raw), weights_only=True)
    state = warm[1324]["pre_step"] if declaration["case"] == "warm1324" else cold[1]["pre_step"]
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        initial = _model(state, "g")(state["prior"]["z"]).detach()
    if not torch.equal(initial, torch.tensor(result["outcome"]["initial_support"])):
        raise RuntimeError("initial clean support differs from saved model")
    means = mode_hold.ring_means()
    stream = torch.Generator().set_state(state["rng"]["data"])
    grades, accepts, failure_rows = [], 0, []
    previous = initial
    for index, row in enumerate(result["outcome"]["records"], 1):
        mode_hold.sample_ring(means, 128, mode_hold.SIGMA, stream)
        points = torch.tensor(row["points"])
        fixed_index, noise = fixed_draw(240 + index, points)
        grade = score_support(points, fixed_index, noise, means)
        if row["step"] != index or grade != row["grade"]:
            raise RuntimeError(f"fixed late-noise grade differs at {index}")
        should_accept = row["target_A"] < row["pre_A"] and row["target_B"] < row["pre_B"]
        if row["accepted"] != should_accept:
            raise RuntimeError(f"split acceptance differs at {index}")
        if not should_accept and not torch.equal(points, previous):
            raise RuntimeError(f"rest changed free outputs at {index}")
        accepts += int(should_accept)
        good = grade["modes"] == 8 and grade["hq"] >= .9
        grades.append(good)
        if not good:
            pre_index, pre_noise = fixed_draw(240 + index, previous)
            same_noise_pre = score_support(previous, pre_index, pre_noise, means)
            failure_rows.append(dict(step=index, accepted=should_accept,
                                     modes=grade["modes"], hq=grade["hq"],
                                     same_noise_pre_hq=same_noise_pre["hq"],
                                     maximum_particle_motion=float((points-previous).norm(dim=1).max()),
                                     A_improvement=row["pre_A"] - row["target_A"],
                                     B_improvement=row["pre_B"] - row["target_B"]))
        previous = points
    stream_sha = sha(stream.get_state().numpy().tobytes())
    outcome = result["outcome"]
    if (stream_sha != outcome["data_stream_after1200_sha256"]
            or accepts != outcome["accepted"]
            or sum(grades) != outcome["passing_checks"]
            or all(grades) != outcome["all1200"]
            or all(grades[-5:]) != outcome["terminal5"]):
        raise RuntimeError("draw-count, acceptance count or gate arithmetic differs")
    first_fail = outcome["first_failure"]
    first_raw = gzip.decompress((args.run / first_fail["path"]).read_bytes())
    if sha(first_raw) != first_fail["raw_sha256"]:
        raise RuntimeError("first failure sidecar SHA differs")
    failure = torch.load(io.BytesIO(first_raw), weights_only=True)
    batch_stream = torch.Generator().set_state(failure["data_rng_before"])
    regenerated = mode_hold.sample_ring(means, 128, mode_hold.SIGMA, batch_stream)
    if not torch.equal(regenerated, failure["real128"]):
        raise RuntimeError("first failure native real batch does not replay")
    selected = failure["selected"]
    first_record = outcome["records"][first_fail["step"] - 1]
    if (not torch.equal(selected, torch.tensor(first_record["points"]))
            or failure["grade"] != first_record["grade"]):
        raise RuntimeError("first failure cloud or fixed grade differs")
    a, b = regenerated[:64], regenerated[64:]
    obj = failure["objective"]
    if any(abs(value - expected) > 1e-12 for value, expected in (
        (_cost(a, failure["before"]), obj["pre_A"]),
        (_cost(b, failure["before"]), obj["pre_B"]),
        (_cost(a, failure["target"]), obj["target_A"]),
        (_cost(b, failure["target"]), obj["target_B"]))):
        raise RuntimeError("first failure split objective does not replay")
    audit = dict(case=declaration["case"], scope="read-only verification of recorded free-output run",
                 raw_result_sha256=sha(raw), source_hashes_verified=len(declaration["source"]),
                 exact_initial_support=True, fixed_grades_verified=1200,
                 strict_acceptance_decisions_verified=1200,
                 real_draw_schedule_and_final_rng_verified=1200,
                 first_fail_batch_and_objectives_exact=True,
                 passing=sum(grades), accepted=accepts,
                 all1200=all(grades), terminal5=all(grades[-5:]),
                 failure_rows=failure_rows)
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="SPLIT_OUTPUT_INDEPENDENT_GRADE", case=audit["case"],
                          passing=audit["passing"], accepted=audit["accepted"],
                          failures=[r["step"] for r in failure_rows])), flush=True)


if __name__ == "__main__":
    main()

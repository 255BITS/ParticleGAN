"""Fixed-budget free-output split-batch C+Q time-stability diagnostic.

Each update samples native ring real128, allocates and makes one quadratic C+Q
target using A64, then accepts that exact output cloud only if both A64 and
untouched B64 strictly improve. No GAN, neural fit, oracle grade, learning-rate
schedule, seed variation, or adaptive filter length enters the controller.
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
from reports.toy100.chamfer_discrete_reallocation import greedy_real_reallocate, _cost
from reports.toy100.chamfer_pullback import chamfer_targets
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
from reports.toy100.pr84_early_geometry import _model


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_archive(output):
    names = ("reports/toy100/pr84_split_output_stability_filter.py",
             "reports/toy100/chamfer_discrete_reallocation.py",
             "reports/toy100/chamfer_pullback.py",
             "reports/toy100/coverage_fixed_eval.py",
             "reports/toy100/pr84_early_geometry.py",
             "benchmarks/locked_shared/mode_hold.py",
             "benchmarks/locked_shared/mlp.py")
    hashes = {}
    for name in names:
        raw = (ROOT / name).read_bytes()
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        hashes[name] = sha(raw)
    return hashes


def initial_support(state):
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        return _model(state, "g")(state["prior"]["z"]).detach()


def save_failure(output, label, step, payload):
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    path = output / "first-fail.pt.gz"
    path.write_bytes(gzip.compress(raw, mtime=0))
    return dict(path=path.name, raw_sha256=sha(raw), step=step, label=label)


def run(state, *, label, output):
    means = mode_hold.ring_means()
    with torch.no_grad():
        points = initial_support(state)
        if points.shape != (12, 2):
            raise RuntimeError("expected native 12x2 clean output cloud")
        stream = torch.Generator().set_state(state["rng"]["data"])
        first_points = points.detach().clone()
        init_index, init_noise = fixed_draw(240, points)
        initial_grade = score_support(points, init_index, init_noise, means)
        rows, accepted = [], 0
        first_fail = None
        for step in range(1, 1201):
            data_rng_before = stream.get_state().clone()
            real = mode_hold.sample_ring(means, 128, mode_hold.SIGMA, stream)
            train, validation = real[:64], real[64:]
            before = points.detach().clone()
            allocated, allocation = greedy_real_reallocate(train, before)
            target, counts, _, _ = chamfer_targets(train.double(), allocated.double())
            target = target.to(points.dtype)
            pre_a, pre_b = _cost(train, before), _cost(validation, before)
            target_a, target_b = _cost(train, target), _cost(validation, target)
            choose = target_a < pre_a and target_b < pre_b
            if choose:
                points = target
                accepted += 1
            index, noise = fixed_draw(240 + step, points)
            grade = score_support(points, index, noise, means)
            good = grade["modes"] == 8 and grade["hq"] >= .9
            row = dict(step=step, accepted=choose, allocation_moves=len(allocation["moves"]),
                       pre_A=pre_a, target_A=target_a, pre_B=pre_b, target_B=target_b,
                       assigned_counts=counts.tolist(), grade=grade,
                       points=points.tolist())
            rows.append(row)
            if not good and first_fail is None:
                first_fail = save_failure(output, label, step,
                    dict(step=step, data_rng_before=data_rng_before, real128=real,
                         before=before, allocated=allocated, target=target,
                         selected=points, allocation=allocation,
                         objective=dict(pre_A=pre_a,target_A=target_a,
                                        pre_B=pre_b,target_B=target_b),
                         grade=grade, assigned_counts=counts))
            if step % 100 == 0:
                print(json.dumps(dict(event="PROGRESS", case=label, step=step,
                                      accepted=accepted, passing=sum(
                                          r["grade"]["modes"] == 8
                                          and r["grade"]["hq"] >= .9 for r in rows),
                                      modes=grade["modes"], hq=grade["hq"])), flush=True)
        passing = [row["grade"]["modes"] == 8 and row["grade"]["hq"] >= .9
                   for row in rows]
        first_pass = next((i + 1 for i, good in enumerate(passing) if good), None)
        late_fail_steps = ([] if first_pass is None else [i + 1 for i, good
                           in enumerate(passing) if i + 1 > first_pass and not good])
        return dict(case=label, initial_support=first_points.tolist(),
                    initial_grade=initial_grade, updates=1200,
                    accepted=accepted, rests=1200-accepted,
                    passing_checks=sum(passing), all1200=all(passing),
                    terminal5=all(passing[-5:]), first_pass=first_pass,
                    first_failure=first_fail, late_fail_count=len(late_fail_steps),
                    first_late_fail=late_fail_steps[0] if late_fail_steps else None,
                    worst_hq=min(row["grade"]["hq"] for row in rows),
                    worst_modes=min(row["grade"]["modes"] for row in rows),
                    final=rows[-1]["grade"],
                    final_support=points.tolist(),
                    data_stream_after1200_sha256=sha(stream.get_state().numpy().tobytes()),
                    records=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("warm1324", "cold1"), required=True)
    parser.add_argument("--cold-capture", type=Path, required=True)
    parser.add_argument("--warm-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cold_raw = gzip.decompress(args.cold_capture.read_bytes())
    warm_raw = args.warm_capture.read_bytes()
    cold_receipt = json.loads((args.cold_capture.parent / "result.json").read_text())
    if (sha(cold_raw) != cold_receipt["state_file_sha256"]
            or sha(warm_raw) != "37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47"):
        raise RuntimeError("frozen input sidecar SHA differs")
    sources = source_archive(args.output)
    declaration = dict(scope="free-output time-stability diagnostic; no neural training",
                       case=args.case, source=sources,
                       cold_states_sha256=sha(cold_raw), warm_states_sha256=sha(warm_raw),
                       updates=1200, native_real_batch=128, split="first64 proposal, last64 untouched validation",
                       proposal="greedy real-row reallocation on A then one fixed-assignment C+Q target",
                       accept="strict decrease in unit-mean C+Q on both A and B; otherwise exact output rest",
                       random_stream="copied saved data stream; only one real128 batch per update",
                       evaluation="every update, fixed 4096 draws with late .029 output noise, diagnostic clock 240+t",
                       warm_gate="8 modes and HQ>=.9 at all1200 updates",
                       cold_gate="8 modes and HQ>=.9 at final five updates; report first pass and subsequent failures",
                       no_oracle_objective=True,
                       limitation="real-only sampled stream and free output points do not replay native GAN or prove neural realizability")
    (args.output / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    cold = torch.load(io.BytesIO(cold_raw), weights_only=True)["selected"]
    warm = torch.load(io.BytesIO(warm_raw), weights_only=True)
    state = warm[1324]["pre_step"] if args.case == "warm1324" else cold[1]["pre_step"]
    outcome = run(state, label=args.case, output=args.output)
    verdict = outcome["all1200"] if args.case == "warm1324" else outcome["terminal5"]
    result = dict(status="PASS" if verdict else "FAIL", declaration=declaration, outcome=outcome)
    raw = json.dumps(result, allow_nan=False).encode()
    (args.output / "result.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    summary = {key:value for key,value in outcome.items()
               if key not in ("records", "initial_support", "final_support")}
    summary["status"] = result["status"]
    summary["result_raw_sha256"] = sha(raw)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="DONE", case=args.case, status=result["status"],
                          passing=outcome["passing_checks"], accepted=outcome["accepted"],
                          first_failure=outcome["first_failure"], final=outcome["final"])), flush=True)


if __name__ == "__main__":
    main()

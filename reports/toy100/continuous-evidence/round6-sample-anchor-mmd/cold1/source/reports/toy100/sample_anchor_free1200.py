"""Fixed-stream free-output stability test of sampled MST anchor MM.

This is an objective/geometry diagnostic, not a GAN update. Starting at the
archived warm-1324 or cold-1 clean support, each update consumes exactly one
native 128-real minibatch, infers its groups, and takes one exact output-space
active-quadratic MM target only if the *whole current-bank objective* decreases.
The evaluation is outside the update: the pinned 4096-draw late-noise grade at
clock 240+t. No target means, configured group count, D, NN, or LR enters it.
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
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
from reports.toy100.pr84_early_geometry import _model
from reports.toy100.sample_group_anchor import field, mst_groups, output_mm_step

SOURCE_NAMES = (
    "reports/toy100/sample_anchor_free1200.py",
    "reports/toy100/sample_group_anchor.py",
    "reports/toy100/coverage_fixed_eval.py",
    "reports/toy100/pr84_early_geometry.py",
    "benchmarks/locked_shared/mode_hold.py",
    "benchmarks/locked_shared/mlp.py",
)
COLD_ARCHIVE = ROOT / "reports/toy100/continuous-evidence/pr84-finite-cold-prefix100"
WARM_ARCHIVE = ROOT / "reports/toy100/continuous-evidence/pr84-stationary-failure-diagnosis"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_states():
    cold_compressed = (COLD_ARCHIVE / "prefix-states.pt.gz").read_bytes()
    cold_manifest = json.loads((COLD_ARCHIVE / "manifest.json").read_text())
    if sha(cold_compressed) != cold_manifest["files"]["prefix-states.pt.gz"]["sha256"]:
        raise RuntimeError("cold archive compressed SHA mismatch")
    cold_raw = gzip.decompress(cold_compressed)
    if sha(cold_raw) != cold_manifest["decompressed_state_sha256"]:
        raise RuntimeError("cold archive raw SHA mismatch")
    warm_compressed = (WARM_ARCHIVE / "compact-states.pt.gz").read_bytes()
    warm_manifest = json.loads((WARM_ARCHIVE / "manifest.json").read_text())
    if sha(warm_compressed) != warm_manifest["files"]["compact-states.pt.gz"]["sha256"]:
        raise RuntimeError("warm archive compressed SHA mismatch")
    cold = torch.load(io.BytesIO(cold_raw), weights_only=True, map_location="cpu")
    warm = torch.load(io.BytesIO(gzip.decompress(warm_compressed)), weights_only=True,
                      map_location="cpu")
    return (cold["selected"][1]["pre_step"], warm[1324]["pre_step"],
            dict(cold_compressed_sha256=sha(cold_compressed), cold_raw_sha256=sha(cold_raw),
                 warm_compressed_sha256=sha(warm_compressed),
                 warm_full_sidecar_sha256=warm_manifest["original_replay_sha256"]["selected-states.pt"]))


def initial_support(state):
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        return _model(state, "g")(state["prior"]["z"]).detach()


def source_archive(output):
    hashes = {}
    for name in SOURCE_NAMES:
        raw = (ROOT / name).read_bytes()
        path = output / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        hashes[name] = sha(raw)
    return hashes


def first_fail(output, step, data_before, real, points_before, centers, proposal, selected,
               grouping, grade):
    payload = dict(step=step, data_rng_before=data_before, real128=real,
                   points_before=points_before, inferred_centers=centers,
                   proposal=proposal, selected=selected, grouping=grouping, grade=grade)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    path = output / "first-fail.pt.gz"
    compressed = gzip.compress(raw, mtime=0)
    path.write_bytes(compressed)
    return dict(step=step, path=path.name, compressed_sha256=sha(compressed),
                raw_sha256=sha(raw), raw_bytes=len(raw))


def run(state, case, output, fixture):
    means = mode_hold.ring_means()
    with torch.no_grad():
        points = initial_support(state).float()
        if points.shape != (12, 2):
            raise RuntimeError("expected native 12x2 clean particle cloud")
        stream = torch.Generator().set_state(state["rng"]["data"])
        data_start_sha = sha(stream.get_state().numpy().tobytes())
        first_points = points.clone()
        idx0, noise0 = fixed_draw(240, points)
        initial_grade = score_support(points, idx0, noise0, means)
        rows = []
        first_failure = None
        group_counts = {}
        accepted = 0
        status = "COMPLETE"
        error = None
        for step in range(1, 1201):
            data_before = stream.get_state().clone()
            real = mode_hold.sample_ring(means, 128, mode_hold.SIGMA, stream)
            centers, grouping = mst_groups(real)
            k = len(centers)
            group_counts[str(k)] = group_counts.get(str(k), 0) + 1
            if k >= len(points):
                status = "ERROR"
                error = dict(step=step, reason="inferred groups >= free particles",
                             inferred_groups=k, particles=len(points),
                             real_bank_sha256=sha(real.contiguous().numpy().tobytes()),
                             data_rng_before_sha256=sha(data_before.numpy().tobytes()))
                break
            before_points = points.clone()
            with torch.enable_grad():
                mm = output_mm_step(points.double(), centers)
                objective_before = field(points, centers)["total"]
                proposal = torch.tensor(mm["target"], dtype=points.dtype)
                objective_proposal = field(proposal, centers)["total"]
            epsilon = 64 * torch.finfo(torch.float64).eps * max(1., abs(objective_before))
            choose = objective_proposal < objective_before - epsilon
            if choose:
                points = proposal
                accepted += 1
            index, noise = fixed_draw(240 + step, points)
            grade = score_support(points, index, noise, means)
            good = grade["modes"] == 8 and grade["hq"] >= .9
            if not good and first_failure is None:
                first_failure = first_fail(output, step, data_before, real, before_points,
                                           centers, proposal, points, grouping, grade)
            rows.append(dict(step=step, groups=k, member_sizes=grouping["member_sizes"],
                             bank_sha256=sha(real.contiguous().numpy().tobytes()),
                             objective_before=objective_before,
                             objective_mm_double=mm["after"],
                             objective_proposal=objective_proposal,
                             accepted=choose, max_output_displacement=(
                                 float((proposal-before_points).norm(dim=1).max()) if choose else 0.),
                             grade=grade, points=points.tolist()))
            if step % 100 == 0:
                print(json.dumps(dict(event="PROGRESS", case=case, step=step,
                                      accepted=accepted, passing=sum(
                                          r["grade"]["modes"] == 8 and r["grade"]["hq"] >= .9
                                          for r in rows), modes=grade["modes"], hq=grade["hq"])),
                      flush=True)
        passing = [r["grade"]["modes"] == 8 and r["grade"]["hq"] >= .9 for r in rows]
        first_pass = next((i+1 for i, good in enumerate(passing) if good), None)
        late_fails = ([] if first_pass is None else [i+1 for i, good in enumerate(passing)
                       if i+1 > first_pass and not good])
        summary = dict(case=case, status=("ERROR" if status == "ERROR" else
                       "PASS" if (all(passing) if case == "warm1324" else all(passing[-5:]))
                       else "FAIL"), error=error,
                       initial_support=first_points.tolist(), initial_grade=initial_grade,
                       completed_updates=len(rows), accepted=accepted, rests=len(rows)-accepted,
                       passing_checks=sum(passing), all1200=len(rows) == 1200 and all(passing),
                       terminal5=len(rows) == 1200 and all(passing[-5:]),
                       first_pass=first_pass, first_failure=first_failure,
                       late_fail_count=len(late_fails), first_late_fail=(late_fails[0] if late_fails else None),
                       worst_hq=(min(r["grade"]["hq"] for r in rows) if rows else None),
                       worst_modes=(min(r["grade"]["modes"] for r in rows) if rows else None),
                       final=(rows[-1]["grade"] if rows else None), final_support=points.tolist(),
                       group_counts=group_counts, data_stream_start_sha256=data_start_sha,
                       data_stream_after_sha256=sha(stream.get_state().numpy().tobytes()),
                       records=rows)
    result = dict(declaration=fixture, outcome=summary)
    result_raw = canonical(result)
    result_path = output / "result.json.gz"
    compressed = gzip.compress(result_raw, mtime=0)
    result_path.write_bytes(compressed)
    short = {key:value for key,value in summary.items()
             if key not in ("records", "initial_support", "final_support")}
    short.update(result_raw_sha256=sha(result_raw), result_raw_bytes=len(result_raw),
                 result_compressed_sha256=sha(compressed), result_compressed_bytes=len(compressed))
    (output / "summary.json").write_text(json.dumps(short, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="DONE", case=case, status=summary["status"],
                          passing=summary["passing_checks"], accepted=accepted,
                          first_failure=first_failure, final=summary["final"])), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("warm1324", "cold1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cold, warm, input_hashes = load_states()
    source = source_archive(args.output)
    fixture = dict(scope="free-output sampled-anchor stochastic time-stability diagnostic; no neural training",
                   case=args.case, input_hashes=input_hashes, source_sha256=source,
                   initial_state=("warm1324 pre_step" if args.case == "warm1324" else "cold1 pre_step"),
                   data="same saved data RNG; one native real128 draw per update",
                   algorithm="infer MST groups each bank; one exact output-space active-quadratic MM target",
                   objective="unit-mean distinct group-anchor squared distance plus nearest-group precision squared distance",
                   accept="whole current-bank objective strictly descends beyond float64 rounding tolerance, else exact rest",
                   unsupported="inferred groups >= 12 particles -> ERROR and stop",
                   updates=1200, evaluation="every update fixed 4096 draws, late output sigma .029, clock 240+t",
                   warm_gate="8 modes and HQ>=.9 at every update",
                   cold_gate="8 modes and HQ>=.9 at final five updates",
                   no_oracle_control=True,
                   limitations=["free outputs, no G/prior Jacobian or GAN", "MST group count may vary across batches",
                                "objective changes the frozen game", "fixed bank MM proof does not imply stochastic stability"])
    (args.output / "declaration.json").write_text(json.dumps(fixture, indent=2) + "\n")
    run(warm if args.case == "warm1324" else cold, args.case, args.output, fixture)


if __name__ == "__main__":
    main()

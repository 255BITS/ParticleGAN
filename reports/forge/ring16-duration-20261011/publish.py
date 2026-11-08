"""Verify saved continuation artifacts and publish compact metrics and GIFs.

This script adds no training updates or model sampling draws.
"""
import argparse
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from run import load_protocol, sha, write


def suffix(curve):
    count = 0
    for point in reversed(curve):
        if not point["passed"]:
            break
        count += 1
    return count


def publish(raw, destination):
    protocol = load_protocol()
    assert json.loads((raw / "frozen-protocol.json").read_text()) == protocol
    sys.path.insert(0, protocol["parent_snapshot"])
    import numpy as np
    import torch
    from benchmarks.toy_audit.api_run import render_gif
    from benchmarks.toy_audit.api_vectors import _bounds
    from benchmarks.toy_audit.ring16_quality import score_samples
    from benchmarks.transfer_suite.vector_tasks import sample_target
    from experiments.forge.state import require_same_formulation, require_optimizer_steps, state_digest
    torch.set_num_threads(1)
    destination.mkdir(parents=True, exist_ok=True)
    events = [json.loads(line) for line in (raw/"progress.jsonl").read_text().splitlines()]
    process_seconds = {}
    for row in protocol["runs"]:
        family = row["family"]
        start = next(e for e in events if e["family"] == family and e["event"] == "start")
        finish = next(e for e in events if e["family"] == family and e["event"] == "finished")
        assert finish["returncode"] == 0
        process_seconds[family] = finish["time"] - start["time"]
    results = []
    for row in protocol["runs"]:
        family = row["family"]
        directory = raw / family
        receipt = json.loads((directory / "receipt.json").read_text())
        assert receipt["protocol_sha256"] == sha(HERE / "protocol.json")
        assert receipt["runner_sha256"] == sha(HERE / "execute.py")
        for name, digest in receipt["artifacts"].items():
            assert sha(directory / name) == digest
        request = json.loads((Path(protocol["parent_archive"]) / "certificates" / row["attempt_id"] / "request.json").read_text())["request"]
        task = request["tasks"]["ring16_acquisition"]
        spec = {**task["execution"]["host_definition"], "thresholds": task["evaluation"]["thresholds"]}
        assert spec["thresholds"] == protocol["evaluation"]["thresholds"]
        records = torch.load(directory / "observations.pt", weights_only=True, map_location="cpu")
        curve = json.loads((directory / "curve.json").read_text())
        assert len(records) == len(curve) == 384
        assert [r["step"] for r in records] == [p["step"] for p in curve]
        assert all(p["passed"] == (not _bounds(p["metrics"], spec["thresholds"])) for p in curve)
        # Check saved scored samples at budget cuts and all final five observations.
        checks = {1600, 3200, 4800, 6400} | {p["step"] for p in curve[-5:]}
        recomputations = 0
        for observed, point in zip(records, curve):
            if point["step"] not in checks:
                continue
            metrics = score_samples(observed["samples"], spec, point["step"])
            for name, _, _ in spec["thresholds"]:
                assert math.isclose(metrics[name], point["metrics"][name], rel_tol=0, abs_tol=1e-7)
            recomputations += 1
        parent_dir = Path(protocol["parent_archive"]) / "queue/recipe-prior-refactor-selected-leaders-v1" / row["attempt_id"]
        parent = torch.load(parent_dir / "provenance/provenance-state.pt", weights_only=True, map_location="cpu")
        final = torch.load(directory / "state.pt", weights_only=True, map_location="cpu")
        assert state_digest(parent) == receipt["resume_proof"]["original_state_sha256"]
        require_same_formulation(parent, final)
        require_optimizer_steps(final, 6400)
        assert final["trainer"]["completed_steps"] == final["trainer"]["max_steps"] == 6400
        assert final["prior_contract"] == parent["prior_contract"]
        cuts = []
        for cap in (1600, 3200, 4800, 6400):
            prefix = [p for p in curve if p["step"] <= cap]
            point = prefix[-1]
            cuts.append(dict(updates=cap, terminal_passing_observations=suffix(prefix),
                             verdict="PASS" if suffix(prefix) >= 5 else "FAIL",
                             failed_bounds=_bounds(point["metrics"], spec["thresholds"]),
                             metrics={k:v for k,v in point["metrics"].items() if isinstance(v,(int,float))}))
        first = None
        longest = streak = 0
        for p in curve:
            streak = streak + 1 if p["passed"] else 0
            longest = max(longest, streak)
            if streak == 5 and first is None:
                first = p["step"]
        reference = sample_target(spec, 4096, torch.Generator().manual_seed(78013), 0)
        frames = []
        for index in np.linspace(0,len(records)-1,12).round().astype(int):
            observed, point = records[index], curve[index]
            view = dict(kind="scatter", title="Target and actual sixteen-Gaussian ring",
                        target=reference, samples=observed["samples"], xlim=[-4,4], ylim=[-4,4],
                        xlabel="x", ylabel="y", caption="Exact archived checkpoint continuation; clean live MoG/G samples. Schedule horizon stays 400.")
            frames.append(dict(step=point["step"], views=[view], passed=point["passed"],
                               metrics={k:point["metrics"][k] for k in ["modes","hq","mass_tv","component_covariance_error","component_min_eigen_ratio"]}))
        gif = family + ".gif"
        verdict = cuts[-1]["verdict"]
        render_gif(dict(id=family+"-ring16-duration", goal="Does unchanged training through 6400 updates acquire sustained full ring quality?",default_steps=6400),
                   frames,destination/gif,full_budget=True,requested_steps=6400,final_verdict=verdict)
        result = {**receipt, "process_wall_seconds":process_seconds[family],
                  "raw_receipt_sha256":sha(directory/"receipt.json"),
                  "verdict":verdict,"budget_cuts":cuts,"first_five_passing_checks_update":first,
                  "longest_passing_streak":longest,"passing_observations":sum(p["passed"] for p in curve),
                  "final_state_sha256":state_digest(final),"saved_sample_gate_recomputations":recomputations,
                  "final_failed_bounds":cuts[-1]["failed_bounds"],"gif":gif,"gif_sha256":sha(destination/gif)}
        results.append(result)
    assert len({r["new_data_sequence_sha256"] for r in results}) == 1
    report = dict(schema_version=1, scope=protocol["scope"], qualification_input=False,
                  original_grades_preserved=True, continuations=3, additional_updates=14400,
                  continuation_loop_seconds=sum(r["continuation_loop_seconds"] for r in results),
                  new_paid_process_seconds=sum(process_seconds.values()),
                  matched_continuation_batches=True, protocol_sha256=sha(HERE/"protocol.json"),
                  runner_sha256=sha(HERE/"execute.py"), execution_runner_source="execute.py",
                  publisher_sha256=sha(Path(__file__)),
                  budget=protocol["budget"], raw_artifact_root=str(raw.resolve()),
                  runs=results)
    write(destination/"results.json", report)
    print(json.dumps(dict(results=[dict(family=r["family"],verdict=r["verdict"],cuts=r["budget_cuts"],first_stable=r["first_five_passing_checks_update"],longest=r["longest_passing_streak"]) for r in results],seconds=report["continuation_loop_seconds"]),indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,default=ROOT/"runs/api/ring16-duration-20261011")
    parser.add_argument("--output",type=Path,default=HERE)
    args=parser.parse_args()
    publish(args.raw,args.output)

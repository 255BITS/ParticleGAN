"""Independent read-only grade of the frozen 44-update virtual-D saved filter."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100.pr84_prediction_state_filter import state_hash


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-filter", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary_bytes = (args.saved_filter / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    declaration = summary["declaration"]
    states_file = args.capture / "selected-states.pt"
    diagnosis_file = args.capture / "diagnosis.json"
    states_bytes, diagnosis_bytes = states_file.read_bytes(), diagnosis_file.read_bytes()
    if (sha(states_bytes) != declaration["states_sha256"]
            or sha(diagnosis_bytes) != declaration["capture_sha256"]):
        raise RuntimeError("input capture SHA differs from declaration")
    states = torch.load(states_file, weights_only=True)
    diagnosis = json.loads(diagnosis_bytes)
    if diagnosis["status"] != "EXACT_REFERENCE_PARITY":
        raise RuntimeError("original long-hold capture lacks reference parity")
    captured = {row["step"]: row for row in diagnosis["rows"]}
    for path, expected in declaration["source"].items():
        raw = (args.saved_filter / "source" / path).read_bytes()
        if sha(raw) != expected:
            raise RuntimeError(f"frozen filter source SHA differs: {path}")
    if not summary["original_all_exact"]:
        raise RuntimeError("producer did not assert original parity")
    rows = []
    full_capture_checks = 0
    ordinary_support_checks = ordinary_record_checks = 0
    for branch in summary["branches"]:
        start, end = branch["start"], branch["end"]
        variants = branch["variants"]
        count = end - start + 1
        for name in ("original", "unrolled"):
            raw = json.loads((args.saved_filter / f"{name}-{start}-{end}.json").read_text())
            if raw != variants[name]:
                raise RuntimeError(f"{name} branch file differs from embedded summary")
            if len(raw["points"]) != count or len(raw["accepted_states"]) != count:
                raise RuntimeError("incomplete branch chronology")
            expected_rates = {(step, "d", (0.00425,)) for step in range(start, end + 1)}
            expected_rates |= {(step, "g_prior", (0.00425, 0.0085))
                               for step in range(start, end + 1)}
            actual_rates = {(row["step"], row["role"], tuple(row["rates"]))
                            for row in raw["rates"]}
            if actual_rates != expected_rates:
                raise RuntimeError("a role rate differs from constant declared rates")
            if raw["moment_steps"] != {"d": [end], "g": [end]}:
                raise RuntimeError("one moment update per role/step not reflected")
        original, unrolled = variants["original"], variants["unrolled"]
        if (original["rng_final_sha256"] != unrolled["rng_final_sha256"]
                or original["noise"] != unrolled["noise"]):
            raise RuntimeError("counterfactual RNG/noise endpoint differs")
        for index, point in enumerate(original["points"]):
            step = start + index
            if point["step"] != step or point["support"] != captured[step]["stages"]["bounded_joint"]:
                raise RuntimeError(f"ordinary support differs at {step}")
            ordinary_support_checks += 1
            recorded = original["dynamics"]["records"][index]
            if recorded != dict(captured[step]["stages"]["record"], outer_step=index + 1):
                raise RuntimeError(f"ordinary curvature receipt differs at {step}")
            ordinary_record_checks += 1
        for row in original["accepted_states"]:
            if row["step"] in states:
                expected = state_hash(states[row["step"]]["post_bounded_g"])
                if row["accepted_state_sha256"] != expected:
                    raise RuntimeError(f"ordinary captured full state differs at {row['step']}")
                full_capture_checks += 1
        grades = unrolled["local_gate"]
        recomputed = [p["grade"]["modes"] == 8 and p["grade"]["hq"] >= .9
                      for p in unrolled["points"]]
        if (grades["checks"] != count or grades["passing_checks"] != sum(recomputed)
                or grades["pass_all"] != all(recomputed)):
            raise RuntimeError("candidate gate arithmetic differs")
        rows.append(dict(start=start, end=end, checks=count,
                         candidate_passing=sum(recomputed),
                         candidate_failing_steps=grades["failing_steps"],
                         candidate_min_hq=grades["min_hq"],
                         candidate_min_modes=grades["min_modes"]))
    result = dict(scope="saved-state local diagnostic; no cold/warm promotion",
                  summary_sha256=sha(summary_bytes),
                  captured_state_hashes_verified=full_capture_checks,
                  ordinary_support_arrays_verified=ordinary_support_checks,
                  ordinary_update_records_verified=ordinary_record_checks,
                  all_constant_role_rates=True, same_rng_noise_endpoints=True,
                  candidate_total_passing=sum(row["candidate_passing"] for row in rows),
                  candidate_total_checks=sum(row["checks"] for row in rows),
                  local_gate_pass=all(row["candidate_passing"] == row["checks"] for row in rows),
                  branches=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

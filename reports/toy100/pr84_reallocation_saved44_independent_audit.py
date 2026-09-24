"""Independent exact grade of the frozen PR84 reallocation saved-state filter."""

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
    if sha(states_file.read_bytes()) != declaration["states_sha256"]:
        raise RuntimeError("original state sidecar hash differs")
    states = torch.load(states_file, weights_only=True)
    diagnosis = json.loads(diagnosis_file.read_text())
    if diagnosis["status"] != "EXACT_REFERENCE_PARITY":
        raise RuntimeError("original observer lacks exact parity")
    reference = {row["step"]: row for row in diagnosis["rows"]}
    for path, expected in declaration["sources"].items():
        source = args.saved_filter / "source" / path
        if sha(source.read_bytes()) != expected:
            raise RuntimeError(f"frozen source differs: {path}")
    rows, full_capture_checks = [], 0
    for branch in summary["branches"]:
        start, end = branch["start"], branch["end"]
        count = end - start + 1
        original, candidate = (branch["variants"][name]
                               for name in ("original", "reallocation"))
        for name, arm in (("original", original), ("reallocation", candidate)):
            if (len(arm["points"]) != count or len(arm["accepted_states"]) != count
                    or len(arm["dynamics"]["records"]) != count):
                raise RuntimeError(f"{name} incomplete at {start}")
            raw = json.loads((args.saved_filter / f"{name}-{start}-{end}.json").read_text())
            if raw != arm:
                raise RuntimeError(f"{name} branch differs from embedded summary")
            expected_rates = [(step, role, rates)
                              for step in range(start, end + 1)
                              for role, rates in (("d", [0.00425]),
                                                  ("g_prior", [0.00425, 0.0085]))]
            actual_rates = [(r["step"], r["role"], r["rates"]) for r in arm["rates"]]
            if actual_rates != expected_rates:
                raise RuntimeError(f"{name} nominal rate trace differs")
            if arm["moment_steps"] != {"d": [end], "g": [end]}:
                raise RuntimeError(f"{name} moment counter differs")
        for index, point in enumerate(original["points"]):
            step = start + index
            if point["support"] != reference[step]["stages"]["bounded_joint"]:
                raise RuntimeError(f"original support differs at {step}")
            if original["dynamics"]["records"][index] != dict(
                    reference[step]["stages"]["record"], outer_step=index + 1):
                raise RuntimeError(f"original update record differs at {step}")
        for row in original["accepted_states"]:
            if row["step"] in states:
                if row["accepted_state_sha256"] != state_hash(states[row["step"]]["post_bounded_g"]):
                    raise RuntimeError(f"captured full state differs at {row['step']}")
                full_capture_checks += 1
        if (original["rng_final_sha256"] != candidate["rng_final_sha256"]
                or original["noise"] != candidate["noise"]):
            raise RuntimeError("original and candidate RNG/noise endpoint differs")
        corrections = candidate["dynamics"]["corrections"]
        if len(corrections) != count or any(row["step"] != start + i
                                            for i, row in enumerate(corrections)):
            raise RuntimeError("correction chronology differs")
        if any(row["final_cost"] > row["pre_cost"] + 1e-10
               for row in corrections):
            raise RuntimeError("a selected whole-map update increased its native batch cost")
        checks = [row["grade"]["modes"] == 8 and row["grade"]["hq"] >= .9
                  for row in candidate["points"]]
        grade = candidate["local_gate"]
        if (grade["checks"] != count or grade["passing_checks"] != sum(checks)
                or grade["failing_steps"] != [start + i for i, good in enumerate(checks)
                                                 if not good]):
            raise RuntimeError("candidate grade arithmetic differs")
        rows.append(dict(start=start, end=end, passing=sum(checks), checks=count,
                         failing=grade["failing_steps"], min_hq=grade["min_hq"],
                         selections={name:sum(row["selected"] == name for row in corrections)
                                     for name in ("joint_fit", "native_gan", "rest")}))
    result = dict(scope="saved-state local filter only; no warm/cold promotion",
                  frozen_summary_sha256=sha(summary_bytes),
                  original_full_captured_state_hashes_verified=full_capture_checks,
                  original_supports_and_records_verified=44,
                  all_constant_role_rates=True, one_moment_update_per_role_per_step=True,
                  same_original_candidate_rng_noise_endpoints=True,
                  passing=sum(row["passing"] for row in rows),
                  checks=sum(row["checks"] for row in rows),
                  all_pass=all(row["passing"] == row["checks"] for row in rows),
                  branches=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="REALLOCATION_SAVED44_AUDIT",
                          passing=result["passing"], checks=result["checks"],
                          captured_states=full_capture_checks)), flush=True)


if __name__ == "__main__":
    main()

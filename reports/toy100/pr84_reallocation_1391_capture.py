"""Passive exact saved-state replay/capture of the failed reallocation update 1391.

This runs only updates 1380..1391 on the frozen original and candidate arms.
It saves the pre-step state, actual native real batch, free-output target and
post-native/post-selected states without altering either policy's update.
"""

from contextlib import contextmanager
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--saved-filter", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from reports.toy100 import pr84_prediction_state_filter as replay
    from reports.toy100 import reallocation_state_filter as frozen_filter
    from reports.toy100 import reallocation_smoothed_candidate as candidate
    from reports.toy100.chamfer_discrete_reallocation import greedy_real_reallocate, _cost
    from reports.toy100.chamfer_pullback import chamfer_targets

    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    frozen_bytes = (args.saved_filter / "summary.json").read_bytes()
    frozen = json.loads(frozen_bytes)
    declaration = frozen["declaration"]
    for path, expected in declaration["sources"].items():
        if sha((root / path).read_bytes()) != expected:
            raise RuntimeError(f"root source differs from frozen run: {path}")
    states_file = args.capture / "selected-states.pt"
    diagnosis_file = args.capture / "diagnosis.json"
    state_bytes, diagnosis_bytes = states_file.read_bytes(), diagnosis_file.read_bytes()
    if (sha(state_bytes) != declaration["states_sha256"]
            or sha(state_bytes) != frozen_filter.EXPECTED_CAPTURE):
        raise RuntimeError("source warm state differs")
    diagnosis = json.loads(diagnosis_bytes)
    if (diagnosis["status"] != "EXACT_REFERENCE_PARITY"
            or diagnosis["selected_states_sha256"] != declaration["states_sha256"]):
        raise RuntimeError("original capture lacks exact parity")
    states = torch.load(io.BytesIO(state_bytes), weights_only=True)
    reference = {row["step"]: row for row in diagnosis["rows"]}
    config = json.loads((root / "configs/toy100/constraints_simple_regularization.json").read_text())
    branch = next(row for row in frozen["branches"] if row["start"] == 1380)
    captured = {}
    source_dir = args.output / "source"
    source_dir.mkdir()
    for path in declaration["sources"]:
        dest = source_dir / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((root / path).read_bytes())

    @contextmanager
    def selected(*, task, prediction):
        with candidate.reallocation_smoothed_candidate(task=task, correction=prediction) as value:
            recorder, _ = value
            if prediction:
                ordinary_phases = recorder.phases

                def observed_phases(step, opt_d, opt_g, local):
                    if step + 1 == 1391:
                        captured["pre_step"] = replay.snapshot(local)
                    for phase in ordinary_phases(step, opt_d, opt_g, local):
                        yield phase

                recorder.phases = observed_phases
                ordinary_correct = recorder.correct

                def observed_correct(optimizer):
                    local = recorder._local
                    if local["step"] + 1 == 1391:
                        with torch.no_grad():
                            clean = getattr(local["generator"], "model", local["generator"])
                            real = recorder.real.detach().clone()
                            pre = recorder.pre_points.detach().clone()
                            native = clean(local["prior"].z).detach().clone()
                            allocated, allocation = greedy_real_reallocate(real, pre)
                            target, counts, assignment, nearest_real = chamfer_targets(
                                real.double(), allocated.double())
                        captured.update(real=real, pre_points=pre, native_points=native,
                                        allocated_points=allocated, target_points=target,
                                        target_assigned_counts=counts,
                                        target_assignment=assignment,
                                        target_nearest_real=nearest_real,
                                        allocation=allocation,
                                        post_native=replay.snapshot(local))
                    result = ordinary_correct(optimizer)
                    if local["step"] + 1 == 1391:
                        with torch.no_grad():
                            clean = getattr(local["generator"], "model", local["generator"])
                            captured["selected_points"] = clean(local["prior"].z).detach().clone()
                        captured["post_selected"] = replay.snapshot(local)
                    return result

                recorder.correct = observed_correct
            yield value

    local_results = {}
    with patch.object(replay.prediction_module, "pr84_opponent_prediction", selected):
        for name, switch in (("original", "current"), ("reallocation", "predicted")):
            value, _ = replay.run_local(config, states[1380]["pre_step"],
                                        start=1380, end=1391, opponent=switch,
                                        source_dir=source_dir)
            archived = branch["variants"][name]
            for key in ("points", "accepted_states", "rates"):
                expected = archived[key][:24 if key == "rates" else 12]
                if value[key] != expected:
                    pairs = list(zip(value[key], expected))
                    first = next(((index, a, b) for index, (a, b) in enumerate(pairs)
                                  if a != b), None)
                    raise RuntimeError(f"{name} {key} differs from archived prefix: "
                                       f"lengths {len(value[key])}/{len(expected)}, "
                                       f"first {first}")
            if value["dynamics"]["records"] != archived["dynamics"]["records"][:12]:
                raise RuntimeError(f"{name} update records differ from archived prefix")
            if name == "original":
                for index, point in enumerate(value["points"]):
                    step = 1380 + index
                    if (point["support"] != reference[step]["stages"]["bounded_joint"]
                            or value["dynamics"]["records"][index]
                            != dict(reference[step]["stages"]["record"], outer_step=index + 1)):
                        raise RuntimeError(f"original host reference differs at {step}")
            else:
                if value["dynamics"]["corrections"] != archived["dynamics"]["corrections"][:12]:
                    raise RuntimeError("candidate correction records differ from archived prefix")
                if replay.state_hash(captured["post_selected"]) != value["accepted_states"][-1]["accepted_state_sha256"]:
                    raise RuntimeError("captured post-selected full state differs from exact replay")
            local_results[name] = value
            payload = json.dumps(value, allow_nan=False).encode()
            (args.output / f"{name}-1380-1391.json.gz").write_bytes(gzip.compress(payload, mtime=0))
            print(json.dumps(dict(event="BRANCH_PREFIX_EXACT", variant=name,
                                  checks=len(value["points"]), end=1391)), flush=True)
    if (local_results["original"]["rng_final_sha256"]
            != local_results["reallocation"]["rng_final_sha256"]
            or local_results["original"]["noise"] != local_results["reallocation"]["noise"]):
        raise RuntimeError("original/candidate RNG or noise differs")
    correction = local_results["reallocation"]["dynamics"]["corrections"][-1]
    if (correction["step"] != 1391
            or correction["allocation"] != captured["allocation"]
            or correction["target_assigned_counts"] != captured["target_assigned_counts"].tolist()
            or abs(correction["target_cost"] - _cost(captured["real"], captured["target_points"])) > 1e-10):
        raise RuntimeError("captured allocation/target differs from candidate receipt")
    buffer = io.BytesIO()
    torch.save(captured, buffer)
    state_raw = buffer.getvalue()
    (args.output / "update-1391.pt.gz").write_bytes(gzip.compress(state_raw, mtime=0))
    result = dict(status="EXACT_PREFIX_REPLAY_AND_PASSIVE_CAPTURE",
                  scope="updates1380..1391 only; no full warm/cold training",
                  frozen_summary_sha256=sha(frozen_bytes),
                  source_sha256=declaration["sources"],
                  original_reference_supports_and_records=12,
                  original_archived_prefix_records_and_states=12,
                  candidate_archived_prefix_records_and_states_and_corrections=12,
                  same_candidate_original_rng_noise_endpoint=True,
                  capture_state_raw_sha256=sha(state_raw),
                  update=1391, grade=local_results["reallocation"]["points"][-1]["grade"],
                  selected=correction["selected"],
                  objective=dict(pre=correction["pre_cost"],native=correction["native_cost"],
                                 target=correction["target_cost"],fitted=correction["fitted_cost"],
                                 final=correction["final_cost"]),
                  capture_stages=["pre_step", "post_native", "post_selected"],
                  data="actual native D real128, pre/native/allocated/target/selected clean clouds")
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="REALLOCATION_1391_CAPTURE", status=result["status"],
                          grade=result["grade"], objective=result["objective"])), flush=True)


if __name__ == "__main__":
    main()

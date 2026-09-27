"""Compare the packaged-hook replay against the qualified frozen run.

Usage: python audit_package_parity.py REPLAY_ROOT REFERENCE_ROOT OUTPUT_JSON
Both roots contain jobs.json, manifest.json, and runs/<job-id>/result.json.
Large checkpoints remain external; this writes a compact, inspectable receipt.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a, b)
        return 1
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return sum(equal(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        return sum(equal(x, y) for x, y in zip(a, b))
    assert a == b, (a, b)
    return 0


def audit(root, reference):
    rows = []
    for job in json.loads((root / "jobs.json").read_text())["jobs"]:
        current = root / "runs" / job["id"]
        original = reference / "runs" / job["id"]
        a, b = [json.loads((directory / "result.json").read_text())
                for directory in (current, original)]
        assert a["status"] == b["status"], job["id"]
        row = dict(id=job["id"], status=a["status"])
        for key in ("worker_sha256", "driver_sha256", "candidate_hashes"):
            if key in a:
                assert a[key] == b[key], (job["id"], key)
        if "randomness" in a:
            assert a["randomness"]["sha256"] == b["randomness"]["sha256"]
            row["full_rng_digest_equal"] = True
        if job["kind"] == "toy":
            row["equal_initial_tensors"] = equal(*[
                torch.load(p / "initial-values.pt", map_location="cpu", weights_only=True)
                for p in (current, original)])
            checkpoints = [torch.load(p / "final-state.pt", map_location="cpu", weights_only=False)
                           for p in (current, original)]
            row["equal_final_state_tensors"] = {
                key: equal(checkpoints[0][key], checkpoints[1][key])
                for key in ("parameters", "optimizers", "response_history", "cpu_rng", "cuda_rng")}
        elif job["kind"] == "native":
            row["equal_sample_elements"] = {}
            for name in ("final_samples.npz", "holdout_samples.npz"):
                with np.load(current / "native" / job["task"] / name) as x, np.load(
                        original / "native" / job["task"] / name) as y:
                    assert set(x.files) == set(y.files)
                    for key in x.files:
                        assert np.array_equal(x[key], y[key]), (job["id"], name, key)
                    row["equal_sample_elements"][name] = sum(x[key].size for key in x.files)
        else:
            keys = (["gate", "good_hold_updates", "stopped", "hold_window", "post_failure_diagnostic",
                     "dense", "coarse", "ema", "last_live", "last_ema"] if job["kind"] == "hold" else
                    ["terminal_grade", "stationary", "continued_hold", "shift_recovery", "diagnostic",
                     "final", "ema", "shift_pair", "optimizer_final", "rate_ranges", "steps", "shift_step", "shift"])
            for key in keys:
                assert a[key] == b[key], (job["id"], key)
            row["equal_recorded_stress_trajectory"] = True
            row["final_checkpoint_available"] = False
        rows.append(row)
    manifests = []
    for directory in (root, reference):
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, expected in manifest.items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
        manifests.append(dict(root=str(directory), files=len(manifest), unchanged=True,
                              sha256=hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()))
    return dict(scope="Packaged replay hook on the frozen research trainer; not a current-public-trainer qualification",
                rows=rows, manifests=manifests)


if __name__ == "__main__":
    replay, reference, output = map(Path, sys.argv[1:])
    result = audit(replay, reference)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(jobs=len(result["rows"]), manifests=result["manifests"])))

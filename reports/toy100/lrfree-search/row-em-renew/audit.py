"""Summarize frozen native100 receipts and check learner parity with critic floor."""

import hashlib
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT if (ROOT / "runs").exists() else Path(
    "/ml2/hypergan/gan-attempts/row-em-renew-20260928")
BASE = RUN_ROOT.parent / "combined-h1-h2-20260928/runs/h2-handoff-critic-floor"
TASKS = ("grid100", "rotated100", "staggered100")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(a, b, path=""):
    if isinstance(a, torch.Tensor):
        equal = (isinstance(b, torch.Tensor) and a.shape == b.shape
                 and a.dtype == b.dtype and
                 (torch.equal(a, b) or
                  (torch.is_floating_point(a) and
                   torch.allclose(a, b, rtol=0, atol=0, equal_nan=True))))
        return [] if equal else [path]
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return [path + "/keys"]
        return [item for key in a for item in differences(a[key], b[key],
                                                          path + "/" + str(key))]
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return [path + "/length"]
        return [item for i, (x, y) in enumerate(zip(a, b))
                for item in differences(x, y, path + "/" + str(i))]
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if a == b else [path]


def audit_task(task):
    directory = RUN_ROOT / "runs" / task
    baseline = BASE / task
    result = json.loads((directory / "result.json").read_text())
    header = json.loads((directory / "job-header.json").read_text())
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    rates = (directory / "rates.jsonl").read_bytes()
    assert rates == (baseline / "rates.jsonl").read_bytes()
    assert len(rates.splitlines()) == 7000
    assert result["observations"] == 34
    assert [x["step"] for x in metrics[-5:]] == [6000, 6250, 6500, 6750, 7000]
    assert result["stream_deviations"] == 0
    assert result["native"]["terminal_accuracy"] == [x["acc_accuracy_pass"] for x in metrics[-5:]]

    a = torch.load(baseline / "final-state.pt", map_location="cpu", weights_only=False)["trainer"]
    b = torch.load(directory / "final-state.pt", map_location="cpu", weights_only=False)["trainer"]
    ignored = {"schema", "row_em"}
    diff = [item for key in a if key not in ignored and key != "models"
            for item in differences(a[key], b[key], key)]
    for name, state in a["models"].items():
        if name in ("prior", "ema_prior"):
            diff += differences(state["z"], b["models"][name]["z"], "models/" + name + "/z")
        else:
            diff += differences(state, b["models"][name], "models/" + name)
    assert not diff, diff[:20]
    row_em = b["row_em"]
    assert row_em["committed"] and row_em["fits"] > 0

    return {
        "task": task,
        "status": result["status"],
        "package_sha256": header["package_sha256"],
        "observations": result["observations"],
        "accuracy_passing_checks": result["native"]["accuracy_passing_checks"],
        "terminal_accuracy": result["native"]["terminal_accuracy"],
        "holdout_pass": result["native"]["holdout_pass"],
        "stream_deviations": result["stream_deviations"],
        "learner_parity": True,
        "rates_identical": True,
        "fits": row_em["fits"],
        "sampling_sigma": row_em["sampling_sigma"],
        "training_sigma": float(b["output_noise"]["log_sigma"].exp()),
        "last_fit": row_em["last"],
        "seconds": result["seconds"],
        "final": {key: result["final"][key] for key in (
            "precision", "acc_center_rms_sigma", "min_cov_eig_ratio",
            "max_cov_eig_ratio", "acc_abs_cov_trace_bias", "acc_radial_ks")},
        "terminal_centers": [x["acc_center_rms_sigma"] for x in metrics[-5:]],
        "sample_sha256": {
            "final": sha256(directory / "native-noisy/final_samples.npz"),
            "holdout": sha256(directory / "native-noisy/holdout_samples.npz"),
        },
    }


if __name__ == "__main__":
    report = {task: audit_task(task) for task in TASKS}
    hashes = {x["package_sha256"] for x in report.values()}
    assert len(hashes) == 1, hashes
    (ROOT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    for task, value in report.items():
        print(task, value["status"], value["terminal_accuracy"],
              "holdout", value["holdout_pass"], "fits", value["fits"],
              "seconds", value["seconds"])

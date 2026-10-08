"""Inspect the pinned BCAP Gaussian samples without constructing a learner."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr
import torch


ATTEMPT = "4a5687bf78db4967aebe1ca068163a07"
ARCHIVE_SHA256 = "a2519b3f68d2cbf8bf96e929242c21876db301df3df48d51cfd3b4e1957410fc"
RAW_SHA256 = "167e3b9cdb7afe900f48ed0087a2e409faeef1ce858d80c93c53b24fac5af0a5"
SCORER_SHA256 = "7ec72e07c1aea87e77c85401b7e822f23d5a248ba45d718180045a1ad64ccd8b"
RECEIPT_HASHES = {
    "evidence.json": "32e199ad4b8992cdb367e4ba294710d1986975247b310cf019ba35730ac2ceec",
    "request.json": "eb7362177b9c079e602997e8ba4b21dd3f926878559ecff0b1eef885c483b4d8",
    "result.json": "a4e6c51a93f4797a5a15c343ac756f039016c56012c86e3d8ea9a266ee6d75a2",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def moment_pass(metrics):
    return metrics["mean_error_sigma"] <= .2 and .8 <= metrics["std_ratio"] <= 1.2


def full_pass(metrics):
    return (metrics["sample_count"] >= 4096 and metrics["finite_fraction"] == 1
            and moment_pass(metrics) and metrics["cdf_ks"] <= .05)


def describe(samples, metrics):
    ordered = np.sort(samples.numpy().ravel().astype(np.float64))
    ranks = np.arange(len(ordered), dtype=np.float64) / len(ordered)
    target = ndtr((ordered - 2.) / .5)
    above = ranks + 1 / len(ordered) - target
    below = target - ranks
    empirical_above = above.max() >= below.max()
    index = int(np.argmax(above if empirical_above else below))
    fitted = ndtr((ordered - metrics["mean"]) / metrics["std"])
    return {
        "metrics": metrics,
        "largest_cdf_gap": {
            "x": float(ordered[index]),
            "empirical_cdf": float(ranks[index] + (1 / len(ordered) if empirical_above else 0)),
            "target_cdf": float(target[index]),
            "direction": "empirical_above_target" if empirical_above else "target_above_empirical",
        },
        "fitted_normal_ks": max(float(np.max(fitted - ranks)),
                                float(np.max(ranks + 1 / len(ordered) - fitted))),
        "median": float(np.median(ordered)),
    }


def plot(records, output):
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "bcap-gaussian-smoke"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
    trained = records[1:]
    for key, label, color in [("metrics", "Primary", "#2563eb"),
                              ("confirmation", "Confirmation", "#d97706")]:
        metrics = [row[key] if key == "metrics" else row[key]["metrics"] for row in trained]
        axes[0].plot([row["step"] for row in trained], [m["cdf_ks"] for m in metrics],
                     marker="o", markersize=3, label=label, color=color, linewidth=1.3)
    axes[0].axhline(.05, color="#dc2626", linestyle="--", label="KS bound 0.05")
    axes[0].set(xlabel="Training update", ylabel="KS against target Gaussian",
                title="24 scheduled observations", ylim=(0, .38))
    axes[0].legend(fontsize=8)
    for axis, step in zip(axes[1:], [167, 1000]):
        row = next(row for row in records if row["step"] == step)
        for key, label, color in [("samples", "Primary", "#2563eb"),
                                  ("confirmation_samples", "Confirmation", "#d97706")]:
            ordered = np.sort(row[key].numpy().ravel().astype(np.float64))
            residual = np.arange(1, len(ordered) + 1) / len(ordered) - ndtr((ordered - 2.) / .5)
            keep = np.unique(np.concatenate([np.arange(0, len(ordered), 16),
                                            [len(ordered) - 1, np.argmax(residual), np.argmin(residual)]]))
            axis.plot(ordered[keep], residual[keep], label=label, color=color, linewidth=1.1)
        axis.axhline(0, color="#111827", linewidth=.8)
        for bound in [-.05, .05]:
            axis.axhline(bound, color="#dc2626", linestyle="--", linewidth=1)
        axis.set(xlabel="Generated value", ylabel="Empirical CDF minus target CDF",
                 title=f"Saved state at update {step}", xlim=(.5, 3.5), ylim=(-.085, .085))
        axis.legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.tight_layout()
    svg = output / "cdf-diagnosis.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/software/bcap-gaussian-smoke"))
    args = parser.parse_args()
    with args.archive.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == ARCHIVE_SHA256
    scorer_path = Path(__file__).resolve().parents[3] / "benchmarks/toy_audit/gaussian1d_quality.py"
    assert sha256(scorer_path.read_bytes()) == SCORER_SHA256, "Use the pinned public scorer"
    spec = importlib.util.spec_from_file_location("pinned_gaussian_scorer", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    with tarfile.open(args.archive) as archive:
        for name, expected in RECEIPT_HASHES.items():
            data = archive.extractfile(f"reports/forge/attempts/{ATTEMPT}/{name}").read()
            assert sha256(data) == expected
        prefix = f"runs/forge-word-split/technique-inventory-word-split-v1/{ATTEMPT}"
        raw_bytes = archive.extractfile(f"{prefix}/raw-result.json").read()
        assert sha256(raw_bytes) == RAW_SHA256
        raw = json.loads(raw_bytes)
        sample_bytes = archive.extractfile(f"{prefix}/evaluator/observed-samples.pt").read()
        assert sha256(sample_bytes) == raw["evidence"]["saved_observer_outputs"]["sha256"]
    records = torch.load(io.BytesIO(sample_bytes), map_location="cpu", weights_only=True)
    assert [r["step"] for r in records] == [0] + [(i * 1000 + 23) // 24 for i in range(1, 25)]
    recomputed = 0
    for row in records:
        for samples, metrics in [(row["samples"], row["metrics"]),
                                 (row["confirmation_samples"], row["confirmation"]["metrics"])]:
            assert scorer.score_samples(samples, raw["evidence"]["host"]["definition"]) == metrics
            recomputed += 1
        confirm = row["confirmation"]
        assert row["training_state_sha256"] == confirm["primary_state_sha256"] == confirm["confirmed_state_sha256"]
        assert confirm["training_state_unchanged"]
        assert confirm["independent_stream"] == "eval/live/smoke_confirmation"
    assert raw["evidence"]["observations"] == [dict(step=r["step"], **r["metrics"]) for r in records[1:]]
    assert raw["evidence"]["confirmations"] == [r["confirmation"] for r in records[1:]]
    primary = [r["metrics"] for r in records[1:]]
    confirmation = [r["confirmation"]["metrics"] for r in records[1:]]
    counts = {
        "scheduled_pairs": len(primary),
        "primary_full_passes": sum(map(full_pass, primary)),
        "confirmation_full_passes": sum(map(full_pass, confirmation)),
        "confirmed_pairs": sum(full_pass(p) and full_pass(c) for p, c in zip(primary, confirmation)),
        "primary_moment_passes": sum(map(moment_pass, primary)),
        "confirmation_moment_passes": sum(map(moment_pass, confirmation)),
        "primary_ks_failures": sum(m["cdf_ks"] > .05 for m in primary),
        "confirmation_ks_failures": sum(m["cdf_ks"] > .05 for m in confirmation),
    }
    grade = raw["gaussian_grade"]
    assert grade["gate_status"] == "FAIL"
    assert counts["primary_full_passes"] == grade["evaluator_result"]["passing_observations"] == 1
    assert counts["confirmed_pairs"] == 0 and grade["evaluator_result"]["first_confirmed_step"] is None
    selected = []
    for row in records:
        if row["step"] in (167, 1000):
            selected.append({"step": row["step"], "training_state_sha256": row["training_state_sha256"],
                             "primary": describe(row["samples"], row["metrics"]),
                             "confirmation": describe(row["confirmation_samples"], row["confirmation"]["metrics"])})
    summary = {
        "schema_version": 1, "scope": "saved_sample_analysis", "qualification_input": False,
        "qualification_reuse": False, "optimizer_updates_added": 0, "sampling_draws_added": 0,
        "attempt_id": ATTEMPT, "source_commit": "737592c128ef84cf596c7f198b0ce8aad7c65700",
        "source_digest": "cbb19c5e55e93aff93abd4092bbbe79e10c03a8b9f65a3f5db3b97f0f557d1ec",
        "archive_sha256": ARCHIVE_SHA256, "raw_result_sha256": RAW_SHA256,
        "observed_samples_sha256": sha256(sample_bytes), "scorer_sha256": SCORER_SHA256,
        "analyzer_sha256": sha256(Path(__file__).read_bytes()), "original_receipt_sha256": RECEIPT_HASHES,
        "exact_sample_metric_recomputations": recomputed, "unchanged_observation_states": len(records),
        "initial_observation_is_qualification": False, "recorded_gate": grade["gate_status"],
        "counts": counts, "selected_states": selected,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot(records, args.output)
    print(json.dumps({"verified_sample_sets": recomputed, **counts}))


if __name__ == "__main__":
    main()

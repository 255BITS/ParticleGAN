"""Audit four completed image diagnostics and publish their saved training outputs."""
from __future__ import annotations

from collections import Counter
import importlib.util
from io import BytesIO
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.tier1_media import _indices, _scored_outputs
from reports.forge.regenerate_technique_inventory import project_receipt

STUDY = "bcap-convolution-images-v1"
OUTPUT = ROOT / "reports/forge/bcap-convolution"


def helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(actual, expected):
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(close(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(close(a, b) for a, b in zip(actual, expected))
    if type(expected) is float:
        return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7)
    return actual == expected


def audit_rates(recipe, checkpoint):
    """Verify the declared schedule at every clock and its saved endpoint."""
    import torch
    from particlegan.recipes import Recipe, learning_rate_scales

    declaration = Recipe(**recipe)
    scales = [learning_rate_scales(step, declaration) for step in range(601)]
    assert set(scales) == {(1.0, 1.0)}
    state = torch.load(Path(checkpoint["artifact_root"]) / checkpoint["path"],
                       map_location="cpu", weights_only=False)["trainer"]
    assert state["completed_steps"] == 600
    schedules = {}
    for role, initial, optimizer in zip(("generator_and_prior", "discriminator"),
                                        state["initial_lrs"], state["optimizers"], strict=True):
        groups = optimizer["param_groups"]
        assert [group["lr"] for group in groups] == initial
        metadata = optimizer["dualnorm"]
        assert metadata["convolution"] == "per_offset"
        assert metadata["momentum"] == 0 and metadata["smoothing"] == 1e-5
        schedules[role] = {"initial_group_rates": initial,
                           "checkpoint_group_rates": [group["lr"] for group in groups],
                           "kernel_layouts": [group["dualnorm_convolution"]
                                              for group in groups if "dualnorm_convolution" in group]}
    assert state["initial_lrs"][0] == [recipe["lr"]] * 7 + [recipe["lr"] * recipe["prior_lr_mult"]]
    assert state["initial_lrs"][1] == [recipe["lr"] * recipe["d_lr_mult"]] * 4
    return {"method": "declared_schedule_all_601_clocks_and_checkpoint_endpoint",
            "clock_count": len(scales), "network_scale_range": [1, 1],
            "prior_scale_range": [1, 1], "optimizer_groups": schedules}


def render_image(task, row, local, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from benchmarks.transfer_suite.image_tasks import image_metrics

    records, inputs = _scored_outputs(task, row["evidence"], local)
    observed = row["evidence"]["observations"]
    assert len(records) == len(observed) == 24
    assert [point["step"] for point in observed] == list(range(25, 601, 25))
    for record, point in zip(records, observed):
        assert point == {"step": record["step"], **record["metrics"]}
        measured = image_metrics(record["samples"], record["targets"], task["evaluation"]["measurement"])
        assert close(measured, record["metrics"]), (measured, record["metrics"])

    def grid(images, columns):
        values = images[:, 0].numpy()
        rows = math.ceil(len(values) / columns)
        sheet = np.ones((rows * 9 - 1, columns * 9 - 1))
        for index, value in enumerate(values):
            y, x = divmod(index, columns)
            sheet[y * 9:y * 9 + 8, x * 9:x * 9 + 8] = value
        return sheet

    frames = []
    indices = _indices(len(records))
    steps = [point["step"] for point in observed]
    for index in indices:
        record = records[index]
        figure, axes = plt.subplots(3, 1, figsize=(8, 7), constrained_layout=True,
                                    gridspec_kw={"height_ratios": [1, 3, 2]})
        for axis, values, columns, label in (
                (axes[0], record["targets"], len(record["targets"]), "Declared target templates"),
                (axes[1], record["samples"], 8, "Actual clean outputs: all 32 learned prior centers")):
            axis.imshow(grid(values, columns), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(label, fontsize=10)
            axis.axis("off")
        axis = axes[2]
        axis.plot(steps[:index + 1], [p["hq"] for p in observed[:index + 1]], label="Quality fraction")
        axis.axhline(.9, color="black", linestyle="--", label="Required quality ≥ .9")
        axis.set(xlim=(0, 600), ylim=(0, 1.05), xlabel="Training update", ylabel="Quality fraction")
        modes = axis.twinx()
        modes.plot(steps[:index + 1], [p["modes"] for p in observed[:index + 1]], color="tab:orange", label="Covered modes")
        required = task["evaluation"]["measurement"]["modes"]
        modes.axhline(required, color="tab:orange", linestyle=":")
        modes.set(ylim=(0, required + .25), ylabel=f"Covered modes (required {required})")
        axis.legend(loc="lower left", fontsize=8)
        figure.suptitle(f"{task['id']} · update {record['step']} · sustained gate {row['gate_status']}\n"
                       f"hq={record['metrics']['hq']:.3f}, modes={record['metrics']['modes']}, "
                       f"mean RMSE={record['metrics']['mean_rmse']:.3f}", fontsize=10)
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=100)
        plt.close(figure)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("RGB"))
    output.parent.mkdir(exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=400, loop=0)
    receipt = {"schema_version": 1, "task_id": task["id"], "recorded_grade": row["gate_status"],
               "kind": "actual_training_saved_image_outputs_gif", "observation_count": 24,
               "saved_sample_sets_recomputed": 24, "selected_observation_indices": indices,
               "source_inputs": inputs, "observations_sha256": stable_hash(observed),
               "gif_sha256": file_hash(output), "optimizer_updates_added": 0,
               "sampling_draws_added": 0, "qualification_input": False,
               "renderer_sha256": file_hash(Path(__file__))}
    atomic_json(output.with_suffix(".json"), receipt)
    return receipt


def main():
    queue_root = ROOT / "runs/forge"
    state = read_json(queue_root / "queue/state.json")
    entries = [entry for entry in state["submissions"].values() if entry["request"].get("campaign_id") == STUDY]
    assert len(entries) == 1 and entries[0]["status"] not in {"queued", "running", "paused"}
    request = entries[0]["request"]
    assert request["view"]["evidence_scope"] == "research_diagnostic"
    campaign = state["campaigns"][STUDY]
    assert campaign["reserved_seconds"] == 0
    rows, audits, inputs, media = [], [], [], []
    guard = helper("saved_media_guard", ROOT / "reports/forge/gaussian-smoke-inventory/export_media.py")
    for definition in request["jobs"]:
        job = state["jobs"][definition["compatibility_key"]]
        assert job["status"] == "terminal" and len(job["attempts"]) == 1
        attempt = job["result"]["attempt_id"]
        directory = ROOT / "reports/forge/attempts" / attempt
        name = definition["task_id"]
        receipt = project_receipt(ROOT, attempt)
        atomic_json(OUTPUT / "receipts" / (name + ".json"), receipt)
        envelope = read_json(directory / "request.json")
        assert envelope["request"]["source"] == request["source"]
        assert envelope["request"]["candidate_revision"] == request["candidate_revision"]
        local = Path(envelope["worker"]["directory"])
        raw = read_json(local / "raw-result.json")
        applied = raw.get("applied", raw)
        recipe = applied["recipe"]
        assert recipe["optimizer_convolution"] == "per_offset" and recipe["optimizer_smoothing"] == 1e-5
        assert recipe["optimizer_family"] == "dualnorm" and recipe.get("optimizer_momentum", 0) == 0
        assert recipe["lr"] == .012 and recipe["d_lr_mult"] == 1.5 and recipe["prior_lr_mult"] == 2.5
        assert recipe["lr_floor"] == recipe["network_lr_floor"] == 1
        row, = job["result"]["task_results"]
        assert row["gate_status"] in {"PASS", "FAIL"}
        evidence = row["evidence"]
        checkpoint = evidence["provenance_checkpoint"]
        verify_artifacts(Path(checkpoint["artifact_root"]), checkpoint["artifact_manifest"])
        schedules = audit_rates(recipe, checkpoint)
        assert checkpoint["completed_steps"] == 600
        assert evidence["guards"]["optimizer_updates"] == {"generator": 600, "discriminator": 600, "prior": 600}
        assert evidence["guards"]["unintended_rng_deviations"] == 0
        observed = evidence["observations"]
        from experiments.forge.decision_contracts import OPS
        bounds = request["tasks"][name]["evaluation"]["thresholds"]
        passing = [point["step"] for point in observed if all(OPS[op](point[metric], bound) for metric, op, bound in bounds)]
        compact, = receipt["task_results"]
        rows.append({**compact, "attempt_id": attempt, "original_task_tier": 2,
                     "completed_steps": 600, "passing_checks": len(passing), "total_checks": 24,
                     "passing_check_steps": passing})
        audits.append({"task_id": name, "receipt_validated": True, "guards": evidence["guards"],
                       "constant_rate_groups": schedules, "recipe_sha256": stable_hash(recipe),
                       "checkpoint_sha256": checkpoint["state_sha256"]})
        with guard.forbid_live_execution():
            rendered = render_image(request["tasks"][name], row, local, OUTPUT / "media" / (name + ".gif"))
        media.append({"task_id": name, "gif": name + ".gif", "gif_sha256": rendered["gif_sha256"],
                      "renderer_receipt": name + ".json", "recorded_grade": row["gate_status"]})
        inputs.extend((directory, local))
        print(name, row["gate_status"], row["metrics"], flush=True)
    assert len(rows) == 4
    counts = dict(Counter(row["gate_status"] for row in rows))
    from experiments.forge import knowledge
    compile_memory = knowledge.compile_memory
    knowledge.compile_memory = lambda *a, **k: None
    try:
        record = knowledge.readout(ROOT, request["candidate"]["id"],
            f"Per-offset smoothed DualNorm completed all four unchanged image tasks: {counts}; all 2400 declared updates ran. "
            "The prior four optimizer setup errors remain under the original source. Constant rates, fixed smoothing, seed0, no retries or annealing.",
            "One global recipe differs from the selected incumbent only by optimizer_convolution=per_offset. "
            "Four explicitly scoped Tier2 image diagnostics preserve all gates and task conditions; no source pooling or ordinary qualification credit.",
            "Use the recorded quality/coverage trajectories to assess the convolution adaptation. "
            "Keep the six original Tier1 passes and full original Tier2 readout; other eleven failures and Tier3 remain unaddressed.", study_id=STUDY)
    finally:
        knowledge.compile_memory = compile_memory
    atomic_json(OUTPUT / "media/index.json", {"schema_version": 1, "study_id": STUDY, "tasks": media,
        "optimizer_updates_added": 0, "sampling_draws_added": 0, "qualification_input": False})
    inputs.extend((queue_root / STUDY, queue_root / "queue", queue_root / "events.jsonl",
                   Path(request["source"]["snapshot_path"]), OUTPUT / "run.py", Path(__file__),
                   ROOT / "configs/forge/studies" / (STUDY + ".json")))
    archiver = helper("bcap_archive", ROOT / "reports/forge/bcap-six/publish.py")
    archive = archiver.archive(ROOT, inputs, ROOT / "artifacts/bcap-convolution-images-v1.tar.gz")
    atomic_json(OUTPUT / "readout.json", {"schema_version": 1, "study_id": STUDY,
        "candidate_id": request["candidate"]["id"], "candidate_revision": request["candidate_revision"],
        "source_digest": request["source"]["digest"], "source_origin_commit": request["source"]["origin_commit"],
        "runtime": request["runtime"], "compute_profiles": request["compute_profiles"],
        "request_id": request["request_id"], "campaign": campaign, "counts": counts, "tasks": rows, "audits": audits,
        "archive": archive, "new_attempt_count": 4, "scientific_retries": 0, "tier1_reruns": 0,
        "completed_training_updates": 2400, "saved_sample_sets_recomputed": 96,
        "concluded_readout_record": "reports/forge/records/" + record["record_id"] + ".json",
        "evidence_scope": "research_diagnostic", "qualification_input": False, "qualification_reuse": False,
        "default_adoption": False, "publisher_sha256": file_hash(Path(__file__))})
    print(counts, "paid seconds", campaign["spent_seconds"], flush=True)


if __name__ == "__main__":
    main()

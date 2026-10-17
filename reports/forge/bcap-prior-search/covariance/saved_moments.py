"""Measure final live location moments using checkpointed initial references.

Saved tensors only: no model construction/forward, sampling, or optimization.
These full-table diagnostics are not the minibatch objective or a Forge gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.knowledge import _attempts
from experiments.forge.state import state_digest


ROOT = Path(__file__).resolve().parents[4]
REFERENCES = ("moment_tether_mean", "moment_tether_covariance", "moment_tether_count")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def live_tables(saved):
    """Select live priors explicitly; never inspect averaged state as live."""
    if "trainer" in saved:
        return {"prior": saved["trainer"]["models"]["prior"]}
    if "fixture" in saved:
        return {"prior": saved["fixture"]["api_state"]["models"]["prior"]}
    return {name: values for name, values in saved.get("models", {}).items() if name.startswith("prior")}


def initial_reference(tables):
    require(tables and all(all(k in values for k in REFERENCES) for values in tables.values()),
            "An active live prior lacks its checkpointed initial reference")
    counts = [values[REFERENCES[2]] for values in tables.values()]
    require(all(int(n) == len(v["z"]) for n, v in zip(counts, tables.values())),
            "Fixed initial counts differ from task prior capacity")
    if len(tables) == 1:
        values = next(iter(tables.values()))
        return values[REFERENCES[0]], values[REFERENCES[1]]
    total = sum(counts)
    mean = sum(v[REFERENCES[0]] * n for n, v in zip(counts, tables.values())) / total
    scatter = sum((n - 1) * v[REFERENCES[1]]
                  + n * torch.outer(v[REFERENCES[0]] - mean, v[REFERENCES[0]] - mean)
                  for n, v in zip(counts, tables.values()))
    covariance = scatter / (total - 1).clamp_min(1)
    return mean, covariance


def discrepancy(tables, reference, eps):
    locations = torch.cat([values["z"] for values in tables.values()])
    mean = locations.mean(0)
    centered = locations - mean
    covariance = centered.T @ centered / max(len(locations) - 1, 1)
    initial_mean, initial_covariance = reference
    scale = initial_covariance.diagonal().clamp_min(0) + eps
    mean_error = ((mean - initial_mean).square() / scale).mean()
    covariance_error = ((covariance - initial_covariance).square() / torch.outer(scale, scale)).mean()
    return dict(full_location_count=len(locations), latent_dimension=locations.shape[1],
                normalized_mean_discrepancy=float(mean_error),
                normalized_covariance_discrepancy=float(covariance_error),
                unweighted_moment_discrepancy=float(mean_error + covariance_error),
                mean_sha256=state_digest(mean), covariance_sha256=state_digest(covariance))


def read_checkpoint(row):
    descriptor = row.get("evidence", {}).get("provenance_checkpoint")
    require(descriptor is not None, "Certified completion lacks its saved provenance state")
    root = Path(descriptor["artifact_root"])
    verify_artifacts(root, descriptor["artifact_manifest"])
    path = root / descriptor["path"]
    require(file_hash(path) == descriptor["sha256"], "Checkpoint bytes changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    require(state_digest(saved) == descriptor["state_sha256"], "Checkpoint tensor/state hash differs")
    return saved, dict(artifact_root=str(root), path=descriptor["path"], sha256=descriptor["sha256"],
                       state_sha256=descriptor["state_sha256"], completed_steps=descriptor["completed_steps"])


def actual_metadata(saved, row):
    if "applied" in saved:
        return saved["applied"]
    if "initialization" in saved:
        return saved
    return row  # WordFixture's initialization receipt accompanies its saved policy state.


def write_markdown(receipt, output):
    """Render only the compact, already validated scalar diagnostics."""
    text = ["# Saved full-location moment diagnostics", "",
        "Every score below is diagnostic only, with **no qualification target**. "
        "Candidate and control use the same checkpointed original reference after "
        "initializer tensor receipts and prior-law/capacity checks. All live rows "
        "are included; EMA state, serving noise and standardization are excluded.", "",
        "[Full hashes, reference buffers, own-prefix scope and receipts](moment-diagnostics.json). "
        "[Readout](readout.md). No model execution, sampling or optimizer update is added.", "",
        "| Strength | Task | Rows × dimensions | Mean candidate / control | Covariance candidate / control | Total candidate / control | Total ratio | Target |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    def fmt(value):
        return "unavailable" if value is None else f"{value:.9g}"
    for row in receipt["diagnostics"]:
        after, before = row["candidate"], row.get("control") or {}
        pairs = [f'{fmt(after[field])} / {fmt(before.get(field))}' for field in (
            "normalized_mean_discrepancy", "normalized_covariance_discrepancy", "unweighted_moment_discrepancy")]
        shape = f'{after["full_location_count"]} × {after["latent_dimension"]}'
        text.append(f'| {row["strength"]} | {row["task"]} | {shape} | ' + " | ".join(pairs)
                    + f' | {fmt(row.get("candidate_to_control_discrepancy_ratio"))} | No gate |')
    text += ["", "Counts are frozen prior capacities and dimensions. Unequal own "
        "continuation prefixes remain explicit in each JSON receipt; these rows "
        "do not claim equal trained starting weights or equal consumed prefixes. "
        "The full-table diagnostic differs from the native selected-row objective.", "",
        "## Not applicable or unavailable", "",
        "| Global strength | Task | Recorded gate | Reason |", "| --- | --- | --- | --- |"]
    for row in receipt["not_applicable_or_unavailable"]:
        text.append(f'| {row["strength"]} | {row["task"]} | {row["gate_status"]} | {row["reason"]} |')
    output.with_suffix(".md").write_text("\n".join(text) + "\n")


def run(archive, output):
    torch.set_num_threads(1)
    progress = json.loads((archive / "progress.json").read_text())
    state = json.loads((archive / "queue/queue/state.json").read_text())
    require(progress["phase"] == "complete"
            and not any(e["status"] in {"queued", "running", "paused"} for e in state["submissions"].values()),
            "Only the completed registered comparison can receive a final diagnostic")
    spec = importlib.util.spec_from_file_location("moment_diagnostic_workflow", ROOT / "reports/forge/bcap-prior-search/workflow.py")
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    originals, _ = _attempts(ROOT)
    attempts = {a["attempt_id"]: a for a in originals}
    control_entry = state["submissions"][progress["control_request"]]
    control_rows = {r["task_id"]: r for r in workflow.validated_submission_rows(state, control_entry, attempts)}
    diagnostics, skipped = [], []
    for request_id in progress["search_requests"]:
        entry = state["submissions"][request_id]
        request = entry["request"]
        require(request["source"]["digest"] == progress["source_digest"], "Diagnostic source cohort changed")
        rows = workflow.validated_submission_rows(state, entry, attempts)
        for row in rows:
            task = row["task_id"]
            policy = row.get("evidence", {}).get("prior_policy", {})
            identity = dict(task=task, request_id=request_id, candidate_id=request["candidate"]["id"],
                            configuration_id=request["candidate"]["configuration_id"], gate_status=row["gate_status"],
                            strength=request["candidate"]["recipe_overrides"]["prior_reg"])
            if row["gate_status"] not in {"PASS", "FAIL"} or policy.get("applicability") != "latent_prior":
                skipped.append(dict(identity, reason="No certified completed live latent-prior objective", prior_policy=policy))
                continue
            require(policy["regularizer"] == "moment_tether" and policy["weight"] == identity["strength"] > 0
                    and policy["update"] == "learned", "Positive moment-tether policy differs")
            saved, certificate = read_checkpoint(row)
            actual = actual_metadata(saved, row)
            tables = live_tables(saved)
            reference = initial_reference(tables)
            score = discrepancy(tables, reference, policy["eps"])
            record = dict(identity, strength=policy["weight"], reference_sha256=state_digest(reference),
                checkpoint=certificate, original_receipt_binding=row["receipt_binding"], actual_prior_policy=policy,
                initialization_sha256=stable_hash(actual["initialization"]), initial_prior=actual["prior"],
                actual_recipe_sha256=stable_hash(actual["recipe"]),
                continuation=row.get("evidence", {}).get("continuity"),
                reference_tables={name: dict(count=int(v[REFERENCES[2]]),
                    mean_sha256=state_digest(v[REFERENCES[0]]), covariance_sha256=state_digest(v[REFERENCES[1]]))
                    for name, v in tables.items()}, candidate=score,
                weighted_full_table_diagnostic=policy["weight"] * score["unweighted_moment_discrepancy"],
                target="diagnostic only; no qualification credit")
            control = control_rows.get(task)
            if control and control["gate_status"] in {"PASS", "FAIL"}:
                before, control_certificate = read_checkpoint(control)
                control_actual = actual_metadata(before, control)
                require(control_actual["initialization"] == actual["initialization"]
                        and control_actual["prior"] == actual["prior"],
                        task + ": cannot borrow initial references across unlike initializer/prior laws")
                before_tables = live_tables(before)
                require(set(before_tables) == set(tables)
                        and all(before_tables[k]["z"].shape == tables[k]["z"].shape for k in tables),
                        "Matched live prior architecture/capacity differs")
                control_score = discrepancy(before_tables, reference, policy["eps"])
                record.update(control=control_score, control_gate_status=control["gate_status"],
                    control_checkpoint=control_certificate,
                    control_original_receipt_binding=control["receipt_binding"],
                    control_continuation=control.get("evidence", {}).get("continuity"),
                    control_reference_binding="Exact candidate reference after equal public-initializer tensor receipts and prior-law/capacity checks",
                    equal_completed_prefix=(control_certificate["completed_steps"] == certificate["completed_steps"]),
                    own_prefix_scope="Continuations retain their own candidate-certified acquisition; this diagnostic compares drift from the original public initialization, not equal trained starting weights.",
                    candidate_to_control_discrepancy_ratio=(score["unweighted_moment_discrepancy"]
                        / control_score["unweighted_moment_discrepancy"] if control_score["unweighted_moment_discrepancy"] else None))
            else:
                record["control"] = None
                record["control_unavailable_reason"] = "No certified completed same-source control cell"
            diagnostics.append(record)
    receipt = dict(schema_version=1, scope="saved_final_live_full_location_initial_moment_diagnostic",
        source_commit=progress["source_commit"], source_digest=progress["source_digest"], protocol_seed=0,
        analysis_device="CPU over hash-verified saved tensors", analysis_threads=1,
        coordinates="raw learned locations, without serving noise or standardization",
        objective="mean_i((mu_i-mu0_i)^2/(C0_ii+eps)) + mean_ij((C_ij-C0_ij)^2/((C0_ii+eps)*(C0_jj+eps))); unbiased covariance; initial diagonal clamped nonnegative",
        diagnostics=diagnostics, not_applicable_or_unavailable=skipped,
        qualification_input=False, optimizer_updates_added=0, sampling_draws_added=0, model_executions_added=0,
        limitations="The trainer retains its declared row selection; full-table scores differ from native sampled-row objectives. Leader DualNorm scales each unique sampled row's combined gradient g by 1/sqrt(||g||^2+0.001^2); its nonlinear smoothed saturation means the coefficient does not bound displacement. Unsampled rows remain fixed despite dense auxiliary gradients. Moment preservation does not establish output coverage, quality, or retention.")
    atomic_json(output, receipt)
    write_markdown(receipt, output)
    print(json.dumps(dict(event="saved_moments", measured_positive_tasks=len(diagnostics), skipped=len(skipped))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("moment-diagnostics.json"))
    args = parser.parse_args()
    run(args.archive, args.output)

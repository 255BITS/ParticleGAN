"""Compare completed fresh controls with the archived leader, without reuse or training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASELINE = ROOT / "reports/forge/recipe-prior-refactor/publication/results.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archived = json.loads(BASELINE.read_text())
    reference = {row["task_id"]: row for row in archived["certified_results"]
                 if row["family"] == "bcap-pure" and row["scope"] == "ordinary"}
    records = []
    for name in ("proximal", "coulomb", "sigreg", "covariance", "mmd"):
        path = OUT / name / "results.json"
        result = json.loads(path.read_text())
        actual = {row["task_id"]: row for row in result["control_results"]}
        rows = []
        for task, before in sorted(reference.items()):
            after = actual.get(task, {})
            binding, metrics = after.get("receipt_binding", {}), after.get("metrics", {})
            valid = bool(binding.get("valid_receipt"))
            changed = [key for key in sorted(before["metrics"])
                       if key not in metrics
                       or before["metrics"][key] != metrics[key]]
            rows.append(dict(task=task, gate_equal=valid and before["gate_status"] == after.get("gate_status"),
                             valid_fresh_receipt=valid, changed_archived_metrics=changed,
                             additional_metrics=sorted(set(metrics) - set(before["metrics"])),
                             fresh_attempt_id=binding.get("attempt_id"),
                             archived_attempt_id=before["attempt_id"]))
        records.append(dict(name=name, source_commit=result["source_commit"], source_digest=result["source_digest"],
            results_sha256=digest(path), tasks=rows, all_control_tasks_present=set(actual) == set(reference),
            exact_gate_matches=sum(row["gate_equal"] for row in rows),
            exact_archived_metric_matches=sum(row["valid_fresh_receipt"] and not row["changed_archived_metrics"] for row in rows),
            exact_metric_dictionary_matches=sum(row["valid_fresh_receipt"] and not (row["changed_archived_metrics"] or row["additional_metrics"]) for row in rows)))
    outcome = dict(schema_version=1, scope="saved_evidence_baseline_reproduction_diagnostic",
        baseline_path=str(BASELINE.relative_to(ROOT)), baseline_sha256=digest(BASELINE),
        archived_source_digest=archived["source_digest"], records=records,
        archived_evidence_reused=False, qualification_input=False,
        optimizer_updates_added=0, sampling_draws_added=0)
    (OUT / "baseline_audit.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    assert len(reference) == 28, "The frozen ordinary leader baseline must contain 28 tasks including the optional clock."
    assert all(record["all_control_tasks_present"]
               and record["exact_gate_matches"] == len(reference)
               and record["exact_archived_metric_matches"] == len(reference)
               for record in records), "Fresh controls differ from the archived leader; inspect baseline_audit.json."


if __name__ == "__main__":
    main()

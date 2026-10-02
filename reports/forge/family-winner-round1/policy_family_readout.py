"""Read-only projection of an already executed policy-family study.

No API calls, training, sampling, metric rescoring or leaderboard publication.
The original toy gate and additional study gate remain separate. Timeline
diagnostics use only the recorded primary-observation boolean verdicts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


SCHEMA = "particlegan_policy_family_search_v1"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detail(row):
    return deepcopy({key: row[key] for key in
                     ("step", "elapsed_seconds", "passed", "failed_bounds", "metrics")})


def timeline(receipt):
    """Describe the saved verdict stream; supply no new scientific gate."""
    steps = receipt["protocol"]["metric_evaluation_steps"]
    rows = [row for row in receipt["observations"]
            if row["step"] > 0 and row["step"] in steps]
    if [row["step"] for row in rows] != steps[1:]:
        raise ValueError("complete primary metric schedule required for timeline diagnosis")
    streak = longest = 0
    acquired = None
    for index, row in enumerate(rows):
        if type(row["passed"]) is not bool:
            raise ValueError("recorded observation verdict must be boolean")
        streak = streak + 1 if row["passed"] else 0
        longest = max(longest, streak)
        if acquired is None and streak == 5:
            acquired = index
    failures = [row for row in rows if not row["passed"]]
    after = [] if acquired is None else rows[acquired + 1:]
    broken = [row for row in after if not row["passed"]]
    return {"primary_checks": len(rows), "passing_checks": sum(row["passed"] for row in rows),
            "longest_passing_streak": longest, "terminal_passing_suffix": streak,
            "original_required_terminal_checks": receipt["protocol"]["terminal_observations"],
            "first_primary_failure": detail(failures[0]) if failures else None,
            "first_acquisition_window": ([rows[acquired - 4]["step"], rows[acquired]["step"]]
                                         if acquired is not None else None),
            "post_acquisition_checks": len(after),
            "post_acquisition_passed": sum(row["passed"] for row in after),
            "first_post_acquisition_failure": detail(broken[0]) if broken else None,
            "terminal_observation": detail(rows[-1]) if rows else None,
            "recorded_primary_observations": [detail(row) for row in rows]}


def project(study_path):
    study_path = Path(study_path).resolve()
    payload = study_path.read_bytes()
    study = json.loads(payload)
    if study.get("schema") != SCHEMA or study.get("executed_family") not in {"atlas", "e22"}:
        raise ValueError("one registered policy-family execution archive required")
    if len(study["trials"]) != 8 or len({trial["id"] for trial in study["trials"]}) != 8:
        raise ValueError("all eight declared trial identities must remain visible")
    expected = [(row["id"], row["tier"]) for row in study["spec"]["cases"]]
    trials = []
    for trial in study["trials"]:
        if [(row["id"], row["tier"]) for row in trial["cases"]] != expected:
            raise ValueError("changed or missing required case denominator")
        exported = {key: deepcopy(trial[key]) for key in
                    ("id", "family", "recipe_overrides", "status")}
        exported["paid_wall_seconds"] = trial.get("paid_wall_seconds", 0.)
        exported["cases"] = []
        for row in trial["cases"]:
            item = deepcopy(row)
            if row["status"] not in {"PASS", "FAIL"}:
                exported["cases"].append(item)
                continue
            path = Path(row["receipt_path"])
            if file_hash(path) != row["receipt_sha256"]:
                raise ValueError("certified original receipt changed")
            receipt = json.loads(path.read_bytes())
            if (receipt["status"] != "COMPLETE" or not receipt["default_protocol_complete"]
                    or receipt["source"]["commit"] != study["source"]["commit"]
                    or any(receipt["source"]["files_sha256"].get(name) != sha
                           for name, sha in study["source"]["files_sha256"].items())):
                raise ValueError("complete original source-bound protocol required")
            if (receipt["verdict"] != row["original_gate"]
                    or receipt["recipe"] != row["recipe"]
                    or receipt["runtime"] != row["runtime"]
                    or receipt["artifacts"] != row["artifacts"]
                    or receipt["observations"][-1]["metrics"] != row["final_metrics"]):
                raise ValueError("display differs from certified original evidence")
            artifacts = {}
            for name, binding in receipt["artifacts"].items():
                artifact = path.parent / name
                if file_hash(artifact) != binding["sha256"] or artifact.stat().st_size != binding["bytes"]:
                    raise ValueError("original media/state/observation artifact changed")
                artifacts[name] = {"path": str(artifact.resolve()), **binding}
            item.update(original_failed_bounds=deepcopy(receipt["failed_bounds"]),
                        original_sustained_metric_passed=receipt["sustained_metric_passed"],
                        original_sampling=deepcopy(receipt["case"]["sampling"]),
                        protocol=deepcopy(receipt["protocol"]),
                        diagnostic=timeline(receipt), bound_artifacts=artifacts)
            exported["cases"].append(item)
        trials.append(exported)
    output = {"schema": "policy_family_display_readout_v1", "qualification_input": False,
              "training_updates": 0, "rescoring": False, "speed_ranking": False,
              "study": {"path": str(study_path), "sha256": hashlib.sha256(payload).hexdigest()},
              "executed_family": study["executed_family"], "spec_sha256": study["spec_sha256"],
              "source": deepcopy(study["source"]), "lane_runtime": deepcopy(study["lane_runtime"]),
              "required_cases_per_config": len(expected), "declared_configs": len(trials),
              "family_declared_configs": sum(trial["family"] == study["executed_family"] for trial in trials),
              "paid_seconds": study["measured_paid_seconds"],
              "unmeasured_interrupt_reservation_seconds": study["unmeasured_interrupt_reservation_seconds"],
              "selection": deepcopy(study["selection"]), "trials": trials}
    json.dumps(output, allow_nan=False)
    return output


def markdown(readout):
    family = readout["executed_family"]
    lines = [f"# {family.title()} ordinary policy-family readout", "",
             "Original toy gates and the separate first-acquisition retention requirement retain "
             "their own verdicts. This projection supplies no training, rescoring, qualification "
             "or speed ranking.", "",
             "| Config / LR / prior rate | Case | Original gate | Study gate | Primary passing / streak / terminal suffix | First hold failure or terminal bounds |",
             "| --- | --- | --- | --- | --- | --- |"]
    for trial in readout["trials"]:
        if trial["family"] != family:
            continue
        knobs = trial["recipe_overrides"]
        label = f"`{trial['id'].split('--')[1][:12]}` / {knobs['lr']} / {knobs['prior_lr_mult']}"
        for row in trial["cases"]:
            diagnostic = row.get("diagnostic")
            if diagnostic is None:
                values = (label, row["id"], row.get("original_gate", "UNMEASURED"),
                          row["status"], "—", row.get("reason", "Unmeasured"))
            else:
                first = diagnostic["first_post_acquisition_failure"]
                bounds = first["failed_bounds"] if first else row["original_failed_bounds"]
                counts = (f"{diagnostic['passing_checks']}/{diagnostic['primary_checks']} / "
                          f"{diagnostic['longest_passing_streak']} / {diagnostic['terminal_passing_suffix']}")
                why = (f"update {first['step']}: " if first else "") + ("; ".join(bounds) or "No recorded failing bound")
                values = (label, row["id"], row["original_gate"], row["study_gate"], counts, why)
            lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    lines += ["", "Every required case remains in the denominator. UNKNOWN cells are unmeasured; "
              "a stopped prerequisite supplies no later-task result. Original terminal recovery "
              "does not erase a failure after the study's first acquired window.", "",
              f"Paid child execution: {readout['paid_seconds']:.3f}s. Convergence speed remains "
              "unranked under external contention. This public served-policy cohort supplies no "
              "clean-MoG or historical qualification reuse.", "",
              f"Original study `{readout['study']['path']}`; SHA256 `{readout['study']['sha256']}`.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data = project(args.study)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "readout.json").write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
    (args.output / "README.md").write_text(markdown(data))
    print(json.dumps({"family": data["executed_family"], "paid_seconds": data["paid_seconds"],
                      "training_updates": 0, "source_study_sha256": data["study"]["sha256"]}))


if __name__ == "__main__":
    main()

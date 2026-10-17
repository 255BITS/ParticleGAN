"""Model-free current task-pin and compatibility audit; never enqueues work."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import torch

from experiments.forge.contracts import file_hash, read_json
from experiments.forge.planning import rekey_jobs, resolve_idea
from experiments.forge.views import (
    load_tasks, load_view, task_evaluation_fingerprint, task_execution_fingerprint,
)


TASK = "five_word_joint_acquisition"
TASK_PATH = "configs/forge/tasks/five_word_joint_acquisition.json"
HELPER = "benchmarks/toy_audit/reproducibility.py"
OLD_SOURCE = "3bf51eab2a80eef3645ca5c7df9fa0583cc7053332bc94b19ec874e2e19f98a0"
EXPECTED_SOURCE = "6269a18ac4f82564cb16ba19afa4b3dd2f836a2b4085fbe5aeb81a35a453e895"
REPRESENTATIVE = "bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9"


def forbidden(*args, **kwargs):
    raise AssertionError("metadata audit attempted model or optimizer construction")


def audit(root, baseline, round_path):
    tasks, original_tasks = load_tasks(root), load_tasks(baseline)
    before, after = read_json(baseline / TASK_PATH), read_json(root / TASK_PATH)
    normalized = deepcopy(after)
    normalized["evaluation"]["sources"][HELPER] = before["evaluation"]["sources"][HELPER]
    normalized["evaluation"]["evaluator_revision"] = before["evaluation"]["evaluator_revision"]
    assert normalized == before, "changed fields outside source pin and revision provenance"
    assert after["evaluation"]["evaluator_revision"]["previous_revision"] == before["evaluation"]["evaluator_revision"]
    assert before["evaluation"]["sources"][HELPER] != file_hash(root / HELPER)
    assert after["evaluation"]["sources"][HELPER] == file_hash(root / HELPER)
    assert task_execution_fingerprint(before) == task_execution_fingerprint(after)
    assert task_evaluation_fingerprint(before) != task_evaluation_fingerprint(after)
    unchanged = []
    for folder in ("tasks", "task-variants"):
        for path in sorted((root / "configs/forge" / folder).rglob("*.json")):
            relative = path.relative_to(root)
            if relative.as_posix() == TASK_PATH:
                continue
            assert path.read_bytes() == (baseline / relative).read_bytes(), relative
            unchanged.append(relative.as_posix())
    for folder in ("particlegan", "experiments", "benchmarks", "lib"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".toml", ".yaml", ".yml", ".sh"} and "__pycache__" not in path.parts:
                assert path.read_bytes() == (baseline / path.relative_to(root)).read_bytes(), path
    view = load_view(root, "discriminator_stability")
    assert view == load_view(baseline, "discriminator_stability")
    selected = {row["task"]: row["qualification_tier"] for row in view["assignments"]
                if row["importance"] == "required" and row["qualification_tier"] <= 2}
    source_checks = 0
    for name in selected:
        for path, expected in tasks[name]["evaluation"].get("sources", {}).items():
            assert file_hash(root / path) == expected, (name, path)
            source_checks += 1
    round_definition = read_json(round_path)
    ids = round_definition["candidate_ids"]
    assert len(ids) == len(set(ids)) == 52
    rows = []
    changed_jobs = unchanged_jobs = 0
    source_digests = set()
    representative = None
    for index, name in enumerate(ids, 1):
        options = dict(view_id="discriminator_stability", through_tier=2,
                       execution_backend="cuda", cuda_model="NVIDIA RTX A6000")
        try:
            old = resolve_idea(baseline, name, **options)
        except ValueError as error:
            try:
                resolve_idea(root, name, **options)
            except ValueError as current_error:
                assert str(current_error) == str(error)
            else:
                raise AssertionError("declaration refusal changed: " + name)
            rows.append(dict(candidate=name, status="UNCHANGED_DECLARATION_REFUSAL", reason=str(error)))
            continue
        new = resolve_idea(root, name, **options)
        assert old["source"]["digest"] == OLD_SOURCE
        assert new["source"]["digest"] == EXPECTED_SOURCE
        differences = {path for path in old["source"]["files"].keys() | new["source"]["files"].keys()
                       if old["source"]["files"].get(path) != new["source"]["files"].get(path)}
        assert differences == {TASK_PATH}
        source_digests.add(new["source"]["digest"])
        assert old["candidate_revision"] != new["candidate_revision"]
        assert old["protocol"] == new["protocol"] and old["rng"] == new["rng"]
        assert old["runtime"] == new["runtime"] and old["compute_profiles"] == new["compute_profiles"]
        assert old["preflight_blockers"] == new["preflight_blockers"]
        old_jobs = {job["execution_group"]: job for job in old["jobs"]}
        new_jobs = {job["execution_group"]: job for job in new["jobs"]}
        semantic_jobs = deepcopy(new["jobs"])
        for job in semantic_jobs:
            job["science"]["candidate_revision"] = old["candidate_revision"]
        rekey_jobs(semantic_jobs)
        semantic_jobs = {job["execution_group"]: job for job in semantic_jobs}
        changed = []
        for group, old_job in old_jobs.items():
            new_job = new_jobs[group]
            assert old_job["compatibility_key"] != new_job["compatibility_key"], (name, group)
            if TASK in old_job["task_ids"]:
                assert old_job["compatibility_key"] != new_job["compatibility_key"]
                changed.append(group)
                changed_jobs += 1
            else:
                assert old_job["science"] == semantic_jobs[group]["science"], (name, group)
                assert old_job["compatibility_key"] == semantic_jobs[group]["compatibility_key"], (name, group)
                unchanged_jobs += 1
                for member in old_job["task_ids"]:
                    assert old["tasks"][member]["preflight_blockers"] == new["tasks"][member]["preflight_blockers"]
        blockers = new["tasks"][TASK]["preflight_blockers"]
        assert not any("evaluator source changed" in reason or "word source binding differs" in reason for reason in blockers)
        row = dict(candidate=name, status="PLANNED", word_blockers=blockers,
                   candidate_revision=new["candidate_revision"], changed_groups=changed,
                   source_origins={"baseline": old["source"]["origin_commit"], "repair": new["source"]["origin_commit"]})
        rows.append(row)
        if name == REPRESENTATIVE:
            assert not blockers
            for task_id in selected:
                assert not new["tasks"][task_id]["preflight_blockers"], task_id
            representative = dict(candidate=name, required_tasks_with_clear_preflight=len(selected),
                word_before_key=old_jobs[TASK]["compatibility_key"], word_after_key=new_jobs[TASK]["compatibility_key"],
                word_before_blockers=old["tasks"][TASK]["preflight_blockers"], word_after_blockers=blockers)
        print(json.dumps({"event": "metadata_plan", "index": index, "total": len(ids),
                          "candidate": name, "word_blockers": blockers}), flush=True)
    assert representative is not None
    return dict(schema_version=1, status="PASS", scope="task_source_binding_metadata_repair",
        training_updates=0, model_constructors=0, optimizer_constructors=0, cuda_kernels=0,
        qualification_input=False, source_digest=EXPECTED_SOURCE, previous_source_digest=OLD_SOURCE, roster_count=len(ids),
        planned_count=sum(row["status"] == "PLANNED" for row in rows),
        unchanged_declaration_refusals=sum(row["status"] == "UNCHANGED_DECLARATION_REFUSAL" for row in rows),
        required_tier1_and_tier2_tasks_checked=len(selected), source_pins_checked=source_checks,
        unchanged_other_task_files=len(unchanged),
        unchanged_nonword_semantics_after_candidate_normalization=unchanged_jobs,
        actual_unchanged_job_keys=0, automatic_cross_source_reuse_available=False,
        changed_word_job_keys=changed_jobs, representative=representative,
        old_task_sha256=file_hash(baseline / TASK_PATH), new_task_sha256=file_hash(root / TASK_PATH),
        helper_sha256={"previous": before["evaluation"]["sources"][HELPER], "current": file_hash(root / HELPER)},
        task_execution_fingerprint=task_execution_fingerprint(after),
        task_evaluation_fingerprint={"previous": task_evaluation_fingerprint(before), "current": task_evaluation_fingerprint(after)},
        roster_declaration_sha256=file_hash(round_path), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--round", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with patch.object(torch.nn.Module, "__init__", forbidden), patch.object(torch.optim.Optimizer, "__init__", forbidden):
        result = audit(args.root.resolve(), args.baseline.resolve(), args.round.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "roster_count", "planned_count", "source_digest",
                       "actual_unchanged_job_keys", "changed_word_job_keys", "training_updates")}), flush=True)


if __name__ == "__main__":
    main()

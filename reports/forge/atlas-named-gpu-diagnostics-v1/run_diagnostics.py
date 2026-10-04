"""Finite GPU1 continuation for three explicit named Atlas host adaptations.

The public Forge runtime owns all updates; the independent evaluator owns all
numerical decisions. This wrapper owns source reconstruction, disjoint lanes,
durable admission and retained-array goal media. It grants no qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = "reports/forge/atlas-named-gpu-diagnostics-v1"
SELF = DIRECTORY + "/run_diagnostics.py"
DELEGATE = "reports/forge/atlas-current-gpu-diagnostics-v1/run_diagnostics.py"
SCHEMA = "particlegan_atlas_named_gpu_diagnostics_v1"
OVERRIDES = {"lr": .0053125, "prior_lr_mult": 1.5}
ORDINARY_DECLARATION = "configs/forge/ideas/atlas-c6-observed-policy-current-v1.json"
LEGACY_ADMISSION_BLOCKER = "new v1 declaration is not immutable legacy evidence; use a v2 decision_contract"
PRIOR_SUMMARY = "reports/forge/atlas-named-gpu-diagnostics-invalid-20261003/summary.json"
PRIOR_SOURCE = {"commit": "aee59bea7f1a629d71052fbb010d930650a4ed26",
                "digest": "73b00a75863383e4038a3731c8213a7fb072971712a05aab79efa181a93c0280"}
PRIOR_DEBITS = {"0": 12.873334385920316, "1": 12.449620655039325}
ENGINEERING_REFERENCE = {"summary": {"path": PRIOR_SUMMARY, "sha256": "7fee64a49aa56b2cec25ffcf4861eccd91ba6886fd33f9e95565dce1b701cfd7", "bytes": 235580},
                         "source": PRIOR_SOURCE, "paid_seconds_by_lane": PRIOR_DEBITS,
                         "reserved_seconds": 0., "qualification_input": False, "authorized_successors": 1}
PRIOR_PUBLICATION = "reports/forge/atlas-named-gpu-diagnostics-native-v3-20261003/results.json"
V3_SOURCE = {"commit": "ff94453b45e02fd451b21c26d5991f6ed435c292",
             "digest": "87fcd4e28bdcd9f347379b7db1fbcf0af3edcc0fbd5e97e4a98d9a49f8987a87"}
V3_DEBITS = {"0": 221.9527463898994, "1": 13.754588949028403}
HISTORICAL_LANE_DEBITS = {key: PRIOR_DEBITS[key] + V3_DEBITS[key] for key in PRIOR_DEBITS}
CONTINUATION_REFERENCE = {
    "publication": {"path": PRIOR_PUBLICATION, "sha256": "8ba7230a71041e3a0185528c9e76c5f39bff6eeadfb98b59d715ef72ff76b9e0", "bytes": 518012},
    "source": V3_SOURCE, "current_paid_seconds_by_lane": V3_DEBITS,
    "inclusive_historical_paid_seconds_by_lane": HISTORICAL_LANE_DEBITS,
    "reserved_seconds": 0., "qualification_input": False, "outcomes_reused": False,
    "completed_cases_reexecuted": False,
}
FLAGS = ("qualification_input", "ordinary_tier_credit", "calibration_credit",
         "default_adoption", "cross_cohort_pooling", "speed_ranking")
ENVIRONMENT = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
# A fixed family/cohort/physical-lane declaration, not a configurable sweep.
FAMILIES = {
    "atlas_conditional": {"cohort": "conditional_policy_selected_cloud_v1", "gpu": "0", "cap": 7200,
                          "parents": ("trajectory", "residual_student", "unipolar", "mid_scale_identity")},
    "atlas_ae_routed": {"cohort": "ae_routed_policy_v1", "gpu": "0", "cap": 300,
                        "parents": ("ae_gan_hold",)},
    "atlas_routed": {"cohort": "routed_policy_selected_cloud_v1", "gpu": "1", "cap": 300,
                     "parents": ("unused_token_hold",)},
    "atlas_multibank": {"cohort": "multibank_policy_v1", "gpu": "1", "cap": 1800,
                        "parents": ("cover_leftover",)},
    "atlas_word_joint_min11": {"cohort": "word_joint_policy_min11_v1", "gpu": "1", "cap": 900,
                              "parents": ("five_word_joint_acquisition",)},
}
ACTIVE_FAMILIES = ("atlas_routed", "atlas_multibank", "atlas_word_joint_min11")
PRESERVED_FAMILIES = ("atlas_conditional", "atlas_ae_routed")
FIXED_HOSTS = {
    "trajectory": (400, 1800), "residual_student": (400, 1800), "unipolar": (400, 1800),
    "mid_scale_identity": (800, 1800), "ae_gan_hold": (250, 300), "unused_token_hold": (200, 300),
    "cover_leftover": (800, 1800), "five_word_joint_acquisition": (20001, 900),
}
_legacy = None


def legacy():
    """Only generic immutable pin/lease/import/charge/selection utilities."""
    global _legacy
    if _legacy is None:
        loader = importlib.util.spec_from_file_location("_atlas_named_diagnostic_retained_utilities", ROOT / DELEGATE)
        _legacy = importlib.util.module_from_spec(loader)
        sys.modules[loader.name] = _legacy
        loader.loader.exec_module(_legacy)
    return _legacy


def forge(name):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("experiments.forge." + name)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    forge("contracts").atomic_json(Path(path), value)


def pin(path):
    return legacy().pin(path)


def check_pin(value):
    return legacy().check_pin(value)


def active_rows(spec, family):
    return [row for row in spec["cases"] if row["family"] == family]


def validate_spec(spec, root=None):
    fixed = {"schema": SCHEMA, "id": "atlas-named-hosts-current-gpu-diagnostics-v4",
             "view": "discriminator_stability", "recipe_preset": "atlas", "recipe_overrides": OVERRIDES,
             "seed": 0, "cuda_model": "NVIDIA RTX A6000", "required_slots_per_family": 26,
             "families": list(FAMILIES), "executable_families": list(ACTIVE_FAMILIES),
             "preserved_v3_families": list(PRESERVED_FAMILIES), "executable_physical_gpus": ["1"],
             "executable_slots": 3, "executable_jobs": 3,
             "diagnostic_cap_seconds": 10500, "lane_cap_seconds": {"0": 7500, "1": 3000},
             "full_view_tiers": {"1": 5, "2": 19, "3": 2}, "frames": 9, "export_grace_seconds": 0,
             "failure_policy": "continue_completed_numerical_FAIL_halt_invalid_lane_no_retry",
             "evidence_use": "named_policy_host_diagnostic", "engineering_carryover": ENGINEERING_REFERENCE,
             "continuation_carryover": CONTINUATION_REFERENCE,
             "resources": {"host_memory_mb": 2048, "cpu_threads": 1, "minimum_free_gpu_memory_mib": 12288,
                           "maximum_gpu_temperature_c": 82, "memory_fraction": .2}}
    if any(canonical(spec.get(k)) != canonical(v) for k, v in fixed.items()) or any(spec.get(k) is not False for k in FLAGS):
        raise ValueError("the one finite named-family/device/budget/nonqualification contract changed")
    rows = spec.get("cases")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("exactly the three remaining GPU1 named questions are required")
    expected = [(family, parent) for family in ACTIVE_FAMILIES for parent in FAMILIES[family]["parents"]]
    if [(r.get("family"), r.get("parent_id")) for r in rows] != expected:
        raise ValueError("missing, duplicated, borrowed or reordered named family question")
    for row in rows:
        family, parent = row["family"], row["parent_id"]
        info = FAMILIES[family]; name = parent + "_" + info["cohort"]
        path = "configs/forge/task-variants/" + info["cohort"] + "/" + name + ".json"
        if (row.get("id") != name or row.get("cohort") != info["cohort"] or row.get("gpu") != info["gpu"]
                or row.get("definition") != path or type(row.get("steps")) is not int
                or type(row.get("timeout_seconds")) is not int
                or (row["steps"], row["timeout_seconds"]) != FIXED_HOSTS[parent]
                or any(type(row.get(k)) is not str or len(row[k]) != 64 or any(c not in "0123456789abcdef" for c in row[k])
                       for k in ("sha256", "evaluation_sha256", "execution_sha256"))):
            raise ValueError("named task identity/horizon/allowance/source changed")
        if root is not None:
            actual = Path(root) / path
            if actual.is_symlink() or file_hash(actual) != row["sha256"]:
                raise ValueError("named task source drift: " + name)
            task = read(actual)
            forge("policy_cohorts").validate_policy_task(task, root=root, allow_compiler_annotations=False)
            if (task["id"] != name or task["policy_family"] != family or task["policy_parent"]["id"] != parent
                    or digest(task["evaluation"]) != row["evaluation_sha256"]
                    or digest(task["execution"]) != row["execution_sha256"]
                    or task["execution"]["steps"] != row["steps"]
                    or task["resources"]["timeout_seconds"] != row["timeout_seconds"]
                    or task["execution"]["device"] != "cuda" or task["resources"]["gpus"] != 1
                    or task["resources"]["cpu_threads"] != 1 or task["resources"].get("allow_cpu") is not False
                    or task["evaluation"]["observations"] != 24 or task["evaluation"]["minimum_stable_checks"] != 5):
                raise ValueError("actual named task law/gates/cadence/resources differ")
    return spec


def engineering_carryover(spec, root, *, durable=False):
    """Old INVALID receipts supply cost only, never current task outcomes."""
    if canonical(spec.get("engineering_carryover")) != canonical(ENGINEERING_REFERENCE):
        raise ValueError("the exact old engineering debit/source cannot reset or change")
    path = Path(root) / PRIOR_SUMMARY; summary_pin = ENGINEERING_REFERENCE["summary"]
    if path.is_symlink() or file_hash(path) != summary_pin["sha256"] or path.stat().st_size != summary_pin["bytes"]:
        raise ValueError("old engineering publication missing or changed")
    prior = read(path)
    if (prior.get("status") != "INVALID" or prior["source"].get("origin_commit") != PRIOR_SOURCE["commit"]
            or prior["source"].get("digest") != PRIOR_SOURCE["digest"]
            or prior.get("qualification_input") is not False or prior["counts"].get("completed_updates") != 0
            or prior["counts"].get("all_planned_family_cells") != {"INVALID": 2, "NOT_RUN": 128}
            or prior["cost"].get("reserved_seconds") != 0. or len(prior["attempts"]) != 2):
        raise ValueError("old engineering attribution changed")
    records = []
    for item, (physical, family) in zip(prior["attempts"], (("0", "atlas_conditional"), ("1", "atlas_routed"))):
        expected_cost = PRIOR_DEBITS[physical]
        if (item.get("family") != family or item.get("physical_gpu") != physical or item.get("status") != "INVALID"
                or item.get("scientific_gate") != "UNAVAILABLE" or item.get("completed_updates") != 0
                or item.get("observations") != 0 or item.get("goal_gifs") != 0
                or item.get("supervisor_status") != "completed" or item.get("child_returncode") != 1
                or any(item["cost"].get(key) != value for key, value in
                       (("paid_seconds", expected_cost), ("charged_seconds", expected_cost), ("reserved_seconds", 0.)))):
            raise ValueError("old attempts supply only the two exact measured engineering debits")
        if durable:
            for artifact in item["artifacts"].values():
                check_pin(artifact)
            terminal = read(check_pin(item["artifacts"]["terminal"]))
            supervisor = read(check_pin(item["artifacts"]["supervisor_request"]))
            old_state = read(check_pin(item["artifacts"]["study"]))
            old_row = old_state["jobs"][0]
            if (hashlib.sha256(terminal["token"].encode()).hexdigest() != item["token_sha256"]
                    or supervisor["token"] != terminal["token"] or old_row["token"] != terminal["token"]
                    or supervisor["source"].get("digest") != PRIOR_SOURCE["digest"]
                    or supervisor["source"].get("origin_commit") != PRIOR_SOURCE["commit"]
                    or terminal.get("attempt_status") != "completed" or terminal.get("child_returncode") != 1
                    or terminal.get("paid_wall_seconds") != expected_cost
                    or old_state.get("status") != "INVALID" or old_row.get("status") != "INVALID"
                    or old_row.get("outcome") is not None or old_row.get("charged_seconds") != expected_cost
                    or old_row.get("unmeasured_interrupt_reserved_seconds") != 0.):
                raise ValueError("old durable terminal/source/cost proof changed")
        records.append({"physical_gpu": physical, "family": family, "status": "INVALID",
                        "paid_seconds": expected_cost, "reserved_seconds": 0.,
                        "artifacts": deepcopy(item["artifacts"]), "qualification_input": False})
    if prior["cost"].get("paid_seconds") != sum(PRIOR_DEBITS.values()):
        raise ValueError("old aggregate debit differs from the two durable measurements")
    return {"reference": deepcopy(ENGINEERING_REFERENCE), "attempts": records, "qualification_input": False}


def continuation_carryover(spec, root, *, durable=False):
    """The immutable v3 cut supplies paid costs and separate old outcomes only."""
    if canonical(spec.get("continuation_carryover")) != canonical(CONTINUATION_REFERENCE):
        raise ValueError("the exact v3 continuation debit/source cannot reset or change")
    reference = CONTINUATION_REFERENCE["publication"]
    path = Path(root) / reference["path"]
    if path.is_symlink() or file_hash(path) != reference["sha256"] or path.stat().st_size != reference["bytes"]:
        raise ValueError("the completed v3 publication is missing or changed")
    previous = read(path)
    if (previous.get("schema") != "particlegan_atlas_named_gpu_diagnostics_publication_v3"
            or previous.get("source") != {"origin_commit": V3_SOURCE["commit"], "digest": V3_SOURCE["digest"]}
            or any(previous.get(key) is not False for key in (*FLAGS, "reuse"))
            or previous.get("counts") != {"attempted_adapted_questions": 6, "declared_adapted_questions": 8,
                "required_family_cells": 130, "required_slots_per_family": 26,
                "statuses": {"INVALID": 1, "NOT_RUN": 124, "PASS": 5}, "terminal_families": 3,
                "tiers_per_family": {"1": 5, "2": 19, "3": 2}}
            or previous.get("engineering_carryover", {}).get("reference") != ENGINEERING_REFERENCE
            or set(previous.get("families", {})) != set(FAMILIES)):
        raise ValueError("v3 source/full denominators/outcome-only scope changed")
    cost = previous.get("cost", {})
    if (cost.get("original_cap_seconds") != 10500 or cost.get("current_paid_seconds") != sum(V3_DEBITS.values())
            or cost.get("current_reserved_seconds") != 0.
            or cost.get("prior_engineering_paid_seconds") != sum(PRIOR_DEBITS.values())
            or cost.get("inclusive_charged_seconds") != sum(V3_DEBITS.values()) + sum(PRIOR_DEBITS.values())
            or set(cost.get("lanes", {})) != {"0", "1"}):
        raise ValueError("v3 aggregate cost may not reset, duplicate or hide a reserve")
    for physical in ("0", "1"):
        if cost["lanes"][physical] != {"cap_seconds": spec["lane_cap_seconds"][physical],
                "current_paid_seconds": V3_DEBITS[physical], "current_reserved_seconds": 0.,
                "inclusive_charged_seconds": HISTORICAL_LANE_DEBITS[physical],
                "prior_engineering_paid_seconds": PRIOR_DEBITS[physical]}:
            raise ValueError("v3 costs cannot borrow another physical lane")
    attempts, preserved, per_lane = [], [], {"0": 0., "1": 0.}
    for family, info in FAMILIES.items():
        old = previous["families"][family]
        expected_status = "COMPLETE_DIAGNOSTIC" if family in PRESERVED_FAMILIES else "INVALID" if family == "atlas_routed" else "NOT_RUN"
        expected_counts = {"PASS": len(info["parents"]), "NOT_RUN": 26 - len(info["parents"])} if family in PRESERVED_FAMILIES else {"INVALID": 1, "NOT_RUN": 25} if family == "atlas_routed" else {"NOT_RUN": 26}
        expected_attempts = len(info["parents"]) if family in PRESERVED_FAMILIES or family == "atlas_routed" else 0
        slot_names = [parent + "_" + info["cohort"] if parent in info["parents"] else parent for parent in legacy().PARENTS]
        active_names = {parent + "_" + info["cohort"] for parent in info["parents"]}
        slots = old.get("slots", {})
        active_status = "PASS" if family in PRESERVED_FAMILIES else "INVALID" if family == "atlas_routed" else "NOT_RUN"
        if (old.get("family") != family or old.get("cohort") != info["cohort"]
                or old.get("physical_gpu") != info["gpu"] or old.get("family_cap_seconds") != info["cap"]
                or old.get("required_slots") != 26 or old.get("tiers") != {"1": 5, "2": 19, "3": 2}
                or old.get("status") != expected_status or old.get("counts") != expected_counts
                or set(slots) != set(slot_names)
                or any(row.get("qualification_input") is not False
                    or row.get("diagnostic_status") != (active_status if name in active_names else "NOT_RUN")
                    for name, row in slots.items())
                or old.get("reserved_seconds") != 0.
                or old.get("charged_seconds") != old.get("paid_seconds")
                or len(old.get("attempts", [])) != expected_attempts
                or any(old.get(key) is not False for key in FLAGS) or old.get("ordinary_qualified_tier") != 0):
            raise ValueError("v3 family/source scope or complete terminal prefix changed")
        if durable and expected_attempts:
            state = read(check_pin(old["study_input"]))
            source = state.get("source", {})
            if (source.get("origin_commit") != V3_SOURCE["commit"] or source.get("digest") != V3_SOURCE["digest"]
                    or digest(source.get("files")) != source.get("digest")
                    or len(source.get("files", {})) != previous.get("source_file_count")
                    or any(Path(name).is_absolute() or ".." in Path(name).parts for name in source.get("files", {}))
                    or state.get("execution_source") != source or state.get("family") != family
                    or state.get("status") != expected_status or state.get("spec_sha256") != previous["spec_sha256"]
                    or state.get("spec", {}).get("id") != "atlas-named-hosts-current-gpu-diagnostics-v3"
                    or state.get("measured_paid_seconds") != old["paid_seconds"]
                    or state.get("unmeasured_interrupt_reserved_seconds") != 0.
                    or state.get("spent_seconds") != old["paid_seconds"]
                    or state.get("lane_runtime") != old.get("runtime") or state.get("slots") != old["slots"]
                    or state.get("qualification_input") is not False or len(state.get("jobs", [])) != expected_attempts):
                raise ValueError("v3 durable study/source/cost attribution changed")
            manifest = Path(source["snapshot_path"]) / "forge-source.json"
            if read(manifest) != {key: value for key, value in source.items() if key != "snapshot_path"}:
                raise ValueError("v3 frozen source manifest differs from its executed study")
        paid = 0.
        for index, (item, parent) in enumerate(zip(old["attempts"], info["parents"])):
            name = parent + "_" + info["cohort"]
            complete = family in PRESERVED_FAMILIES
            amount = legacy().number(item.get("paid_seconds"), "v3 measured paid")
            if (item.get("status") != ("COMPLETE" if complete else "INVALID")
                    or item.get("task_ids") != [name] or item.get("allowance_seconds") != FIXED_HOSTS[parent][1]
                    or item.get("reserved_seconds") != 0. or item.get("charged_seconds") != amount
                    or item.get("terminal_status") != "completed" or item.get("child_returncode") != (0 if complete else 1)
                    or item.get("qualification_input") is not False
                    or (complete and (item.get("outcome", {}).get("status") != "PASS"
                        or item["outcome"].get("completed_steps") != FIXED_HOSTS[parent][0]
                        or item["outcome"].get("observations") != 24))):
                raise ValueError("v3 attempt terminal/allowance/grade cannot change")
            if durable:
                row = state["jobs"][index]; terminal = read(check_pin(item["terminal"]))
                # The exact supervisor bytes are bound by the publication input index below.
                supervisor = read(Path(item["terminal"]["path"]).with_name("supervisor-request.json"))
                if (row.get("terminal") != item["terminal"] or row.get("attempt_key") != item.get("attempt_key")
                        or row.get("compatibility_key") != item.get("compatibility_key") or row.get("task_ids") != [name]
                        or row.get("status") != item["status"]
                        or row.get("paid_wall_seconds") != amount or row.get("charged_seconds") != amount
                        or row.get("unmeasured_interrupt_reserved_seconds") != 0.
                        or terminal.get("token") != row.get("token") or supervisor.get("token") != row.get("token")
                        or hashlib.sha256(str(row.get("token")).encode()).hexdigest() != item.get("token_sha256")
                        or terminal.get("attempt_status") != "completed" or terminal.get("child_returncode") != item["child_returncode"]
                        or terminal.get("paid_wall_seconds") != amount or supervisor.get("source") != source):
                    raise ValueError("v3 durable terminal/supervisor source or paid proof changed")
                job = next(job for job in state["request"]["jobs"] if job["task_ids"] == [name])
                if job.get("budget_seconds") != item["allowance_seconds"] or job.get("compatibility_key") != item["compatibility_key"]:
                    raise ValueError("v3 original job identity/allowance changed")
                if complete:
                    verify_predecessor_outcome(state, row, job)
                    outcome = item["outcome"]
                    if (row["outcome"]["raw"] != outcome["raw_input"]
                            or row["outcome"]["grading"] != outcome["grading_input"]
                            or row["outcome"]["media"][name] != outcome["media_input"]):
                        raise ValueError("v3 publication outcome differs from its retained original grade/media")
                else:
                    check_pin(item["raw_error_input"])
                    if "outcome" in row or item.get("numerical_gate") != "UNAVAILABLE" or item.get("goal_gif") is not None:
                        raise ValueError("v3 engineering error cannot acquire a numerical gate or media")
            attempts.append({"family": family, "physical_gpu": info["gpu"], "task_ids": [name],
                "status": item["status"], "paid_seconds": amount, "reserved_seconds": 0.,
                "terminal": deepcopy(item["terminal"]), "token_sha256": item["token_sha256"],
                "study": deepcopy(old["study_input"]), "qualification_input": False})
            if complete:
                preserved.append({"family": family, "task_id": name, "status": "PASS",
                    "original_source": deepcopy(V3_SOURCE), "grade": deepcopy(item["outcome"]["grade_publication"]),
                    "gif": deepcopy(item["outcome"]["gif_publication"]), "qualification_input": False,
                    "current_slots_reused": False, "reexecution_authorized": False})
            paid += amount
        if not math.isclose(old["paid_seconds"], paid, rel_tol=0, abs_tol=1e-10):
            raise ValueError("v3 family debit differs from its exact measured terminals")
        per_lane[info["gpu"]] += paid
    if any(not math.isclose(per_lane[key], V3_DEBITS[key], rel_tol=0, abs_tol=1e-10) for key in per_lane):
        raise ValueError("v3 lane cost does not match all six retained attempts")
    if durable:
        index_ref = previous["input_index"]
        index_path = path.parent / index_ref["path"]
        if index_path.is_symlink() or file_hash(index_path) != index_ref["sha256"] or index_path.stat().st_size != index_ref["bytes"]:
            raise ValueError("v3 original source/artifact input index changed")
        entries = read(index_path)
        files = entries.get("files", [])
        if (entries.get("raw_files_changed") is not False or entries.get("file_count") != index_ref["file_count"]
                or len(files) != index_ref["file_count"] or len({item["path"] for item in files}) != len(files)):
            raise ValueError("v3 input source/artifact closure is incomplete or duplicated")
        identities = {item["path"]: item for item in files}
        if identities.get(previous["trusted_cut"]["path"]) != previous["trusted_cut"]:
            raise ValueError("v3 trusted terminal cut is absent from its original input closure")
        for item in files:
            check_pin(item)
        for family in (*PRESERVED_FAMILIES, "atlas_routed"):
            state = read(check_pin(previous["families"][family]["study_input"]))
            snapshot = Path(state["source"]["snapshot_path"])
            required = [snapshot / "forge-source.json", *[snapshot / name for name in state["source"]["files"]]]
            required += [Path(item["terminal"]["path"]).with_name("supervisor-request.json")
                         for item in previous["families"][family]["attempts"]]
            if any(str(item) not in identities for item in required):
                raise ValueError("v3 source or supervisor proof is absent from the original input closure")
            if any(identities[str(snapshot / name)]["sha256"] != sha for name, sha in state["source"]["files"].items()):
                raise ValueError("v3 actual source bytes differ from the executed manifest")
    return {"reference": deepcopy(CONTINUATION_REFERENCE), "attempts": attempts,
            "preserved_outcomes": preserved, "qualification_input": False, "outcomes_reused": False}


def declaration(root, spec, family):
    """Inline source-bound diagnostic declaration; ordinary draft stays intact."""
    base = forge("planning").load_idea(Path(root), "atlas-c6-observed-policy-current-v1")
    base = deepcopy(base)
    ordinary_reference = {"candidate_id": base["id"], "path": ORDINARY_DECLARATION,
                          "sha256": file_hash(Path(root) / ORDINARY_DECLARATION),
                          "decision_contract": base.pop("decision_contract"), "execution_authorized": False}
    base.update(schema_version=1, id=family + "-c6-named-gpu-diagnostic-v1", trainer_family=family,
                task_cohort=FAMILIES[family]["cohort"], execution_path="public_trainer",
                recipe_preset="atlas", recipe_overrides=deepcopy(OVERRIDES),
                requires_capabilities=["a2", "named_rng", "policy_controls", "policy_serving"],
                hypothesis="Execute the explicitly named host adaptation on GPU under one fixed Atlas pair; preserve original gates and all 26 required questions without qualification credit.",
                changed_factors=["Explicit named family/serving/routing/resource law for its original question, independent of ordinary prerequisite promotion"],
                qualification_scope={"diagnostic_only": True, "ordinary_credit": False, "seed": 0,
                                     "full_required_roster": 26, "historical_reuse": False},
                reference_context_scope="Generic Atlas public_trainer Recipe metadata only; all adapted executions are explicitly task-owned public_components with their own prior/row/encoder laws.",
                ordinary_decision_reference=ordinary_reference)
    # The unchanged v2 contract is inactive provenance. Schema-v1 metadata
    # resolution intentionally remains ineligible for ordinary legacy admission;
    # only this separately fixed diagnostic dispatcher authorizes these jobs.
    base["source_files"] = sorted(set(base.get("source_files", ())) | {ORDINARY_DECLARATION})
    return base


def request_from_base(base, spec, family, source):
    validate_spec(spec)
    info = FAMILIES[family]; candidate = base["candidate"]
    if (candidate.get("id") != family + "-c6-named-gpu-diagnostic-v1"
            or candidate.get("trainer_family") != family or candidate.get("task_cohort") != info["cohort"]
            or candidate.get("recipe_preset") != "atlas" or candidate.get("execution_path") != "public_trainer"
            or canonical(candidate.get("recipe_overrides")) != canonical(OVERRIDES)
            or base["protocol"].get("seed") != 0 or base.get("execution_backend") != "cuda"):
        raise ValueError("actual named candidate/Recipe/seed/backend differs")
    reference = candidate.get("ordinary_decision_reference", {})
    if (candidate.get("schema_version") != 1 or "decision_contract" in candidate
            or reference.get("candidate_id") != "atlas-c6-observed-policy-current-v1"
            or reference.get("path") != ORDINARY_DECLARATION or reference.get("execution_authorized") is not False
            or reference.get("decision_contract", {}).get("status") != "draft"
            or not isinstance(reference.get("sha256"), str) or len(reference["sha256"]) != 64
            or base.get("decision_review") is not None or base.get("decision_admission") is not None):
        raise ValueError("ordinary decision reference must stay inactive and source-bound")
    blockers = list(base.get("preflight_blockers", []))
    if blockers.count(LEGACY_ADMISSION_BLOCKER) != 1:
        raise ValueError("expected immutable-legacy admission refusal is absent or ambiguous")
    blockers.remove(LEGACY_ADMISSION_BLOCKER)
    if blockers:
        raise ValueError("nondecision source/API blockers: " + "; ".join(blockers))
    assignments = base["view"]["assignments"]
    mapping = {parent: parent + "_" + info["cohort"] for parent in info["parents"]}
    expected = [mapping.get(parent, parent) for parent in legacy().PARENTS]
    if (list(base["tasks"]) != expected or [a["task"] for a in assignments] != expected
            or base["view"].get("revision") != 4 or base["view"].get("policy_family") != family
            or [sum(a["qualification_tier"] == tier for a in assignments) for tier in (1, 2, 3)] != [5, 19, 2]
            or any(a["importance"] != "required" for a in assignments)):
        raise ValueError("each actual named family retains its own complete 5/19/2 view")
    for row in active_rows(spec, family):
        task = base["tasks"][row["id"]]
        if (task.get("policy_family") != family or task.get("task_cohort") != info["cohort"]
                or task["policy_parent"]["id"] != row["parent_id"]
                or task["execution"].get("execution_path") != "public_components"):
            raise ValueError("borrowed named question")
    request = deepcopy(base)
    request.update(source=deepcopy(source), diagnostic_contract=deepcopy(spec), diagnostic_contract_sha256=digest(spec),
                   evidence_scope=spec["evidence_use"], qualification_reuse=False,
                   campaign_id=spec["id"] + "--" + family, ordinary_request_sha256=digest(base),
                   ordinary_admission={"status": "BLOCKED", "blockers": [LEGACY_ADMISSION_BLOCKER],
                                       "execution_authorized": False})
    request["candidate_revision"] = forge("planning").candidate_revision_for(source["digest"], candidate)
    for job in request["jobs"]:
        job["science"].update(candidate_revision=request["candidate_revision"], evidence_use=spec["evidence_use"],
                             diagnostic_contract_sha256=digest(spec), named_family=family)
    forge("planning").rekey_jobs(request["jobs"])
    for row in active_rows(spec, family):
        jobs = [j for j in request["jobs"] if row["id"] in j["task_ids"]]
        if len(jobs) != 1 or jobs[0]["task_ids"] != [row["id"]] or jobs[0]["budget_seconds"] != row["timeout_seconds"]:
            raise ValueError("an adapted host must be one unchanged full-budget job")
    return json.loads(canonical(request))


def build_requests(root, spec, source=None):
    validate_spec(spec, root)
    requests = {family: forge("planning").resolve_idea(Path(root), family + "-c6-named-gpu-diagnostic-v1",
                declaration=declaration(root, spec, family), view_id=spec["view"], through_tier=1,
                freeze_source=False, execution_backend="cuda", cuda_model=spec["cuda_model"]) for family in FAMILIES}
    if source is None:
        extras = {SELF, DELEGATE, DIRECTORY + "/protocol.json", DIRECTORY + "/README.md", PRIOR_SUMMARY,
                  PRIOR_PUBLICATION, str(Path(PRIOR_PUBLICATION).with_name("input-index.json"))}
        extras.update(str(p.relative_to(root)) for p in (Path(root) / "configs/forge").rglob("*.json"))
        for request in requests.values():
            extras.update(request["source"]["files"])
        source = forge("sources").inspect_source(Path(root), sorted(extras))
    elif any(source["files"].get(p) != h for request in requests.values() for p, h in request["source"]["files"].items()):
        raise ValueError("named request reconstruction escaped the frozen source")
    return {family: request_from_base(base, spec, family, source) for family, base in requests.items()}, source


def case_definitions(requests, spec):
    result = {}
    for row in spec["cases"]:
        request = requests[row["family"]]
        job = next(j for j in request["jobs"] if row["id"] in j["task_ids"])
        result[row["id"]] = {"task": request["tasks"][row["id"]], "job_science": job["science"],
                              "family": row["family"], "evidence_use": spec["evidence_use"]}
    return result


def card_for(requests, spec, source):
    return {"schema": SCHEMA + "_structural_readiness", "source_digest": source["digest"], "contract_sha256": digest(spec),
            "task_preflight": {r["id"]: requests[r["family"]]["tasks"][r["id"]].get("preflight_blockers", []) for r in spec["cases"]},
            "required_questions_per_family": 26, "declared_adapted_questions": 3,
            "preserved_old_outcomes": 5, "old_outcomes_are_current_credit": False,
            "model_capacity_proved": False, "learned_quality_proved": False, "optimizer_updates": 0,
            "qualification_input": False, "claim": "Metadata/API/source readiness only; no capacity, CPU numerical or learned-quality proof."}


def preparation_path(output):
    output = Path(output).resolve()
    return output.parent / ("." + output.name + ".atlas-named-gpu-diagnostics.json")


def prepare(spec_path, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("prepare a new empty output; no overwritten evidence or failed retry")
    spec = validate_spec(read(spec_path), ROOT)
    requests, source = build_requests(ROOT, spec)
    snapshot = forge("sources").snapshot_source(ROOT, output.parent / "named-diagnostic-source", source)
    source = {**source, "snapshot_path": str(snapshot)}
    requests, _ = build_requests(snapshot, spec, source)
    card = card_for(requests, spec, source)
    card_path = output.parent / ("." + output.name + ".named-structural-readiness.json")
    write(card_path, card)
    runtimes = [r["runtime"] for r in requests.values()]
    if any(runtime != runtimes[0] for runtime in runtimes):
        raise ValueError("one frozen software runtime is required")
    packet = {"schema": SCHEMA, "spec": {**spec, "representation_card": pin(card_path)}, "spec_sha256": digest(spec),
              "requests": requests, "source": source, "execution_source": source, "capacity_preflight": card,
              "case_definitions": case_definitions(requests, spec), "runtime_contract": runtimes[0],
              "family_paid_budget_seconds": {family: info["cap"] for family, info in FAMILIES.items()},
              "engineering_carryover": engineering_carryover(spec, ROOT, durable=True),
              "continuation_carryover": continuation_carryover(spec, ROOT, durable=True)}
    path = preparation_path(output)
    if path.exists() and read(path) != packet:
        raise ValueError("preserve previous preparation/source; declare a new output")
    write(path, packet)
    return packet


def verify_packet(packet):
    spec = deepcopy(packet["spec"]); card_pin = spec.pop("representation_card")
    validate_spec(spec)
    if packet.get("schema") != SCHEMA or packet.get("spec_sha256") != digest(spec) or packet.get("source") != packet.get("execution_source"):
        raise ValueError("prepared source/contract identity changed")
    source = packet["source"]; snapshot = Path(source["snapshot_path"])
    for relative, path in ((SELF, Path(__file__)), (DELEGATE, ROOT / DELEGATE)):
        if source["files"].get(relative) != file_hash(path):
            raise ValueError("current wrapper/delegated utility differs from the frozen source")
    forge("sources").verify_snapshot(snapshot, source)
    if read(snapshot / "forge-source.json") != {k: v for k, v in source.items() if k != "snapshot_path"}:
        raise ValueError("snapshot metadata changed")
    if (source["files"].get(PRIOR_SUMMARY) != ENGINEERING_REFERENCE["summary"]["sha256"]
            or packet.get("engineering_carryover") != engineering_carryover(spec, snapshot, durable=True)):
        raise ValueError("frozen engineering debit/artifacts/source changed")
    if (source["files"].get(PRIOR_PUBLICATION) != CONTINUATION_REFERENCE["publication"]["sha256"]
            or packet.get("continuation_carryover") != continuation_carryover(spec, snapshot, durable=True)):
        raise ValueError("frozen v3 source/terminal costs or separate preserved outcomes changed")
    requests, _ = build_requests(snapshot, spec, source)
    if canonical(requests) != canonical(packet["requests"]):
        raise ValueError("candidate/task/Recipe/runtime/compatibility reconstruction changed")
    card = card_for(requests, spec, source)
    if (read(check_pin(card_pin)) != card or packet.get("capacity_preflight") != card
            or packet.get("case_definitions") != case_definitions(requests, spec)
            or packet.get("family_paid_budget_seconds") != {f: i["cap"] for f, i in FAMILIES.items()}
            or any(r["runtime"] != packet.get("runtime_contract") for r in requests.values())):
        raise ValueError("readiness/full-roster/quota/runtime identity changed")
    legacy().guard_imports(source, execution_root=ROOT)
    return spec


def family_packet(packet, family, predecessor_paths=()):
    if family not in ACTIVE_FAMILIES:
        raise ValueError("v4 executes only the three remaining GPU1 families; preserved families cannot rerun")
    result = deepcopy(packet)
    result.update(family=family, request=deepcopy(packet["requests"][family]),
                  lane_runtime=lane_runtime(packet["requests"][family], FAMILIES[family]["gpu"]),
                  lane_predecessors=[{"family": name, "study": pin(path)} for name, path in predecessor_paths])
    return result


def lane_runtime(request, physical):
    compute = request["compute_profiles"]["cuda"]
    if physical not in {"0", "1"} or compute.get("availability") == "unavailable" or compute.get("model") != "NVIDIA RTX A6000":
        raise ValueError("fixed GPU runtime unavailable")
    return {**request["runtime"], "device": "cuda:0", "physical_gpu": physical,
            "cuda_device_model": compute["model"], "compute": compute, "torch_threads": 1}


def configure_environment(physical=None):
    if physical is not None and physical != "1":
        raise ValueError("v4 has only the declared physical GPU1 continuation lane")
    os.environ.update(ENVIRONMENT)
    os.environ["CUDA_VISIBLE_DEVICES"] = physical or ""


def gpu_readiness(spec, physical, query=None):
    if physical != "1":
        raise ValueError("v4 cannot admit the preserved physical GPU0 lane")
    command = ["nvidia-smi", "--id=" + physical, "--query-gpu=index,name,memory.free,temperature.gpu", "--format=csv,noheader,nounits"]
    value = query(command) if query else subprocess.check_output(command, text=True)
    rows = [line.strip().split(",") for line in value.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("unavailable/ambiguous physical lane telemetry")
    device, model, free, temperature = [v.strip() for v in rows[0]]
    free = legacy().number(float(free), "free GPU memory"); temperature = legacy().number(float(temperature), "GPU temperature")
    if (device != physical or model != spec["cuda_model"] or free < spec["resources"]["minimum_free_gpu_memory_mib"]
            or temperature > spec["resources"]["maximum_gpu_temperature_c"]):
        raise ValueError("unsafe/wrong physical lane; zero admission/reservation")
    return {"physical_gpu": device, "model": model, "free_memory_mib": free, "temperature_c": temperature}


def selected_jobs(packet):
    names = [r["id"] for r in active_rows(packet["spec"], packet["family"])]
    return [next(j for j in packet["request"]["jobs"] if j["task_ids"] == [name]) for name in names]


def executable_jobs(packet):
    return [job for job in selected_jobs(packet)
            if not packet["request"]["tasks"][job["task_id"]].get("preflight_blockers")]


def prior_lane_families(family):
    if family not in ACTIVE_FAMILIES:
        raise ValueError("preserved family has no current executable lane")
    names = list(ACTIVE_FAMILIES)
    return names[:names.index(family)]


def verify_predecessor_outcome(state, row, job):
    """Read certified artifacts/grades only; never replay or rescore a model."""
    outcome = row["outcome"]
    resolved = read(check_pin(outcome["resolved"]))
    raw = read(check_pin(outcome["raw"])); grade = read(check_pin(outcome["grading"]))
    packet = resolved["packet"]
    if (resolved.get("packet_sha256") != digest(packet) or packet.get("source") != state["source"]
            or packet.get("family") != state["family"] or packet.get("lane_predecessors") != state.get("lane_predecessors")
            or resolved.get("request") != state["request"] or resolved.get("job") != job
            or resolved.get("worker", {}).get("token") != row["token"]
            or grade.get("raw_hash") != digest(raw) or grade.get("source_digest") != state["source"]["digest"]
            or set(grade.get("grades", {})) != set(job["task_ids"])
            or {name: value.get("gate_status") for name, value in grade["grades"].items()} != outcome["statuses"]
            or set(outcome.get("media", {})) != set(job["task_ids"])):
        raise ValueError("predecessor original grade/source/task attribution changed")
    for name, media in outcome["media"].items():
        receipt = read(check_pin(media["receipt"])); check_pin(media["gif"])
        if (receipt.get("family") != state["family"] or receipt.get("task") != name
                or receipt.get("source_digest") != state["source"]["digest"]
                or receipt.get("original_gate") != outcome["statuses"][name]
                or receipt.get("qualification_input") is not False or receipt.get("gif") != media["gif"]):
            raise ValueError("predecessor media/original verdict changed")
        for item in receipt.get("inputs", []):
            check_pin(item)


def lane_accounting(state, *, require_ready=False):
    family = state["family"]; physical = FAMILIES[family]["gpu"]
    if (canonical(state["spec"].get("engineering_carryover")) != canonical(ENGINEERING_REFERENCE)
            or state.get("engineering_carryover", {}).get("reference") != ENGINEERING_REFERENCE):
        raise ValueError("historical lane engineering debit cannot disappear or reset")
    if (state["spec"].get("continuation_carryover") != CONTINUATION_REFERENCE
            or state.get("continuation_carryover") != continuation_carryover(state["spec"], ROOT)):
        raise ValueError("the full v3 paid/source/outcome history cannot disappear or become current credit")
    required = prior_lane_families(family); refs = state.get("lane_predecessors", [])
    names = [item.get("family") for item in refs]
    if names != required[:len(names)] or (require_ready and names != required):
        raise ValueError("admission requires exactly the complete predecessor-family lane prefix")
    paid = reserved = 0.
    for item in refs:
        predecessor = read(check_pin(item["study"]))
        name = item["family"]
        if (predecessor.get("family") != name or predecessor.get("status") != "COMPLETE_DIAGNOSTIC"
                or predecessor.get("source") != state["source"] or predecessor.get("execution_source") != state["execution_source"]
                or predecessor.get("spec_sha256") != state["spec_sha256"] or predecessor.get("spec") != state["spec"]
                or predecessor.get("engineering_carryover") != state["engineering_carryover"]
                or predecessor.get("continuation_carryover") != state["continuation_carryover"]
                or predecessor.get("request") != state["requests"][name]
                or predecessor.get("requests") != state["requests"]
                or predecessor.get("lane_runtime") != lane_runtime(state["requests"][name], physical)):
            raise ValueError("foreign, stale or unfinished predecessor cannot supply lane costs")
        verify_state(predecessor)
        for row, job in zip(predecessor["jobs"], executable_jobs(predecessor)):
            terminal = read(check_pin(row["terminal"]))
            if row.get("status") != "COMPLETE" or terminal.get("child_returncode") != 0:
                raise ValueError("predecessor did not complete with its original numerical grade")
            verify_predecessor_outcome(predecessor, row, job)
        paid += predecessor["measured_paid_seconds"]
        reserved += predecessor["unmeasured_interrupt_reserved_seconds"]
    charged = legacy().number(state.get("spent_seconds", 0.), "current family charged")
    return {"physical_gpu": physical, "lane_cap_seconds": state["spec"]["lane_cap_seconds"][physical],
            "historical_engineering_debit_seconds": PRIOR_DEBITS[physical],
            "historical_v3_paid_seconds": V3_DEBITS[physical],
            "historical_lane_debit_seconds": HISTORICAL_LANE_DEBITS[physical],
            "predecessor_paid_seconds": paid, "predecessor_reserved_seconds": reserved,
            "predecessor_charged_seconds": paid + reserved, "current_family_charged_seconds": charged,
            "inclusive_lane_charged_seconds": HISTORICAL_LANE_DEBITS[physical] + paid + reserved + charged,
            "all_predecessor_families_complete": names == required, "qualification_input": False}


def full_allowance_fits(state, job):
    ledger = lane_accounting(state, require_ready=True)
    return (state["spent_seconds"] + job["budget_seconds"] <= FAMILIES[state["family"]]["cap"]
            and ledger["inclusive_lane_charged_seconds"] + job["budget_seconds"] <= ledger["lane_cap_seconds"])


def initial_state(packet):
    state = deepcopy(packet); family = packet["family"]
    active = {r["id"] for r in active_rows(packet["spec"], family)}
    state.update(status="PREPARED", jobs=[], ordinary_qualified_tier=0, qualification_input=False, default_adoption=False,
                 measured_paid_seconds=0., unmeasured_interrupt_reserved_seconds=0., spent_seconds=0., slots={})
    for assignment in packet["request"]["view"]["assignments"]:
        name = assignment["task"]; blockers = packet["request"]["tasks"][name].get("preflight_blockers", [])
        state["slots"][name] = {"tier": assignment["qualification_tier"], "in_diagnostic_batch": name in active,
            "diagnostic_status": "BLOCKED" if name in active and blockers else "NOT_RUN",
            "preflight_blockers": blockers, "qualification_input": False}
    state["historical_original_word"] = {"prior_rows": 5, "status": "BLOCKED", "execution_credit": False,
        "claim": "Original N5 cannot enable all public row owners; min11 is a distinct declared resource/joint-code law."}
    state["lane_accounting"] = lane_accounting(state)
    return state


def verify_state(state):
    family = state["family"]; allowed = selected_jobs(state)
    expected = initial_state(state)["slots"]
    executable = [j for j in allowed if not expected[j["task_id"]]["preflight_blockers"]]
    if len(state["jobs"]) > len(executable):
        raise ValueError("extra/duplicate named job")
    ledger = lane_accounting(state, require_ready=bool(state["jobs"]))
    paid = reserved = 0.; halted = False
    for index, row in enumerate(state["jobs"]):
        job = executable[index]
        if (paid + reserved + job["budget_seconds"] > FAMILIES[family]["cap"]
                or ledger["historical_lane_debit_seconds"] + ledger["predecessor_charged_seconds"]
                   + paid + reserved + job["budget_seconds"] > ledger["lane_cap_seconds"]):
            raise ValueError("the full original allowance was unavailable before this attempt")
        if halted or row.get("compatibility_key") != job["compatibility_key"] or row.get("task_ids") != job["task_ids"]:
            raise ValueError("each family executes an uninterrupted prefix of its own declared questions")
        if "terminal" in row:
            terminal = read(check_pin(row["terminal"]))
            if terminal.get("token") != row.get("token"):
                raise ValueError("foreign durable supervisor token")
            cost = legacy().charge(terminal["paid_wall_seconds"], terminal, job["budget_seconds"])
        else:
            measurement = 0.
            if "launch_error" in row:
                error = read(check_pin(row["launch_error"]))
                if error.get("token") != row.get("token") or error.get("source_digest") != state["source"]["digest"]:
                    raise ValueError("foreign parent interruption")
                measurement = error["paid_wall_seconds"]
            cost = legacy().charge(measurement, None, job["budget_seconds"])
        if any(row.get(k) != v for k, v in cost.items()):
            raise ValueError("paid/reserved/conservative charge differs from durable semantics")
        paid += cost["paid_wall_seconds"]; reserved += cost["unmeasured_interrupt_reserved_seconds"]
        if row.get("status") == "COMPLETE":
            outcome = row.get("outcome", {})
            if (set(outcome.get("statuses", {})) != set(job["task_ids"])
                    or any(v not in {"PASS", "FAIL"} for v in outcome["statuses"].values())
                    or outcome.get("qualification_input") is not False):
                raise ValueError("only complete original numerical grades may advance a diagnostic")
            statuses = outcome["statuses"]
        elif row.get("status") in {"INCOMPLETE", "INVALID"} and "outcome" not in row:
            statuses = {name: row["status"] for name in job["task_ids"]}; halted = True
        else:
            raise ValueError("unknown terminal status")
        for name, status in statuses.items():
            expected[name]["diagnostic_status"] = status
    if expected != state["slots"]:
        raise ValueError("forged grades or credit for unexecuted/cross-family questions")
    for key, value in (("measured_paid_seconds", paid), ("unmeasured_interrupt_reserved_seconds", reserved), ("spent_seconds", paid + reserved)):
        if not math.isclose(legacy().number(state.get(key), key), value, rel_tol=0, abs_tol=1e-8):
            raise ValueError("study totals do not match durable charged costs")
    if state.get("lane_accounting") != ledger:
        raise ValueError("inclusive lane/current paid/reserve/engineering ledger changed")
    overrun = paid + reserved > FAMILIES[family]["cap"] or ledger["inclusive_lane_charged_seconds"] > ledger["lane_cap_seconds"]
    if (state.get("family_paid_budget_seconds") != {f: info["cap"] for f, info in FAMILIES.items()}
            or state.get("qualification_input") is not False or state.get("default_adoption") is not False
            or state.get("ordinary_qualified_tier") != 0):
        raise ValueError("finite family quota/nonqualification changed")
    if overrun and state.get("status") != "BUDGET_EXCEEDED":
        raise ValueError("a durable measured overshoot must be retained as BUDGET_EXCEEDED and halt")
    if state.get("status") == "COMPLETE_DIAGNOSTIC" and (halted or len(state["jobs"]) != len(executable)):
        raise ValueError("incomplete lane cannot claim diagnostic completion")


def validate_resolved(resolved):
    packet = resolved["packet"]; verify_packet(packet)
    family = packet.get("family")
    if (family not in ACTIVE_FAMILIES or resolved.get("packet_sha256") != digest(packet)
            or packet.get("request") != packet["requests"][family] or resolved.get("request") != packet["request"]):
        raise ValueError("child named family/request attribution changed")
    jobs = [j for j in selected_jobs(packet) if j["compatibility_key"] == resolved["job"].get("compatibility_key")]
    worker = resolved["worker"]; physical = FAMILIES[family]["gpu"]
    if (len(jobs) != 1 or jobs[0] != resolved["job"]
            or packet["request"]["tasks"][jobs[0]["task_id"]].get("preflight_blockers")
            or worker.get("device") != physical or worker.get("lane_runtime") != lane_runtime(packet["request"], physical)
            or packet.get("lane_runtime") != worker["lane_runtime"] or resolved.get("prerequisites") != {}):
        raise ValueError("child job/physical lane/runtime/prerequisite changed")
    for dep in packet["request"]["tasks"][jobs[0]["task_id"]].get("dependencies", []):
        if dep.get("kind") != "gate":
            raise ValueError("this diagnostic imports no unknown/failed checkpoint prerequisite")
    return packet


def child_environment(lease_fd, resolved):
    physical = FAMILIES[resolved["packet"]["family"]]["gpu"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != physical or any(os.environ.get(k) != v for k, v in ENVIRONMENT.items()):
        raise ValueError("child visibility/determinism/one-thread environment changed")
    legacy().verify_lease(lease_fd, resolved)
    os.environ["FORGE_LEASE_FD"] = str(lease_fd)


def stage(path, *, execute):
    resolved = read(path); packet = validate_resolved(resolved)
    legacy().guard_imports(packet["source"])
    legacy().verify_lease(int(os.environ["FORGE_LEASE_FD"]), resolved)
    if any(os.environ.get(k) != v for k, v in ENVIRONMENT.items()):
        raise ValueError("stage determinism/thread environment changed")
    if execute:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != FAMILIES[packet["family"]]["gpu"]:
            raise ValueError("numerical child must use its declared physical GPU")
        import torch
        torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.2, 0)
        code = forge("runtime").execute(Path(path))
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("independent grading/media requires a clean CPU process")
        grade = forge("evaluate").evaluate(Path(path))
        render_media(resolved, Path(path).parent, grade); code = 0
    legacy().guard_imports(packet["source"])
    return code


def child(path, lease_fd, runner=None):
    resolved = read(path); validate_resolved(resolved); child_environment(lease_fd, resolved)
    command = [sys.executable, "-u", str(Path(__file__).resolve())]; run = runner or subprocess.run
    executed = run(command + ["--execute", str(path)], pass_fds=(lease_fd,), check=False)
    if executed.returncode != 0:
        return executed.returncode
    environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = ""
    evaluated = run(command + ["--evaluate", str(path)], env=environment, pass_fds=(lease_fd,), check=False)
    if evaluated.returncode != 0:
        return evaluated.returncode
    grade = read(Path(path).parent / "graded-result.json")
    return 0 if all(g.get("gate_status") in {"PASS", "FAIL"} for g in grade["grades"].values()) else 2


def media_views(task, arrays):
    """Lossless views of recorded contexts and outputs; no metric computation."""
    import numpy as np
    family = task["policy_family"]; parent = task["policy_parent"]["id"]
    context = "Actual retained selected-policy output; original known training contexts only, no unseen-context claim."
    def view(kind, title, target, samples, caption):
        return {"kind": kind, "title": title, "target": np.asarray(target), "samples": np.asarray(samples),
                "caption": caption, "xlabel": "x" if kind == "line" or kind == "scatter" else "Feature index",
                "ylabel": "y" if kind == "line" or kind == "scatter" else "Context/value"}
    if family == "atlas_conditional":
        if parent in {"trajectory", "residual_student"}:
            paths = {key: np.asarray(arrays[key]).reshape(len(arrays[key]), 8, 2) for key in ("given_context", "target", "samples")}
            return [view("line", "Given slow trajectory → desired fast trajectory", paths["given_context"], paths["target"],
                         "Each path is one original paired episode; gray is the given source, red the desired target."),
                    view("line", "Desired fast trajectory → actual selected prediction", paths["target"], paths["samples"], context)]
        return [view("image", "Desired responses and selected responses at each declared scale", arrays["target"], arrays["samples"],
                     context + " Rows retain the saved scale order: " + str(arrays["given_context"].reshape(-1).tolist()) + "; all feature values are shown.")]
    if family == "atlas_routed":
        desired = np.stack((arrays["neu"][0], np.asarray(arrays["concept_target"]).reshape(-1)))
        return [view("scatter", "Unused slot stays; used slot moves along its concept", desired, arrays["embeds"],
                     "Desired slot 0 is its retained neutral input; desired slot 1 is its retained concept target. Both actual slots are plotted."),
                view("image", "Every coordinate of the two protected/edited slots", desired, arrays["embeds"], context)]
    if family == "atlas_multibank":
        return [view("image", "Desired pole residuals and selected complete-bank residuals", arrays["targets"] - arrays["neu"], arrays["residual"],
                     "Two rows are + and − concept polarity; columns retain all four original content/leftover coordinates. Desired residual equals retained pole minus neutral.")]
    if family == "atlas_ae_routed":
        return [view("scatter", "Original paired inputs and selected AE reconstructions", arrays["target"], arrays["reconstruction"],
                     "All recorded input/reconstruction coordinates; original reconstruction-MSE gate is scored independently."),
                view("scatter", "Protected anchors and actual selected generated population", arrays["anchors"], arrays["samples"],
                     "Declared fixed-width MoG .025 and routed AE family; output noise is off in this observation. Original anchor-hold gate remains separate.")]
    if family == "atlas_word_joint_min11":
        definition = task["execution"]["host_definition"]; chars = definition["characters"]; length = definition["length"]
        def words(value):
            shaped = np.asarray(value).reshape(-1, len(chars), length)
            return ["".join(chars[int(i)] for i in row) for row in shaped.argmax(1)]
        target = arrays["target"]; labels = words(target)
        first = view("text", "Five canonical words and the first eight actual generated draws", target, arrays["generated"],
                     "Argmax text is a display only; the original quality, mass and token-confidence gates use all 1024 recorded draws. Eleven actual prior rows preserve five target words.")
        first.update(target_labels=labels, sample_labels=words(arrays["generated"])[:8])
        second = view("text", "Correctly paired reconstruction of every canonical word", target, arrays["reconstruction"],
                      "The actual free E encodes each known word; selected G uses the same effective code under public DV12. No reconstruction loss is added to training; padding is displayed.")
        second.update(target_labels=labels, sample_labels=words(arrays["reconstruction"]))
        probabilities = view("image", "Paired token probabilities, including padding", np.asarray(target).reshape(-1, 1, len(chars), length),
                             np.asarray(arrays["reconstruction"]).reshape(-1, 1, len(chars), length),
                             "Each tile is one canonical word. Rows follow the declared 28-character alphabet, columns its six token positions. Actual reconstruction probabilities retain confidence and padding, beyond argmax text.")
        probabilities.update(vmin=0., vmax=1.)
        return [first, second, probabilities]
    raise ValueError("unknown named goal media")


def render_media(resolved, directory, grading):
    import numpy as np
    from benchmarks.toy_audit.api_run import render_gif
    name = resolved["job"]["task_id"]; task = resolved["request"]["tasks"][name]
    if set(grading.get("grades", {})) != {name} or grading["grades"][name].get("gate_status") not in {"PASS", "FAIL"}:
        raise ValueError("only independently complete numerical outcomes receive goal media")
    raw = read(directory / "raw-result.json"); evidence = raw["evidence"]
    forge("artifacts").verify_artifacts(Path(evidence["artifact_root"]), evidence["artifact_manifest"])
    points, selected, paths, metadata = legacy().media_selection(task, evidence)
    if len(points) != 24 or metadata:
        raise ValueError("named media requires its complete 24-observation original cadence")
    records = []
    for index, path in zip(selected, paths):
        with np.load(path, allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("nonfinite retained media array")
        point = points[index]
        records.append({"step": point["step"], "metrics": {k: v for k, v in point.items() if k != "step"},
                        "passed": legacy().point_pass(point, task["evaluation"]["thresholds"]), "views": media_views(task, arrays)})
    status = grading["grades"][name]["gate_status"]; goal = task["policy_parent"]["id"] + " — " + task["policy_family"]
    feature_views = [view for record in records for view in record["views"] if view["kind"] == "image" and "vmin" not in view]
    if feature_views:
        low = min(float(np.min(view[key])) for view in feature_views for key in ("target", "samples"))
        high = max(float(np.max(view[key])) for view in feature_views for key in ("target", "samples"))
        if high == low:
            low -= .5; high += .5
        for view in feature_views:
            view.update(vmin=low, vmax=high)
            view["caption"] += f" Fixed display range across all retained frames: [{low:.6g}, {high:.6g}]; signed values are preserved."
    gif = directory / (name + "-goal.gif")
    render_gif({"id": name, "goal": goal + " — original terminal gate; named diagnostic only, no ordinary/default/speed credit", "default_steps": task["execution"]["steps"]}, records, gif,
               full_budget=True, requested_steps=task["execution"]["steps"], final_verdict=status)
    write(directory / (name + "-media.json"), {"schema": SCHEMA + "_media", "task": name, "family": task["policy_family"],
        "source_digest": resolved["request"]["source"]["digest"], "renderer_sha256": file_hash(Path(__file__)),
        "original_gate": status, "qualification_input": False, "draws": 0, "optimizer_updates": 0,
        "actual_steps": [r["step"] for r in records], "inputs": [pin(path) for path in paths], "gif": pin(gif)})


def certified_outcome(path, grader=None, policy_guard=None):
    resolved = read(path); packet = validate_resolved(resolved); directory = Path(path).parent
    raw = read(directory / "raw-result.json"); grade = read(directory / "graded-result.json")
    name = resolved["job"]["task_id"]; task = packet["request"]["tasks"][name]
    if (grade.get("raw_hash") != digest(raw) or grade.get("source_digest") != packet["source"]["digest"]
            or set(grade.get("grades", {})) != {name}):
        raise ValueError("raw/independent grade/source/question changed")
    evidence = raw["evidence"]; artifacts = Path(evidence["artifact_root"]).resolve()
    if not artifacts.is_relative_to(directory.resolve()):
        raise ValueError("foreign candidate/task artifacts")
    forge("artifacts").verify_artifacts(artifacts, evidence["artifact_manifest"])
    guard = (policy_guard or forge("views")._policy_guards)(task, evidence)
    if guard is not None:
        raise ValueError("invalid/incomplete owner/runtime/purity cannot count as numerical FAIL: " + canonical(guard))
    expected = (grader or forge("views").grade_result)(task, raw)
    if expected != grade["grades"][name] or expected.get("gate_status") not in {"PASS", "FAIL"}:
        raise ValueError("complete original numerical grade differs")
    media_path = directory / (name + "-media.json"); receipt = read(media_path)
    points, selected, paths, metadata = legacy().media_selection(task, evidence)
    gif = directory / (name + "-goal.gif")
    fixed = {"schema": SCHEMA + "_media", "task": name, "family": packet["family"],
             "source_digest": packet["source"]["digest"], "renderer_sha256": packet["source"]["files"][SELF],
             "original_gate": expected["gate_status"], "qualification_input": False, "draws": 0, "optimizer_updates": 0,
             "actual_steps": [points[i]["step"] for i in selected], "inputs": [pin(p) for p in paths], "gif": pin(gif)}
    if len(points) != 24 or metadata or receipt != fixed:
        raise ValueError("media source/original verdict/cadence/retained inputs changed")
    from PIL import Image
    with Image.open(gif) as image:
        if image.n_frames != len(selected):
            raise ValueError("GIF frame count differs from actual observations")
    return {"statuses": {name: expected["gate_status"]}, "resolved": pin(path), "raw": pin(directory / "raw-result.json"),
            "grading": pin(directory / "graded-result.json"), "media": {name: {"receipt": pin(media_path), "gif": pin(gif)}}, "qualification_input": False}


def save_state(output, state):
    state["measured_paid_seconds"] = sum(r["paid_wall_seconds"] for r in state["jobs"])
    state["unmeasured_interrupt_reserved_seconds"] = sum(r["unmeasured_interrupt_reserved_seconds"] for r in state["jobs"])
    state["spent_seconds"] = state["measured_paid_seconds"] + state["unmeasured_interrupt_reserved_seconds"]
    state["lane_accounting"] = lane_accounting(state, require_ready=bool(state["jobs"]))
    if (state["spent_seconds"] > FAMILIES[state["family"]]["cap"]
            or state["lane_accounting"]["inclusive_lane_charged_seconds"] > state["lane_accounting"]["lane_cap_seconds"]):
        state.update(status="BUDGET_EXCEEDED", stop_reason="Durable measured supervisor overrun retained; no further admission or budget reset.")
    verify_state(state); write(Path(output) / "study.json", state)
    lines = ["# " + state["family"] + " GPU diagnostic", "", "All 26 required questions remain visible. Only this family's named adapted subset is executed; no ordinary/default/speed credit.", "",
             f"Measured paid {state['measured_paid_seconds']:.6f}s; reserve {state['unmeasured_interrupt_reserved_seconds']:.6f}s; charged {state['spent_seconds']:.6f}/{FAMILIES[state['family']]['cap']}s.", "",
             f"Historical v1 engineering debit {state['lane_accounting']['historical_engineering_debit_seconds']:.15g}s; historical v3 paid {state['lane_accounting']['historical_v3_paid_seconds']:.15g}s; completed v4 predecessor families charged {state['lane_accounting']['predecessor_charged_seconds']:.15g}s; inclusive GPU{state['lane_accounting']['physical_gpu']} charge {state['lane_accounting']['inclusive_lane_charged_seconds']:.15g}/{state['lane_accounting']['lane_cap_seconds']}s. No earlier outcome supplies current qualification credit.", "",
             "The five original GPU0 PASS outcomes remain only in the pinned v3 publication, under its original source. GPU0 is not executable in this continuation and no old PASS is assigned to these v4 slots.", "",
             "| Actual question | Tier | Original diagnostic gate | Goal GIF |", "|---|---:|---|---|"]
    for assignment in state["request"]["view"]["assignments"]:
        name = assignment["task"]; media = ""
        for job in state["jobs"]:
            if name in job.get("outcome", {}).get("media", {}):
                relative = Path(job["outcome"]["media"][name]["gif"]["path"]).relative_to(output).as_posix()
                media = f"[actual goal GIF]({relative})"
        lines.append(f"| {name} | {assignment['qualification_tier']} | {state['slots'][name]['diagnostic_status']} | {media} |")
    lines += ["", "Original five-row word source remains BLOCKED under full-owner guards. Named min11 is a different explicit resource/joint-code law; neither variant grants the original source credit.", ""]
    (Path(output) / "README.md").write_text("\n".join(lines))


def coordinator_for(queue_root):
    return forge("policy_execution").PolicyCoordinator(Path(queue_root).resolve(), report_root=ROOT / "reports/forge")


def run_family(output, packet, queue_root):
    if packet.get("family") not in ACTIVE_FAMILIES:
        raise ValueError("preserved GPU0 families are not executable in v4")
    output = Path(output).resolve(); verify_packet(packet); family = packet["family"]
    physical = FAMILIES[family]["gpu"]; runtime = lane_runtime(packet["request"], physical)
    state = read(output / "study.json") if (output / "study.json").exists() else initial_state(packet)
    for key in ("spec", "spec_sha256", "source", "execution_source", "request", "requests", "family", "case_definitions", "runtime_contract", "family_paid_budget_seconds", "lane_runtime", "engineering_carryover", "continuation_carryover", "lane_predecessors"):
        if state.get(key) != packet.get(key):
            raise ValueError("resume source/family/quota/runtime changed")
    verify_state(state)
    lane_accounting(state, require_ready=True)
    if state["status"] == "BUDGET_EXCEEDED":
        return state
    for row in state["jobs"]:
        if "outcome" in row and certified_outcome(check_pin(row["outcome"]["resolved"])) != row["outcome"]:
            raise ValueError("retained scientific/media result changed")
        if row["status"] != "COMPLETE":
            return state  # No failed infrastructure retry.
    gpu_readiness(packet["spec"], physical)
    coordinator = coordinator_for(queue_root)
    key, actual = coordinator.register(packet, output, family, runtime)
    if actual != output:
        raise ValueError("compatible named study already has a canonical output")
    state.update(executed_family=family, coordinator=packet.get("coordinator")); save_state(output, state)
    with coordinator.study_lease(key) as study_lease:
        if study_lease is None:
            raise RuntimeError("another owner holds this exact family study")
        for job in executable_jobs(packet)[len(state["jobs"]):]:
            if state.get("continuation_carryover") != continuation_carryover(state["spec"], ROOT, durable=True):
                raise ValueError("v3 terminal/source cost history must verify before each new full allowance")
            if not full_allowance_fits(state, job):
                state.update(status="INCOMPLETE", stop_reason="next full allowance cannot fit the unchanged family and inclusive lane ceilings"); break
            telemetry = gpu_readiness(packet["spec"], physical)
            row = {"id": job["task_id"], "timeout_seconds": job["budget_seconds"]}
            attempt = coordinator.attempt_key(packet, {"family": family, "recipe_overrides": OVERRIDES}, row)
            with coordinator.admit(attempt, packet, row, "cuda:0") as (admission, lease):
                if admission["status"] == "busy":
                    state.update(status="INCOMPLETE", stop_reason=admission["reason"]); break
                directory = output / "attempts" / job["execution_group"]; directory.mkdir(parents=True, exist_ok=True)
                path = directory / "resolved.json"; token = admission["token"]; launch_error = None
                if admission["status"] == "running" and lease is not None:
                    gpu_readiness(packet["spec"], physical)
                    resolved = {"schema_version": 1, "packet": packet, "packet_sha256": digest(packet), "request": packet["request"],
                        "job": job, "prerequisites": {}, "worker": {"device": physical, "token": token, "attempt": attempt,
                        "lane_runtime": runtime, "lease_path": admission["lease_path"]}}
                    write(path, resolved)
                    command = [sys.executable, "-u", str(Path(packet["source"]["snapshot_path"]) / SELF), "--child", str(path), "--lease-fd", str(lease.fileno())]
                    try:
                        coordinator.launch(command, packet, directory / "run.log", (study_lease, lease), job["budget_seconds"])
                    except (Exception, KeyboardInterrupt) as error:
                        launch_error = {"token": token, "source_digest": packet["source"]["digest"], "type": type(error).__name__,
                            "message": str(error), "paid_wall_seconds": legacy().number(getattr(error, "paid_wall_seconds", 0.), "parent paid")}
                        write(directory / "launch-error.json", launch_error)
                else:
                    launch_error = read(directory / "launch-error.json") if (directory / "launch-error.json").exists() else None
                terminal_path = Path(admission["lease_path"]).parent / "supervisor-terminal.json"
                terminal = read(terminal_path) if terminal_path.exists() else None
                if terminal is not None and terminal.get("token") != token:
                    raise ValueError("foreign durable supervisor terminal")
                paid = terminal.get("paid_wall_seconds", 0.) if terminal else (launch_error or {}).get("paid_wall_seconds", 0.)
                result = {"compatibility_key": job["compatibility_key"], "task_ids": job["task_ids"], "attempt_key": attempt,
                          "token": token, "status": "INCOMPLETE", "admission": telemetry,
                          **legacy().charge(paid, terminal, job["budget_seconds"])}
                if terminal is not None:
                    result["terminal"] = pin(terminal_path)
                elif launch_error is not None:
                    result["launch_error"] = pin(directory / "launch-error.json")
                if terminal and terminal["attempt_status"] == "completed" and terminal.get("child_returncode") == 0:
                    try:
                        result["outcome"] = certified_outcome(path); result["status"] = "COMPLETE"
                    except Exception as error:
                        result.update(status="INVALID", reason=f"{type(error).__name__}: {error}")
                elif terminal and terminal["attempt_status"] == "completed":
                    result.update(status="INVALID", reason="runtime/grading/media did not produce a valid complete original outcome")
                if admission["status"] in {"running", "awaiting_certification"}:
                    coordinator.complete(attempt, result)
                elif admission.get("charged_seconds") != result["charged_seconds"]:
                    raise ValueError("retained local/central conservative charges differ")
                state["jobs"].append(result)
                for name in job["task_ids"]:
                    state["slots"][name]["diagnostic_status"] = result["outcome"]["statuses"][name] if result["status"] == "COMPLETE" else result["status"]
                state["status"] = "RUNNING" if result["status"] == "COMPLETE" else result["status"]
                save_state(output, state)
                if result["status"] != "COMPLETE" or state["status"] == "BUDGET_EXCEEDED":
                    return state
        else:
            state["status"] = "COMPLETE_DIAGNOSTIC"
        save_state(output, state)
    return state


def run_lane(output, queue_root, physical):
    if physical != "1":
        raise ValueError("the five preserved GPU0 outcomes cannot be re-executed")
    packet = read(preparation_path(output)); verify_packet(packet)
    summaries = []; predecessors = []
    for family in ACTIVE_FAMILIES:
        state = run_family(Path(output) / family, family_packet(packet, family, predecessors), queue_root)
        summaries.append({"family": family, "status": state["status"], "paid_seconds": state["measured_paid_seconds"],
                          "reserved_seconds": state["unmeasured_interrupt_reserved_seconds"],
                          "lane_accounting": state["lane_accounting"], "qualification_input": False})
        if state["status"] != "COMPLETE_DIAGNOSTIC":
            break  # Infrastructure/invalid evidence stops the physical lane.
        predecessors.append((family, Path(output) / family / "study.json"))
    return summaries


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, default=ROOT / DIRECTORY / "protocol.json")
    p.add_argument("--output", type=Path); p.add_argument("--queue-root", type=Path)
    p.add_argument("--prepare-only", action="store_true"); p.add_argument("--gpus", choices=("1",))
    stages = p.add_mutually_exclusive_group()
    for flag in ("child", "execute", "evaluate"):
        stages.add_argument("--" + flag, type=Path)
    p.add_argument("--lease-fd", type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.child:
        if args.lease_fd is None:
            raise ValueError("child needs an admitted inherited lease descriptor")
        return child(args.child, args.lease_fd)
    if args.execute or args.evaluate:
        return stage(args.execute or args.evaluate, execute=args.execute is not None)
    if args.output is None:
        raise ValueError("declare a fresh external output")
    configure_environment(None if args.prepare_only else args.gpus)
    if args.prepare_only:
        packet = prepare(args.spec, args.output)
        print(canonical({"status": "PREPARED", "source_digest": packet["source"]["digest"], "spec_sha256": packet["spec_sha256"], "qualification_input": False}), flush=True)
        return 0
    if args.gpus is None or args.queue_root is None:
        raise ValueError("run needs an explicit physical lane and the existing shared queue")
    rows = run_lane(args.output, args.queue_root, args.gpus)
    print(canonical({"physical_gpu": args.gpus, "families": rows, "qualification_input": False}), flush=True)
    return 0 if all(r["status"] == "COMPLETE_DIAGNOSTIC" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())

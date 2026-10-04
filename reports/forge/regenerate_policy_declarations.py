"""Refresh current declaration source pins; never alter archived evidence.

All original parents, metrics and full horizons remain inputs to the fixed
metadata transformations. No evaluator, model, artifact or queue is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.forge import policy_contracts as independent
from experiments.forge import conditional_policy_contracts as conditional
from experiments.forge import routed_policy_contracts as routed
from experiments.forge import multibank_policy_contracts as multibank
from experiments.forge import ae_routed_policy_contracts as ae
from experiments.forge import word_joint_policy_contracts as word
from experiments.forge.policy_declaration_sources import DECLARATION_SOURCES


def declarations(root):
    """Yield exactly 26 independent and eight explicitly named questions."""
    root = Path(root).resolve()
    directory = root / "configs/forge/task-variants" / independent.COHORT
    # Retain the established support-source roster and add the new common
    # execution/grading dispatch. Historical pinned declarations are untouched.
    source_names = independent.REQUIRED_POLICY_SOURCES | DECLARATION_SOURCES
    for path in sorted(directory.glob("*.json")):
        source_names |= set(json.loads(path.read_bytes())["execution"]["policy_contract"]["sources"])
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in sorted(source_names)}
    for name in independent.PARENT_TASK_IDS:
        raw = (root / "configs/forge/tasks" / (name + ".json")).read_bytes()
        parent = json.loads(raw)
        task = independent._prospective_variant(parent,
            independent._parent_record(parent, hashlib.sha256(raw).hexdigest()), sources)
        yield task
    for name in conditional.HOSTS:
        yield conditional.make_conditional_variant(root, name)
    yield routed.make_unused_variant(root)
    yield multibank.make_variant(root)
    yield ae.make_ae_variant(root)
    yield word.make_variant(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    tasks = list(declarations(args.root))
    if len(tasks) != 34 or len({task["id"] for task in tasks}) != 34:
        raise ValueError("declaration roster must contain exactly 34 distinct tasks")
    changed = []
    for task in tasks:
        path = args.root / "configs/forge/task-variants" / task["task_cohort"] / (task["id"] + ".json")
        data = (json.dumps(task, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if not path.is_file() or path.read_bytes() != data:
            changed.append(path.relative_to(args.root).as_posix())
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    protocol_path = args.root / "reports/forge/atlas-current-gpu-diagnostics-v1/protocol.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_bytes())
        independent_tasks = {task["id"]: task for task in tasks
                             if task["task_cohort"] == independent.COHORT}
        if {case["id"] for case in protocol["cases"]} != set(independent_tasks):
            raise ValueError("current independent diagnostic protocol must retain exactly 26 slots")
        for case in protocol["cases"]:
            task = independent_tasks[case["id"]]
            expected_path = "configs/forge/task-variants/" + independent.COHORT + "/" + task["id"] + ".json"
            if case["definition"] != expected_path:
                raise ValueError("diagnostic definition path differs from its exact variant")
            data = (json.dumps(task, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            case["sha256"] = hashlib.sha256(data).hexdigest()
        data = (json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if protocol_path.read_bytes() != data:
            changed.append(protocol_path.relative_to(args.root).as_posix())
            if not args.check:
                protocol_path.write_bytes(data)
    named_protocol_path = args.root / "reports/forge/atlas-named-gpu-diagnostics-v1/protocol.json"
    if named_protocol_path.is_file():
        protocol = json.loads(named_protocol_path.read_bytes())
        named_tasks = {task["id"]: task for task in tasks
                       if task["task_cohort"] != independent.COHORT}
        if {case["id"] for case in protocol["cases"]} != set(named_tasks):
            raise ValueError("named diagnostic protocol must retain exactly eight adapted slots")
        for case in protocol["cases"]:
            task = named_tasks[case["id"]]
            expected_path = "configs/forge/task-variants/" + task["task_cohort"] + "/" + task["id"] + ".json"
            if case["definition"] != expected_path:
                raise ValueError("named diagnostic definition differs from its exact variant")
            data = (json.dumps(task, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            case["sha256"] = hashlib.sha256(data).hexdigest()
            for section in ("execution", "evaluation"):
                encoded = json.dumps(task[section], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                case[section + "_sha256"] = hashlib.sha256(encoded).hexdigest()
        data = (json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if named_protocol_path.read_bytes() != data:
            changed.append(named_protocol_path.relative_to(args.root).as_posix())
            if not args.check:
                named_protocol_path.write_bytes(data)
    print(json.dumps({"status": "STALE" if args.check and changed else "CURRENT",
                      "declarations": len(tasks), "changed": changed,
                      "scientific_evidence_regraded": False, "compute_reserved": False}, indent=2))
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Classify unavailable own-checkpoint diagnostics in saved publications.

The generic diagnostic queue can finish with unsatisfied optional dependencies
still pending. Its publisher calls those cells UNMEASURED. This read-only
adapter records the known checkpoint blocker, preserving every certified row,
attempt, source, cost and gate. No training, grading, sampling or retry occurs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def annotate(collection):
    cells = collection["cells"]
    statuses = {(cell["scope"], cell["role"], cell["task_id"]): cell["gate_status"] for cell in cells}
    requests = {(context["scope"], role): submission["request"]
                for context in collection["scopes"]
                for role, submission in context["submissions"].items()}
    for cell in cells:
        if cell["gate_status"] != "UNMEASURED":
            continue
        request = requests.get((cell["scope"], cell["role"]))
        if request is None:
            continue
        task = request["tasks"].get(cell["task_id"], {})
        dependencies = [dependency["task"] for dependency in task.get("dependencies", [])
                        if isinstance(dependency, dict) and dependency.get("kind") == "checkpoint"]
        blocked = [(task_id, statuses.get((cell["scope"], cell["role"], task_id), "UNMEASURED"))
                   for task_id in dependencies
                   if statuses.get((cell["scope"], cell["role"], task_id)) != "PASS"]
        if blocked:
            cell.update(gate_status="BLOCKED", reason=[
                f"Own passing checkpoint producer {task_id} is {status}; no state was borrowed or measured."
                for task_id, status in blocked])
    return collection


def install(module):
    loader = module.phase2.publisher
    def publisher(root):
        saved = loader(root)
        collect = saved.collect
        saved.collect = lambda options: annotate(collect(options))
        return saved
    module.phase2.publisher = publisher


def main():
    # Accept the unchanged phase3 CLI. Load its own frozen recipe/API in this
    # separate process; this file is solely a saved-publication adapter.
    if "--publish" not in sys.argv or "--repository" not in sys.argv:
        raise ValueError("Use --publish and an explicit frozen --repository")
    root = Path(sys.argv[sys.argv.index("--repository") + 1]).resolve()
    path = root / "reports/forge/bcap-three-phase/phase3.py"
    spec = importlib.util.spec_from_file_location("adapted_frozen_phase3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    install(module)
    module.main()


if __name__ == "__main__":
    main()

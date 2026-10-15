"""Reproduce compatibility against the exact executed source; no training."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="checkout of a0493ca3c213d559523eadc76e244ec9f9c1f81f")
    args = parser.parse_args()
    source = args.source_root.resolve()
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(source))
    from experiments.forge.planning import resolve_idea, task_preflight
    from experiments.forge.views import load_tasks, task_execution_fingerprint, task_evaluation_fingerprint
    from experiments.forge.contracts import atomic_json, read_json, stable_hash
    publication = read_json(root / "reports/forge/technique-inventory.json")
    progress = publication["family_progress"]
    views = {view["id"] for view in progress["families"][0]["cohorts"][0]["views"]}
    names = sorted({assignment["task"] for view in progress["views"] if view["id"] in views
                    for assignment in view["assignments"] if assignment["importance"] == "required"})
    all_tasks, rows = load_tasks(source), []
    for selected in publication["rows"][:7]:
        request = resolve_idea(source, selected["candidate_id"], view_id="clockfree_continuous",
                               through_tier=3, execution_backend="cuda")
        assert request["source"]["digest"] == selected["bindings"]["source_digest"]
        assert request["candidate_revision"] == selected["candidate_revision"]
        tasks = {}
        for name in names:
            task = all_tasks[name]
            tasks[name] = {
                "blockers": task_preflight(task, request["candidate"], request["protocol"], root=source, tasks=all_tasks),
                "execution_sha256": task_execution_fingerprint(task),
                "evaluation_sha256": task_evaluation_fingerprint(task), "dependencies": task.get("dependencies", [])}
        family = selected["trainer_family"]
        rows.append({"family": "bcap-pure" if family == "bcap-dualnorm" else family,
            "candidate_id": selected["candidate_id"], "candidate_revision": request["candidate_revision"],
            "source_digest": request["source"]["digest"],
            "recipe_sha256": stable_hash(request["candidate"]["resolved_recipe"]), "tasks": tasks})
    atomic_json(Path(__file__).with_suffix(".json"), {
        "schema_version": 1, "scope": "selected_formulation_task_preflight_only", "qualification_input": False,
        "qualification_reuse": False, "source_commit": "a0493ca3c213d559523eadc76e244ec9f9c1f81f", "families": rows})


if __name__ == "__main__":
    main()

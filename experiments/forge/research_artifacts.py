"""Join published research artifacts by explicit question and task identities.

This is a navigation index, never a qualification reducer. Related API demos
retain their own scopes; only exact task IDs select recorded Forge outcomes.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .contracts import file_hash, read_json, stable_hash
from .views import task_evaluation_fingerprint, task_execution_fingerprint


API_ROOT = Path("reports/toy_audit/api_contract")
REVIEW_ARTIFACTS = {
    "solutions": "reports/forge/technique-inventory.md",
    "solution_evidence": "reports/forge/technique-inventory.json",
    "memory": "reports/forge/EXPERIMENT_MEMORY.md",
    "questions": "reports/toy_audit/api_contract/QUESTION_RANKING.md",
    "gallery": "reports/toy_audit/api_contract/GALLERY.md",
    "api_readout": "reports/toy_audit/api_contract/RUN_REPORT.md",
    "api_contract": "reports/toy_audit/api_contract/README.md",
    "later_questions": "reports/toy_audit/api_contract/recent_prs/README.md",
    "caption_questions": "reports/toy_audit/api_contract/caption_prs/README.md",
}


def build_artifacts(root: Path, tasks: dict, task_paths: dict, *, publication: dict | None = None) -> dict:
    """Read compact, committed indexes without model construction or regrading."""
    root = Path(root).resolve()
    inputs = {}

    def load(path):
        if publication is not None and str(path) == REVIEW_ARTIFACTS["solution_evidence"]:
            content = json.dumps(publication, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            inputs[str(path)] = hashlib.sha256(content.encode()).hexdigest()
            return publication
        source = root / path
        if not source.is_file():
            return {}
        inputs[str(path)] = file_hash(source)
        return read_json(source)

    def records(path, key="cases"):
        rows = load(path).get(key, [])
        result = {}
        for row in rows:
            if row["id"] in result:
                raise ValueError(f"{path}: duplicate artifact ID {row['id']!r}")
            result[row["id"]] = row
        return result

    catalog = records(Path("reports/toy_audit/catalog.json"))
    definitions = records(API_ROOT / "cases.json")
    readouts = records(API_ROOT / "readout.json")
    completed = records(API_ROOT / "runs.json")
    failed = records(API_ROOT / "failed-runs.json")
    locations = {name: {"definition": str(API_ROOT / "cases.json"),
                       "readout": str(API_ROOT / "readout.json"),
                       "receipt": str(API_ROOT / ("runs.json" if name in completed else "failed-runs.json")),
                       "media_base": API_ROOT}
                 for name in definitions}
    # New questions publish separate compact evidence rather than rewriting the
    # frozen campaign. A task explicitly names its supplement; no path guessing.
    supplements = {task.get("research_artifacts", {}).get("api_publication")
                   for task in tasks.values()} - {None}
    for source in sorted(supplements):
        path = Path(source)
        if path.is_absolute() or not (root / path).resolve().is_relative_to(root / API_ROOT):
            raise ValueError("supplemental API publication must stay inside the API report directory")
        document = load(path)
        if not document:
            continue  # Registered, as yet unmeasured question.
        additions = records(path)
        if additions.keys() & definitions.keys():
            raise ValueError("supplemental API publication duplicates an existing variant")
        definitions.update(additions)
        for key, destination in (("readouts", readouts), ("runs", completed)):
            evidence = records(path, key)
            if evidence.keys() - additions.keys():
                raise ValueError("supplemental evidence must belong to its own variant definitions")
            destination.update(evidence)
        locations.update({name: {"definition": str(path), "readout": str(path),
                                "receipt": str(path), "media_base": path.parent}
                          for name in additions})
    if completed.keys() & failed.keys():
        raise ValueError("API publication has conflicting completed and failed receipts")
    receipts = {**completed, **failed}
    if (readouts.keys() | receipts.keys()) - definitions.keys():
        raise ValueError("published API evidence has no matching variant definition")

    by_question = defaultdict(list)
    for case in definitions.values():
        location = locations[case["id"]]
        readout = readouts.get(case["id"], {})
        receipt = receipts.get(case["id"], {})
        recipe = receipt.get("recipe", {})
        prior_kind = {"mog": "mog", "particles": "particle_cloud"}.get(recipe.get("prior_kind"))
        recorded_prior = ({"kind": "particle_cloud", "sigma": 0.} if prior_kind == "particle_cloud" else
                          {"kind": "mog", **{key: recipe[key] for key in ("sigma_rel", "standardize")
                                             if key in recipe}} if prior_kind == "mog" else None)
        media = readout.get("gif") or receipt.get("gif")
        media_path = location["media_base"] / media if media else None
        if media_path is not None:
            resolved = (root / media_path).resolve()
            if not resolved.is_relative_to(root / API_ROOT) or resolved.suffix != ".gif":
                raise ValueError(f"{case['id']}: invalid published GIF path")
        variant = {
            "id": case["id"], "goal": readout.get("goal", case["goal"]),
            "scope": readout.get("scope", case["scope"]),
            "definition_source": location["definition"],
            "evidence_source": location["readout"] if readout else None,
            "receipt_source": location["receipt"] if receipt else None,
            "recipe": receipt.get("recipe", {}).get("name", case.get("default_recipe", "undeclared")),
            "prior": recorded_prior,
            "source_commit": readout.get("source_commit"),
            "source_identity": receipt.get("source_identity"),
            "runtime": receipt.get("runtime", {}), "sampling": case.get("sampling"),
            "execution_status": readout.get("execution_status", "UNMEASURED"),
            "verdict": readout.get("verdict", "UNKNOWN"),
            "completed_updates": readout.get("completed_updates"),
            "default_updates": readout.get("default_updates", case.get("default_steps")),
            "failed_bounds": readout.get("failed_bounds", []),
            "gif": str(media_path) if media_path is not None else None,
            "media_available": media_path is not None and (root / media_path).is_file(),
            "qualification_input": False,
        }
        for question in set(case["legacy_ids"]):
            by_question[question].append(variant)

    publication_path = "reports/forge/technique-inventory.json"
    publication = load(publication_path)
    by_task = defaultdict(list)
    if publication.get("publication_scope") == "current_technique_inventory":
        for row in publication["rows"]:
            config = next((str(path) for path in (
                Path("configs/forge/configurations") / f"{row['candidate_id']}.json",
                Path("configs/forge/ideas") / f"{row['candidate_id']}.json",
            ) if (root / path).is_file()), None)
            for outcome in row["tasks"]:
                # UNKNOWN/BLOCKED remain visible in the single authoritative board.
                if outcome["status"] not in {"PASS", "FAIL", "ERROR", "INVALID", "INCOMPLETE", "UNCONFIRMED"}:
                    continue
                if outcome["task_id"] not in tasks:
                    continue
                recorded_contract = row.get("bindings", {}).get("task_contracts", {}).get(outcome["task_id"])
                contract = publication.get("task_contracts", {}).get(recorded_contract, {})
                current = tasks[outcome["task_id"]]
                declaration_match = (
                    contract.get("execution_sha256") == task_execution_fingerprint(current)
                    and contract.get("evaluation_sha256") == task_evaluation_fingerprint(current)
                    and contract.get("timeout_seconds") == current["resources"].get("timeout_seconds")
                ) if contract else None
                by_task[outcome["task_id"]].append({
                    "task_id": outcome["task_id"], "status": outcome["status"],
                    "family": row["trainer_family"], "candidate_id": row["candidate_id"],
                    "label": row.get("technique", row["trainer_family"]),
                    "candidate_revision": row["candidate_revision"], "cohort": row["cohort"],
                    "backend": row.get("runtime_cohort", {}).get("execution_backend"),
                    "source_commit": row.get("bindings", {}).get("source_origin_commit"),
                    "config_source": config, "evidence_source": publication_path,
                    "recorded_task_contract": recorded_contract,
                    "prior": contract.get("prior"),
                    "prior_applicability": contract.get("sampling", {}).get("prior_applicability"),
                    "declaration_match": declaration_match,
                })

    grouped = defaultdict(list)
    for task in tasks.values():
        execution = task["execution"]
        grouped[execution.get("host") or execution.get("problem") or task["id"]].append(task)
    guides = []
    for question, members in sorted(grouped.items()):
        identities = [f"develop-{question}", f"atlas-{question}"]
        identities += sorted({name for task in members for name in task.get("retained_question_ids", [])})
        original = next((catalog[name] for name in identities if name in catalog), {})
        variants = {case["id"]: case for name in identities for case in by_question[name]}
        goal = next((task["description"] for task in members if isinstance(task.get("description"), str)
                     and task["description"].strip()), None)
        goal = goal or original.get("verifies") or next((case["goal"] for case in variants.values()), None)
        if goal is None:
            conditions = members[0]["evaluation"].get("conditions", [])
            goal = ("Check saved public trainer state under " + ", ".join(conditions) + " perturbations."
                    if conditions else f"Execute the declared {members[0]['evaluation']['kind']} gate.")
        explanations = sorted({task.get("research_artifacts", {}).get("readout") for task in members} - {None})
        for explanation in explanations:
            resolved = (root / explanation).resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError("task research readout must name an existing file in the checkout")
            inputs[explanation] = file_hash(resolved)
        guides.append({
            "id": question, "goal": goal,
            "readout_sources": explanations,
            "original_question_ids": [name for name in identities if name in catalog],
            "tasks": [{"id": task["id"], "source": task_paths[task["id"]],
                       "evaluation": task["evaluation"], "execution": task["execution"]}
                      for task in sorted(members, key=lambda task: task["id"])],
            "api_variants": [variants[name] for name in sorted(variants)],
            "forge_results": sorted((row for task in members for row in by_task[task["id"]]),
                                    key=lambda row: (row["task_id"], row["family"], row["candidate_id"])),
        })
        for variant in variants.values():
            if variant["media_available"] and variant["gif"] not in inputs:
                inputs[variant["gif"]] = file_hash(root / variant["gif"])
    return {
        "experiment_guides": guides,
        "review_artifacts": {name: path for name, path in REVIEW_ARTIFACTS.items() if (root / path).is_file()},
        "artifact_input_hashes": dict(sorted(inputs.items())),
        "artifact_input_digest": stable_hash(inputs),
        "api_variant_count": len(definitions),
    }

"""Explicit ordinary-evidence publication after a prior-ownership migration.

This module cannot train or rank recipes. Old pins remain in their original
policy archive; a declared successor supplies one complete new ordinary row.
"""
from __future__ import annotations

import argparse
import importlib.util
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.trainer_families import (
    CURRENT_SELECTION, current_family_candidates, family_for_candidate, family_row_pin,
    load_current_selection, scientific_row_hash)

SCOPE = "selected_leader_recipe_prior_ownership_v1"
PRIOR_DEFAULTS = dict(prior_update="learned", prior_regularizer="vicreg",
                      prior_reg_target_std=1.0, prior_reg_eps=1e-4, prior_l2=0.0)
EXPECTED_FAMILIES = {"atlas", "bcap-pure", "e22", "k3p", "ka2", "r1r2", "release07-gan-v3"}
REPORT = Path("reports/forge/recipe-prior-refactor")


def require(condition, message):
    if not condition:
        raise ValueError("ownership migration: " + message)


def bound_file(root, descriptor):
    relative = Path(descriptor["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe evidence path")
    path = root / relative
    require(path.is_file() and file_hash(path) == descriptor["sha256"], "evidence file hash mismatch")
    return read_json(path)


def descriptor(root, path):
    return dict(path=path.relative_to(root).as_posix(), sha256=file_hash(path))


def _load_map(root, path):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    require(path.resolve().is_relative_to(root), "migration declaration must be inside repository")
    data = read_json(path)
    require(data.get("schema_version") == 1 and data.get("scope") == SCOPE
            and data.get("qualification_transfer") is False, "unsupported declared migration")
    return data, descriptor(root, path.resolve())


def _validate_bindings(root, data):
    from experiments.forge.api import resolve_public_recipe
    from experiments.forge.planning import load_idea
    baseline = bound_file(root, data["baseline"])
    proof = bound_file(root, data["binding_proof"])
    registration = bound_file(root, data["registration"])
    require(baseline["source_commit"] == "03466efa4de8271b9a2c964406dc6f4f0f260792",
            "original develop lineage changed")
    require(proof.get("status") == "PASS" and proof.get("baseline_sha256") == data["baseline"]["sha256"]
            and proof.get("original_source_commit") == baseline["source_commit"]
            and proof.get("current_source_digest") == data["source_digest"]
            and proof.get("registration_sha256") == data["registration"]["sha256"],
            "actual task-binding proof/source/registration mismatch")
    ordinary_tasks = [a["task"] for a in baseline["view"]["assignments"] if a["qualification_tier"] <= 2]
    require(data["ordinary_task_ids"] == ordinary_tasks and len(ordinary_tasks) == 28,
            "original ordinary task roster changed")
    require(registration.get("scope") == "recipe_prior_refactor_ordinary_measurement"
            and registration.get("baseline_sha256") == data["baseline"]["sha256"]
            and registration.get("through_tier") == 2 and registration.get("view_revision") == 9
            and set(registration.get("families", {})) == EXPECTED_FAMILIES, "ordinary registration scope changed")
    pairs = proof.get("pairs", [])
    expected = {(leader["selection"]["family"], name)
                for leader in baseline["leaders"] if leader["selection"]["family"] not in {"atlas", "e22"}
                for name in data["ordinary_task_ids"]}
    require(len(pairs) == len(expected) == 140
            and {(p["family"], p["task"]) for p in pairs} == expected
            and all(p["original_nonprior_sha256"] == p["current_nonprior_sha256"] for p in pairs),
            "incomplete or changed actual nonprior task bindings")
    entries = data.get("leaders", [])
    require(len(entries) == 7 and {x["family"] for x in entries} == EXPECTED_FAMILIES
            and len({x["configuration_family"] for x in entries}) == 7
            and len({x["candidate_id"] for x in entries}) == 7, "require exactly seven distinct selected leaders")
    original = {leader["selection"]["family"]: leader for leader in baseline["leaders"]}
    spec = importlib.util.spec_from_file_location("ownership_migration_workflow", Path(__file__).with_name("workflow.py"))
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    requests = workflow.resolve_plans(root, registration)
    pair_lookup = {(p["family"], p["task"]): p for p in pairs}
    from experiments.forge.api import task_formulation_context
    for entry in entries:
        leader = original[entry["family"]]
        old = leader["selection"]
        require(entry["configuration_family"] == old["configuration_family"]
                and entry["original_candidate_id"] == old["candidate_id"], "selected configuration family changed")
        require(entry["original_declaration"] == dict(path=leader["declaration_path"], sha256=leader["declaration_sha256"]),
                "original declaration identity changed")
        bound_file(root, entry["original_declaration"])
        candidate = bound_file(root, entry["candidate"])
        require(candidate == load_idea(root, entry["candidate_id"]), "candidate declaration mismatch")
        unchanged = entry["family"] in {"atlas", "e22"}
        require(entry["blocked_without_execution"] is unchanged, "known unsupported cohort changed")
        if unchanged:
            require(candidate == leader["declaration"] and entry["candidate_id"] == old["candidate_id"],
                    "unsupported original declaration must remain exact")
        else:
            require(candidate.get("schema_version") == 3 and candidate.get("parent") == old["candidate_id"]
                    and candidate.get("trainer_family") == old["configuration_family"], "successor lineage/family mismatch")
        actual = asdict(resolve_public_recipe(candidate))
        require(all(stable_hash(actual[key]) == stable_hash(value) for key, value in leader["portable_recipe"].items()),
                "selected global nonprior Recipe or prior weight changed")
        require(all(actual[key] == value for key, value in PRIOR_DEFAULTS.items())
                and actual["prior_reg"] == leader["prior_reg"], "undeclared prior policy or weight changed")
        require(stable_hash(actual) == entry["recipe_sha256"], "public Recipe binding changed")
        study = bound_file(root, entry["study"])
        require(study["id"] == entry["study_id"] and study["candidate"] == entry["candidate_id"]
                and study["scope"] == dict(view="discriminator_stability", through_tier=2,
                    execution_backend="cuda", cuda_model="NVIDIA RTX A6000"), "ordinary study scope changed")
        plan, request = registration["families"][entry["family"]], requests[entry["family"]]
        require(plan["candidate_id"] == entry["candidate_id"] and plan["candidate_revision"] == entry["candidate_revision"]
                and plan["study_id"] == entry["study_id"] and plan["source_digest"] == data["source_digest"]
                and stable_hash(plan["study_admission"]) == entry["study_admission_sha256"]
                and len(plan["tasks"]) == len(ordinary_tasks)
                and {task["task"] for task in plan["tasks"]} == set(ordinary_tasks),
                "family plan differs from its exact ordinary registration")
        if not unchanged:
            for name in ordinary_tasks:
                current = task_formulation_context(request["candidate"], request["tasks"][name], request["protocol"], root=root)
                old_task = read_json(root / REPORT / "legacy-task-cards" / f"{name}.json")
                require(file_hash(root / REPORT / "legacy-task-cards" / f"{name}.json") == baseline["task_cards"][name]["sha256"],
                        "original initial-law task bytes changed")
                original_context = task_formulation_context(leader["declaration"], old_task, request["protocol"], root=root)
                prior_hash = pair_lookup[entry["family"], name].get("initial_prior_sha256")
                require(prior_hash == stable_hash(current.prior_config) == stable_hash(original_context.prior_config),
                        "actual original/current initial prior law differs")
    return baseline


def build_map(root, registration_path, proof_path):
    from experiments.forge.api import resolve_public_recipe
    root = Path(root).resolve()
    registration_path = Path(registration_path).resolve()
    require(registration_path.is_relative_to(root), "exact registration must be committed with reproduction metadata")
    registration = read_json(registration_path)
    baseline_path = root / REPORT / "baseline.json"
    baseline = read_json(baseline_path)
    card = read_json(root / CURRENT_SELECTION)
    require(file_hash(root / CURRENT_SELECTION) == baseline["family_current_sha256"], "original selected pin card changed")
    entries = []
    for leader in baseline["leaders"]:
        choice = leader["selection"]
        plan = registration["families"][choice["family"]]
        candidate_id = plan["candidate_id"]
        candidate_path = root / (leader["declaration_path"] if choice["family"] in {"atlas", "e22"}
                                 else f"configs/forge/ideas/{candidate_id}.json")
        candidate = read_json(candidate_path)
        pin = next(x for x in card["selections"] if x["trainer_family"] == choice["configuration_family"])
        study_path = root / f'configs/forge/studies/{plan["study_id"]}.json'
        entries.append(dict(family=choice["family"], configuration_family=choice["configuration_family"],
            original_candidate_id=choice["candidate_id"], original_pin_sha256=stable_hash(pin),
            original_declaration=dict(path=leader["declaration_path"], sha256=leader["declaration_sha256"]),
            candidate_id=candidate_id, candidate=descriptor(root, candidate_path),
            candidate_revision=plan["candidate_revision"], recipe_sha256=stable_hash(asdict(resolve_public_recipe(candidate))),
            study_id=plan["study_id"], study=descriptor(root, study_path),
            study_sha256=stable_hash(read_json(study_path)), study_admission_sha256=stable_hash(plan["study_admission"]),
            blocked_without_execution=choice["family"] in {"atlas", "e22"}))
    ordinary_task_ids = [a["task"] for a in baseline["view"]["assignments"] if a["qualification_tier"] <= 2]
    families = {x["configuration_family"] for x in entries}
    data = dict(schema_version=1, scope=SCOPE, qualification_transfer=False,
        baseline=descriptor(root, baseline_path), binding_proof=descriptor(root, proof_path.resolve()),
        registration=descriptor(root, registration_path),
        original_selection=dict(path=CURRENT_SELECTION.as_posix(), sha256=file_hash(root / CURRENT_SELECTION)),
        original_policy_fingerprint=card["policy_fingerprint"],
        new_policy_fingerprint=stable_hash(read_json(root / "configs/forge/views/discriminator_stability.json")),
        source_digest=next(iter(registration["families"].values()))["source_digest"],
        ordinary_task_ids=ordinary_task_ids, leaders=entries,
        archived_only_selections=[deepcopy(pin) for pin in card["selections"] if pin["trainer_family"] not in families])
    _validate_bindings(root, data)
    return data


def advance_selection(root, manifest, report, rows, prior_reports, path, *, validate_row):
    from experiments.forge.trainer_families import _current_pin
    from experiments.forge.views import load_tasks, load_view, task_evaluation_fingerprint, task_execution_fingerprint
    root = Path(root).resolve()
    data, declaration = _load_map(root, path)
    _validate_bindings(root, data)
    card = bound_file(root, data["original_selection"])
    require(manifest["view"] == report["view"] == "discriminator_stability"
            and manifest["view_revision"] == 8 and report["view_revision"] == 9
            and manifest["policy_fingerprint"] == data["original_policy_fingerprint"]
            and report["policy_fingerprint"] == data["new_policy_fingerprint"], "ownership policy revisions differ")
    require(report.get("publication_scope") == "frozen_source"
            and data["source_digest"] in report.get("frozen_source", {}).get("source_digests", []),
            "require independently reconstructed frozen ordinary source")
    require(card["historical_selections"] and len(card["selections"]) == 20,
            "original whole selection/history changed")
    pins = load_current_selection(root, view_id=manifest["view"], policy_fingerprint=manifest["policy_fingerprint"])
    visible = current_family_candidates(root)
    require({(x["family"], x["configuration_family"], x["candidate_id"]) for x in visible} ==
            {(x["family"], x["configuration_family"], x["original_candidate_id"]) for x in data["leaders"]},
            "visible selected roster changed")
    for family, pin in pins.items():
        matches = [{**row, "trainer_family": family} for _, _, old_rows in prior_reports for row in old_rows.values()
                   if family_row_pin({**row, "trainer_family": family}, selection_kind=pin["selection_kind"],
                       reason=pin["reason"], measurement_views=pin.get("measurement_views"),
                       measurement_tasks=pin.get("measurement_tasks")) == pin]
        require(matches, "old selection lacks exact verified whole-row evidence")
    active_families = {entry["configuration_family"] for entry in data["leaders"]}
    require(data["archived_only_selections"] == [pin for pin in card["selections"] if pin["trainer_family"] not in active_families]
            and len(data["archived_only_selections"]) == 13, "archived-only original pins changed")
    tasks, view = load_tasks(root), load_view(root, report["view"])
    require(stable_hash(view) == data["new_policy_fingerprint"], "current view changed")
    selections = []
    for entry in data["leaders"]:
        family = entry["configuration_family"]
        old = pins[family]
        require(stable_hash(old) == entry["original_pin_sha256"], "exact old pin changed")
        row = rows.get(entry["candidate_id"])
        require(row is not None and row["candidate_revision"] == entry["candidate_revision"]
                and row.get("runtime_cohort", {}).get("execution_backend") == old["execution_backend"] == "cuda",
                "missing or mismatched successor whole runtime row")
        row = {**deepcopy(row), "trainer_family": family}
        require(row.get("bindings", {}).get("source_digest") == data["source_digest"]
                and row["bindings"].get("recipe_sha256") == entry["recipe_sha256"], "successor source/Recipe differs")
        candidate = bound_file(root, entry["candidate"])
        require(family_for_candidate(root, entry["candidate_id"], candidate, current_presentation=True)["id"] == family,
                "successor belongs to another configuration family")
        validate_row(root, report, row)
        # Check all 28 current contracts, including unexecuted cells, rather
        # than treating a matching score or Tier1 summary as source identity.
        for name in data["ordinary_task_ids"]:
            digest = row["bindings"].get("task_contracts", {}).get(name)
            contract = report.get("task_contracts", {}).get(digest, {})
            task = tasks[name]
            require(digest == stable_hash(contract) and contract.get("execution_sha256") == task_execution_fingerprint(task)
                    and contract.get("evaluation_sha256") == task_evaluation_fingerprint(task)
                    and contract.get("timeout_seconds") == task["resources"]["timeout_seconds"], "current task contract mismatch")
        if entry["blocked_without_execution"]:
            require(not row.get("attempt_ids") and row.get("qualified_tier") == 0 and row.get("status") == "BLOCKED",
                    "unsupported original received fabricated execution or credit")
        else:
            require(row.get("attempt_ids"), "successor requires actual ordinary receipts")
            proofs = [proof for proof in report.get("registered_study_reconstruction", [])
                      if proof.get("candidate_id") == entry["candidate_id"] and proof.get("study_id") == entry["study_id"]]
            require(len(proofs) == 1 and proofs[0].get("study_sha256") == entry["study_sha256"]
                    and proofs[0].get("admission_sha256") == entry["study_admission_sha256"]
                    and proofs[0].get("source_digest") == data["source_digest"] and proofs[0].get("original_receipts"),
                    "missing or mixed ordinary admitted-study receipts")
        measurements = old.get("measurement_views", [manifest["view"]])
        required = {a["task"] for name in measurements for a in load_view(root, name)["assignments"]
                    if a["importance"] == "required" and a["qualification_tier"] == 1}
        required.update(old.get("measurement_tasks", []))
        observed = row.get("tasks", []) + row.get("nonrequired_tasks", [])
        statuses = {task["task_id"]: task["status"] for task in observed}
        complete = bool(row.get("attempt_ids")) and len(statuses) == len(observed) and bool(required) and all(
            statuses.get(task) in {"PASS", "FAIL"} for task in required)
        reason = "Retain the declared selected recipe after explicit prior-ownership migration; fresh ordinary evidence only, no outcome ranking or archived qualification transfer."
        pin = family_row_pin(row, selection_kind="current_measurement" if complete else "historical_incumbent", reason=reason,
            measurement_views=measurements if complete else None,
            measurement_tasks=old.get("measurement_tasks") if complete else None)
        _current_pin(root, family, [row], pin, view_id=report["view"], catalogs=report)
        selections.append(pin)
    original = (root / CURRENT_SELECTION).read_bytes().decode("utf-8")
    digest = data["original_selection"]["sha256"]
    relative = CURRENT_SELECTION.parent / "history" / f"family-current-{digest}.json"
    require(not (root / relative).exists() or (root / relative).read_bytes() == original.encode(), "original archive conflicts")
    receipt = dict(declaration=declaration, original_selection_archive=dict(path=relative.as_posix(), sha256=digest),
        archived_only_selections=deepcopy(data["archived_only_selections"]), source_digest=data["source_digest"],
        qualification_transfer=False, selection_scope="seven_visible_selected_leaders")
    return {**deepcopy(card), "policy_fingerprint": report["policy_fingerprint"], "selections": selections,
            "ownership_migration": receipt}, (relative, original, digest)


@contextmanager
def archived_presentation(root, card):
    """Old searches cannot select a new-policy row for archived alternatives."""
    from experiments.forge import trainer_families
    receipt = (card or {}).get("ownership_migration")
    if not receipt:
        yield
        return
    root = Path(root).resolve()
    data = bound_file(root, receipt["declaration"])
    require(receipt["qualification_transfer"] is False and data["scope"] == SCOPE
            and receipt["archived_only_selections"] == data["archived_only_selections"], "archived presentation identity changed")
    archive = root / receipt["original_selection_archive"]["path"]
    original = archive if archive.is_file() else root / CURRENT_SELECTION
    require(file_hash(original) == receipt["original_selection_archive"]["sha256"], "old pin-card archive changed")
    excluded = {pin["trainer_family"] for pin in receipt["archived_only_selections"]}
    original_search = trainer_families._search_pin
    def search(directory, family_id, backend, rows, *args, **kwargs):
        if Path(directory).resolve() == root and family_id in excluded:
            require(all(trainer_families._declaration_only_row(row) for row in rows),
                    "archived-only presentation cannot hide a new measurement")
            return None
        return original_search(directory, family_id, backend, rows, *args, **kwargs)
    trainer_families._search_pin = search
    try:
        yield
    finally:
        trainer_families._search_pin = original_search


def label_archived_declarations(family_result, card):
    from experiments.forge.trainer_families import _declaration_only_row
    excluded = {pin["trainer_family"] for pin in card["ownership_migration"]["archived_only_selections"]}
    for row in family_result["rows"]:
        if row["trainer_family"] in excluded:
            require(_declaration_only_row(row), "archived alternative received new-policy gate credit")
            row["selection"] = dict(selection_kind="archived_policy_declaration", qualified=False, default_adoption=False,
                qualification_input=False, qualification_reuse=False,
                reason="Current unmeasured declaration only; selected optimizer evidence remains in its exact original-policy archive.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--binding-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    data = build_map(options.repository.resolve(), options.registration, options.binding_proof)
    atomic_json(options.output, data)
    print(dict(scope=SCOPE, leaders=len(data["leaders"]), archived_only=len(data["archived_only_selections"])), flush=True)


if __name__ == "__main__":
    main()

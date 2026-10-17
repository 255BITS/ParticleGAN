"""Register a completed ordinary pair and replace only BCAP's research pin.

No training, new sampling, or default-qualification claim. Invoke only after the
coordinator publishes the complete saved results and independent audit. Original
Forge request/evidence/result receipts must remain hydrated locally; the compact
receipt projections produced here are display evidence, never grading inputs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.planning import load_idea
from experiments.forge.trainer_families import (
    CURRENT_SELECTION, _current_pin, comparison_cohort, family_for_candidate,
    family_row_pin, scientific_row_hash,
)
from experiments.forge.views import load_tasks, task_evaluation_fingerprint, task_execution_fingerprint
from reports.forge import regenerate_technique_inventory as publication

REPORT = Path("reports/forge/bcap-default-baseline")
FAMILY = "bcap-dualnorm"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_row(root, report, name, commit, digest):
    matches = [deepcopy(row) for row in report["rows"] if row["candidate_id"] == name
               and row.get("bindings", {}).get("source_digest") == digest]
    require(len(matches) == 1, f"need one exact measured source/runtime row: {name}")
    row = matches[0]
    require(bool(row.get("attempt_ids")), f"ordinary measured receipts missing: {name}")
    require(row.get("runtime_cohort", {}).get("execution_backend") == "cuda", "CUDA cohort required")
    require(commit in row["bindings"].get("recorded_source_origin_commits", []), "wrong executed origin")
    for attempt in row["attempt_ids"]:
        # Recheck the originals, not merely a compact report's self-hash.
        actual = publication.project_receipt(root, attempt)
        proof = report["provenance"]["qualified_receipts"][attempt]
        provenance = actual["provenance"]
        require(provenance["canonical_result_hash"] == proof["canonical_result_hash"], "result proof changed")
        require({key: value["sha256"] for key, value in provenance["original_files"].items()}
                == proof["original_file_sha256"], "original receipt hashes changed")
        request = read_json(root / "reports/forge/attempts" / attempt / "request.json")
        source = request.get("request", request)["source"]
        require(source["origin_commit"] == commit and source["digest"] == digest, "mixed executed sources")
    publication._validate_published_row(root, report, row)
    family = family_for_candidate(root, name, load_idea(root, name), current_presentation=True)
    require(family["id"] == FAMILY, "ordinary candidate must explicitly belong to bcap-dualnorm")
    row["trainer_family"] = FAMILY
    return row


def validate_saved_pair(audit, results, names):
    """A valid audit may contain unverified pairs; those cannot select a winner."""
    selected = {(item["role"], item["task_id"]): item for item in results["task_results"]
                if item.get("role") in {"control", "candidate"} and item.get("task_id") in names}
    require(len(selected) == 2 * len(names), "need both certified results for every compared task")
    proofs = {}
    for proof in audit["saved_state_comparisons"]:
        name = proof["task_id"]
        if name not in names or proof["saved_variant"] != "final":
            continue
        require(name not in proofs, "duplicate final consumed-state proof")
        require(set(proof["completed_roles"]) == {"control", "candidate"}
                and proof["all_declared_arms_present"] is True
                and proof["initialization_and_prior_equal"] is True
                and proof["named_training_bindings_equal"] is True
                and proof["consumed_non_eval_streams_and_batches_equal"] is True,
                f"unverified paired initial/prior/RNG/batch history: {name}")
        for role in ("control", "candidate"):
            saved = selected[role, name]["provenance_checkpoint"]
            require(proof["saved_state_references"][role] == saved
                    and proof["completed_steps"][role] == saved["completed_steps"],
                    f"audit references a different consumed state: {role}/{name}")
        proofs[name] = proof
    require(set(proofs) == set(names), "missing final consumed-state comparison for a required task")
    require(not any(item["task_id"] in names for item in audit["unavailable_complete_states"]),
            "unavailable complete state in the compared task cohort")
    return {"verified_final_consumed_state_pairs": len(proofs),
            "saved_state_comparisons_sha256": stable_hash([proofs[name] for name in sorted(proofs)])}


def validate_pair(root, report, candidate, control, results, audit):
    required = report["tier_requirements"]
    require([len(required[str(tier)]) for tier in (1, 2, 3)] == [6, 21, 2], "ordinary denominator changed")
    names = [*required["1"], *required["2"]]
    declarations = load_tasks(root)
    statuses = {}
    for role, row in (("candidate", candidate), ("control", control)):
        cells = {task["task_id"]: task["status"] for task in row["tasks"]}
        require(len(cells) == len(row["tasks"]), "duplicate ordinary task cells")
        require(all(cells.get(name) in {"PASS", "FAIL"} for name in names), f"unresolved Tier1/Tier2: {role}")
        require(all(cells[name] == "PASS" for name in required["1"]), f"all six Tier1 gates must PASS: {role}")
        for tier in ("1", "2"):
            cell = row["tiers"][tier]
            require(cell["total"] == len(required[tier])
                    and cell["passed"] == sum(cells[name] == "PASS" for name in required[tier]), "tier summary mismatch")
        for name in names:
            task = declarations[name]
            contract = report["task_contracts"][row["bindings"]["task_contracts"][name]]
            require(contract["execution_sha256"] == task_execution_fingerprint(task)
                    and contract["evaluation_sha256"] == task_evaluation_fingerprint(task)
                    and contract["timeout_seconds"] == task["resources"]["timeout_seconds"], "task contract changed")
        statuses[role] = cells
    require(comparison_cohort(candidate, report, task_ids=names)
            == comparison_cohort(control, report, task_ids=names), "pair differs in source/runtime/task/prior/RNG contract")
    recipes = {role: deepcopy(report["recipe_contracts"][row["bindings"]["recipe_sha256"]])
               for role, row in (("candidate", candidate), ("control", control))}
    require(recipes["candidate"].pop("constraint_geometry_mode", "none") == "direction_blend", "candidate geometry changed")
    require(recipes["control"].pop("constraint_geometry_mode", "none") == "none", "control geometry changed")
    require(recipes["candidate"] == recipes["control"], "resolved recipes differ beyond declared geometry delta")
    # Saved publication must describe these same ordinary results, not an older
    # child diagnostic or a new served/initialization cohort.
    compact = {}
    for cell in results["task_cells"]:
        role = cell.get("role")
        name = cell.get("task_id", cell.get("task"))
        if role in statuses and name in names:
            require((role, name) not in compact, "duplicate compact result cell")
            compact[role, name] = cell["gate_status"]
    require(all(compact.get((role, name)) == cells[name]
                for role, cells in statuses.items() for name in names), "compact result differs from ordinary receipts")
    old_passes = {name for name in required["2"] if statuses["control"][name] == "PASS"}
    new_passes = {name for name in required["2"] if statuses["candidate"][name] == "PASS"}
    require(old_passes < new_passes, "candidate must preserve every matched-control Tier2 pass and add a pass")
    state_proof = validate_saved_pair(audit, results, names)
    return {**state_proof, "control_tier2_passes": sorted(old_passes), "candidate_tier2_passes": sorted(new_passes),
            "repaired_tier2_tasks": sorted(new_passes - old_passes), "regressed_tier2_tasks": [],
            "unresolved_tier1_tier2_tasks": [], "tier3": "Retain exact recorded gates; ordinary veto remains effective."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--candidate", default="bcap-default-baseline-direction-v1")
    parser.add_argument("--control", default="bcap-default-baseline-control-v1")
    parser.add_argument("--results", type=Path, default=REPORT / "results.json")
    parser.add_argument("--audit", type=Path, default=REPORT / "audit.json")
    parser.add_argument("--artifact-root", type=Path, required=True, help="archive for the independently regraded source evidence")
    parser.add_argument("--validated-report", type=Path, help="reuse an intact frozen report after rechecking original receipts")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    resolve = lambda path: path.resolve() if path.is_absolute() else (root / path).resolve()
    result_path, audit_path = resolve(args.results), resolve(args.audit)
    results, audit = read_json(result_path), read_json(audit_path)
    commit = publication._resolve_commit(root, args.source_commit)
    require(results.get("scope") == "ordinary_bcap_default_baseline_pair", "ordinary saved pair required")
    digest = results["source_digest"]
    require(results["source_commit"] == audit["source_commit"] == commit
            and audit["source_digest"] == digest and audit["status"] == "PASS", "independent audit/source mismatch")
    require(audit["task_results_sha256"] == stable_hash(results["task_results"]), "audit results binding changed")
    require(results["original_placement_counts"] == {"tier1": 6, "tier2": 21, "tier3": 2}, "original task placement changed")
    if args.validated_report:
        metadata = {"json": str(resolve(args.validated_report)), "source_commit": commit}
    else:
        metadata = publication.regenerate(root, source_commit=commit, execution_backend="cuda",
                                          output_prefix=resolve(args.artifact_root) / "source-evidence")
    report, _ = publication._published_report(root, Path(metadata["json"]))
    require(report["publication_scope"] == "frozen_source" and report["frozen_source"]["commit"] == commit
            and report["frozen_source"]["source_digests"] == [digest], "wrong frozen evidence report")
    manifest = read_json(root / publication.EVIDENCE_MANIFEST)
    require(all(report[key] == manifest[key] for key in publication.POLICY_FIELDS), "policy differs; do not silently advance it")
    candidate = exact_row(root, report, args.candidate, commit, digest)
    control = exact_row(root, report, args.control, commit, digest)
    evidence = validate_pair(root, report, candidate, control, results, audit)
    card_path = root / CURRENT_SELECTION
    original_bytes, original_sha = card_path.read_bytes(), file_hash(card_path)
    card = read_json(card_path)
    matches = [pin for pin in card["selections"] if pin["trainer_family"] == FAMILY]
    require(len(matches) == 1 and card["default_adoption"] is False, "need one existing BCAP research selection")
    previous = deepcopy(matches[0])
    pin = family_row_pin(candidate, selection_kind="configured_standard",
        reason="Completed seed-0 ordinary matched pair: both global recipes pass 6/6 Tier1; direction_blend preserves all matched-control Tier2 passes and adds passes. Select this whole measured research baseline, retaining Tier2 failures and ordinary Tier3 veto. Calibration remains provisional; no qualified public-default claim.")
    _current_pin(root, FAMILY, [candidate], pin, view_id=report["view"], catalogs=report)
    card["selections"] = [pin if item["trainer_family"] == FAMILY else item for item in card["selections"]]
    evidence_sha = file_hash(Path(metadata["json"]))
    original_regenerate = publication.regenerate

    def verified_same_report(checkout, *, source_commit, execution_backend, view_id, output_prefix):
        require(Path(checkout) == root and source_commit == commit and execution_backend == "cuda"
                and view_id == report["view"] and file_hash(Path(metadata["json"])) == evidence_sha, "staged report changed")
        return metadata

    atomic_json(card_path, card)
    publication.regenerate = verified_same_report
    try:
        published = publication.publish_current(root, source_commit=commit, execution_backend="cuda")
    except Exception:
        card_path.write_bytes(original_bytes)
        raise
    finally:
        publication.regenerate = original_regenerate
    atomic_json(root / REPORT / "selection-change.json", {
        "schema_version": 1, "source_commit": commit, "source_digest": digest,
        "previous_card_sha256": original_sha, "current_card_sha256": file_hash(card_path),
        "previous_pin": previous, "selected_pin": pin,
        "preserved_other_family_pins_sha256": stable_hash([item for item in card["selections"] if item["trainer_family"] != FAMILY]),
        "historical_selections_unchanged_sha256": stable_hash(card.get("historical_selections", [])),
        "results_sha256": file_hash(result_path), "audit_sha256": file_hash(audit_path),
        "source_evidence_sha256": evidence_sha, "source_evidence_input_digest": report["provenance"]["input_digest"],
        "selected_scientific_row_sha256": scientific_row_hash(candidate), "selection_evidence": evidence,
        "publication": published, "default_adoption": False, "qualification_input": False,
        "helper_sha256": file_hash(Path(__file__)),
        "history": "Previous pin retained verbatim here; all immutable source snapshots and other family selections preserved.",
    })
    print(published)


if __name__ == "__main__":
    main()

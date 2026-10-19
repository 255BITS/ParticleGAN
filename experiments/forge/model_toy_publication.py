"""Compact API-mode research on the existing board, without qualification reuse."""
from pathlib import Path
import os

from .contracts import file_hash, read_json, stable_hash

PUBLICATION = Path("reports/toy_audit/api_contract/model_modes/publication.json")
PROTOCOL = Path("reports/forge/api-model-toys/protocol.json")


def load_publication(root):
    root = Path(root)
    if not (root / PUBLICATION).is_file():
        return None
    data = read_json(root / PUBLICATION)
    protocol = read_json(root / PROTOCOL)
    if (data.get("schema") != "forge_api_model_research_v1" or data.get("qualification_input") is not False
            or data.get("seed") != 0 or data.get("protocol_sha256") != file_hash(root / PROTOCOL)):
        raise ValueError("model research publication has changed its frozen protocol")
    if {row["task_id"] for row in data["readouts"]} != set(protocol["task_contracts"]):
        raise ValueError("model research publication must retain its complete declared denominator")
    definitions = {case["id"]: case for case in data["cases"]}
    receipts = {row["id"]: row for row in data["runs"]}
    for row in data["readouts"]:
        task = protocol["task_contracts"][row["task_id"]]
        if (row["id"] not in definitions or row["id"] not in receipts
                or row["task_contract_sha256"] != stable_hash(task)
                or row["qualification_input"] is not False):
            raise ValueError("model research result lacks its exact case and task contract")
        receipt = receipts[row["id"]]
        if receipt["task_id"] != row["task_id"] or receipt["recipe_sha256"] != stable_hash(receipt["recipe"]):
            raise ValueError("model research changed its resolved recipe")
        adapted = set(task["execution"]["recipe_fields"]) | {
            "num_particles", "z_dim", "batch_size", "total_steps", "prior_kind", "sigma_rel", "standardize"}
        # Public serialization omits inactive/default fields such as
        # adam_variant. Rehydrate through that same API before comparison.
        from particlegan import Recipe
        resolved = Recipe(**receipt["recipe"])
        if any(stable_hash(getattr(resolved, key)) != stable_hash(value) for key, value in protocol["candidate_contract"]["recipe_overrides"].items()
               if key not in adapted):
            raise ValueError("model research changed the one global baseline trainer")
        if row["verdict"] == "PASS" and not (row["full_protocol_complete"] and row["terminal_passes"] == 5):
            raise ValueError("partial model research cannot claim trained quality PASS")
        if row.get("gif"):
            path = (root / PUBLICATION.parent / row["gif"]).resolve()
            if not path.is_relative_to((root / PUBLICATION.parent).resolve()) or file_hash(path) != receipt["gif_sha256"]:
                raise ValueError("model research actual-training GIF changed")
    return data


def board_section(root, page):
    data = load_publication(root)
    if data is None:
        return []
    root, page = Path(root), Path(page)
    def link(label, target):
        return f"[{label}]({os.path.relpath(root / target, page.parent)})"
    lines = ["## AE, VAE and DDGAN API research", "",
             "One fixed BISPF recipe on new model-mode tasks, protocol seed 0. These source-bound CPU research results "
             "keep their own gates, prior, initialization, budget and sampling; they add no selected-family qualification credit. "
             + link("Proposal, failed bounds and recommendations", "reports/forge/api-model-toys/README.md") + ".", "",
             "| Task | Declared tier | Full-budget result | Completed / required updates | Terminal passing checks | Actual training |",
             "| --- | ---: | --- | ---: | ---: | --- |"]
    for row in data["readouts"]:
        gif = link("GIF", PUBLICATION.parent / row["gif"]) if row.get("gif") else "—"
        lines.append(f"| `{row['task_id']}` | {row['tier']} | **{row['verdict']}** | "
                     f"{row['completed_updates']} / {row['default_updates']} | {row['terminal_passes']} / 5 | {gif} |")
    lines.extend(["", "Tier 2 full runs require every basic model task to pass. A blocked task may have a separately "
                  "declared three-update software GIF; that prefix is INCOMPLETE and grants no deeper training credit.", ""])
    return lines


def memory_inputs(root):
    """Track compact evidence and display code without hydrating execution logs."""
    data = load_publication(root)
    if data is None:
        return []
    paths = {Path(root) / path for path in (PUBLICATION, PROTOCOL,
             Path("experiments/forge/model_toy_publication.py"))}
    paths.update(Path(root) / PUBLICATION.parent / row["gif"]
                 for row in data["readouts"] if row.get("gif"))
    paths.update((Path(root) / "reports/forge/api-model-toys").glob("*.md"))
    return sorted(paths)


def memory_records(root):
    """Project the published round, keeping software prefixes out of full gates."""
    from .publication_memory import _base, _source

    root = Path(root)
    data = load_publication(root)
    if data is None:
        return []
    protocol = read_json(root / PROTOCOL)
    record = _base(_source(root, root / PUBLICATION, "reports/forge/technique-inventory.md"),
                   "api-model-toys-v1", data["baseline"],
                   {"publication": str(PUBLICATION), "protocol_sha256": data["protocol_sha256"]},
                   protocol["goal"], "published_study")
    receipts = {row["id"]: row for row in data["runs"]}
    record.update(
        mechanism_class="public_api_model_mode_research", trainer_family="bispf",
        hypothesis="Test AE reconstruction/generation, hard/categorical VAE, full DDGAN and AE-DDGAN chains through the public API.",
        settings=protocol["candidate_contract"]["recipe_overrides"],
        provenance={"protocol": str(PROTOCOL), "protocol_sha256": data["protocol_sha256"],
                    "source_commits": sorted({row["source_commit"] for row in data["readouts"]}),
                    "seed": data["seed"], "execution_backend": protocol["execution_backend"],
                    "cross_cohort_pooling": False},
        recorded_cost={"wall_seconds": data["paid_cpu_seconds"],
                       "scope": "Summed CPU API initialization, training and numerical evaluation; imports, export and media excluded. No speed ranking."},
        conclusion="Three PASS (both AE cases and categorical VAE), three full-budget FAIL (hard VAE terminal stability and both Gaussian chains), two BLOCKED bimodal full runs. "
                   "AE-DDGAN paired reconstruction passes its terminal subgates while independent generation fails. Three-update deeper GIFs are INCOMPLETE software prefixes. No qualification or default credit.",
        next_action="Retain scoped positives and failures. Inspect saved VAE usage/residuals and DDGAN per-timestep transitions before one new bounded mechanism hypothesis; no seed sweep or unchanged rerun.")
    record["task_results"] = [{
        "task_id": row["task_id"], "gate_status": row["verdict"], "metrics": row["final_metrics"] if row["full_protocol_complete"] else {},
        "failed_bounds": row["failed_bounds"], "full_protocol_complete": row["full_protocol_complete"],
        "completed_updates": row["completed_updates"], "required_updates": row["default_updates"],
        "terminal_passes": row["terminal_passes"], "qualification_tier": row["tier"],
        "task_contract_sha256": row["task_contract_sha256"],
        "recipe_sha256": receipts[row["id"]]["recipe_sha256"],
        "gif": str(PUBLICATION.parent / row["gif"]),
        "software_prefix": None if row["full_protocol_complete"] else {
            "status": "INCOMPLETE", "completed_updates": row["completed_updates"],
            "diagnostic_metrics": row["final_metrics"]},
    } for row in data["readouts"]]
    return [record]

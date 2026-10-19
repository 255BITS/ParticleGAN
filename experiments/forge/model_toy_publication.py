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
        if any(receipt["recipe"][key] != value for key, value in protocol["candidate_contract"]["recipe_overrides"].items()
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

"""Deterministic task inventory from Forge's current qualification view policy."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shlex
from urllib.parse import quote

from .contracts import atomic_json, atomic_text, file_hash, read_json, stable_hash
from .knowledge import DECLARATION_ONLY_BOARD, leaderboard_path
from .priors import PRIOR_CODE_PATHS, task_prior
from .research_artifacts import build_artifacts
from .family_reports import evaluation_notes, evaluation_rows, number
from .views import load_tasks, load_view


REPORT_PATH = Path("reports/forge/EXPERIMENTS_BY_TIER.md")
TIER_NAMES = {1: "smoke", 2: "quality", 3: "endurance"}
IMPORTANCES = ("required", "ranking", "diagnostic")


def build_report(root: Path, view_id: str | None = None, *, publication: dict | None = None) -> dict:
    """Read validated declarations, including tasks not assigned to any view."""
    root = Path(root)
    tasks = load_tasks(root)
    for task in tasks.values():
        for dependency in task["dependencies"]:
            if dependency["task"] not in tasks:
                raise ValueError(f"{task['id']}: missing catalog dependency {dependency['task']!r}")
    task_paths = {read_json(path)["id"]: path.relative_to(root).as_posix()
                  for path in sorted([*(root / "configs/forge/tasks").glob("*.json"),
                                      *(root / "configs/forge/task-variants").glob("*/*.json")])}
    view_paths = sorted((root / "configs/forge/views").glob("*.json"))
    if not view_paths:
        raise ValueError("no Forge views found")
    views = [load_view(root, path.stem) for path in view_paths]
    if view_id is not None and view_id not in {view["id"] for view in views}:
        raise ValueError(f"unknown Forge view {view_id!r}")
    assigned = {assignment["task"] for view in views for assignment in view["assignments"]}

    def task_row(name):
        task = tasks[name]
        execution = task["execution"]
        return {
            "id": name, "source": task_paths[name], "adapter": task["adapter"],
            "guide_id": execution.get("host") or execution.get("problem") or name,
            "evaluation_kind": task["evaluation"]["kind"],
            "prior": task_prior(task),
            "prior_code_path": PRIOR_CODE_PATHS[execution["prior"]["kind"]],
            "prior_applicability": execution.get("prior_applicability"),
            "initializer": execution["initializer"],
            "fixed_initialization": execution.get("fixed_initialization"),
            "host_initialization": execution.get("host_definition", {}).get("initialization"),
            "protocol": execution.get("protocol"),
            "steps": execution.get("steps"),
            "incremental_steps": execution.get("incremental_steps"),
            "extension_steps": execution.get("extension_steps"),
            "max_total_steps": execution.get("max_total_steps"),
            "timeout_seconds": task["resources"].get("timeout_seconds"),
            "execution_group": execution.get("execution_group"),
            "uninterrupted": execution.get("uninterrupted", False),
            "dependencies": task["dependencies"],
        }

    result_views = []
    for view, path in zip(views, view_paths):
        if view_id is not None and view["id"] != view_id:
            continue
        tiers = []
        for tier, name in TIER_NAMES.items():
            assignments = sorted(
                (a for a in view["assignments"] if a["qualification_tier"] == tier),
                key=lambda a: (a["order"], a["task"]))
            counts = Counter(a["importance"] for a in assignments)
            tiers.append({
                "qualification_tier": tier, "name": name,
                "counts": {importance: counts[importance] for importance in IMPORTANCES},
                "tasks": [{**task_row(a["task"]), "importance": a["importance"], "order": a["order"]}
                          for a in assignments],
            })
        result_views.append({
            "id": view["id"], "revision": view["revision"], "goal": view["goal"],
            "evidence_scope": view.get("evidence_scope"),
            "calibration": view.get("calibration", {}),
            "source": path.relative_to(root).as_posix(), "tiers": tiers,
        })
    input_paths = list(task_paths.values()) + [path.relative_to(root).as_posix() for path in view_paths]
    inputs = {name: file_hash(root / name) for name in sorted(input_paths)}
    return {
        "schema_version": 1, "task_count": len(tasks), "assigned_task_count": len(assigned),
        "view_count": len(views), "views": result_views,
        "prior_counts": {kind: sum(task["execution"]["prior"]["kind"] == kind for task in tasks.values())
                         for kind in PRIOR_CODE_PATHS},
        "nonsampled_prior_count": sum(task["execution"].get("prior_applicability") == "not_sampled"
                                     for task in tasks.values()),
        "unassigned_tasks": [task_row(name) for name in sorted(tasks.keys() - assigned)],
        "task_sources": task_paths,
        "input_hashes": inputs, "input_digest": stable_hash(inputs),
        **build_artifacts(root, tasks, task_paths, publication=publication),
    }


def _cell(value) -> str:
    return "—" if value is None else str(value).replace("|", "\\|").replace("\n", " ")


def _prior_cell(prior, applicability=None):
    """Label the declared/recorded kind, including zero-sigma MoG evidence."""
    if not prior or prior.get("kind") not in PRIOR_CODE_PATHS:
        return "unrecorded"
    fields = []
    for key in ("sigma", "sigma_rel"):
        if key in prior:
            fields.append(f"{key}={prior[key]:g}")
    if applicability == "not_sampled":
        fields.append("not sampled")
    return PRIOR_CODE_PATHS[prior["kind"]] + (" (" + "; ".join(fields) + ")" if fields else "")


def render_markdown(report: dict, root: Path, output_path: Path | None = None) -> str:
    """Make source links relative to the report's destination or checkout root."""
    root = Path(root).resolve()
    output = root / (output_path if output_path is not None else REPORT_PATH)
    base = output.parent if output_path is not None else root

    def link(label, source):
        relative = os.path.relpath(root / source, base)
        return f"[{_cell(label)}]({quote(relative, safe='/._-')})"

    command_parts = ["python", "-m", "experiments.forge", "experiments-by-tier"]
    if len(report["views"]) != report["view_count"]:
        command_parts += ["--view", report["views"][0]["id"]]
    command_parts += ["--output", output.relative_to(root).as_posix() if output.is_relative_to(root) else str(output)]
    command = shlex.join(command_parts)
    lines = [
        "# Forge experiments by tier", "",
        "Current task assignments, grouped by goal view and qualification tier. "
        "Required tasks gate progression; ranking and diagnostic tasks retain their declared roles.", "",
        f"Catalog: **{report['task_count']} tasks**; **{report['assigned_task_count']} assigned** to at least one view; "
        f"**{len(report['unassigned_tasks'])} unassigned**. Showing **{len(report['views'])}/{report['view_count']} views**.", "",
        f"Declared priors across the catalog: **{report['prior_counts']['mog']} MoGParticlePrior**, "
        f"**{report['prior_counts']['particle_cloud']} ParticlePrior** "
        f"(including **{report['nonsampled_prior_count']} nonsampled parameter controls**). "
        "Every experiment defines `execution.prior` explicitly; candidate and API defaults cannot supply it. "
        "`kind: mog` selects `MoGParticlePrior`; `kind: particle_cloud` selects `ParticlePrior`. "
        "Sigma alone does not identify the code path. Ordinary Forge MoG tasks require positive sigma; "
        "archived zero-sigma MoG evidence keeps its recorded kind. "
        "Task sigma is absolute; API demonstrations may instead record the recipe's relative `sigma_rel`.", "",
        "Reproducible comparisons use the fixed screening seed `0` and candidate-independent named RNG streams. "
        "Within each task, candidates share architecture, data law, batch size, prior, initialization, "
        "training budget, evaluation cadence and sampling law. Only the declared trainer change varies. "
        "The initialization column exposes fixed controls and component policies that take precedence over "
        "the deterministic orthogonal fallback; these are separate comparison cohorts. "
        "Historical results retain their original bindings.", "",
        f"Regenerate from the repository root with `{command}`. Add `--json` for machine-readable output "
        "(use a `.json` output path when saving). Regeneration reads declarations and published artifacts and launches no training.", "",
        "Tier 1 is smoke, Tier 2 is quality, and Tier 3 is endurance. "
        "Views may leave later tiers empty. Placement follows each view's policy.", "",
        "Steps and timeouts are declared per task, rather than measured costs. "
        "Tasks in an uninterrupted execution group share one run; their budgets must not be added together. "
        "Continuation rows distinguish total steps from additional or extension steps.", "",
        "| View | Revision | Tier 1 | Tier 2 | Tier 3 | Declared calibration |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    artifacts = report["review_artifacts"]
    review = ["## Review experiments and solutions", "",
              "Use the tier tables to see the current requirements, then open each task's "
              "experiment guide for its question, numerical gates, recorded configurations and related training GIFs.", ""]
    for name, label, purpose in (
        ("solutions", "Current solution leaderboard", "selected configurations and complete Forge tier denominators"),
        ("memory", "Research memory", "hypotheses, comparisons, failure explanations and recommendations"),
        ("questions", "All retained public-API questions", "the wider research catalog, explanations, results and goal GIFs"),
        ("api_readout", "Public-API experiment readout", "completed protocols and failed numerical bounds"),
        ("api_contract", "Public-API test contract", "reproduction commands and exact scope of the demos"),
    ):
        if name in artifacts:
            review.append(f"- {link(label, artifacts[name])}: {purpose}.")
    review += ["", "Recorded Forge outcomes below are lookups by exact task ID from the current solution publication; "
               "their source revision and runtime remain explicit. Related API variants illustrate the question "
               "under their own gates, recipes, priors, initialization, budgets and sampling laws. "
               "Their PASS results do not qualify a different Forge task or the latest checkout. "
               "The solution leaderboard remains the single ranking for its goal.", "",
               "This report follows changing declarations and published evidence; it selects no release winner. "
               "Release selection requires full Forge qualification and review of the winner. "
               "A smoke-qualified entry or historical pass alone does not establish release readiness.", "",
               "## Current tier assignments", ""]
    # Put the review path before the assignment summary, after the introduction.
    index = lines.index("| View | Revision | Tier 1 | Tier 2 | Tier 3 | Declared calibration |")
    lines[index:index] = review

    def count_text(tier):
        return ", ".join(f"{tier['counts'][importance]} {importance}" for importance in IMPORTANCES
                         if tier["counts"][importance]) or "0 tasks"

    for view in report["views"]:
        counts = [count_text(tier) for tier in view["tiers"]]
        lines.append("| " + " | ".join([
            link(view["id"], view["source"]), str(view["revision"]), *counts,
            _cell(view["calibration"].get("status", "undeclared")),
        ]) + " |")

    def task_table(tasks, assigned=True):
        headers = ["Task"] + (["Importance"] if assigned else [])
        headers += ["Prior code path", "Initialization / protocol", "Experiment guide", "Adapter / gate", "Declared steps", "Timeout (s)", "Dependencies / shared execution"]
        table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for task in tasks:
            steps = _cell(task["steps"])
            if task["max_total_steps"] is not None:
                steps = f"up to {task['max_total_steps']} total"
            if task["incremental_steps"] is not None:
                if task["max_total_steps"] is None:
                    steps += " total"
                steps += f"; {task['incremental_steps']} additional"
            if task["extension_steps"] is not None:
                steps += f"; {task['extension_steps']} extension"
            notes = [link(dep["task"], report["task_sources"][dep["task"]])
                     + f" ({_cell(dep['kind'])})" for dep in task["dependencies"]]
            if task["execution_group"]:
                notes.append("group: " + _cell(task["execution_group"])
                             + (" (uninterrupted)" if task["uninterrupted"] else ""))
            cells = [link(task["id"], task["source"])] + ([_cell(task["importance"])] if assigned else [])
            cells += [_prior_cell(task["prior"], task["prior_applicability"]),
                      _cell(task["initializer"]) +
                      ("; fixed: " + _cell(json.dumps(task["fixed_initialization"], sort_keys=True)) if task["fixed_initialization"] else "") +
                      ("; component policy" if task["host_initialization"] else "") +
                      "; " + _cell(task["protocol"]),
                      f"[Question, results, GIFs](#experiment-{task['guide_id'].replace('_', '-')})",
                      _cell(task["adapter"]) + " / " + _cell(task["evaluation_kind"]),
                      _cell(steps), _cell(task["timeout_seconds"]), "; ".join(notes) or "—"]
            table.append("| " + " | ".join(cells) + " |")
        return table

    for view in report["views"]:
        lines += ["", f"## {_cell(view['id'])}", "",
                  f"Declaration: {link(view['id'], view['source'])}; revision {view['revision']}; "
                  f"goal: `{_cell(view['goal'])}`.", "",
                  f"Declared calibration status: **{_cell(view['calibration'].get('status', 'undeclared'))}**.", ""]
        if view["calibration"].get("adoption_blocker"):
            lines += [_cell(view["calibration"]["adoption_blocker"]), ""]
        if view["evidence_scope"] is not None:
            lines += [f"Declared evidence scope: `{_cell(view['evidence_scope'])}`.", ""]
        if view["evidence_scope"] == "calibration_diagnostic":
            lines += ["This view records calibration diagnostics and grants no ordinary qualification.", ""]
        board = leaderboard_path(root, view["id"])
        if (root / board).is_file() and DECLARATION_ONLY_BOARD in (root / board).read_text():
            lines += [f"Declaration only: {link('view pointer', board.as_posix())}; no published candidate outcomes, metrics or measured costs."]
        elif (root / board).is_file():
            lines += [f"Candidate outcomes, metrics and measured costs: {link('leaderboard', board.as_posix())}."]
        else:
            lines += ["No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification."]
        for tier in view["tiers"]:
            lines += ["", f"### Tier {tier['qualification_tier']}: {tier['name']}", "", count_text(tier) + ".", ""]
            lines += task_table(tier["tasks"]) if tier["tasks"] else ["No tasks assigned."]
    lines += ["", "## Tasks unassigned to any view", "",
              "These catalog tasks have no tier placement. Add an assignment to a view to include them in its policy.", ""]
    lines += task_table(report["unassigned_tasks"], assigned=False) if report["unassigned_tasks"] else ["None."]
    lines += ["", "## Experiment guides", "",
              "Task variants share a guide when their declarations name the same host or problem. "
              "This grouping is for navigation; it does not assert matching scientific contracts. "
              "Public-API variants join only through their explicit retained question IDs."]
    for guide in report["experiment_guides"]:
        lines += ["", f"### Experiment: {guide['id'].replace('_', '-')}", "", _cell(guide["goal"]), ""]
        if guide.get("readout_sources"):
            lines += ["Explanation, interpretation and reproduction: " + ", ".join(
                link("experiment readout", source) for source in guide["readout_sources"]) + ".", ""]
        lines += ["Forge declarations: " + ", ".join(link(task["id"], task["source"]) for task in guide["tasks"]) + ".", ""]
        contracts = {}
        for task in guide["tasks"]:
            evaluation, execution = task["evaluation"], task["execution"]
            fields = ("kind", "thresholds", "coverage_thresholds", "accuracy_limits", "minimum_stable_checks",
                      "conditions", "confirmation_checks", "hold_budget", "extension_steps", "recovery_deadline",
                      "stationary_checks", "deadline_checks", "minimum_frozen_passing", "requires_pair_artifacts",
                      "guards", "observations")
            numeric = {key: evaluation[key] for key in fields if key in evaluation}
            sampling = {"prior": execution.get("prior"), "sampling_law": evaluation.get("sampling_law"),
                        "scoring_weights": evaluation.get("scoring_weights"),
                        "eval_output_noise": evaluation.get("eval_output_noise")}
            key = stable_hash({"numeric": numeric, "sampling": sampling})
            contracts.setdefault(key, {"tasks": [], "numeric": numeric, "sampling": sampling})["tasks"].append(task)
        lines += ["Declared Forge numerical gates and sampling:", ""]
        for contract in contracts.values():
            lines += [", ".join(link(task["id"], task["source"]) for task in contract["tasks"]), ""]
            bounds = evaluation_rows(contract["numeric"])
            if bounds:
                lines += ["| Metric | Required bound |", "| --- | --- |"]
                lines += [f"| {_cell(metric)} | {_cell(op)} {number(bound)} |" for metric, op, bound in bounds]
                lines.append("")
            lines += evaluation_notes(contract["numeric"])
            lines.append("")
            sampling = contract["sampling"]
            lines += ["| Measurement | Declared condition |", "| --- | --- |",
                      "| Prior | " + _prior_cell(sampling["prior"]) + " |",
                      "| Sampling law | " + _cell(sampling["sampling_law"]) + " |",
                      "| Scoring weights | " + _cell(sampling["scoring_weights"]) + " |",
                      "| Evaluation output noise | " + _cell(sampling["eval_output_noise"]) + " |", ""]
        if guide["forge_results"]:
            lines += ["Recorded Forge task outcomes (exact saved configuration/source/runtime):", "",
                      "| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for result in guide["forge_results"]:
                config = link(result["label"], result["config_source"]) if result["config_source"] else _cell(result["candidate_id"])
                source = result["source_commit"][:12] if result["source_commit"] else "unbound"
                binding = f"{result['backend']} / {source} / {result['cohort'][:12]}"
                match = {True: "matches; source remains frozen", False: "CHANGED; earlier contract", None: "unbound"}[result["declaration_match"]]
                lines.append("| " + " | ".join([
                    _cell(result["task_id"]), config, _prior_cell(result["prior"], result["prior_applicability"]),
                    _cell(result["status"]), match, _cell(binding),
                    link("source-bound receipt index", result["evidence_source"]),
                ]) + " |")
        else:
            lines += ["No measured Forge outcome for these exact task IDs in the current solution publication. "
                      "Consult the solution leaderboard for unknown requirements and capability blockers."]
        lines.append("")
        if guide["api_variants"]:
            lines += ["Related public-API demonstrations, with their own recorded contracts:", "",
                      "| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for variant in guide["api_variants"]:
                media = link(variant["id"], variant["gif"]) if variant["media_available"] else _cell(variant["id"]) + " (GIF unavailable)"
                outcome = f"{variant['execution_status']} / {variant['verdict']}"
                if variant["completed_updates"] is not None:
                    outcome += f"; {variant['completed_updates']}/{variant['default_updates']} updates"
                if variant["failed_bounds"]:
                    outcome += "; " + ", ".join(variant["failed_bounds"])
                identity = variant["source_identity"] or variant["source_commit"] or "unbound"
                binding = f"{variant['recipe']} / {variant['runtime'].get('device', 'unrecorded')} / {identity[:12]}"
                evidence = [link("definition", variant["definition_source"])]
                for field, label in (("evidence_source", "readout"), ("receipt_source", "recipe and provenance")):
                    if variant[field]:
                        evidence.append(link(label, variant[field]))
                lines.append("| " + " | ".join([
                    media, _cell(variant["goal"] + " Scope: " + variant["scope"]),
                    _prior_cell(variant["prior"]), _cell(outcome), _cell(binding), "; ".join(evidence),
                ]) + " |")
        else:
            lines += ["No related published API training GIF. This task retains its own declared numerical audit."]
    lines += ["", "## Keep this view current", "",
              "After editing task/view declarations or publishing new compact results and media indexes, "
              f"run `{command}` and commit this same report. New tasks and explicit API question mappings "
              "are discovered automatically; no training, rescoring or queue access is needed. "
              "The report freshness test detects stale generated content.", "",
              "The wider question review also links standalone experiments outside the Forge tier catalog. "
              "Adding a diagnostic there does not assign it to a Forge tier.", ""]
    for name in ("questions", "later_questions", "caption_questions"):
        if name in artifacts:
            lines.append(f"- {link(name.replace('_', ' ').capitalize(), artifacts[name])}")
    lines += ["", f"Declaration input digest: `{report['input_digest']}`. "
              "The JSON form includes the individual task and view file hashes.", "",
              f"Published artifact input digest: `{report['artifact_input_digest']}`. "
              "Artifact hashes and exact recipe/source/runtime bindings are included in the JSON form.", ""]
    return "\n".join(lines)


def write_report(report: dict, root: Path, output_path: Path, *, as_json: bool = False) -> Path:
    """Atomically publish a report without touching evidence or the queue."""
    path = (Path(root) / output_path).resolve()
    if as_json:
        if not path.exists() or read_json(path) != report:
            atomic_json(path, report)
    else:
        content = render_markdown(report, root, path)
        if not path.exists() or path.read_text() != content:
            atomic_text(path, content)
    return path

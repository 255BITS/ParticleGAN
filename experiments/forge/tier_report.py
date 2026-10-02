"""Deterministic task inventory from Forge's current qualification view policy."""
from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import shlex
from urllib.parse import quote

from .contracts import atomic_json, atomic_text, file_hash, read_json, stable_hash
from .knowledge import leaderboard_path
from .views import load_tasks, load_view


REPORT_PATH = Path("reports/forge/EXPERIMENTS_BY_TIER.md")
TIER_NAMES = {1: "smoke", 2: "quality", 3: "endurance"}
IMPORTANCES = ("required", "ranking", "diagnostic")


def build_report(root: Path, view_id: str | None = None) -> dict:
    """Read validated declarations, including tasks not assigned to any view."""
    root = Path(root)
    tasks = load_tasks(root)
    for task in tasks.values():
        for dependency in task["dependencies"]:
            if dependency["task"] not in tasks:
                raise ValueError(f"{task['id']}: missing catalog dependency {dependency['task']!r}")
    task_paths = {read_json(path)["id"]: path.relative_to(root).as_posix()
                  for path in sorted((root / "configs/forge/tasks").glob("*.json"))}
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
            "evaluation_kind": task["evaluation"]["kind"],
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
        "unassigned_tasks": [task_row(name) for name in sorted(tasks.keys() - assigned)],
        "task_sources": task_paths,
        "input_hashes": inputs, "input_digest": stable_hash(inputs),
    }


def _cell(value) -> str:
    return "—" if value is None else str(value).replace("|", "\\|").replace("\n", " ")


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
        f"Regenerate from the repository root with `{command}`. Add `--json` for machine-readable output "
        "(use a `.json` output path when saving). Regeneration reads declarations and launches no training.", "",
        "Tier 1 is smoke, Tier 2 is quality, and Tier 3 is endurance. "
        "Views may leave later tiers empty. Placement follows each view's policy.", "",
        "Steps and timeouts are declared per task, rather than measured costs. "
        "Tasks in an uninterrupted execution group share one run; their budgets must not be added together. "
        "Continuation rows distinguish total steps from additional or extension steps.", "",
        "| View | Revision | Tier 1 | Tier 2 | Tier 3 | Declared calibration |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]

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
        headers += ["Adapter / gate", "Declared steps", "Timeout (s)", "Dependencies / shared execution"]
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
            cells += [_cell(task["adapter"]) + " / " + _cell(task["evaluation_kind"]),
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
        lines += [f"Candidate outcomes, metrics and measured costs: {link('leaderboard', leaderboard_path(root, view['id']).as_posix())}."]
        for tier in view["tiers"]:
            lines += ["", f"### Tier {tier['qualification_tier']}: {tier['name']}", "", count_text(tier) + ".", ""]
            lines += task_table(tier["tasks"]) if tier["tasks"] else ["No tasks assigned."]
    lines += ["", "## Tasks unassigned to any view", "",
              "These catalog tasks have no tier placement. Add an assignment to a view to include them in its policy.", ""]
    lines += task_table(report["unassigned_tasks"], assigned=False) if report["unassigned_tasks"] else ["None."]
    lines += ["", f"Declaration input digest: `{report['input_digest']}`. "
              "The JSON form includes the individual task and view file hashes.", ""]
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

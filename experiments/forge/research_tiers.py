"""Display retained experiments in their recorded tiers, without qualification.

Each row keeps one recipe/source/seed. Observation checks never enter task
counts, and untiered historical studies remain on the family detail pages.
"""
from pathlib import Path
from posixpath import relpath

from .contracts import read_json, file_hash

COVERAGE = Path("reports/forge/tier-coverage-20261014/results.json")


def coverage_report(root):
    path = Path(root) / COVERAGE
    if not path.is_file():
        return None
    report = read_json(path)
    if (report.get("schema_version") != 1 or report.get("scope") != "selected_family_diagnostic_coverage"
            or report.get("qualification_input") is not False or report.get("qualification_reuse") is not False):
        raise ValueError("selected family coverage must remain diagnostic-only")
    for relative, digest in report["inputs"].items():
        target = Path(root) / relative
        if not target.resolve().is_relative_to(Path(root).resolve()) or file_hash(target) != digest:
            raise ValueError("diagnostic coverage input changed: " + relative)
    return report


def _link(root, page, label, target):
    return f"[{label}]({relpath(str(Path(root) / target), str(Path(page).parent))})"


def _row(root, page, label, target, tasks, requirements):
    from .family_reports import score
    if set(tasks) - set().union(*map(set, requirements.values())):
        raise ValueError("research result has no recorded tier assignment")
    counts = []
    for tier in ("1", "2", "3"):
        required = requirements[tier]
        if not required:
            counts.append("—")
            continue
        passed = sum(tasks.get(task) == "PASS" for task in required)
        counts.append(_link(root, page, score({"passed": passed, "total": len(required),
            "not_run": sum(tasks.get(task) in {None, "UNKNOWN", "NOT_RUN"} for task in required),
            "incompatible": sum(tasks.get(task) == "BLOCKED" for task in required)}), target))
    required = set().union(*map(set, requirements.values()))
    passed = sum(tasks.get(task) == "PASS" for task in required)
    counts.append(_link(root, page, score({"passed": passed, "total": len(required),
        "not_run": sum(tasks.get(task) in {None, "UNKNOWN", "NOT_RUN"} for task in required),
        "incompatible": sum(tasks.get(task) == "BLOCKED" for task in required)}), target))
    return "| " + " | ".join(["↳ † " + _link(root, page, label, target), *counts]) + " |"


def _confidence_rows(root, page, requirements):
    from .test_performance import Inputs, SUPPLEMENTAL
    from .test_performance_render import _render_confidence_packets
    inputs = Inputs(root)
    if not (Path(root) / SUPPLEMENTAL).is_file():
        return []
    names = {
        "pr399-confidence-lr-tier1-6-20261013": "screen",
        "pr399-confidence-lr-own-retention-2-20261013": "retention",
        "pr399-gaussian-finite-cap-1-20261013": "finite_cap",
        "pr399-critic-orientation-2-20261013": "orientation",
        "pr399-signed-confidence-2-20261013": "signed",
        "pr399-positive-evidence-2-20261013": "positive",
    }
    packets = {}
    for pin in inputs.read(SUPPLEMENTAL)["publications"]:
        document = inputs.read(pin["path"], sha256=pin["sha256"])
        key = names.get(document.get("discovery_summary", {}).get("id"))
        if key:
            if key in packets or any(document.get(flag) is not False for flag in
                                     ("qualification_input", "qualification_reuse")):
                raise ValueError("confidence tier rows require unique nonqualifying packets")
            packets[key] = pin, document
    # Reuse the existing receipt/continuity/count validators before projection.
    _render_confidence_packets(packets, lambda name: name)
    if not packets:
        return []
    rows = []
    for label, keys, anchor in (
        ("Confidence LR; seed 0", ("screen", "retention"), "confidence-lr-correction-six-task-screen-and-own-retention"),
        ("Finite critic cap; seed 0", ("finite_cap",), "later-gaussian-corrections-one-failed-acquisition-one-failed-retention"),
        ("Critic orientation; seed 0", ("orientation",), "later-gaussian-corrections-one-failed-acquisition-one-failed-retention"),
        ("Signed confidence; seed 0", ("signed",), "signed-and-positive-evidence-responses-retention-remains-unsolved"),
        ("Positive evidence; seed 0", ("positive",), "signed-and-positive-evidence-responses-retention-remains-unsolved"),
    ):
        tasks, identities = {}, set()
        for key in keys:
            document = packets[key][1]
            for cell in document["cells"]:
                task = cell["task_id"]
                if task in tasks:
                    raise ValueError("research tier row repeats a task")
                tasks[task] = cell["status"]
                condition = document["condition_contracts"][cell["conditions_sha256"]]
                identities.add((cell["candidate_revision"], condition["source_digest"], condition["seed"]))
        if len(identities) != 1:
            raise ValueError("research tier row cannot pool recipes, sources or seeds")
        rows.append(_row(root, page, label, "reports/forge/families/bcap-pure.md#" + anchor,
                         tasks, requirements))
    return rows


def tier_rows(root, page):
    """Rows keyed by owning family; no modification of selected publication data."""
    rows = {}
    from .source_bound_board import BOARD_SOURCE, validate_board
    from .confidence_coherence_scores import (
        PAIRED, _pinned, validate_original_rows, validate_paired_results,
    )
    if (Path(root) / BOARD_SOURCE).is_file():
        board = validate_board(read_json(Path(root) / BOARD_SOURCE))
        validate_original_rows(root, board)
        requirements = {str(tier): {a["task"] for a in board["recorded_view"]["assignments"]
                                   if a["importance"] == "required" and a["qualification_tier"] == tier}
                        for tier in (1, 2, 3)}
        if ({tier: len(tasks) for tier, tasks in requirements.items()} != {"1": 6, "2": 21, "3": 2}
                or len(set().union(*requirements.values())) != 29):
            raise ValueError("research tiers must retain the recorded revision 9 requirements")
        bcap = rows.setdefault("bcap-pure", [])
        bcap.extend(_confidence_rows(root, page, requirements))
        for record in board["rows"]:
            tasks = {task: cell["status"] for task, cell in record["benchmark_tasks"].items()
                     if cell["status"] is not None}
            bcap.append(_row(root, page, record["label"], record["readout"], tasks, requirements))
        if (Path(root) / PAIRED).is_file():
            paired = _pinned(root, PAIRED)
            validate_paired_results(paired)
            for seed in range(1, 7):
                for arm in ("candidate", "control"):
                    study = f"paired-g-coherence-seed{seed}-{arm}"
                    tasks = {r["task"]: r["gate"]["status"] for r in paired["records"] if r["study"] == study}
                    bcap.append(_row(root, page, f"G-coherence {arm}; research seed {seed}",
                                     "reports/forge/families/bcap-pure.md#g-only-coherence-registered-paired-research",
                                     tasks, requirements))
        from .pr399_scores import tier_rows as pr399_tier_rows
        for family, additions in pr399_tier_rows(root, page, requirements).items():
            rows.setdefault(family, []).extend(additions)
    from .atlas_catalog_scores import ATLAS, _read, atlas_score_lines
    if (Path(root) / ATLAS / "results.json").is_file():
        atlas_score_lines(root)  # Existing byte pins, original verdicts and scope.
        result = _read(root, f"{ATLAS}/results.json")
        tasks = {row["task"]: row["status"] for row in result["rows"]}
        tasks.update({task: "BLOCKED" for task in result["blocked_unstarted"]})
        if len(tasks) != 7 or any("tier1_policy_selected_cloud_v1" not in task for task in tasks):
            raise ValueError("Atlas policy questions must retain their original Tier 1 scope")
        rows["atlas"] = [_row(root, page, "Selected-cloud policy; seed 0 (separate scope)",
                              f"{ATLAS}/README.md", tasks, {"1": set(tasks), "2": set(), "3": set()})]
    coverage = coverage_report(root)
    for record in (coverage or {}).get("diagnostic_rows", []):
        requirements = {tier: set(names) for tier, names in record["requirements"].items()}
        rows.setdefault(record["family"], []).append(_row(root, page, record["label"],
            "reports/forge/tier-coverage-20261014/README.md", record["tasks"], requirements))
    return rows


def family_research_lines(root, page, family_id):
    """Keep detailed scores and untiered historical diagnostics off the scoreboard."""
    if family_id == "atlas":
        from .atlas_catalog_scores import atlas_score_lines
        from .pr399_scores import family_lines
        return atlas_score_lines(root) + family_lines(root, page, family_id)
    if family_id != "bcap-pure":
        return []
    from .atlas_catalog_scores import catalog_score_lines
    from .source_bound_board import main_board_rows
    from .test_performance_render import render_discovery_navigation
    lines = catalog_score_lines(root, page) + main_board_rows(root, page)
    lines.append(render_discovery_navigation(root, page))
    from .pr399_scores import family_lines
    lines += family_lines(root, page, family_id)
    text = "\n".join(lines)
    for heading, anchor in (
        ("G-only coherence: registered paired research", "g-only-coherence-registered-paired-research"),
        ("Confidence-LR correction: six-task screen and own retention", "confidence-lr-correction-six-task-screen-and-own-retention"),
        ("Later Gaussian corrections: one failed acquisition, one failed retention", "later-gaussian-corrections-one-failed-acquisition-one-failed-retention"),
        ("Signed and positive-evidence responses: retention remains unsolved", "signed-and-positive-evidence-responses-retention-remains-unsolved"),
    ):
        text = text.replace("## " + heading, f'<a name="{anchor}"></a>\n\n## ' + heading, 1)
    return [text]

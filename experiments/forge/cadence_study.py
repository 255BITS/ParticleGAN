"""Matched lazy-regularization studies using Forge's existing public recipe grid.

This layer owns the comparison, not a trainer. Ordinary qualification and
explicitly nonqualifying cross-tier coverage use separate evidence namespaces.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from . import configuration_search as search
from .contracts import atomic_json, identifier, positive_number, read_json, stable_hash
from .queue import Queue, drain
from .views import load_view, view_fingerprint


FIELDS = frozenset({"schema_version", "id", "base_candidate", "trainer_family", "view",
                    "through_tier", "execution_backend", "cuda_model", "protocol", "intervals",
                    "control_interval", "hypothesis", "campaign", "coverage_view", "lanes"})
REQUIRED = FIELDS - {"cuda_model", "coverage_view", "lanes"}


def load_spec(root, spec, *, validate_current=True):
    """Validate a bounded declaration; planning never writes or starts training."""
    if isinstance(spec, dict):
        value = deepcopy(spec)
    else:
        path = Path(spec)
        if path.suffix != ".json" and len(path.parts) == 1:
            path = Path("configs/forge/cadence-studies") / f"{path}.json"
        value = read_json(Path(root) / path)
    if not isinstance(value, dict) or set(value) - FIELDS or REQUIRED - value.keys():
        raise ValueError("cadence study has missing or unsupported fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("cadence study requires schema_version 1")
    for key in ("id", "base_candidate", "trainer_family", "view", "protocol"):
        identifier(value[key], f"cadence {key}")
    if type(value["through_tier"]) is not int or value["through_tier"] < 1:
        raise ValueError("through_tier must be an integer >= 1")
    if value["execution_backend"] not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be cpu or cuda")
    intervals = value["intervals"]
    if (not isinstance(intervals, list) or not 2 <= len(intervals) <= 16
            or any(type(k) is not int or k < 1 for k in intervals)
            or len(set(intervals)) != len(intervals)):
        raise ValueError("intervals must contain 2 to 16 distinct positive integers")
    if type(value["control_interval"]) is not int or value["control_interval"] != 1 or 1 not in intervals:
        raise ValueError("cadence studies require the eager control_interval 1")
    if not isinstance(value["hypothesis"], str) or not value["hypothesis"].strip() or "TODO" in value["hypothesis"]:
        raise ValueError("cadence study requires a finished hypothesis")
    lanes = value.get("lanes", ["ordinary"])
    if (not isinstance(lanes, list) or not lanes or len(set(lanes)) != len(lanes)
            or any(lane not in {"ordinary", "coverage"} for lane in lanes)):
        raise ValueError("lanes must explicitly select ordinary and/or coverage")
    if "coverage" in lanes:
        identifier(value.get("coverage_view"), "coverage_view")
    campaign = value["campaign"]
    if not isinstance(campaign, dict) or set(campaign) != {"id", "budget_seconds", "candidate_budget_seconds"}:
        raise ValueError("campaign requires id and explicit campaign/candidate budgets")
    identifier(campaign["id"], "cadence campaign")
    for key in ("budget_seconds", "candidate_budget_seconds"):
        positive_number(campaign[key], f"cadence {key}")
    if campaign["candidate_budget_seconds"] > campaign["budget_seconds"]:
        raise ValueError("candidate budget exceeds campaign budget")
    if validate_current:
        protocol = read_json(Path(root) / f"configs/forge/protocols/{value['protocol']}.json")
        if type(protocol.get("seed")) is not int or protocol["seed"] != 0:
            raise ValueError("cadence comparisons require protocol seed 0; no seed experiments")
        source_view = load_view(root, value["view"])
        if source_view.get("evidence_scope") in {"research_diagnostic", "calibration_diagnostic"}:
            raise ValueError("view must name the original tiered ordinary task view")
        if "coverage" in lanes:
            coverage = load_view(root, value["coverage_view"])
            original = {a["task"] for a in source_view["assignments"]
                        if a["qualification_tier"] <= value["through_tier"]}
            if (coverage.get("evidence_scope") != "research_diagnostic"
                    or any(a["importance"] != "diagnostic" or a["qualification_tier"] != 1
                           for a in coverage["assignments"])
                    or {a["task"] for a in coverage["assignments"]} != original):
                raise ValueError("coverage_view must declare the same scoped tasks as tier-1 research diagnostics")
    return value


def search_spec(root, spec, lane="ordinary"):
    """Compile only reg_every; base strength and all other recipe fields persist."""
    value = load_spec(root, spec)
    if lane not in value.get("lanes", ["ordinary"]):
        raise ValueError("lane is not declared by this cadence study")
    return {"schema_version": 2, "id": f"{value['id']}-{lane}",
            "trainer_family": value["trainer_family"], "base_candidate": value["base_candidate"],
            "grid": {"reg_every": value["intervals"]},
            "view": value["coverage_view"] if lane == "coverage" else value["view"],
            "tuning_through_tier": 1 if lane == "coverage" else value["through_tier"],
            "execution_backend": value["execution_backend"], "protocol": value["protocol"],
            "hypothesis": value["hypothesis"], "campaign": deepcopy(value["campaign"]),
            **({"cuda_model": value["cuda_model"]} if value.get("cuda_model") else {})}


def _registration_path(root, study_id):
    return Path(root) / "reports/forge/cadence-studies" / f"{study_id}-registration.json"


def _context(root, spec):
    view = load_view(root, spec["view"])
    return {"schema_version": 1, "study_id": spec["id"], "spec": deepcopy(spec),
            "spec_hash": stable_hash(spec), "source_view_fingerprint": view_fingerprint(view),
            "source_assignments": [deepcopy(a) for a in view["assignments"]
                                   if a["qualification_tier"] <= spec["through_tier"]]}


def _registered(root, spec):
    path = _registration_path(root, spec["id"])
    if not path.exists():
        return None
    saved = read_json(path)
    if saved.get("input_digest") != stable_hash({k: v for k, v in saved.items() if k != "input_digest"}):
        raise ValueError("cadence registration digest mismatch")
    if saved.get("spec_hash") != stable_hash(spec):
        raise ValueError("cadence declaration is immutable after enqueue; use a new id")
    return saved


def plan_study(root, queue_root, spec, *, queue=None):
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    value = load_spec(root, spec)
    context = _context(root, value)
    registered = _registered(root, value)
    if registered and registered["source_view_fingerprint"] != context["source_view_fingerprint"]:
        raise ValueError("cadence source view changed; use a new study id")
    queue = queue or Queue(queue_root)
    lanes = {lane: search.plan_search(root, queue_root, search_spec(root, value, lane), queue=queue)
             for lane in value.get("lanes", ["ordinary"])}
    ceilings = {}
    for summary in lanes.values():
        for trial in summary["trials"]:
            k = trial["settings"]["reg_every"]
            ceilings[k] = ceilings.get(k, 0) + trial["declared_worst_case_seconds"]
    total = sum(ceilings.values())
    if total > value["campaign"]["budget_seconds"] or max(ceilings.values()) > value["campaign"]["candidate_budget_seconds"]:
        raise ValueError("cadence campaign/candidate budget cannot cover every declared lane")
    return {**context, "stage": "planned", "lanes": lanes,
            "declared_worst_case_seconds": total, "per_interval_worst_case_seconds": ceilings,
            "training_launched": False, "logs": str(queue_root / "events.jsonl"),
            "interpretation": "Public lazy policy including application-clock controllers; internal k-fold strength compensation. "
                              "Base coefficient, optimizer updates and task protocols remain fixed.",
            "qualification_note": "Coverage results are research diagnostics; source tiers are display context only."}


def enqueue_study(root, queue_root, spec, *, queue=None):
    """Validate all lanes, freeze their requests, then admit without training."""
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    queue = queue or Queue(queue_root, report_root=root / "reports/forge")
    planned = plan_study(root, queue_root, spec, queue=queue)
    value = planned["spec"]
    frozen = {lane: search._freeze_search(root, queue_root, summary["spec"], queue=queue)
              for lane, summary in planned["lanes"].items()}
    registration = {**_context(root, value), "queue_root": str(queue_root),
                    "lane_specs": {lane: summary["spec"] for lane, summary in planned["lanes"].items()}}
    registration["input_digest"] = stable_hash(registration)
    old = _registered(root, value)
    if old and old != registration:
        raise ValueError("cadence registration bindings changed; use a new id")
    atomic_json(_registration_path(root, value["id"]), registration)
    for requests, summary in frozen.values():
        search._admit_search(root, queue, requests, summary)
    return report_study(root, queue_root, value, queue=queue)


def comparison_report(planned):
    """Produce the same explicit unavailable cells for a pre-run preview."""
    from .cadence_report import build_report
    return {"schema_version": 1, "artifact_kind": "forge_cadence_comparison_v1",
            "study_id": planned["study_id"], "stage": planned["stage"],
            "hypothesis": planned["spec"]["hypothesis"], "logs": planned["logs"],
            "declared_worst_case_seconds": planned.get("declared_worst_case_seconds"),
            "interpretation": planned.get("interpretation"),
            "cost_note": "New-paid cost is campaign-wide and shared across lane reports; never add it across lanes. "
                         "Observed evidence cost retains its original attempt scope. Neither cost is a speed ranking.",
            "lanes": {lane: build_report(summary, control_value=planned["spec"]["control_interval"],
                       source_assignments=planned["source_assignments"] if lane == "coverage" else None)
                      for lane, summary in planned["lanes"].items()},
            "training_launched": False, "default_adoption": False}


def report_study(root, queue_root, spec, *, queue=None):
    """Read frozen original bindings, never rebind an old study to current source."""
    value = load_spec(root, spec, validate_current=False)
    registration = _registered(root, value)
    if registration is None:
        raise ValueError("cadence study has not been enqueued; use cadence plan")
    if Path(registration["queue_root"]).resolve() != Path(queue_root).resolve():
        raise ValueError("report queue differs from the frozen cadence study queue")
    summaries = {lane: search.report_search(root, queue_root, lane_spec, queue=queue)
                 for lane, lane_spec in registration["lane_specs"].items()}
    return comparison_report({**registration, "stage": "reported", "lanes": summaries,
                              "logs": str(Path(queue_root) / "events.jsonl"),
                              "declared_worst_case_seconds": sum(s["declared_worst_case_seconds"] for s in summaries.values()),
                              "interpretation": "Public lazy policy; coefficient scaling and application-clock controller effects are coupled."})


def run_study(root, queue_root, spec, *, devices=None, queue=None):
    value = load_spec(root, spec)
    devices = list(devices or (["cpu"] if value["execution_backend"] == "cpu" else ["0", "1"]))
    if (value["execution_backend"] == "cpu") != (devices == ["cpu"]):
        raise ValueError("cadence devices must agree with execution_backend")
    queue = queue or Queue(queue_root, report_root=Path(root) / "reports/forge")
    enqueue_study(root, queue_root, value, queue=queue)
    drain(queue, devices, campaign=value["campaign"]["id"])
    result = report_study(root, queue_root, value, queue=queue)
    result["training_launched"] = True
    return result


def write_report(report, output):
    """Create an export pair, or atomically refresh this study's own export."""
    from .cadence_report import render_markdown
    from .contracts import atomic_text
    output = Path(output)
    if output.suffix != ".md":
        raise ValueError("cadence report output must end in .md")
    study_id = identifier(report.get("study_id"), "cadence report study_id")
    kind = "forge_cadence_comparison_v1"
    if report.get("artifact_kind") != kind:
        raise ValueError("cadence report requires its generated comparison artifact identity")
    try:
        # Resolve both lexical destinations independently: either the Markdown
        # file, its companion or a parent directory may itself be a symlink.
        companion = output.with_suffix(".json").resolve()
        output = output.resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError("invalid cadence report destination") from error
    if output.suffix != ".md" or companion.suffix != ".json":
        raise ValueError("resolved cadence report destinations must be .md and .json")
    if output.exists() or companion.exists():
        try:
            existing = read_json(companion) if companion.is_file() else None
            owned = (output.is_file() and isinstance(existing, dict)
                     and existing.get("artifact_kind") == kind
                     and existing.get("study_id") == study_id
                     and output.read_text(encoding="utf-8").startswith(f"# {study_id}\n"))
        except (OSError, ValueError, UnicodeError):
            owned = False
        if not owned:
            raise ValueError("refusing to overwrite files outside this study's generated report pair")
    # Render and validate strict JSON before creating directories or modifying
    # either file, including when an existing owned pair is being refreshed.
    stable_hash(report)
    sections = [f"# {study_id}\n\n{report['hypothesis']}\n",
                f"Stage: {report['stage']}. No default adoption.\n",
                f"Reserved ceiling: {report['declared_worst_case_seconds']} seconds.\n",
                f"{report.get('interpretation') or ''}\n"]
    if report.get("cost_note"):
        sections.append(f"{report['cost_note']}\n")
    for lane, details in report["lanes"].items():
        sections.append(f"## {lane}\n\n" + render_markdown(details))
    atomic_text(output, "\n".join(sections))
    atomic_json(companion, report)
    return {"markdown": str(output), "json": str(companion)}

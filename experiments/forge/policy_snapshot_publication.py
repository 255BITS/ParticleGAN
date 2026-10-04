"""Read-only declaration checks for published policy-cohort display rows.

This validates compact metadata against live or explicitly pinned source bytes.
It never hydrates checkpoints, reconstructs scientific results, imports frozen
Python, samples a model, or grants qualification to historical rows.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from .contracts import stable_hash
from . import policy_contracts as policy


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_PATH = re.compile(r"[A-Za-z0-9_./-]+\Z")
_POLICY_FIELDS = {"task_cohort", "qualification_view", "task_slot_map"}
_BINDING_FIELDS = _POLICY_FIELDS | {
    "qualification_view_revision", "qualification_view_sha256", "qualification_policy_fingerprint"}
_STATUSES = {"PASS", "FAIL", "INCOMPLETE", "INVALID", "BLOCKED", "UNKNOWN"}


def _same(left, right):
    # Canonical JSON preserves false/0 and integer/float distinctions.
    return stable_hash(left) == stable_hash(right)


def _relative(path):
    if (not isinstance(path, str) or not _PATH.fullmatch(path)
            or path.startswith("-") or PurePosixPath(path).is_absolute()
            or any(part in {".", ".."} for part in path.split("/"))
            or str(PurePosixPath(path)) != path):
        raise ValueError(f"policy publication unsafe source path: {path!r}")
    return path


@lru_cache(maxsize=4096)
def _git_bytes(root, commit, relative):
    """Cache only immutable, fully identified Git objects, never live files."""
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError("policy publication needs a full pinned source commit")
    relative = _relative(relative)
    try:
        result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=root,
                                capture_output=True, check=False)
    except OSError as exc:
        raise ValueError("policy publication pinned source is unavailable") from exc
    if result.returncode:
        raise ValueError(f"policy publication pinned source is unavailable: {commit}:{relative}")
    return result.stdout


class _Sources:
    def __init__(self, root, report):
        self.root = Path(root).resolve()
        self.commit = None
        if "frozen_source" in report:
            frozen = report["frozen_source"]
            if (not isinstance(frozen, dict) or not isinstance(frozen.get("commit"), str)
                    or not _COMMIT.fullmatch(frozen["commit"])):
                raise ValueError("policy publication needs its exact frozen source commit")
            self.commit = frozen["commit"]
        elif report.get("publication_scope") == "frozen_source":
            raise ValueError("policy publication frozen source commit is missing")
        self.cache = {}

    def read(self, relative):
        relative = _relative(relative)
        if relative not in self.cache:
            if self.commit is not None:
                data = _git_bytes(str(self.root), self.commit, relative)
            else:
                path = self.root / relative
                if not path.resolve().is_relative_to(self.root) or not path.is_file():
                    raise ValueError(f"policy publication live source is unavailable: {relative}")
                data = path.read_bytes()
            self.cache[relative] = data
        return self.cache[relative]

    def json(self, relative):
        def nonfinite(value):
            raise ValueError(f"policy publication nonfinite source JSON: {value}")
        try:
            result = json.loads(self.read(relative), parse_constant=nonfinite)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"policy publication invalid source JSON: {relative}") from exc
        if not isinstance(result, dict):
            raise ValueError(f"policy publication source JSON must be an object: {relative}")
        return result

    def verify(self, relative, expected):
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise ValueError(f"policy publication invalid source SHA256: {relative}")
        if hashlib.sha256(self.read(relative)).hexdigest() != expected:
            raise ValueError(f"policy publication source binding drift: {relative}")


def _policy_claim(report, row):
    if _POLICY_FIELDS & row.keys():
        return True
    bindings = row.get("bindings", {})
    if isinstance(bindings, dict):
        if _BINDING_FIELDS & bindings.keys():
            return True
        catalog = report.get("task_contracts", {})
        refs = bindings.get("task_contracts", {})
        if isinstance(catalog, dict) and isinstance(refs, dict):
            for ref in refs.values():
                contract = catalog.get(ref) if isinstance(ref, str) else None
                if isinstance(contract, dict) and {"task_cohort", "policy_parent"} & contract.keys():
                    return True
    tasks = row.get("tasks", [])
    from .policy_cohorts import KNOWN_COHORTS
    return isinstance(tasks, list) and any(
        isinstance(task, dict) and isinstance(task.get("task_id"), str)
        and any(task["task_id"].endswith("_" + cohort) for cohort in KNOWN_COHORTS)
        for task in tasks)


def _execution_hash(task):
    fields = {key: task.get(key) for key in (
        "schema_version", "adapter", "execution", "requires_capabilities", "dependencies")}
    if task.get("task_cohort") is not None:
        fields.update(task_cohort=task["task_cohort"], policy_parent=task.get("policy_parent"))
    return stable_hash(fields)


def _expected_contract(task):
    execution, evaluation = task["execution"], task["evaluation"]
    host = execution.get("host_definition", {})
    return {
        "adapter": task.get("adapter"), "execution_sha256": _execution_hash(task),
        "evaluation_sha256": stable_hash(evaluation),
        "host": execution.get("host", execution.get("problem")),
        "host_definition_sha256": stable_hash(host),
        "host_profiles": {key: value for key, value in execution.items() if key.endswith("_profile")},
        "prior": execution.get("prior"),
        "initialization": execution.get("fixed_initialization", host.get("initialization")),
        "steps": execution.get("steps"), "timeout_seconds": task.get("resources", {}).get("timeout_seconds"),
        "sampling": {key: evaluation.get(key) for key in (
            "sampling_contract_version", "sampling_law", "eval_output_noise", "scoring_weights")},
        "task_cohort": task.get("task_cohort"), "policy_parent": task.get("policy_parent"),
        "policy_recipe_overrides": execution.get("policy_recipe_overrides", {}),
        "policy_recipe_overrides_provenance": execution.get("policy_recipe_overrides_provenance"),
    }


def _verify_sources(reader, parent, variant):
    sources = {}
    manifests = (parent["evaluation"].get("sources", {}), variant["evaluation"].get("sources", {}),
                 variant["execution"]["policy_contract"]["sources"],
                 variant["execution"].get("policy_resource_sources", {}))
    for manifest in manifests:
        if not isinstance(manifest, dict):
            raise ValueError("policy publication source manifest must be an object")
        for relative, expected in manifest.items():
            if relative in sources and sources[relative] != expected:
                raise ValueError("policy publication conflicting evaluator/implementation source bindings")
            sources[relative] = expected
    provenance = variant["execution"]["policy_recipe_overrides_provenance"]
    if provenance is not None and "recipe_group" in provenance:
        relative = provenance["source"]
        expected = provenance["source_sha256"]
        if relative in sources and sources[relative] != expected:
            raise ValueError("policy publication conflicting C6 source binding")
        sources[relative] = expected
        reader.verify(relative, expected)
        selection = reader.json(relative)
        group = selection.get("resolved_recipe_groups", {}).get(provenance["recipe_group"])
        if (not isinstance(group, dict) or stable_hash(group.get("recipe")) != group.get("sha256")
                or group.get("sha256") != provenance["reference_recipe_sha256"]
                or not _same({name: group["recipe"].get(name) for name in policy.OVERRIDE_FIELDS},
                             variant["execution"]["policy_recipe_overrides"])):
            raise ValueError("policy publication C6 overrides differ from their source-pinned Recipe")
    elif provenance is not None and "source" in provenance:
        if provenance["source"] not in sources:
            raise ValueError("policy publication host provenance lacks its bound implementation source")
    for relative, expected in sources.items():
        reader.verify(relative, expected)


def _validate(root, report, row):
    if row.get("task_cohort") != policy.COHORT:
        return _validate_named(root, report, row)
    reader = _Sources(root, report)
    common = reader.json(f"configs/forge/views/{policy.PARENT_VIEW_ID}.json")
    parents, variants = {}, {}
    for parent_id in policy.PARENT_TASK_IDS:
        name = parent_id + policy.SUFFIX
        parent_path = f"configs/forge/tasks/{parent_id}.json"
        variant_path = f"configs/forge/task-variants/{policy.COHORT}/{name}.json"
        parent, variant = reader.json(parent_path), reader.json(variant_path)
        pin = policy._parent_record(parent, hashlib.sha256(reader.read(parent_path)).hexdigest())
        if (parent.get("id") != parent_id or variant.get("id") != name
                or not _same(variant.get("policy_parent"), pin)):
            raise ValueError(f"policy publication parent byte/fingerprint binding drift: {name}")
        policy._validate_variant(variant, parent)
        _verify_sources(reader, parent, variant)
        parents[parent_id], variants[name] = parent, variant
    expected_view, _ = policy.resolve_policy_view(common, {**parents, **variants}, {"task_cohort": policy.COHORT})
    actual = row.get("qualification_view")
    slots = {name + policy.SUFFIX: name for name in policy.PARENT_TASK_IDS}
    return _validate_projection(reader, report, row, common, expected_view, slots, variants, policy.COHORT)


def _validate_projection(reader, report, row, common, expected_view, slots, variants, cohort):
    actual = row.get("qualification_view")
    if (row.get("task_cohort") != cohort or not _same(actual, expected_view)
            or not _same(row.get("task_slot_map"), slots)):
        raise ValueError("policy publication actual view/parent-slot mapping changed")
    if (report.get("view") != common["id"] or report.get("view_revision") != common["revision"]
            or report.get("policy_fingerprint") != stable_hash(common)
            or report.get("provenance", {}).get("view_sha256") != stable_hash(common)):
        raise ValueError("policy publication common parent view fingerprint changed")
    required = {str(tier): [entry["task"] for entry in common["assignments"]
                           if entry["qualification_tier"] == tier] for tier in (1, 2, 3)}
    if not _same(report.get("tier_requirements"), required):
        raise ValueError("policy publication full 5/19/2 task requirements changed")
    tasks = row.get("tasks")
    names = [entry["task"] for entry in expected_view["assignments"]]
    if (not isinstance(tasks, list) or any(not isinstance(entry, dict) for entry in tasks)
            or [entry.get("task_id") for entry in tasks] != names
            or row.get("nonrequired_tasks") != []):
        raise ValueError("policy publication must retain all actual scientific task IDs in order")
    for entry in tasks:
        status = entry.get("status")
        if (status not in _STATUSES or entry.get("gate_status") not in _STATUSES | {"NOT_RUN", None}
                or status != ("UNKNOWN" if entry.get("gate_status") in {"NOT_RUN", None} else entry["gate_status"])):
            raise ValueError("policy publication display status differs from its recorded gate")
    cells = {}
    for tier in (1, 2, 3):
        values = [entry["status"] for entry, assignment in zip(tasks, expected_view["assignments"])
                  if assignment["qualification_tier"] == tier]
        counts = dict(Counter(values))
        cells[str(tier)] = {"passed": counts.get("PASS", 0), "total": len(values), "counts": counts}
    if not _same(row.get("tiers"), cells):
        raise ValueError("policy publication tier counts differ from the full task denominator")
    maximum = 0
    for tier in (1, 2, 3):
        if cells[str(tier)]["passed"] != cells[str(tier)]["total"]:
            break
        maximum = tier
    if type(row.get("qualified_tier")) is not int or not 0 <= row["qualified_tier"] <= maximum:
        raise ValueError("policy publication attained tier exceeds its recorded task gates")
    bindings = row.get("bindings")
    if (not isinstance(bindings, dict) or bindings.get("available") is not True
            or bindings.get("task_cohort") != cohort
            or bindings.get("qualification_view_revision") != expected_view["revision"]
            or bindings.get("qualification_view_sha256") != stable_hash(expected_view)
            or bindings.get("qualification_policy_fingerprint") != stable_hash(expected_view)
            or not _same(bindings.get("task_slot_map"), slots)
            or not isinstance(bindings.get("task_contracts"), dict)
            or set(bindings["task_contracts"]) != set(names)):
        raise ValueError("policy publication missing or changed actual task/view bindings")
    catalog = report.get("task_contracts")
    if not isinstance(catalog, dict):
        raise ValueError("policy publication task contract catalog is missing")
    for name in names:
        ref = bindings["task_contracts"][name]
        contract = catalog.get(ref) if isinstance(ref, str) else None
        if (not isinstance(ref, str) or not _SHA256.fullmatch(ref) or not isinstance(contract, dict)
                or stable_hash(contract) != ref or not _same(contract, _expected_contract(variants[name]))):
            raise ValueError(f"policy publication compact task contract differs from pinned task: {name}")


def _validate_named(root, report, row):
    """Check frozen JSON/source bytes without importing a frozen host or scorer."""
    from .named_policy_planning import NAMED_PARENTS, project_view
    cohort = row.get("task_cohort")
    families = {
        "conditional_policy_selected_cloud_v1": "atlas_conditional",
        "routed_policy_selected_cloud_v1": "atlas_routed",
        "multibank_policy_v1": "atlas_multibank",
        "ae_routed_policy_v1": "atlas_ae_routed",
        "word_joint_policy_min11_v1": "atlas_word_joint_min11",
    }
    if not isinstance(cohort, str) or cohort not in families:
        raise ValueError("policy publication unknown explicit named cohort")
    if row.get("bindings", {}).get("trainer_family") != families[cohort]:
        raise ValueError("named policy publication trainer family differs from its explicit cohort")
    reader = _Sources(root, report)
    common = reader.json(f"configs/forge/views/{policy.PARENT_VIEW_ID}.json")
    if ([a.get("task") for a in common.get("assignments", [])] != list(policy.PARENT_TASK_IDS)
            or type(common.get("revision")) is not int or common["revision"] != policy.PARENT_REVISION
            or any(a.get("importance") != "required"
                   or a.get("qualification_tier") != (1 if i < 5 else 2 if i < 24 else 3)
                   or a.get("order") != i - (0 if i < 5 else 5 if i < 24 else 24)
                   for i, a in enumerate(common["assignments"]))):
        raise ValueError("named publication changed the original full 5/19/2 view")
    selected, slots = {}, {}
    for parent_id in policy.PARENT_TASK_IDS:
        parent_path = f"configs/forge/tasks/{parent_id}.json"
        parent = reader.json(parent_path)
        if parent.get("id") != parent_id:
            raise ValueError("named publication original parent identity changed")
        if parent_id not in NAMED_PARENTS[cohort]:
            for relative, sha in parent["evaluation"].get("sources", {}).items():
                reader.verify(relative, sha)
            selected[parent_id], slots[parent_id] = parent, parent_id
            continue
        name = parent_id + "_" + cohort
        variant = reader.json(f"configs/forge/task-variants/{cohort}/{name}.json")
        pin = policy._parent_record(parent, hashlib.sha256(reader.read(parent_path)).hexdigest())
        contract = variant.get("execution", {}).get("policy_contract", {})
        if (variant.get("id") != name or variant.get("task_cohort") != cohort
                or variant.get("policy_family") != families[cohort]
                or not _same(variant.get("policy_parent"), pin)
                or contract.get("cohort") != cohort or contract.get("family") != families[cohort]
                or contract.get("owner") != "particlegan.UpdatePolicy"
                or contract.get("lifecycle") != "ordered_public_update"
                or contract.get("execution_path") != "public_components"
                or contract.get("execution_device") != "cuda"
                or contract.get("cpu_controls_scope") != "structural_only"
                or contract.get("numerical_equivalence_to_parent") is not False):
            raise ValueError("named publication lacks exact parent/family/owner/source scope")
        # These current, whitelisted contract modules perform only metadata
        # transforms. Frozen Python is never imported; the original parent and
        # implementation bytes are read through _Sources at the exact pin.
        from .policy_cohorts import module_for_task
        module = module_for_task(variant)
        sources = contract.get("sources")
        from .policy_declaration_sources import DECLARATION_SOURCES
        required = DECLARATION_SOURCES | (policy.REQUIRED_POLICY_SOURCES - {"experiments/forge/policy_contracts.py"})
        for field in ("SOURCES", "REQUIRED_SOURCES", "COMMON_SOURCES"):
            if hasattr(module, field):
                required = required | set(getattr(module, field))
        if cohort == "conditional_policy_selected_cloud_v1":
            required = required | {module.HOSTS[parent_id]["source"]}
        if not isinstance(sources, dict) or not required <= sources.keys():
            raise ValueError("named publication is missing public implementation source pins")
        expected = module._variant(parent, pin, sources)
        if not _same(variant, expected):
            raise ValueError("named publication changes a frozen parent field or its explicit policy transformation")
        _verify_sources(reader, parent, variant)
        selected[name], slots[name] = variant, parent_id
    expected_view = project_view(common, selected, cohort, families[cohort])
    return _validate_projection(reader, report, row, common, expected_view, slots, selected, cohort)


def validate_policy_publication(root, report, row) -> None:
    """Check a reduced policy row against declarations; ordinary rows are a no-op.

    Frozen rows require the exact ``report.frozen_source.commit``. A missing Git
    object is a publication blocker, with no live/current-source substitution.
    Grade reconstruction and original raw receipts remain the publisher's job.
    """
    if not _policy_claim(report, row):
        return
    try:
        _validate(root, report, row)
    except (KeyError, TypeError, AttributeError, OSError) as exc:
        raise ValueError(f"policy publication malformed or unavailable declaration: {exc}") from exc

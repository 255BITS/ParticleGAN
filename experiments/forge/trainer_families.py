"""Explicit trainer families and whole-configuration publication selection.

Family names never establish scientific compatibility. Selecting a row copies
its complete evidence; alternatives cannot contribute individual passing cells.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .contracts import identifier, read_json, stable_hash

REGISTRY = Path("configs/forge/trainer-families.json")


def load_families(root: Path | str) -> dict:
    path = Path(root) / REGISTRY
    if not path.is_file():
        return {}
    registry = read_json(path)
    if registry.get("schema_version") != 1 or not isinstance(registry.get("families"), list):
        raise ValueError("unsupported trainer family registry")
    result, candidates = {}, set()
    for family in registry["families"]:
        if not isinstance(family, dict):
            raise ValueError("trainer family must be an object")
        name = identifier(family.get("id"), "trainer family")
        if name in result or not isinstance(family.get("label"), str) or not family["label"].strip():
            raise ValueError("trainer families need unique ids and nonempty labels")
        members = family.get("candidates")
        if not isinstance(members, list) or not members or len(set(members)) != len(members):
            raise ValueError("trainer family needs distinct candidate ids")
        for member in members:
            identifier(member, "candidate")
            if member in candidates:
                raise ValueError("candidate belongs to more than one trainer family")
            candidates.add(member)
        if family.get("canonical_candidate") not in members:
            raise ValueError("trainer family canonical candidate must be a family member")
        for backend, study in family.get("active_search_by_backend", {}).items():
            if backend not in {"cpu", "cuda"}:
                raise ValueError("family active search backend must be cpu or cuda")
            identifier(study, "family active search study")
        result[name] = deepcopy(family)
    return result


def family_for_candidate(root: Path | str, candidate_id: str, declaration: dict | None = None) -> dict:
    families = load_families(root)
    for family in families.values():
        if candidate_id in family["candidates"]:
            if declaration and declaration.get("trainer_family", family["id"]) != family["id"]:
                raise ValueError("candidate trainer_family contradicts its explicit registry")
            return family
    explicit = (declaration or {}).get("trainer_family")
    if explicit is not None:
        if explicit not in families:
            raise ValueError(f"unregistered trainer family {explicit}")
        return families[explicit]
    # A new independent mechanism remains visible before a family is declared.
    return {"id": candidate_id, "label": candidate_id, "canonical_candidate": candidate_id,
            "candidates": [candidate_id], "registration": "unclassified_independent_technique"}


def comparison_cohort(row: dict, catalogs: dict, *, task_ids=None) -> str:
    """Identity held fixed when comparing configurations of a trainer family."""
    bindings = row.get("bindings", {})
    contracts = {}
    selected_tasks = bindings.get("task_contracts", {})
    names = sorted(task_ids if task_ids is not None else selected_tasks)
    for name in names:
        digest = selected_tasks.get(name)
        if digest not in catalogs.get("task_contracts", {}):
            # Unknown scientific identities are not comparable by coincidence.
            return stable_hash({"unbound_row": row.get("candidate_id"), "cohort": row.get("cohort"), "task": name})
        contract = deepcopy(catalogs["task_contracts"][digest])
        # The adaptation receipt repeats the varying resolved recipe. Host,
        # prior, initialization, steps and sampling remain bound independently.
        contract.pop("host_adaptation", None)
        contracts[name] = contract
    if not bindings.get("source_digest"):
        return stable_hash({"unbound_row": row.get("candidate_id"), "cohort": row.get("cohort")})
    return stable_hash({"source_digest": bindings["source_digest"], "runtime": row.get("runtime_cohort"),
                        "prior": bindings.get("prior"), "initializer": bindings.get("initializer"),
                        "protocol_sha256": bindings.get("protocol_sha256"), "rng_sha256": bindings.get("rng_sha256"),
                        "claim_contract": bindings.get("claim_contract"), "task_contracts": contracts})


def _declarations(root):
    from .planning import declaration_paths
    # Shared discovery includes full derived configuration cards.
    return {path.stem: read_json(path) for path in declaration_paths(root)}


def _search_pin(root, family_id, backend, rows, declarations, catalogs, *, view_id, policy_fingerprint):
    from .configuration_search import _grid, _load_spec, configuration_id, select_configuration
    from .planning import FORMULATION_FIELDS
    from .views import load_view
    paths = set((Path(root) / "reports/forge/configuration-search").glob("*.json"))
    paths.update(Path(root) / card["search_report"] for card in declarations.values()
                 if card.get("trainer_family") == family_id and isinstance(card.get("search_report"), str))
    active = load_families(root).get(family_id, {}).get("active_search_by_backend", {}).get(backend)
    if active is None:
        return None
    pins = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        report = read_json(path)
        if (report.get("study_id") != active or report.get("trainer_family") != family_id or report.get("view") != view_id
                or report.get("execution_backend") != backend):
            continue
        if report.get("runtime_cohort") != rows[0].get("runtime_cohort"):
            # The same family/backend may have recorded different hardware.
            # A study can select only inside its exact runtime cohort.
            continue
        if report.get("input_digest") != stable_hash({key: value for key, value in report.items() if key != "input_digest"}):
            raise ValueError("search report input digest mismatch")
        spec = _load_spec(root, report["spec"], validate_current_protocol=False)
        if report.get("spec_hash") != stable_hash(spec):
            raise ValueError("search frozen specification hash mismatch")
        frozen_path = Path(root) / "configs/forge/searches" / (spec["id"] + ".json")
        if not frozen_path.is_file() or stable_hash(read_json(frozen_path)) != report["spec_hash"]:
            raise ValueError("search report differs from its registered specification")
        for key, value in (("study_id", spec["id"]), ("trainer_family", spec["trainer_family"]),
                           ("view", spec["view"]), ("execution_backend", spec["execution_backend"]),
                           ("tuning_through_tier", spec["tuning_through_tier"]), ("protocol_hash", spec["protocol_hash"]),
                           ("policy_fingerprint", policy_fingerprint)):
            if report.get(key) != value:
                raise ValueError("search report contradicts its frozen study contract")
        selection = report.get("selection", {})
        if not selection.get("selected_candidate_id"):
            continue
        trials = report.get("trials", [])
        base = report.get("base_declaration", {})
        if base.get("id") != spec["base_candidate"]:
            raise ValueError("search frozen base declaration differs from its specification")
        expected_settings = {stable_hash(settings) for settings in _grid(spec["grid"])}
        observed_settings = [stable_hash(trial.get("settings")) for trial in trials]
        if len(set(observed_settings)) != len(observed_settings) or set(observed_settings) != expected_settings:
            raise ValueError("search report omitted or changed a declared configuration trial")
        # Use immutable public Recipes, never today's public defaults. Cards are
        # globally reusable across studies; their editorial study metadata can
        # retain the first study while the scientific configuration stays fixed.
        expected = {}
        fixed_fields = set(FORMULATION_FIELDS) | {"host_adaptation", "initializer", "execution_path"}
        fixed_fields.discard("recipe_overrides")
        for trial in trials:
            frozen = trial.get("declaration", {})
            recipe = trial.get("resolved_recipe")
            if not isinstance(recipe, dict) or not recipe:
                raise ValueError("search trial lacks its frozen resolved Recipe")
            overrides = {**base.get("recipe_overrides", {}), **trial["settings"]}
            if (frozen.get("recipe_overrides") != overrides
                    or any(frozen.get(key) != base.get(key) for key in fixed_fields)
                    or trial.get("recipe_overrides") != overrides):
                raise ValueError("search trial changed its frozen base formulation or grid settings")
            digest = configuration_id(frozen, resolved_recipe=recipe)
            name = f"{family_id}--{digest}"
            if (trial.get("candidate_id") != name or trial.get("configuration_id") != digest
                    or frozen.get("id") != name or frozen.get("configuration_id") != digest
                    or frozen.get("trainer_family") != family_id
                    or any(stable_hash(recipe.get(key)) != stable_hash(value)
                           for key, value in overrides.items() if key != "name")):
                raise ValueError("search frozen configuration identity differs from its Recipe")
            expected[name] = frozen
        ids = [trial.get("candidate_id") for trial in trials]
        if len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError("search report omitted or changed a declared configuration trial")
        cohort_rows = [row for row in rows if row["candidate_id"] in expected]
        if len(cohort_rows) != len(expected) or {row["candidate_id"] for row in cohort_rows} != set(expected):
            raise ValueError("search comparison is missing verified configuration evidence")
        assignments = {a["task"]: a for a in load_view(root, view_id)["assignments"]}
        verified = []
        for trial in trials:
            name = trial["candidate_id"]
            row = next(item for item in cohort_rows if item["candidate_id"] == name)
            card = declarations.get(name, {})
            if (card.get("configuration_id") != expected[name]["configuration_id"]
                    or configuration_id(card, resolved_recipe=trial["resolved_recipe"]) != card.get("configuration_id")
                    or trial.get("configuration_id") != card.get("configuration_id")
                    or trial.get("trainer_family") != family_id
                    or trial.get("candidate_revision") != row.get("candidate_revision")):
                raise ValueError("search configuration or revision differs from verified evidence")
            bindings = row.get("bindings", {})
            protocol = catalogs.get("protocol_contracts", {}).get(bindings.get("protocol_sha256"))
            if protocol is None or stable_hash(protocol) != spec["protocol_hash"]:
                raise ValueError("search frozen protocol differs from verified protocol contracts")
            if (trial.get("source_digest") != bindings.get("source_digest")
                    or trial.get("runtime_cohort") != row.get("runtime_cohort")
                    or trial.get("protocol_hash") != bindings.get("protocol_sha256")
                    or stable_hash(trial["resolved_recipe"]) != bindings.get("recipe_sha256")):
                raise ValueError("search source/runtime/protocol differs from verified evidence")
            task_rows = trial.get("tasks", [])
            keys = {task["task"]: task.get("compatibility_key") for task in task_rows}
            if len(keys) != len(task_rows) or set(keys) != set(assignments):
                raise ValueError("search trial changed the frozen view task denominator")
            if stable_hash(keys) != bindings.get("task_keys_sha256"):
                raise ValueError("search trial task identities differ from verified evidence")
            grades = {task["task_id"]: task["status"] for task in row.get("tasks", []) + row.get("nonrequired_tasks", [])}
            verified_tasks = []
            for task in task_rows:
                assignment = assignments[task["task"]]
                if any(task.get(key) != assignment[key] for key in ("importance", "qualification_tier")):
                    raise ValueError("search trial changed frozen task importance or tier")
                observed = grades.get(task["task"], "UNKNOWN")
                declared = task.get("gate_status")
                if ("UNKNOWN" if declared == "NOT_RUN" else declared) != observed:
                    raise ValueError("search trial status differs from independently graded publication")
                verified_tasks.append({**assignment, "compatibility_key": task["compatibility_key"], "gate_status": observed})
            required = [task["gate_status"] for task in verified_tasks if task["importance"] == "required"
                        and task["qualification_tier"] <= spec["tuning_through_tier"]]
            terminal = (any(status in {"FAIL", "BLOCKED", "INVALID", "INCOMPLETE"} for status in required)
                        or bool(required) and all(status == "PASS" for status in required))
            verified.append({"candidate_id": name, "configuration_id": card["configuration_id"],
                             "source_digest": bindings["source_digest"], "tasks": verified_tasks,
                             "submission_status": "terminal" if terminal else "pending"})
        cohorts = {comparison_cohort(row, catalogs) for row in cohort_rows}
        if len(cohorts) != 1:
            raise ValueError("search cannot rank different source/runtime/task/prior/protocol cohorts")
        sources = report.get("source_digests", [])
        if set(sources) != {row["bindings"]["source_digest"] for row in cohort_rows}:
            raise ValueError("search report source identity differs from its scientific evidence")
        recomputed = select_configuration(verified, spec["tuning_through_tier"])
        if not recomputed.get("selection_complete"):
            raise ValueError("search configuration selection requires all trials to be terminal")
        if selection != recomputed:
            raise ValueError("search selection differs from verified whole-configuration objective")
        selected = next(row for row in cohort_rows if row["candidate_id"] == recomputed["selected_candidate_id"])
        pins.append((selected, {**recomputed, "study_report": path.relative_to(root).as_posix(),
                                "study_id": report["study_id"], "comparison_cohort": next(iter(cohorts))}))
    if len(pins) > 1:
        raise ValueError("multiple search selections require an explicit family study choice")
    return pins[0] if pins else None


def select_family_rows(root: Path | str, rows: list[dict], catalogs: dict, *, view_id: str,
                       policy_fingerprint: str, declarations: dict | None = None) -> dict:
    """Choose one whole configuration per family/backend, retaining alternatives."""
    declarations = _declarations(root) if declarations is None else declarations
    grouped, families, variants = {}, {}, []
    for original in rows:
        row = deepcopy(original)
        family = family_for_candidate(root, row["candidate_id"], declarations.get(row["candidate_id"]))
        family_id = family["id"]
        families[family_id] = family
        row.update(trainer_family=family_id,
                   configuration_id=declarations.get(row["candidate_id"], {}).get("configuration_id", row["candidate_id"]),
                   comparison_cohort=comparison_cohort(row, catalogs))
        backend = row.get("runtime_cohort", {}).get("execution_backend", "unrecorded")
        runtime_key = stable_hash(row.get("runtime_cohort"))
        grouped.setdefault((family_id, backend, runtime_key), []).append(row)
        variants.append(row)
    selected_rows = []
    for (family_id, backend, _), alternatives in sorted(grouped.items()):
        family = families[family_id]
        pin = _search_pin(root, family_id, backend, alternatives, declarations, catalogs,
                          view_id=view_id, policy_fingerprint=policy_fingerprint)
        if pin:
            selected, metadata = pin
        else:
            canonical = [row for row in alternatives if row["candidate_id"] == family["canonical_candidate"]]
            if len(canonical) > 1:
                raise ValueError("canonical configuration has multiple scientific rows in one runtime; select its recorded revision first")
            if not canonical:
                raise ValueError("family runtime cohort needs its canonical configuration row before selection")
            selected = canonical[0]
            metadata = {"selection_kind": "canonical_fallback" if canonical else "configuration_fallback",
                        "qualified": selected.get("status") == "PASS",
                        "reason": "Canonical configuration preferred; no outcome ranking across incomparable sources."}
        selected_identity = selected["candidate_id"], selected.get("cohort")
        display = deepcopy(selected)
        display.update(technique=family["label"], selection=metadata)
        selected_rows.append(display)
        for row in alternatives:
            identity = row["candidate_id"], row.get("cohort")
            row["selected_configuration"] = identity == selected_identity
            row["alternative_scope"] = ("selected" if row["selected_configuration"] else "comparable_trial"
                                        if row["comparison_cohort"] == selected["comparison_cohort"] else "archived_alternative")
    selected_rows.sort(key=lambda row: (-row.get("qualified_tier", 0), row["technique"],
                                        row.get("runtime_cohort", {}).get("execution_backend", "")))
    return {"rows": selected_rows, "configuration_rows": variants, "trainer_families": families}

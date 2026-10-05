"""Compile finite tagged choices into frozen ordinary Forge searches.

The sampler owns an isolated RNG and never executes training. Categories name
declared structural candidates; numerical trials keep each candidate's technique.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import random

from . import configuration_search as search
from .contracts import atomic_json, identifier, read_json, stable_hash
from .queue import Queue, drain

SPACE_SCHEMA = "forge_search_space_v1"
MANIFEST_SCHEMA = "forge_search_compilation_v1"


def _load(root, value):
    return deepcopy(value) if isinstance(value, dict) else read_json(Path(root) / value)


def _values(tag):
    if not isinstance(tag, dict):
        raise ValueError("search parameters require tagged literal, choice or logspace values")
    kind = tag.get("kind")
    if kind == "literal" and set(tag) == {"kind", "value"}:
        values = [deepcopy(tag["value"])]
    elif kind == "choice" and set(tag) == {"kind", "values"}:
        values = deepcopy(tag["values"])
        if not isinstance(values, list) or not values:
            raise ValueError("choice requires a nonempty values list")
    elif kind == "logspace" and set(tag) == {"kind", "low", "high", "count"}:
        low, high, count = tag["low"], tag["high"], tag["count"]
        if (type(count) is not int or not 2 <= count <= 65_536
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in (low, high))
                or not 0 < low < high):
            raise ValueError("logspace requires 0 < low < high and 2..65536 finite choices")
        values = [math.exp(math.log(low) + (math.log(high) - math.log(low)) * i / (count - 1))
                  for i in range(count)]
        values[0], values[-1] = low, high
    else:
        raise ValueError("unsupported or malformed search parameter tag")
    if len({stable_hash(v) for v in values}) != len(values):
        raise ValueError("duplicate search-space choices")
    return values


def _axes(parameters):
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("each search category requires nonempty parameters")
    from dataclasses import fields
    from particlegan import Recipe
    from .boundaries import TUNABLE_FIELDS
    public = {field.name for field in fields(Recipe)}
    result, assigned = [], set()
    for axis, tag in sorted(parameters.items()):
        identifier(axis, "search-space axis")
        values = _values(tag)
        choices = [{axis: value} for value in values] if axis in public else values
        if not all(isinstance(c, dict) and c for c in choices):
            raise ValueError("grouped choices must assign public Recipe fields")
        names = set(choices[0])
        if any(set(c) != names for c in choices) or assigned & names:
            raise ValueError("grouped search choices must bind identical, nonoverlapping fields")
        if names - TUNABLE_FIELDS:
            raise ValueError("search-space fields must be numerical tuning fields; categories name structural candidates")
        for choice in choices:
            for name, value in choice.items():
                search._validate_grid_value(name, value)
        assigned.update(names)
        result.append(choices)
    return result


def _at(axes, index):
    settings = {}
    for axis in reversed(axes):
        index, offset = divmod(index, len(axis))
        settings.update(deepcopy(axis[offset]))
    if index:
        raise ValueError("search-space index outside the population")
    return settings


def _sample_indices(rng, population, count):
    # Floyd's algorithm works with arbitrarily large integer populations and
    # O(count) storage; range/sample would require a Py_ssize_t-sized population.
    selected = set()
    for upper in range(population - count, population):
        value = rng.randrange(upper + 1)
        selected.add(upper if value in selected else value)
    values = sorted(selected)
    rng.shuffle(values)
    return values


def compile_space(root, queue_root, definition, *, queue=None):
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    space = _load(root, definition)
    required = {"schema", "id", "hypothesis", "protocol", "view", "execution_backend",
                "tuning_through_tier", "campaign", "samples", "candidates"}
    if (not isinstance(space, dict) or required - space.keys()
            or space.keys() - required - {"cuda_model", "parameters", "rationale"}
            or space["schema"] != SPACE_SCHEMA):
        raise ValueError("invalid bounded search-space declaration")
    identifier(space["id"], "search-space id")
    count = space["samples"]
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError("search-space samples must be 1..256")
    protocol = read_json(root / "configs/forge/protocols" / f"{identifier(space['protocol'])}.json")
    if protocol.get("seed") != 0:
        raise ValueError("new search spaces require protocol seed 0")
    candidates = space["candidates"]
    if not isinstance(candidates, list) or not candidates or len(candidates) > 256:
        raise ValueError("search-space categories require 1..256 declared candidates")
    domains, population, seen = [], 0, set()
    for category in candidates:
        if (not isinstance(category, dict) or {"base_candidate", "trainer_family"} - category.keys()
                or category.keys() - {"base_candidate", "trainer_family", "parameters"}):
            raise ValueError("categories name a base_candidate, trainer_family and optional conditional parameters")
        base = identifier(category["base_candidate"])
        family = identifier(category["trainer_family"])
        if base in seen:
            raise ValueError("duplicate structural candidate in search-space roster")
        seen.add(base)
        axes = _axes(category.get("parameters", space.get("parameters")))
        size = math.prod(len(axis) for axis in axes)
        domains.append((category, axes, population, size))
        population += size
    if count > population:
        raise ValueError("sample count exceeds the unique finite population")
    definition_hash = stable_hash(space)
    derivation = {"version": "forge-search-rng-v1", "protocol_seed": 0, "space_id": space["id"],
                  "domain_hash": stable_hash([{"category": row[0], "axes": row[1]} for row in domains])}
    rng = random.Random(int(stable_hash(derivation), 16))
    initial_state = rng.getstate()
    indices = _sample_indices(rng, population, count)
    by_category, draws = {}, []
    for index in indices:
        category, axes, start, size = next(row for row in domains if row[2] <= index < row[2] + row[3])
        settings = _at(axes, index - start)
        base = category["base_candidate"]
        by_category.setdefault(base, []).append(settings)
        draws.append({"index": index, "base_candidate": base, "settings": settings})
    queue = queue or Queue(queue_root)
    specs, bindings, total = [], [], 0
    # Inspect every roster member's capability/ownership contract, even if a
    # finite draw did not select it. Never silently omit an invalid category.
    for number, (category, axes, _, size) in enumerate(domains):
        base = category["base_candidate"]
        spec = {"schema_version": 2, "id": f"{space['id']}--{number}",
                "base_candidate": base, "trainer_family": category["trainer_family"],
                **{k: deepcopy(space[k]) for k in ("hypothesis", "protocol", "view", "execution_backend",
                                                    "tuning_through_tier", "campaign")},
                "grid": [{"configuration": [settings]} for settings in by_category.get(base, [_at(axes, 0)])]}
        if "cuda_model" in space:
            spec["cuda_model"] = space["cuda_model"]
        if "rationale" in space:
            spec["rationale"] = space["rationale"]
        plan = search.plan_search(root, queue_root, spec, queue=queue)
        if base not in by_category:
            # Admission/ownership was checked; retain category visibility but
            # its unsampled control is not an extra paid trial or selection row.
            bindings.append({"base_candidate": base, "population": size, "sampled": 0,
                             "base_declaration_hash": stable_hash(plan["base_declaration"]),
                             "source_digest": plan["source_digest"], "runtime_cohort": plan["runtime_cohort"],
                             "protocol_hash": plan["protocol_hash"], "policy_fingerprint": plan["policy_fingerprint"]})
            continue
        total += plan["declared_worst_case_seconds"]
        specs.append(spec)
        bindings.append({"base_candidate": base, "population": size, "sampled": len(plan["trials"]),
                         "base_declaration_hash": stable_hash(plan["base_declaration"]),
                         "source_digest": plan["source_digest"], "runtime_cohort": plan["runtime_cohort"],
                         "protocol_hash": plan["protocol_hash"], "policy_fingerprint": plan["policy_fingerprint"],
                         "trial_signatures": {t["candidate_id"]: t["scientific_signature"] for t in plan["trials"]}})
    if total > space["campaign"]["budget_seconds"]:
        raise ValueError("shared campaign cannot reserve all sampled categories")
    for key in ("source_digest", "runtime_cohort", "protocol_hash", "policy_fingerprint"):
        if len({stable_hash(row[key]) for row in bindings}) != 1:
            raise ValueError(f"categorical searches require one matched {key} cohort")
    manifest = {"schema": MANIFEST_SCHEMA, "definition": space, "definition_hash": definition_hash,
                "population": population, "draws": draws, "searches": specs, "bindings": bindings,
                "rng": {"derivation": derivation, "initial_state": initial_state, "final_state": rng.getstate()},
                "declared_worst_case_seconds": total, "default_adoption": False,
                "qualification_scope": "sampled complete configurations; provisional ordinary gates"}
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest


def write_compilation(root, queue_root, definition, destination):
    manifest = compile_space(root, queue_root, definition)
    destination = Path(destination)
    if destination.exists():
        if stable_hash(read_json(destination)) != stable_hash(manifest):
            raise ValueError("compiled search manifest is immutable; use a new study identity and output")
    else:
        atomic_json(destination, manifest)
    return manifest


def _verified(root, queue_root, value, queue):
    manifest = _load(root, value)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("expected a compiled search manifest")
    expected = compile_space(root, queue_root, manifest["definition"], queue=queue)
    if stable_hash(expected) != stable_hash(manifest):
        raise ValueError("compiled search source, draws, RNG state or declarations changed; compile a new study")
    return expected


def plan_compilation(root, queue_root, value, *, queue=None):
    queue = queue or Queue(Path(queue_root).resolve())
    manifest = _verified(root, queue_root, value, queue)
    plans = [search.plan_search(root, queue_root, spec, queue=queue) for spec in manifest["searches"]]
    return _summary(manifest, plans, "planned")


def _summary(manifest, plans, stage):
    trials = [trial for plan in plans for trial in plan["trials"]]
    if len({t["candidate_id"] for t in trials}) != len(trials):
        raise ValueError("compiled categories resolve duplicate complete candidates")
    return {"stage": stage, "space_id": manifest["definition"]["id"],
            "manifest_hash": manifest["manifest_hash"], "campaign": manifest["definition"]["campaign"],
            "declared_worst_case_seconds": manifest["declared_worst_case_seconds"],
            "searches": plans, "trials": trials,
            "submitted_count": sum(p.get("submitted_count", 0) for p in plans),
            "selection": search.select_configuration(trials, manifest["definition"]["tuning_through_tier"]),
            "unsampled_categories": [b["base_candidate"] for b in manifest["bindings"] if not b["sampled"]],
            "default_adoption": False}


def enqueue_compilation(root, queue_root, value, *, queue=None):
    queue = queue or Queue(Path(queue_root).resolve(), report_root=Path(root) / "reports/forge")
    manifest = _verified(root, queue_root, value, queue)
    # Freeze the entire roster before admitting any category. Recovery uses
    # the same normal search registration and compatible-request lookup.
    frozen = [search._freeze_search(root, queue_root, spec, queue=queue) for spec in manifest["searches"]]
    for _, plan in frozen:
        binding = next(b for b in manifest["bindings"] if b["base_candidate"] == plan["base_candidate"])
        if {t["candidate_id"]: t["scientific_signature"] for t in plan["trials"]} != binding["trial_signatures"]:
            raise ValueError("compiled search changed while freezing source")
    for _, plan in frozen:
        search._persist(root, plan)
    for requests, plan in frozen:
        search._admit_search(root, queue, requests, plan)
    return report_compilation(root, queue_root, manifest, queue=queue)


def report_compilation(root, queue_root, value, *, queue=None):
    queue = queue or Queue(Path(queue_root).resolve())
    manifest = _load(root, value)
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("manifest_hash") != stable_hash(unsigned):
        raise ValueError("compiled search manifest hash mismatch")
    # Reporting uses frozen search receipts; it need not reproduce a newer
    # checkout's source or turn archived results into current qualification.
    plans = [search.report_search(root, queue_root, spec, queue=queue) for spec in manifest["searches"]]
    return _summary(manifest, plans, "reported")


def run_compilation(root, queue_root, value, *, devices=None, queue=None):
    queue = queue or Queue(Path(queue_root).resolve(), report_root=Path(root) / "reports/forge")
    manifest = _verified(root, queue_root, value, queue)
    backend = manifest["definition"]["execution_backend"]
    devices = list(devices or (["cpu"] if backend == "cpu" else ["0", "1"]))
    if (backend == "cpu") != (devices == ["cpu"]):
        raise ValueError("compiled search devices differ from its runtime cohort")
    summary = enqueue_compilation(root, queue_root, manifest, queue=queue)
    if summary["submitted_count"]:
        drain(queue, devices, campaign=summary["campaign"]["id"])
    return report_compilation(root, queue_root, manifest, queue=queue)

"""Actual declaration controls; every displayed outcome here is software-only.

The parents, full 26-slot view and prospective transforms are real. No trained
receipt, checkpoint, raw array, evaluator, producer or historical Git object is
needed to validate the compact publication boundary.
"""
from collections import Counter
from copy import deepcopy
import builtins
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.forge import ae_routed_policy_contracts as ae
from experiments.forge import conditional_policy_contracts as conditional
from experiments.forge import multibank_policy_contracts as multibank
from experiments.forge import named_policy_planning as planning
from experiments.forge import policy_contracts as independent
from experiments.forge import policy_snapshot_publication as publication
from experiments.forge import routed_policy_contracts as routed
from experiments.forge import technique_board
from experiments.forge import word_joint_policy_contracts as word
from experiments.forge.contracts import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PIN = "2" * 40
COHORTS = (conditional.COHORT, routed.COHORT, multibank.COHORT, ae.COHORT, word.COHORT)
MODULES = {module.COHORT: module for module in (conditional, routed, multibank, ae, word)}
SOFTWARE_ONLY = "synthetic statuses; actual declarations; no learned qualification"


def _write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _variant_path(task):
    return f"configs/forge/task-variants/{task['task_cohort']}/{task['id']}.json"


def _reduce(fixture):
    """Real display reducer, including after deliberately coherent tampering."""
    actual = planning.project_view(fixture.common, fixture.selected, fixture.cohort, fixture.family)
    request = {
        "candidate": {"id": "software-only-" + fixture.family, "trainer_family": fixture.family,
                      "task_cohort": fixture.cohort, "recipe_preset": "atlas",
                      "recipe_overrides": {"lr": .0053125, "prior_lr_mult": 1.5},
                      "resolved_recipe": {"scope": SOFTWARE_ONLY}},
        "view": actual, "tasks": fixture.selected, "jobs": [], "protocol": {}, "rng": {}, "source": {},
    }
    adapted = [a["task"] for a in actual["assignments"] if a["task"] != fixture.slots[a["task"]]]
    outcomes = dict(zip(adapted, ("PASS", "FAIL", "INCOMPLETE", "BLOCKED")))
    statuses = [{"task_id": a["task"], "status": outcomes.get(a["task"], "NOT_RUN")}
                for a in actual["assignments"]]
    original = {
        "candidate_id": request["candidate"]["id"], "status": "INCOMPLETE", "qualified_tier": 0,
        "runtime_cohort": {"execution_backend": "cuda"}, "cost": {},
        "qualification": {"view_revision": actual["revision"], "policy_fingerprint": stable_hash(actual),
                          "tasks": statuses},
        "scientific_bindings": technique_board.request_bindings(request),
        **technique_board.policy_row_metadata(request),
    }
    board = {"view": fixture.common["id"], "view_revision": fixture.common["revision"],
             "policy_fingerprint": stable_hash(fixture.common), "current_rows": [original], "rows": [], "conflicts": []}
    fixture.report = technique_board.reduce_board(board, fixture.common)
    fixture.report.update(publication_scope="live_current", software_fixture_scope=SOFTWARE_ONLY)
    fixture.row = fixture.report["rows"][0]
    return fixture


def _make(tmp_path, cohort):
    """Use exact current maker output; source files are bytes, never imports."""
    module = MODULES[cohort]
    if cohort == conditional.COHORT:
        variants = [conditional.make_conditional_variant(ROOT, host) for host in conditional.HOSTS]
    elif cohort == routed.COHORT:
        variants = [routed.make_unused_variant(ROOT)]
    elif cohort == multibank.COHORT:
        variants = [multibank.make_variant(ROOT)]
    elif cohort == ae.COHORT:
        variants = [ae.make_ae_variant(ROOT)]
    else:
        variants = [word.make_variant(ROOT)]
    parents = {}
    source_files = set()
    for name in independent.PARENT_TASK_IDS:
        relative = f"configs/forge/tasks/{name}.json"
        raw = (ROOT / relative).read_bytes()
        _write(tmp_path, relative, raw)
        parents[name] = json.loads(raw)
        source_files.update(parents[name]["evaluation"].get("sources", {}))
    relative = f"configs/forge/views/{independent.PARENT_VIEW_ID}.json"
    common_raw = (ROOT / relative).read_bytes()
    _write(tmp_path, relative, common_raw)
    common = json.loads(common_raw)
    for task in variants:
        _write(tmp_path, _variant_path(task), task)
        source_files.update(task["execution"]["policy_contract"]["sources"])
        source_files.update(task["execution"].get("policy_resource_sources", {}))
    for relative in source_files:
        _write(tmp_path, relative, (ROOT / relative).read_bytes())
    # The real resolver validates the initial declarations before reducing them.
    _, selected = planning.resolve_task_view(common, {**parents, **{t["id"]: t for t in variants}},
                                             {"task_cohort": cohort})
    fixture = SimpleNamespace(root=tmp_path, cohort=cohort, family=module.FAMILY, module=module,
        common=common, parents=parents, selected=selected,
        slots={name: task.get("policy_parent", {}).get("id", name) for name, task in selected.items()})
    return _reduce(fixture)


@pytest.fixture(params=COHORTS, ids=COHORTS)
def named(request, tmp_path):
    publication._git_bytes.cache_clear()
    return _make(tmp_path, request.param)


def _validate(fixture):
    return publication.validate_policy_publication(fixture.root, fixture.report, fixture.row)


def _first_adapted(fixture):
    return next(task for task in fixture.selected.values() if task.get("task_cohort") == fixture.cohort)


def _coherent_task_change(fixture, change):
    task = _first_adapted(fixture)
    change(task)
    _write(fixture.root, _variant_path(task), task)
    return _reduce(fixture)


def _frozen(fixture, monkeypatch, *, missing=None):
    """A synthetic immutable Git transport; no historical-object CI dependency."""
    blobs = _files(fixture.root)
    if missing is not None:
        blobs.pop(missing)
    calls = []

    def git_show(command, **kwargs):
        assert command[:2] == ["git", "show"]
        commit, relative = command[2].split(":", 1)
        assert commit == PIN
        assert kwargs == {"cwd": str(fixture.root), "capture_output": True, "check": False}
        calls.append(relative)
        return SimpleNamespace(returncode=0 if relative in blobs else 1, stdout=blobs.get(relative, b""))

    monkeypatch.setattr(publication.subprocess, "run", git_show)
    fixture.report.update(publication_scope="frozen_source", frozen_source={"commit": PIN})
    publication._git_bytes.cache_clear()
    return blobs, calls


def _forbid_science_imports(monkeypatch):
    original = builtins.__import__
    forbidden = ("torch", "numpy", "particlegan", "benchmarks", "experiments.forge.runtime",
                 "experiments.forge.evaluate", "experiments.forge.artifacts")

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        package = (globals or {}).get("__package__", "")
        absolute = name if not level else package + ("." + name if name else "")
        assert not any(absolute == x or absolute.startswith(x + ".") for x in forbidden), absolute
        assert not absolute.endswith("_adapters"), absolute
        if package == "experiments.forge":
            assert not any(x in {"runtime", "evaluate", "artifacts"} or x.endswith("_adapters") for x in fromlist)
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_actual_maker_full26_live_projection_preserves_partial_statuses_and_bytes(named, monkeypatch):
    before = _files(named.root)
    metadata = deepcopy(named.report)
    _forbid_science_imports(monkeypatch)
    assert _validate(named) is None
    assert _files(named.root) == before
    assert named.report == metadata
    assert [len(named.report["tier_requirements"][str(t)]) for t in (1, 2, 3)] == [5, 19, 2]
    assert len(named.row["tasks"]) == len(named.row["task_slot_map"]) == 26
    assert named.row["qualified_tier"] == 0
    statuses = Counter(t["status"] for t in named.row["tasks"])
    assert statuses["PASS"] == 1
    assert statuses["UNKNOWN"] == 26 - len(planning.NAMED_PARENTS[named.cohort])
    for parent in independent.PARENT_TASK_IDS:
        assert (named.root / f"configs/forge/tasks/{parent}.json").read_bytes() == (ROOT / f"configs/forge/tasks/{parent}.json").read_bytes()


def test_pinned_projection_reads_frozen_bytes_and_ignores_live_drift(named, monkeypatch):
    _, calls = _frozen(named, monkeypatch)
    (named.root / f"configs/forge/tasks/{independent.PARENT_TASK_IDS[0]}.json").write_text("not current source\n")
    first = _first_adapted(named)
    source = next(p for p in first["execution"]["policy_contract"]["sources"] if p.endswith("_adapters.py"))
    (named.root / source).unlink()
    _forbid_science_imports(monkeypatch)
    before = deepcopy(named.report)
    assert _validate(named) is None
    assert _validate(named) is None
    assert named.report == before
    assert len(calls) == len(set(calls))
    assert all(not name.endswith((".npz", ".pt", ".jsonl")) for name in calls)


def test_missing_frozen_blob_never_substitutes_available_current_source(named, monkeypatch):
    source = next(p for p in _first_adapted(named)["execution"]["policy_contract"]["sources"] if p.endswith("_adapters.py"))
    assert (named.root / source).is_file()
    _frozen(named, monkeypatch, missing=source)
    with pytest.raises(ValueError, match="pinned source is unavailable"):
        _validate(named)


@pytest.mark.parametrize("change", ["horizon", "threshold", "cadence", "stable_checks", "adapter", "initializer",
                                   "owner", "row_policy", "schedule", "sampler", "latent", "output_noise",
                                   "forced_ema", "diagnostic_credit", "prior", "recipe_override", "resource_timeout",
                                   "resource_memory", "cpu_device", "capabilities", "controls", "precision",
                                   "clock", "table_owner", "checkpoint", "routing_sites", "disabled_owner",
                                   "host_definition"])
def test_coherently_rehashed_variant_cannot_change_actual_transform(named, change):
    def tamper(task):
        e, v, c, o = task["execution"], task["evaluation"], task["execution"]["policy_contract"], task["evaluation"]["policy_observation"]
        if change == "horizon": e["steps"] -= 1
        elif change == "threshold": v["thresholds"][0][2] = 999
        elif change == "cadence": v["observations"] = 2
        elif change == "stable_checks": v["minimum_stable_checks"] = 1
        elif change == "adapter": task["adapter"] = "unverified_adapter"
        elif change == "initializer": e["initializer"] = "unverified_init"
        elif change == "owner": c["owner"] = "software_only_fake_owner"
        elif change == "row_policy": c["row_policy"] = "independent" if c["row_policy"] != "independent" else "routed_paired"
        elif change == "schedule": c["schedule"] = "posthoc_horizon"
        elif change == "sampler": o["sampler"] = "fake_clean_sampler"
        elif change == "latent": o["latent_policy"] = "none"
        elif change == "output_noise": o["output_noise"] = not bool(o["output_noise"])
        elif change == "forced_ema": o["weight_selector"] = "forced_ema"
        elif change == "diagnostic_credit": o["diagnostic_credit"] = True
        elif change == "prior": e["prior"]["kind"] = "unverified_prior"
        elif change == "recipe_override": e["policy_recipe_overrides"]["row_policy"] = "fake"
        elif change == "resource_timeout": task["resources"]["timeout_seconds"] += 1
        elif change == "resource_memory": task["resources"]["gpu_memory_mb"] += 1
        elif change == "cpu_device": e["device"] = task["resources"]["device"] = "cpu"
        elif change == "capabilities": task["requires_capabilities"] = []
        elif change == "controls": c["controls"] = "requested_owners_disabled"
        elif change == "precision": c["precision"] = "autocast_low_precision"
        elif change == "clock": c["external_limit"] = 2
        elif change == "table_owner": c["table_owner"] = "unowned_table"
        elif change == "checkpoint": c["checkpoint"] = "parameters_only_no_policy_or_streams"
        elif change == "routing_sites": c.setdefault("routing", {})["sites"] = ["invented_site"]
        elif change == "disabled_owner": e["policy_recipe_overrides"]["particle_birth_death"] = False
        elif change == "host_definition": e["host_definition"]["steps"] = 2
    _coherent_task_change(named, tamper)
    with pytest.raises(ValueError, match="policy publication|named publication"):
        _validate(named)


@pytest.mark.parametrize("field,value", [("num_particles", 5), ("num_particles", 6), ("z_dim", 3), ("batch_size", 128)])
def test_word_resource11_is_physical_and_cannot_be_rehashed_back_to_original5(tmp_path, field, value):
    fixture = _make(tmp_path, word.COHORT)
    task = _first_adapted(fixture)
    assert task["execution"]["resources"] == {"num_particles": 11, "z_dim": 2, "batch_size": 256}
    assert task["execution"]["resource_adaptation"]["original"]["num_particles"] == 5
    def tamper(task):
        task["execution"]["resources"][field] = value
        task["execution"]["resource_adaptation"]["actual"][field] = value
        task["execution"]["policy_contract"]["resource_adaptation"]["actual"][field] = value
    _coherent_task_change(fixture, tamper)
    with pytest.raises(ValueError, match="named publication"):
        _validate(fixture)


@pytest.mark.parametrize("change", ["missing", "duplicate", "order", "tier", "importance", "map", "denominator", "status"])
def test_full_denominator_and_partial_status_projection_fail_closed(named, change):
    row = named.row
    if change == "missing": row["tasks"].pop()
    elif change == "duplicate": row["tasks"][1] = deepcopy(row["tasks"][0])
    elif change == "order": row["tasks"][0], row["tasks"][1] = row["tasks"][1], row["tasks"][0]
    elif change == "tier": row["qualification_view"]["assignments"][0]["qualification_tier"] = 2
    elif change == "importance": row["qualification_view"]["assignments"][0]["importance"] = "diagnostic"
    elif change == "map": row["task_slot_map"][row["tasks"][0]["task_id"]] = "wrong_slot"
    elif change == "denominator": row["tiers"]["2"]["total"] = 4
    elif change == "status": row["tasks"][0]["status"] = "PASS"; row["tasks"][0]["gate_status"] = "NOT_RUN"
    with pytest.raises(ValueError, match="policy publication"):
        _validate(named)


def test_changed_parent_bytes_cannot_preserve_old_parent_identity(named):
    task = _first_adapted(named)
    path = named.root / f"configs/forge/tasks/{task['policy_parent']['id']}.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="parent|scope"):
        _validate(named)


@pytest.mark.parametrize("change", ["wrong_sha", "missing_source", "missing_common_roster", "unsafe_source"])
def test_source_manifest_drift_is_rejected_even_when_display_catalog_is_rehashed(named, change):
    def tamper(task):
        sources = task["execution"]["policy_contract"]["sources"]
        source = "experiments/forge/policy_cohorts.py"
        if change == "wrong_sha": sources[source] = "a" * 64
        elif change == "missing_common_roster": sources.pop(source)
        elif change == "unsafe_source": sources["../foreign.py"] = "b" * 64
        elif change == "missing_source": (named.root / source).unlink()
    _coherent_task_change(named, tamper)
    with pytest.raises(ValueError, match="policy publication|named publication"):
        _validate(named)


@pytest.mark.parametrize("field", ["task_cohort", "policy_family"])
def test_unknown_cohort_or_family_cannot_change_registered_law(named, field):
    task = _first_adapted(named)
    task[field] = "unknown_policy_law"
    # Retain the original valid row/catalog; the source declaration is foreign.
    relative = f"configs/forge/task-variants/{named.cohort}/{task['id']}.json"
    _write(named.root, relative, task)
    with pytest.raises(ValueError, match="named publication"):
        _validate(named)


def test_compact_binding_cannot_claim_an_unknown_trainer_family(named):
    named.row["bindings"]["trainer_family"] = "unknown_policy_law"
    with pytest.raises(ValueError, match="family"):
        _validate(named)


def test_named_ids_cannot_hide_policy_law_by_stripping_outer_and_catalog_markers(named):
    row, report = named.row, named.report
    for field in publication._POLICY_FIELDS:
        row.pop(field, None)
    for field in publication._BINDING_FIELDS:
        row["bindings"].pop(field, None)
    refs = row["bindings"]["task_contracts"]
    for name, old_ref in list(refs.items()):
        contract = deepcopy(report["task_contracts"][old_ref])
        contract.pop("task_cohort", None)
        contract.pop("policy_parent", None)
        new_ref = stable_hash(contract)
        report["task_contracts"][new_ref] = contract
        refs[name] = new_ref
    with pytest.raises(ValueError, match="policy publication"):
        _validate(named)


def test_frozen_metadata_does_not_execute_pinned_producer_python(tmp_path, monkeypatch):
    fixture = _make(tmp_path, word.COHORT)
    source = "experiments/forge/word_joint_policy_adapters.py"
    poison = b"raise AssertionError('Frozen producer must only be read as bytes')\n"
    digest = _write(fixture.root, source, poison)
    _coherent_task_change(fixture, lambda task: task["execution"]["policy_contract"]["sources"].update({source: digest}))
    _frozen(fixture, monkeypatch)
    _forbid_science_imports(monkeypatch)
    assert _validate(fixture) is None


def test_unadapted_original_slots_preserve_live_mog_law_without_policy_credit(named):
    for name, task in named.selected.items():
        if task.get("task_cohort") is not None:
            continue
        assert task == named.parents[name]
        assert "policy_contract" not in task["execution"]
        assert task["evaluation"]["scoring_weights"] == "live"
        status = next(t for t in named.row["tasks"] if t["task_id"] == name)
        assert status == {"task_id": name, "status": "UNKNOWN", "gate_status": "NOT_RUN"}
    assert _validate(named) is None

"""Synthetic declaration-only publication controls; no scientific evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.forge import policy_contracts as policy
from experiments.forge import policy_snapshot_publication as publication
from experiments.forge import technique_board
from experiments.forge.contracts import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PIN = "1" * 40


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def synthetic(tmp_path):
    """The 26 names are a metadata denominator, never learned toy outcomes."""
    source_names = sorted(policy.REQUIRED_POLICY_SOURCES | {
        "benchmarks/locked_shared/two_pole.py", "benchmarks/legacy/locked_shared.py",
        "experiments/forge/policy_adapters.py", "tests/synthetic_evaluator.py"})
    sources = {name: write(tmp_path, name, ("# SYNTHETIC METADATA SOURCE: " + name + "\n").encode())
               for name in source_names}
    # This committed declaration pins C6's host exceptions; no raw artifacts,
    # scientific grader, original Git object, model or package is needed.
    write(tmp_path, policy.SELECTION_SOURCE, (ROOT / policy.SELECTION_SOURCE).read_bytes())
    parents, variants, assignments = {}, {}, []
    for index, name in enumerate(policy.PARENT_TASK_IDS):
        tier, start = (1, 0) if index < 5 else (2, 5) if index < 24 else (3, 24)
        assignments.append({"task": name, "qualification_tier": tier, "importance": "required", "order": index - start})
        parent = {"schema_version": 1, "id": name, "adapter": "synthetic_only",
                  "fixture_scope": "synthetic_metadata_no_scientific_claim",
                  "execution": {"host": name, "steps": 24, "device": "cpu",
                                "host_definition": {"fixture": "synthetic"},
                                "prior": {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}},
                  "evaluation": {"kind": "synthetic_only", "sampling_contract_version": 1,
                                 "sampling_law": "public_prior_without_output_noise", "eval_output_noise": "clean",
                                 "scoring_weights": "live", "thresholds": [["synthetic_error", "<=", 1.]],
                                 "sources": {"tests/synthetic_evaluator.py": sources["tests/synthetic_evaluator.py"]}},
                  "resources": {"gpus": 0, "gpu_memory_mb": 0, "timeout_seconds": 1},
                  "requires_capabilities": ["live_sampling", "mog_prior"], "dependencies": []}
        parent_sha = write(tmp_path, f"configs/forge/tasks/{name}.json", parent)
        variant = policy._prospective_variant(parent, policy._parent_record(parent, parent_sha), sources)
        write(tmp_path, f"configs/forge/task-variants/{policy.COHORT}/{variant['id']}.json", variant)
        parents[name], variants[variant["id"]] = parent, variant
    common = {"schema_version": 1, "id": policy.PARENT_VIEW_ID, "goal": policy.PARENT_VIEW_ID,
              "revision": 3, "assignments": assignments, "eligibility": {},
              "fixture_scope": "synthetic_metadata_no_scientific_claim"}
    write(tmp_path, f"configs/forge/views/{policy.PARENT_VIEW_ID}.json", common)
    actual, _ = policy.resolve_policy_view(common, {**parents, **variants}, {"task_cohort": policy.COHORT})
    request = {"candidate": {"id": "synthetic-policy-row", "task_cohort": policy.COHORT,
                             "resolved_recipe": {"fixture_scope": "synthetic_only"}},
               "view": actual, "tasks": variants, "jobs": [], "protocol": {}, "rng": {}, "source": {}}
    statuses = ["PASS", "BLOCKED", "FAIL"] + ["NOT_RUN"] * 23
    original = {"candidate_id": "synthetic-policy-row", "status": "FAIL", "qualified_tier": 0,
                "runtime_cohort": {"execution_backend": "cuda"}, "cost": {},
                "qualification": {"view_revision": 4, "policy_fingerprint": stable_hash(actual),
                                  "tasks": [{"task_id": entry["task"], "status": status}
                                            for entry, status in zip(actual["assignments"], statuses)]},
                "scientific_bindings": technique_board.request_bindings(request),
                **technique_board.policy_row_metadata(request)}
    source_board = {"view": common["id"], "view_revision": 3, "policy_fingerprint": stable_hash(common),
                    "current_rows": [original], "rows": [], "conflicts": []}
    report = technique_board.reduce_board(source_board, common)
    report["publication_scope"] = "live_current"
    return tmp_path, report, report["rows"][0]


def validate(synthetic):
    root, report, row = synthetic
    return publication.validate_policy_publication(root, report, row)


def update_contract(report, row, name, change):
    refs = row["bindings"]["task_contracts"]
    contract = deepcopy(report["task_contracts"][refs[name]])
    change(contract)
    ref = stable_hash(contract)
    report["task_contracts"][ref] = contract
    refs[name] = ref


def test_synthetic_complete_metadata_is_read_only(synthetic):
    root, report, row = synthetic
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    before_metadata = deepcopy((report, row))
    assert validate(synthetic) is None
    assert (report, row) == before_metadata
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("status", ["PASS", "FAIL", "BLOCKED", "INCOMPLETE", "INVALID", "UNKNOWN"])
def test_synthetic_full_counts_preserve_each_outcome(synthetic, status):
    row = synthetic[2]
    for entry in row["tasks"]:
        entry.update(status=status, gate_status="NOT_RUN" if status == "UNKNOWN" else status)
    row["qualified_tier"] = 3 if status == "PASS" else 0
    for tier, total in (("1", 5), ("2", 19), ("3", 2)):
        row["tiers"][tier] = {"passed": total if status == "PASS" else 0, "total": total,
                               "counts": {status: total}}
    assert validate(synthetic) is None


@pytest.mark.parametrize("row", [{}, {"candidate_id": "atlas", "tasks": [{"task_id": "two_pole", "status": "PASS"}]},
                                {"candidate_id": "atlas", "evidence_scope": "historical", "recorded_counts": {"PASS": 19}}])
def test_ordinary_and_legacy_rows_need_no_sources_or_scientific_reconstruction(tmp_path, row, monkeypatch):
    monkeypatch.setattr(publication._Sources, "__init__", lambda *args: pytest.fail("ordinary rows must skip source reads"))
    assert publication.validate_policy_publication(tmp_path, {}, row) is None


@pytest.mark.parametrize("field", ["task_cohort", "qualification_view", "task_slot_map"])
def test_partial_policy_claim_is_rejected(synthetic, field):
    synthetic[2].pop(field)
    with pytest.raises(ValueError, match="policy publication"):
        validate(synthetic)


def test_stripping_outer_policy_metadata_cannot_hide_policy_contracts(synthetic):
    row = synthetic[2]
    for field in publication._POLICY_FIELDS:
        row.pop(field)
    for field in publication._BINDING_FIELDS:
        row["bindings"].pop(field, None)
    row["tasks"] = []
    with pytest.raises(ValueError, match="policy publication"):
        validate(synthetic)


@pytest.mark.parametrize("change", ["parent_hash", "execution_hash", "evaluation_hash", "override", "provenance", "budget"])
def test_rehashed_compact_contract_cannot_change_pinned_science(synthetic, change):
    _, report, row = synthetic
    name = "vector_two_broad" + policy.SUFFIX
    def tamper(contract):
        if change == "parent_hash":
            contract["policy_parent"]["task_sha256"] = "a" * 64
        elif change == "override":
            contract["policy_recipe_overrides"]["prior_reg"] = 0.
        elif change == "provenance":
            contract["policy_recipe_overrides_provenance"]["evidence_reuse"] = True
        elif change == "budget":
            contract["steps"] = 2
        else:
            contract[change.replace("_hash", "_sha256")] = "b" * 64
    update_contract(report, row, name, tamper)
    with pytest.raises(ValueError, match="compact task contract"):
        validate(synthetic)


@pytest.mark.parametrize("change", ["missing", "duplicate", "parent_id", "reorder", "map", "retier", "importance",
                                  "count", "denominator", "status", "attained_tier", "view_hash"])
def test_full_mappings_view_and_tier_counts_cannot_be_forged(synthetic, change):
    _, report, row = synthetic
    if change == "missing":
        row["bindings"]["task_contracts"].pop(next(iter(row["bindings"]["task_contracts"])))
    elif change == "duplicate":
        row["tasks"][1] = deepcopy(row["tasks"][0])
    elif change == "parent_id":
        row["tasks"][0]["task_id"] = "two_pole"
    elif change == "reorder":
        row["tasks"][0], row["tasks"][1] = row["tasks"][1], row["tasks"][0]
    elif change == "map":
        row["task_slot_map"]["two_pole" + policy.SUFFIX] = "unused_token_hold"
    elif change == "retier":
        row["qualification_view"]["assignments"][0]["qualification_tier"] = 2
    elif change == "importance":
        row["qualification_view"]["assignments"][0]["importance"] = "diagnostic"
    elif change == "count":
        row["tiers"]["1"]["counts"] = {"PASS": 5}
    elif change == "denominator":
        report["tier_requirements"]["1"].pop()
    elif change == "status":
        row["tasks"][1]["status"] = "PASS"
    elif change == "attained_tier":
        row["qualified_tier"] = 1
    else:
        report["policy_fingerprint"] = "c" * 64
    with pytest.raises(ValueError, match="policy publication"):
        validate(synthetic)


def test_live_parent_byte_drift_is_rejected_even_when_json_is_equal(synthetic):
    root, _, _ = synthetic
    path = root / "configs/forge/tasks/two_pole.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="parent byte"):
        validate(synthetic)


def test_coherent_variant_and_catalog_gate_change_cannot_rewrite_original_goal(synthetic):
    root, report, row = synthetic
    name = "two_pole" + policy.SUFFIX
    relative = f"configs/forge/task-variants/{policy.COHORT}/{name}.json"
    task = json.loads((root / relative).read_bytes())
    task["evaluation"]["thresholds"] = [["synthetic_error", "<=", 100.]]
    write(root, relative, task)
    update_contract(report, row, name, lambda contract: contract.update(publication._expected_contract(task)))
    row["qualification_view"]["cohort_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="policy variant changes"):
        validate(synthetic)


@pytest.mark.parametrize("relative", ["tests/synthetic_evaluator.py", "experiments/forge/policy_adapters.py"])
def test_live_evaluator_or_runtime_source_drift_blocks_publication(synthetic, relative):
    (synthetic[0] / relative).write_bytes(b"# changed source\n")
    with pytest.raises(ValueError, match="source binding drift"):
        validate(synthetic)


def frozen(synthetic, monkeypatch, *, missing=None):
    root, report, _ = synthetic
    contents = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    report.update(publication_scope="frozen_source", frozen_source={"commit": PIN})
    calls = []
    def git(args, **kwargs):
        assert args[:2] == ["git", "show"] and kwargs == {"cwd": str(root), "capture_output": True, "check": False}
        assert args[2].startswith(PIN + ":")
        relative = args[2][len(PIN) + 1:]
        calls.append(relative)
        return SimpleNamespace(returncode=1 if relative == missing else 0,
                               stdout=contents.get(relative, b""), stderr=b"synthetic unavailable")
    monkeypatch.setattr(publication.subprocess, "run", git)
    return calls


def test_frozen_reads_only_exact_git_bytes_and_caches_common_sources(synthetic, monkeypatch):
    calls = frozen(synthetic, monkeypatch)
    (synthetic[0] / "configs/forge/tasks/two_pole.json").write_bytes(b"invalid live JSON")
    (synthetic[0] / "experiments/forge/policy_adapters.py").unlink()
    assert validate(synthetic) is None
    count = len(calls)
    assert len(calls) == len(set(calls))
    assert validate(synthetic) is None and len(calls) == count


def test_unavailable_pinned_source_does_not_use_available_live_copy(synthetic, monkeypatch):
    missing = "configs/forge/tasks/two_pole.json"
    frozen(synthetic, monkeypatch, missing=missing)
    assert (synthetic[0] / missing).is_file()
    with pytest.raises(ValueError, match="pinned source is unavailable"):
        validate(synthetic)


def test_frozen_scope_cannot_drop_pin_and_become_live(synthetic):
    synthetic[1]["publication_scope"] = "frozen_source"
    with pytest.raises(ValueError, match="frozen source commit is missing"):
        validate(synthetic)


@pytest.mark.parametrize("commit", [None, "HEAD", "main", "1" * 12, "-x", "1" * 40 + ":bad", "1" * 39 + ";"])
def test_unsafe_or_implicit_frozen_commit_is_rejected_before_git(synthetic, monkeypatch, commit):
    synthetic[1].update(publication_scope="frozen_source", frozen_source={"commit": commit})
    monkeypatch.setattr(publication.subprocess, "run", lambda *args, **kwargs: pytest.fail("unsafe Git call"))
    with pytest.raises(ValueError, match="frozen source commit"):
        validate(synthetic)


@pytest.mark.parametrize("relative", ["../secret", "/absolute", "path/../file", "path//file", "./file", "x:y", "-option", "file\n", "a\\b"])
def test_unsafe_source_paths_are_rejected_before_read_or_git(relative, tmp_path, monkeypatch):
    monkeypatch.setattr(publication.subprocess, "run", lambda *args, **kwargs: pytest.fail("unsafe Git call"))
    with pytest.raises(ValueError, match="unsafe source path"):
        publication._git_bytes(str(tmp_path), PIN, relative)

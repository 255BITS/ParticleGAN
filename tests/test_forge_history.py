"""Historical import must preserve identity and fail closed on evidence gaps."""
from collections import Counter
import json
from pathlib import Path
import subprocess

import pytest

from experiments.forge import history


@pytest.fixture
def tiny_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "reports/toy100/example/README.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Example\nA reported success without a raw result.\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "reports"], check=True)
    return tmp_path


def test_coverage_catches_new_tracked_file_even_in_existing_family(tiny_repo):
    catalog = history.inventory(tiny_repo)
    assert history.validate_inventory(tiny_repo, catalog)["valid"]
    new = tiny_repo / "reports/toy100/example/new-probe.py"
    new.write_text("raise RuntimeError('must not execute during inventory')\n")
    subprocess.run(["git", "-C", str(tiny_repo), "add", str(new)], check=True)
    audit = history.validate_inventory(tiny_repo, catalog)
    assert not audit["valid"]
    assert audit["missing"] == ["reports/toy100/example/new-probe.py"]


def test_generated_outputs_never_recursively_enter_inventory(tiny_repo):
    for name in ("reports/forge/EXPERIMENT_MEMORY.md", "configs/forge/catalog.json",
                 "experiments/forge/history.py"):
        path = tiny_repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    subprocess.run(["git", "-C", str(tiny_repo), "add", "."], check=True)
    files = history.inventory(tiny_repo)["files"]
    assert len(files) == 2
    assert next(f for f in files if f["path"].startswith("experiments/forge/"))["role"] == "engine_support"


def test_unknown_family_is_explicit_and_fails_coverage(tiny_repo):
    path = tiny_repo / "configs/unknown-domain/idea.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    subprocess.run(["git", "-C", str(tiny_repo), "add", str(path)], check=True)
    result = history.inventory(tiny_repo)
    assert history.validate_inventory(tiny_repo, result)["unclassified"] == ["configs/unknown-domain/idea.json"]


def test_raw_error_is_incomplete_unless_capability_refusal_bound():
    row = {"status": "ERROR", "error": "worker died"}
    ordinary = history._task("grid100", row, {})
    blocked = history._task("two_pole", row, {}, capability_reason="unsupported _sigma_intrinsic_scale")
    assert ordinary["gate_status"] == "INCOMPLETE"
    assert blocked["gate_status"] == "BLOCKED"
    assert ordinary["raw_status"] == blocked["raw_status"] == "ERROR"


def test_ema_clean_and_negative_controls_do_not_replace_live_verdict():
    task = history._task("shift", {"status": "FAIL", "ema_final": {"passed": True},
                                   "clean_status": "PASS"}, {}, negative_control=True)
    assert task["gate_status"] == "FAIL"
    assert task["role"] == "negative_control"
    assert task["diagnostics"]["clean_status"] == "PASS"
    assert task["evidence_scope"] == "historical"


def test_nested_result_container_preserves_candidate_identity():
    data = {"arms": {"candidate-a": {"package_sha256": "abc", "gates": {
        "mode_hold": {"status": "PASS"}}, "native": {"grid100": {"status": "FAIL"}}}}}
    groups = history._extract_task_groups(data)
    assert len(groups) == 2
    assert {name for name, _, _ in groups} == {"candidate-a"}
    assert all(context["package_sha256"] == "abc" for _, _, context in groups)


def test_early_stop_and_not_run_remain_distinct():
    data = {"st6": {"grid100": {"harness_result_status": "ERROR", "record_kind": "early_stopped_diagnostic"},
                    "rotated100": {"record_kind": "not_run"}}}
    candidate, rows, _ = history._extract_task_groups(data)[0]
    tasks = [history._task(name, row, {}) for name, row in rows]
    assert candidate == "st6"
    assert [t["gate_status"] for t in tasks] == ["INCOMPLETE", "NOT_RUN"]


@pytest.fixture(scope="module")
def frozen_records():
    root = Path(__file__).resolve().parents[1]
    try:
        source = history.Source(root, history.LRFREE_REVISION, (history.LRFREE_ROOT,))
    except subprocess.CalledProcessError:
        pytest.skip("Pinned PR #155 commit is unavailable in this checkout")
    records, _, _ = history._import_structured(source)
    return source, records


def test_row_em_retains_all22_denominator_and_bound_refusals(frozen_records):
    _, records = frozen_records
    record = next(r for r in records if r["source"]["path"].endswith("row-em-renew/all22-summary.json"))
    assert len(record["task_results"]) == 22
    assert Counter(t["gate_status"] for t in record["task_results"]) == {"PASS": 10, "FAIL": 4, "BLOCKED": 8}
    assert Counter(t["raw_status"] for t in record["task_results"]) == {"PASS": 10, "FAIL": 4, "ERROR": 8}
    assert record["mechanism_class"] == "sampling_only_patch"
    assert record["provenance"]["package_sha256"] == "1fc12d29e2c5964f91c9523006e569cba743eba2e3e9851a68afa063c08cd324"
    assert not record["provenance"]["reuse_eligible"]


def test_native_import_retains_exact_prior_fixture_cost_and_sampling(frozen_records):
    _, records = frozen_records
    record = next(r for r in records if r["source"]["path"].endswith("row-em-renew/grid100-result.json"))
    task = record["task_results"][0]
    assert record["prior"]["kind"] == "particles"
    assert record["prior"]["sigma"] == 0
    assert record["claim_contract"]["sampling_law"]["eval_output_noise"] is True
    assert task["gate_status"] == "PASS"
    assert task["diagnostics"]["clean_status"] == "FAIL"
    assert task["cost"]["seconds"] == 1691.55
    support = record["provenance"]["context"]["supporting_receipts"]
    fixture = next(r for r in support if r["path"].endswith("-fixture.json"))
    assert fixture["content"]["initial"]["prior"]["z"]
    assert fixture["sha256"]


def test_structural_14k_retains_7k_prefix_and_failed_holdout(frozen_records):
    _, records = frozen_records
    record = next(r for r in records if r["candidate_id"] == "st5-scaleaware-qr-14k")
    task = record["task_results"][0]
    assert task["cost"]["steps"] == 14000
    assert task["gate_status"] == "FAIL"
    assert task["metrics"]["seven_k_prefix_verification"]["rates_raw_equal"]
    assert not task["metrics"]["native_gate"]["holdout_pass"]
    assert task["metrics"]["native_budgets"]["7000"]["noisy"]["status"] == "FAIL"
    assert record["provenance"]["context"]["declared_config"]["sha256"]


def test_gapfill_extension_cannot_hide_behind_hold_pass():
    root = Path(__file__).resolve().parents[1]
    try:
        source = history.Source(root, history.BASELINE_REVISION, history.SOURCE_ROOTS)
    except subprocess.CalledProcessError:
        pytest.skip("Pinned baseline commit is unavailable")
    cards, _ = history._import_gap_fill(source)
    record = next(c for c in cards if c["provenance"]["context"].get("attempt_id") == "k3g-hold-mode_hold-0.01-0.05")
    assert [(t["task_id"], t["gate_status"]) for t in record["task_results"]] == [
        ("mode_hold:hold", "PASS"), ("mode_hold:extension", "FAIL")]
    frozen = next(c for c in cards if c["provenance"]["context"].get("attempt_id") == "rg5-a2-shift_frozen-mode_hold-0.1-0.1")
    assert frozen["task_results"][0]["role"] == "negative_control"
    assert frozen["task_results"][0]["raw_status"] == "FAIL"


def test_atomic_output_is_deterministic(tmp_path):
    path = tmp_path / "record.json"
    history._write_json(path, {"b": 2, "a": 1})
    before = path.stat().st_mtime_ns
    history._write_json(path, {"a": 1, "b": 2})
    assert path.stat().st_mtime_ns == before
    assert json.loads(path.read_text()) == {"a": 1, "b": 2}
    assert not path.with_name("record.json.tmp").exists()

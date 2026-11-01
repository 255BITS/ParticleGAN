"""Unit and integration tests for single-command Forge readout refresh (ops/refresh_forge_readouts.py)."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.forge import attempt_coverage
from experiments.forge.contracts import atomic_json, read_json
from ops import refresh_forge_readouts as rfr


ROOT = Path(__file__).resolve().parents[1]


def test_record_to_state_job_losslessly_round_trips_new_attempt_through_coverage_value():
    published_record = {
        "key": "abc123",
        "family": "atlas",
        "candidate_id": "atlas-particle-normal-v4",
        "candidate_revision": "efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd",
        "source_digest": "2e8a97a0aa845c961de00d403fdcd1673b34b33626e4546c3138882a40e7bacf",
        "task": "mode_hold",
        "execution_sha256": "exec-sha",
        "evaluation_sha256": "eval-sha",
        "status": "PASS",
        "attempted": True,
        "origin": "new_attempt",
        "diagnostic": True,
        "evidence": "reports/forge/attempt-coverage.json",
        "original_attempt": "/tmp/runs/forge/attempt-coverage-refresh/attempts/abc123/result.json",
        "updates": 4000,
    }
    retained_record = {
        "key": "def456",
        "family": "atlas",
        "candidate_id": "atlas-particle-normal-v4",
        "candidate_revision": "efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd",
        "source_digest": "2e8a97a0aa845c961de00d403fdcd1673b34b33626e4546c3138882a40e7bacf",
        "task": "gaussian1d_smoke",
        "execution_sha256": "exec-sha-2",
        "evaluation_sha256": "eval-sha-2",
        "status": "PASS",
        "attempted": True,
        "origin": "retained_selected_row",
        "diagnostic": False,
        "evidence": "reports/forge/technique-inventory.json",
    }

    state = {
        "jobs": {
            "abc123": rfr._record_to_state_job(published_record),
            "def456": rfr._record_to_state_job(retained_record),
        }
    }
    recomputed = attempt_coverage.coverage_value(state)
    by_key = {row["key"]: row for row in recomputed["records"]}

    assert by_key["abc123"] == published_record
    assert (
        by_key["abc123"]["original_attempt"]
        == "/tmp/runs/forge/attempt-coverage-refresh/attempts/abc123/result.json"
    )
    assert by_key["def456"]["evidence"] == "reports/forge/technique-inventory.json"
    assert "original_attempt" not in by_key["def456"]


def test_check_queue_status_handles_missing_active_valid_awaiting_and_stale_dangling(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        rfr,
        "catalog_source_manifest",
        lambda root, idea=None: {"digest": "live-digest-0001"},
    )

    # 1. Empty / non-existent queue root is current.
    empty_status = rfr.check_queue_status(tmp_path, tmp_path / "runs/forge")
    assert empty_status == {
        "current": True,
        "active_submissions": [],
        "awaiting_readout_submissions": [],
    }

    # 2. Active submission or valid live-source awaiting_readout marks queue non-current without error.
    state_path = tmp_path / "runs/forge/queue/state.json"
    atomic_json(
        state_path,
        {
            "jobs": {"job-1": {"result": {"task_results": []}}},
            "submissions": {
                "req-active": {
                    "status": "running",
                    "lifecycle": "running",
                    "request": {
                        "candidate": {"id": "cand-a"},
                        "source": {"digest": "live-digest-0001"},
                        "jobs": [],
                    },
                },
                "req-awaiting": {
                    "status": "completed",
                    "lifecycle": "awaiting_readout",
                    "request": {
                        "candidate": {"id": "cand-b"},
                        "source": {"digest": "live-digest-0001"},
                        "jobs": [{"compatibility_key": "job-1"}],
                    },
                },
            },
        },
    )
    busy_status = rfr.check_queue_status(tmp_path, tmp_path / "runs/forge")
    assert busy_status["current"] is False
    assert len(busy_status["active_submissions"]) == 1
    assert len(busy_status["awaiting_readout_submissions"]) == 1

    # 3. Superseded-source awaiting_readout with zero durable results raises CodeChangeRequiredError.
    atomic_json(
        state_path,
        {
            "jobs": {},
            "submissions": {
                "req-dangling": {
                    "status": "completed",
                    "lifecycle": "awaiting_readout",
                    "request": {
                        "candidate": {"id": "cand-stale"},
                        "source": {"digest": "superseded-digest-9999"},
                        "jobs": [{"compatibility_key": "missing-job"}],
                    },
                }
            },
        },
    )
    with pytest.raises(rfr.CodeChangeRequiredError, match="superseded source"):
        rfr.check_queue_status(tmp_path, tmp_path / "runs/forge")


def test_retry_shared_gpu_timeouts_retries_only_single_attempt_worktree_timeouts(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        rfr,
        "_request_matches_worktree",
        lambda root, req: req.get("source", {}).get("digest") == "live-src",
    )
    queue_root = tmp_path / "runs/forge"
    atomic_json(
        queue_root / "queue/state.json",
        {
            "jobs": {
                "job-timeout-1": {
                    "status": "terminal",
                    "attempts": [{"attempt_id": "a1"}],
                    "result": {"raw": {"attempt_status": "timeout"}},
                },
                "job-timeout-2": {
                    "status": "terminal",
                    "attempts": [{"attempt_id": "a1"}, {"attempt_id": "a2"}],
                    "result": {"raw": {"attempt_status": "timeout"}},
                },
                "job-error-1": {
                    "status": "terminal",
                    "attempts": [{"attempt_id": "a1"}],
                    "result": {"raw": {"attempt_status": "error"}},
                },
                "job-stale-1": {
                    "status": "terminal",
                    "attempts": [{"attempt_id": "a1"}],
                    "result": {"raw": {"attempt_status": "timeout"}},
                },
            },
            "submissions": {
                "sub-live": {
                    "lifecycle": "awaiting_readout",
                    "request": {
                        "candidate": {"id": "recipe-prior-refactor-bcap-pure-v1"},
                        "source": {"digest": "live-src"},
                        "jobs": [
                            {"compatibility_key": "job-timeout-1", "task_id": "five_word_joint_smoke"},
                            {"compatibility_key": "job-timeout-2", "task_id": "grid100"},
                            {"compatibility_key": "job-error-1", "task_id": "two_pole"},
                        ],
                    },
                },
                "sub-stale": {
                    "lifecycle": "awaiting_readout",
                    "request": {
                        "candidate": {"id": "old-cand"},
                        "source": {"digest": "stale-src"},
                        "jobs": [{"compatibility_key": "job-stale-1", "task_id": "five_word_joint_smoke"}],
                    },
                },
            },
        },
    )

    retried_calls: list[tuple[str, str]] = []

    class _FakeQueue:
        def retry(self, key: str, *, reason: str):
            retried_calls.append((key, reason))

    retried = rfr._retry_shared_gpu_timeouts(_FakeQueue(), queue_root, tmp_path, json_mode=True)
    assert retried == ["job-timeout-1"]
    assert len(retried_calls) == 1
    assert retried_calls[0][0] == "job-timeout-1"
    assert "workers_per_gpu=1" in retried_calls[0][1]


def test_check_scaling_performance_detects_missing_corrupt_multicohort_and_stale_source(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        rfr,
        "current_family_candidates",
        lambda root: [
            {"family": "atlas", "trainer_family": "atlas", "candidate_id": "atlas-particle-normal-v4"},
            {
                "family": "bcap-pure",
                "trainer_family": "bcap-dualnorm",
                "candidate_id": "recipe-prior-refactor-bcap-pure-v1",
            },
        ],
    )
    monkeypatch.setattr(
        rfr,
        "load_view",
        lambda root, name: {"assignments": [{"task": "dcgan_performance_r32"}]},
    )
    live_files_sha = {"particlegan/__init__.py": "sha-live"}
    monkeypatch.setattr(rfr, "scaling_source_files_sha256", lambda root: live_files_sha)

    # 1. Missing manifest -> needs refresh (not an exception).
    monkeypatch.setattr(rfr.scaling_publication, "load_performance", lambda root: None)
    missing = rfr.check_scaling_performance(tmp_path)
    assert missing["current"] is False
    assert missing["reason"] == "missing_manifest"
    assert missing["missing_cells"] == 2

    # 2. Invalid manifest -> raises CodeChangeRequiredError.
    def _bad_load(root: Path):
        raise ValueError("corrupt performance manifest")

    monkeypatch.setattr(rfr.scaling_publication, "load_performance", _bad_load)
    with pytest.raises(rfr.CodeChangeRequiredError, match="invalid scaling-performance.json"):
        rfr.check_scaling_performance(tmp_path)

    # 3. Complete single-cohort manifest with inventory trainer_family ('bcap-dualnorm') -> current=True.
    def _make_record(fam: str, cid: str, gpu_uuid: str, files_sha: dict[str, str], status: str = "COMPLETE"):
        return {
            "trainer_family": fam,
            "candidate_id": cid,
            "task_id": "dcgan_performance_r32",
            "result": {
                "status": status,
                "identity": {
                    "source": {"sha256": "src-sha-1", "files_sha256": files_sha},
                    "runtime_sha256": "rt-sha-1",
                    "hardware": {"backend": "cuda", "gpu": {"uuid": gpu_uuid, "name": "RTX A6000"}},
                },
            },
        }

    monkeypatch.setattr(rfr.scaling_publication, "performance_status", lambda res: res["status"])
    monkeypatch.setattr(
        rfr.scaling_publication,
        "hardware_key",
        lambda res: res["identity"]["hardware"],
    )

    good_manifest = {
        "unavailable": [],
        "records": [
            _make_record("atlas", "atlas-particle-normal-v4", "GPU-0", live_files_sha),
            _make_record("bcap-dualnorm", "recipe-prior-refactor-bcap-pure-v1", "GPU-0", live_files_sha),
        ],
    }
    monkeypatch.setattr(rfr.scaling_publication, "load_performance", lambda root: good_manifest)
    good = rfr.check_scaling_performance(tmp_path)
    assert good["current"] is True
    assert good["table_cohorts"] == 1

    # 4. Split across GPU-0 and GPU-1 -> table_cohorts == 2 -> current=False (prevents UNKNOWN table split).
    split_manifest = {
        "unavailable": [],
        "records": [
            _make_record("atlas", "atlas-particle-normal-v4", "GPU-0", live_files_sha),
            _make_record("bcap-dualnorm", "recipe-prior-refactor-bcap-pure-v1", "GPU-1", live_files_sha),
        ],
    }
    monkeypatch.setattr(rfr.scaling_publication, "load_performance", lambda root: split_manifest)
    split = rfr.check_scaling_performance(tmp_path)
    assert split["current"] is False
    assert split["table_cohorts"] == 2

    # 5. Stale code file digest -> stale_source_cells == 1 -> current=False.
    stale_manifest = {
        "unavailable": [],
        "records": [
            _make_record("atlas", "atlas-particle-normal-v4", "GPU-0", {"particlegan/__init__.py": "old"}),
            _make_record("bcap-dualnorm", "recipe-prior-refactor-bcap-pure-v1", "GPU-0", live_files_sha),
        ],
    }
    monkeypatch.setattr(rfr.scaling_publication, "load_performance", lambda root: stale_manifest)
    stale = rfr.check_scaling_performance(tmp_path)
    assert stale["current"] is False
    assert stale["stale_source_cells"] == 1


def test_selected_family_preserved_absent_paths_reads_snapshot_closure(tmp_path: Path):
    snap = tmp_path / "reports/forge/technique-evidence/snap.json"
    atomic_json(
        snap,
        {
            "frozen_catalog_support_closure": {
                "preserved_absent_paths": ["reports/transfer_suite/unadjusted/leading_profile.json"]
            }
        },
    )
    pub = {"evidence_sources": {"pk-1": {"snapshot": "reports/forge/technique-evidence/snap.json"}}}
    paths = rfr._selected_family_preserved_absent_paths(tmp_path, pub, {"publication_key": "pk-1"})
    assert paths == ["reports/transfer_suite/unadjusted/leading_profile.json"]
    assert rfr._selected_family_preserved_absent_paths(tmp_path, pub, {"publication_key": "missing"}) == []


def test_pin_checkpoint_pair_gpus_and_expected_preflight_refusals(tmp_path: Path, monkeypatch):
    # 0. _frozen_preflight_blockers forwards PYTHONPATH=frozen_checkout and parses JSON output.
    captured_env: dict[str, str] = {}

    class _FakeCompleted:
        stdout = '{"misgan_block_recovery": ["misgan_block_recovery: unsupported vector target declaration"]}\n'

    def _fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return _FakeCompleted()

    monkeypatch.setattr(rfr.subprocess, "run", _fake_run)
    parsed_blockers = rfr._frozen_preflight_blockers(tmp_path, "atlas-particle-normal-v4")
    assert captured_env.get("PYTHONPATH") == str(tmp_path)
    assert parsed_blockers == {
        "misgan_block_recovery": ["misgan_block_recovery: unsupported vector target declaration"]
    }

    # 1. Fresh PENDING producer -> consumer -> grandchild chain pins all three to pair_gpus[0].
    fresh_jobs: dict[str, dict[str, Any]] = {
        "k1": {"family": "atlas", "task": "gaussian1d_smoke", "status": "PENDING", "dependencies": []},
        "k2": {
            "family": "atlas",
            "task": "gaussian1d_stability",
            "status": "PENDING",
            "dependencies": [{"kind": "checkpoint", "task": "gaussian1d_smoke"}],
        },
        "k3": {
            "family": "atlas",
            "task": "gaussian1d_stage3",
            "status": "PENDING",
            "dependencies": [{"kind": "checkpoint", "task": "gaussian1d_stability"}],
        },
    }
    rfr._pin_checkpoint_pair_gpus(fresh_jobs, "atlas", ["0", "1"])
    assert fresh_jobs["k1"]["required_gpu"] == "0"
    assert fresh_jobs["k2"]["required_gpu"] == "0"
    assert fresh_jobs["k3"]["required_gpu"] == "0"

    # 2. Completed producer on gpu='1' pins PENDING consumer to '1' without mutating completed producer.
    resumed_jobs: dict[str, dict[str, Any]] = {
        "k1": {"family": "atlas", "task": "gaussian1d_smoke", "status": "PASS", "gpu": "1", "dependencies": []},
        "k2": {
            "family": "atlas",
            "task": "gaussian1d_stability",
            "status": "PENDING",
            "dependencies": [{"kind": "checkpoint", "task": "gaussian1d_smoke"}],
        },
    }
    rfr._pin_checkpoint_pair_gpus(resumed_jobs, "atlas", ["0", "1"])
    assert "required_gpu" not in resumed_jobs["k1"]
    assert resumed_jobs["k2"]["required_gpu"] == "1"

    # 3. Certified queue parent with device='cuda:1' pins PENDING consumer to '1'.
    cert_jobs: dict[str, dict[str, Any]] = {
        "k1": {
            "family": "atlas",
            "task": "grid100",
            "status": "PASS",
            "certified_parent": {"result": {"device": "cuda:1"}},
            "dependencies": [],
        },
        "k2": {
            "family": "atlas",
            "task": "grid100_14k",
            "status": "PENDING",
            "dependencies": [{"kind": "checkpoint", "task": "grid100"}],
        },
    }
    rfr._pin_checkpoint_pair_gpus(cert_jobs, "atlas", ["0", "1"])
    assert cert_jobs["k2"]["required_gpu"] == "1"

    # 4. Cross-GPU RNG mismatch vs static preflight blocker recognition.
    rng_err = {
        "family": "atlas",
        "task": "gaussian1d_stability",
        "status": "ERROR",
        "reason": "ValueError: trainer/named RNG checkpoint mismatch",
    }
    preflight_err = {
        "family": "atlas",
        "task": "misgan_block_recovery",
        "status": "ERROR",
        "reason": "ValueError: unsupported vector target declaration",
    }
    preflight_map = {
        ("atlas", "misgan_block_recovery"): ["misgan_block_recovery: unsupported vector target declaration"],
        ("atlas", "gaussian1d_stability"): [],
    }
    assert rfr._is_cross_gpu_rng_mismatch(rng_err) is True
    assert rfr._is_cross_gpu_rng_mismatch(preflight_err) is False
    assert rfr._is_expected_preflight_refusal(preflight_err, preflight_map) is True
    assert rfr._is_expected_preflight_refusal(rng_err, preflight_map) is False


def test_check_attempt_coverage_flags_superseded_selected_records_and_ignores_scoped_records(
    tmp_path: Path, monkeypatch
):
    pub_path = tmp_path / "reports/forge/technique-inventory.json"
    cov_path = tmp_path / "reports/forge/attempt-coverage.json"

    selection = {
        "candidate_id": "atlas-particle-normal-v4",
        "candidate_revision": "rev-v4",
        "publication_key": "pub-atlas",
        "bindings": {"source_digest": "src-v4"},
    }
    atomic_json(
        pub_path,
        {"evidence_sources": {"pub-atlas": {"source_commit": "deadbeef"}}},
    )
    monkeypatch.setattr(
        rfr.attempt_coverage,
        "selected_tasks",
        lambda pub: [("atlas", selection, ["two_pole", "mode_hold"])],
    )
    monkeypatch.setattr(
        rfr,
        "_git_task_names_at_commit",
        lambda root, commit: {"two_pole", "mode_hold"},
    )

    # Old superseded non-scoped record ('atlas' instead of 'atlas-particle-normal-v4') + missing 'mode_hold'.
    atomic_json(
        cov_path,
        {
            "summary": {"pending": 0, "running": 0},
            "records": [
                {
                    "family": "atlas",
                    "candidate_id": "atlas",
                    "candidate_revision": "rev-old",
                    "source_digest": "src-old",
                    "task": "two_pole",
                    "attempted": True,
                    "status": "BLOCKED",
                },
                {
                    "family": "research:legacy",
                    "research_scope": "seed0_leaderboard_research",
                    "candidate_id": "old-research",
                    "candidate_revision": "rev-res",
                    "source_digest": "src-res",
                    "task": "two_pole",
                    "attempted": True,
                    "status": "FAIL",
                },
            ],
        },
    )
    stale = rfr.check_attempt_coverage(tmp_path)
    assert stale["current"] is False
    assert stale["superseded_record_count"] == 1
    assert stale["missing_by_family"] == {
        "atlas": {
            "candidate_id": "atlas-particle-normal-v4",
            "missing_tasks": ["mode_hold", "two_pole"],
        }
    }

    # Replace superseded record with current v4 records covering both tasks while keeping scoped research record.
    atomic_json(
        cov_path,
        {
            "summary": {"pending": 0, "running": 0},
            "records": [
                {
                    "family": "atlas",
                    "candidate_id": "atlas-particle-normal-v4",
                    "candidate_revision": "rev-v4",
                    "source_digest": "src-v4",
                    "task": "two_pole",
                    "attempted": True,
                    "status": "FAIL",
                },
                {
                    "family": "atlas",
                    "candidate_id": "atlas-particle-normal-v4",
                    "candidate_revision": "rev-v4",
                    "source_digest": "src-v4",
                    "task": "mode_hold",
                    "attempted": True,
                    "status": "PASS",
                },
                {
                    "family": "research:legacy",
                    "research_scope": "seed0_leaderboard_research",
                    "candidate_id": "old-research",
                    "candidate_revision": "rev-res",
                    "source_digest": "src-res",
                    "task": "two_pole",
                    "attempted": True,
                    "status": "FAIL",
                },
            ],
        },
    )
    fresh = rfr.check_attempt_coverage(tmp_path)
    assert fresh["current"] is True
    assert fresh["superseded_record_count"] == 0
    assert fresh["missing_by_family"] == {}


def test_check_selected_family_readouts_handles_cohort_dedup_unknown_boards_and_task_errors(
    tmp_path: Path, monkeypatch
):
    rev = "77765918f0b8e49975abeef0068a12cef5c452594aa5d8f94003f0eaedfd3f52"
    fake_req = {
        "candidate_revision": rev,
        "view": {
            "assignments": [
                {"task": "gaussian1d_smoke", "qualification_tier": 1, "importance": "required"},
                {"task": "two_pole", "qualification_tier": 1, "importance": "required"},
                {"task": "mode_hold", "qualification_tier": 2, "importance": "required"},
            ]
        },
        "tasks": {
            "gaussian1d_smoke": {"preflight_blockers": [], "dependencies": []},
            "two_pole": {"preflight_blockers": [], "dependencies": []},
            "mode_hold": {"preflight_blockers": [], "dependencies": []},
        },
    }
    monkeypatch.setattr(
        rfr,
        "current_family_candidates",
        lambda root: [{"family": "bcap-pure", "candidate_id": "recipe-prior-refactor-bcap-pure-v1"}],
    )
    monkeypatch.setattr(rfr.planning, "resolve_idea", lambda *args, **kwargs: deepcopy(fake_req))

    # 1. No readout record yet -> missing_families populated.
    initial = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert initial["current"] is False
    assert len(initial["missing_families"]) == 1

    # 2. Concluded readout exists (FAIL on two_pole stops Tier 2), and cohort-deduplicated leaderboard rows exist.
    readout_path = tmp_path / "reports/forge/records/readout-test.json"
    atomic_json(
        readout_path,
        {
            "lifecycle": "concluded",
            "evidence_scope": "current",
            "candidate_id": "recipe-prior-refactor-bcap-pure-v1",
            "candidate_revision": rev,
            "goal": "discriminator_stability",
            "study": {"scope": {"view": "discriminator_stability", "execution_backend": "cuda"}},
            "task_results": [
                {"task_id": "gaussian1d_smoke", "gate_status": "PASS"},
                {"task_id": "two_pole", "gate_status": "FAIL"},
            ],
        },
    )
    lb_dir = tmp_path / "reports/forge/leaderboards"
    lb_dir.mkdir(parents=True, exist_ok=True)
    for view_id in rfr.ORDINARY_LEADERBOARD_VIEWS:
        (lb_dir / f"{view_id}.md").write_text(
            f"| bcap-prior-coulomb-control-v1 / {rev[:12]} / cuda (NVIDIA RTX A6000) | current | concluded | FAIL | Tier 0 |\n"
        )
    measured = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert measured["current"] is True

    # 3. An '| unknown |' status on one goal leaderboard flags stale_leaderboards.
    (lb_dir / "adaptation.md").write_text(
        f"| bcap-prior-coulomb-control-v1 / {rev[:12]} / cuda (NVIDIA RTX A6000) | current | concluded | unknown | Tier 0 |\n"
    )
    stale_lb = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert stale_lb["current"] is False
    assert stale_lb["stale_leaderboards"] == [
        {
            "family": "bcap-pure",
            "candidate_id": "recipe-prior-refactor-bcap-pure-v1",
            "view": "adaptation",
        }
    ]

    # 3b. A retried task containing both INCOMPLETE (attempt 1) and PASS (attempt 2) is accepted.
    for view_id in rfr.ORDINARY_LEADERBOARD_VIEWS:
        (lb_dir / f"{view_id}.md").write_text(
            f"| bcap-prior-coulomb-control-v1 / {rev[:12]} / cuda (NVIDIA RTX A6000) | current | concluded | FAIL | Tier 0 |\n"
        )
    atomic_json(
        readout_path,
        {
            "lifecycle": "concluded",
            "evidence_scope": "current",
            "candidate_id": "recipe-prior-refactor-bcap-pure-v1",
            "candidate_revision": rev,
            "goal": "discriminator_stability",
            "study": {"scope": {"view": "discriminator_stability", "execution_backend": "cuda"}},
            "task_results": [
                {"task_id": "gaussian1d_smoke", "gate_status": "INCOMPLETE"},
                {"task_id": "gaussian1d_smoke", "gate_status": "PASS"},
                {"task_id": "two_pole", "gate_status": "FAIL"},
            ],
        },
    )
    retried_ok = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert retried_ok["current"] is True

    # 4. Unresolved ERROR / INCOMPLETE in current-source readout raises CodeChangeRequiredError.
    atomic_json(
        readout_path,
        {
            "lifecycle": "concluded",
            "evidence_scope": "current",
            "candidate_id": "recipe-prior-refactor-bcap-pure-v1",
            "candidate_revision": rev,
            "goal": "discriminator_stability",
            "study": {"scope": {"view": "discriminator_stability", "execution_backend": "cuda"}},
            "task_results": [
                {"task_id": "gaussian1d_smoke", "gate_status": "ERROR"},
            ],
        },
    )
    with pytest.raises(rfr.CodeChangeRequiredError, match="has ERROR on gaussian1d_smoke"):
        rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")


def test_main_cli_exit_codes_check_noop_reports_only_and_code_change_required(
    tmp_path: Path, monkeypatch, capsys
):
    current_payload: dict[str, Any] = {
        "status": "CURRENT",
        "source_digest": "2e8a97a0aa84",
        "stale_stages": [],
        "stages": {
            "queue": {"current": True},
            "scaling_performance": {"current": True},
            "attempt_coverage": {"current": True},
            "selected_family_readouts": {"current": True},
            "compiled_reports": {"current": True},
        },
    }
    monkeypatch.setattr(rfr, "check_status", lambda *args, **kwargs: deepcopy(current_payload))

    assert rfr.main(["--root", str(tmp_path), "--check", "--json"]) == rfr.CURRENT_EXIT
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["status"] == "CURRENT"

    assert rfr.main(["--root", str(tmp_path), "--json"]) == rfr.CURRENT_EXIT
    noop = json.loads(capsys.readouterr().out)
    assert noop["action"] == "NOOP"

    stale_gpu_payload = deepcopy(current_payload)
    stale_gpu_payload["status"] = "NEEDS_REFRESH"
    stale_gpu_payload["stale_stages"] = ["scaling_performance"]
    stale_gpu_payload["stages"]["scaling_performance"] = {"current": False}
    monkeypatch.setattr(rfr, "check_status", lambda *args, **kwargs: deepcopy(stale_gpu_payload))
    monkeypatch.setattr(rfr, "check_compiled_reports", lambda root: {"current": True})

    assert rfr.main(["--root", str(tmp_path), "--check"]) == rfr.NEEDS_REFRESH_EXIT
    capsys.readouterr()
    assert rfr.main(["--root", str(tmp_path), "--reports-only", "--json"]) == rfr.NEEDS_REFRESH_EXIT
    partial = json.loads(capsys.readouterr().out)
    assert partial["action"] == "REPORTS_ONLY_PARTIAL"

    def _boom(*args, **kwargs):
        raise rfr.CodeChangeRequiredError("simulated adapter failure")

    monkeypatch.setattr(rfr, "check_status", _boom)
    assert rfr.main(["--root", str(tmp_path), "--check", "--json"]) == rfr.CODE_CHANGE_EXIT
    err_payload = json.loads(capsys.readouterr().out)
    assert err_payload["status"] == "CODE_CHANGE_REQUIRED"
    assert "simulated adapter failure" in err_payload["error"]


def test_live_repository_forge_readouts_are_current_and_source_clean():
    report = rfr.check_status(ROOT)
    assert report["status"] == "CURRENT", json.dumps(report, indent=2, sort_keys=True)
    assert report["stale_stages"] == []
    assert report["stages"]["scaling_performance"]["complete_cells"] == 48
    assert report["stages"]["scaling_performance"]["missing_cells"] == 0
    assert report["stages"]["scaling_performance"]["unavailable_cells"] == 0
    assert report["stages"]["scaling_performance"]["table_cohorts"] == 1
    assert report["stages"]["attempt_coverage"]["pending"] == 0
    assert report["stages"]["attempt_coverage"]["superseded_record_count"] == 0
    assert report["stages"]["attempt_coverage"]["missing_by_family"] == {}
    assert report["stages"]["selected_family_readouts"]["missing_families"] == []
    assert report["stages"]["selected_family_readouts"]["stale_leaderboards"] == []
    assert report["stages"]["compiled_reports"]["unknown_performance_rows"] == 0
    assert report["stages"]["compiled_reports"]["selected_diagnostic_rows_present"] is True
    assert report["stages"]["compiled_reports"]["clean_checkout_safe"] is True
    assert rfr.main(["--root", str(ROOT), "--check"]) == rfr.CURRENT_EXIT


def test_clean_checkout_preserves_leaderboards_and_repairs_missing_attempts(
    tmp_path: Path, monkeypatch
):
    from reports.forge import regenerate_technique_inventory

    monkeypatch.setattr(
        regenerate_technique_inventory,
        "refresh_publication",
        lambda root: {"report": str(Path(root) / "reports/forge/technique-inventory.md")},
    )
    monkeypatch.setattr(rfr, "_ensure_tier_report_markdown", lambda root: None)
    monkeypatch.setattr(rfr, "_ensure_attempt_coverage_markdown", lambda root, json_mode=False: None)
    monkeypatch.setattr(rfr.history, "validate_inventory", lambda root: {"valid": True})

    compile_calls: list[bool] = []

    def _fake_compile_memory(root: Path, *, summaries_only: bool = False):
        compile_calls.append(bool(summaries_only))
        if not summaries_only:
            lb_path = Path(root) / "reports/forge/leaderboards/adaptation.md"
            lb_path.parent.mkdir(parents=True, exist_ok=True)
            lb_path.write_text("WIPED_WITHOUT_ATTEMPTS\n")
        return {"summaries_only": bool(summaries_only)}

    monkeypatch.setattr(rfr.knowledge, "compile_memory", _fake_compile_memory)

    lb_file = tmp_path / "reports/forge/leaderboards/adaptation.md"
    lb_file.parent.mkdir(parents=True, exist_ok=True)
    lb_file.write_text("COMMITTED_LEADERBOARD_CONTENT\n")

    # 1. Clean checkout (local attempt envelopes absent) skips summaries_only=False and preserves leaderboards.
    monkeypatch.setattr(
        rfr,
        "check_selected_family_readouts",
        lambda root, *, cuda_model=rfr.DEFAULT_CUDA_MODEL: {
            "current": True,
            "missing_families": [],
            "stale_leaderboards": [],
            "local_attempts_missing_families": [
                {"family": "atlas", "candidate_id": "atlas-particle-normal-v4", "candidate_revision": "rev"}
            ],
        },
    )
    res_clean = rfr.refresh_compiled_reports(tmp_path)
    assert res_clean["leaderboards_recompiled"] is False
    assert compile_calls == [True]
    assert lb_file.read_text() == "COMMITTED_LEADERBOARD_CONTENT\n"

    # 2. Local attempt envelopes present -> runs full compile_memory(summaries_only=False) then summaries_only=True.
    compile_calls.clear()
    monkeypatch.setattr(
        rfr,
        "check_selected_family_readouts",
        lambda root, *, cuda_model=rfr.DEFAULT_CUDA_MODEL: {
            "current": True,
            "missing_families": [],
            "stale_leaderboards": [],
            "local_attempts_missing_families": [],
        },
    )
    res_full = rfr.refresh_compiled_reports(tmp_path)
    assert res_full["leaderboards_recompiled"] is True
    assert compile_calls == [False, True]

    # 3. Stale leaderboard on clean checkout causes refresh_selected_family_readouts to target families
    #    missing local attempt envelopes so board() will not wipe them.
    resolved_cids: list[str] = []
    monkeypatch.setattr(
        rfr,
        "check_selected_family_readouts",
        lambda root, *, cuda_model=rfr.DEFAULT_CUDA_MODEL: {
            "current": False,
            "missing_families": [],
            "stale_leaderboards": [
                {"family": "atlas", "candidate_id": "atlas-particle-normal-v4", "view": "adaptation"}
            ],
            "local_attempts_missing_families": [
                {"family": "atlas", "candidate_id": "atlas-particle-normal-v4", "candidate_revision": "rev1"},
                {"family": "bispf", "candidate_id": "bispf-optimized-quintic-v1", "candidate_revision": "rev2"},
            ],
        },
    )
    monkeypatch.setattr(rfr, "catalog_source_manifest", lambda root, idea=None: {"digest": "2e8a97a0aa84" * 5 + "0000"})

    def _fake_resolve(root, cid, **kwargs):
        resolved_cids.append(cid)
        return {
            "request_id": f"req-{cid}",
            "candidate_revision": f"rev-{cid}",
            "policy_fingerprint": "pf-1",
            "preflight_blockers": [],
            "source": {"digest": "2e8a97a0aa84" * 5 + "0000"},
            "jobs": [{"compatibility_key": f"ck-{cid}", "budget_seconds": 60}],
        }

    monkeypatch.setattr(rfr.planning, "resolve_idea", _fake_resolve)
    monkeypatch.setattr(
        rfr,
        "_ensure_ordinary_requal_study",
        lambda root, fam, cid, base_req, campaign, **kw: {"id": f"study-{fam}", "campaign": deepcopy(campaign)},
    )
    monkeypatch.setattr(rfr, "snapshot_source", lambda root, qroot, src: Path(qroot) / "snap")
    monkeypatch.setattr(
        rfr.forge_queue,
        "Queue",
        lambda *args, **kwargs: type(
            "_Q",
            (),
            {"submit": lambda self, req, camp: {"request": req, "status": "queued"}},
        )(),
    )
    monkeypatch.setattr(rfr.forge_queue, "drain", lambda *args, **kwargs: None)
    monkeypatch.setattr(rfr, "_conclude_queue_submissions", lambda *args, **kwargs: ["readout-1", "readout-2"])

    concluded = rfr.refresh_selected_family_readouts(
        tmp_path, tmp_path / "runs/forge", gpus=["0", "1"], cuda_model="NVIDIA RTX A6000", json_mode=True
    )
    assert concluded == ["readout-1", "readout-2"]
    assert sorted(set(resolved_cids)) == ["atlas-particle-normal-v4", "bispf-optimized-quintic-v1"]


def test_compiled_reports_dynamic_diagnostics_unregistered_selection_and_markdown_sync(
    tmp_path: Path, monkeypatch
):
    # 1. Dynamic diagnostic heading computation works for any promoted successor (e.g. atlas-particle-normal-v5).
    live_inv = read_json(ROOT / "reports/forge/technique-inventory.json")
    mutated_inv = deepcopy(live_inv)
    for row in mutated_inv.get("rows", []):
        if row.get("candidate_id") == "atlas-particle-normal-v4":
            row["candidate_id"] = "atlas-particle-normal-v5"
    for fam in mutated_inv.get("family_progress", {}).get("families", []):
        if fam.get("id") == "atlas":
            for cohort in fam.get("cohorts", []):
                if cohort.get("candidate_id") == "atlas-particle-normal-v4":
                    cohort["candidate_id"] = "atlas-particle-normal-v5"

    phrases = rfr._expected_selected_diagnostic_phrases(ROOT, mutated_inv)
    assert "Selected recipe 1-for-1 baseline-35 diagnostics (atlas-particle-normal-v5); seed 0" in phrases
    assert "Selected recipe declared-suite Tier 1–3 diagnostics (atlas-particle-normal-v5); seed 0" in phrases
    assert not any("atlas-particle-normal-v4" in p for p in phrases)

    # 2. Unregistered family-current-v1.json modification raises CodeChangeRequiredError (exit code 2).
    sel_rel = rfr.CURRENT_SELECTION
    atomic_json(tmp_path / sel_rel, {"selections": []})
    inv_doc = {"provenance": {"family_current_selection_sha256": "different-registered-sha"}}
    with pytest.raises(rfr.CodeChangeRequiredError, match="differs from reports/forge/technique-inventory.json"):
        rfr._validate_publication_registration_sync(tmp_path, inv_doc)

    # 3. Missing/corrupted ATTEMPT_COVERAGE.md is detected and regenerated cleanly from attempt-coverage.json.
    cov_sample = {
        "schema_version": 1,
        "maximum_tests": 4,
        "maximum_per_gpu": 2,
        "records": [
            {
                "key": "k-atlas-1",
                "family": "atlas",
                "label": "Atlas (atlas-particle-normal-v4; current selected)",
                "candidate_id": "atlas-particle-normal-v4",
                "candidate_revision": "efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd",
                "source_digest": "2e8a97a0aa845c961de00d403fdcd1673b34b33626e4546c3138882a40e7bacf",
                "task": "gaussian1d_smoke",
                "execution_sha256": "e1",
                "evaluation_sha256": "v1",
                "status": "PASS",
                "attempted": True,
                "origin": "retained_selected_row",
                "diagnostic": False,
                "evidence": "reports/forge/technique-inventory.json",
            }
        ],
    }
    atomic_json(tmp_path / "reports/forge/attempt-coverage.json", cov_sample)
    assert rfr._attempt_coverage_markdown_synced(tmp_path, cov_sample) is False
    rfr._ensure_attempt_coverage_markdown(tmp_path, json_mode=True)
    assert (tmp_path / "reports/forge/ATTEMPT_COVERAGE.md").is_file()
    assert rfr._attempt_coverage_markdown_synced(tmp_path, cov_sample) is True


def test_requal_study_and_campaign_avoid_id_collisions_when_candidate_or_policy_changes(
    tmp_path: Path, monkeypatch
):
    src_digest = "2e8a97a0aa845c961de00d403fdcd1673b34b33626e4546c3138882a40e7bacf"
    src12 = src_digest[:12]

    # 1. Campaign creation & collision avoidance when base campaign is already registered in queue state.
    camp1 = rfr._select_or_create_requal_campaign(
        tmp_path, {}, src12=src12, batch_sig="abcd1234", max_allowance=1200, family_count=2
    )
    assert camp1["id"] == f"stale-family-requal-{src12}"

    camp2 = rfr._select_or_create_requal_campaign(
        tmp_path,
        {"campaigns": {camp1["id"]: camp1}},
        src12=src12,
        batch_sig="abcd1234",
        max_allowance=1200,
        family_count=2,
    )
    assert camp2["id"] == f"stale-family-requal-{src12}-abcd1234"

    # 2. Study creation & collision avoidance when legacy study ID already exists for an older candidate or concluded run.
    def _make_req(cid: str, rev: str) -> dict[str, Any]:
        return {
            "candidate_revision": rev,
            "policy_fingerprint": "pf-v14",
            "source": {"digest": src_digest},
            "view": {
                "id": "discriminator_stability",
                "assignments": [{"task": "gaussian1d_smoke", "qualification_tier": 1}],
            },
            "tasks": {
                "gaussian1d_smoke": {
                    "preflight_blockers": [],
                    "evaluation": {"thresholds": [["abs_mean_error", "<=", 0.05]]},
                }
            },
            "jobs": [{"compatibility_key": f"ck-{cid}-{rev}"}],
        }

    req_v1 = _make_req("bispf-v1", "rev-1111")
    study1 = rfr._ensure_ordinary_requal_study(
        tmp_path,
        "bispf",
        "bispf-v1",
        req_v1,
        camp1,
        cuda_model="NVIDIA RTX A6000",
        through_tier=4,
        q_state={},
        concluded_records=[],
    )
    assert study1["id"] == f"requal-{src12}-bispf-ordinary"
    assert study1["scope"]["through_tier"] == 4

    # Promote bispf-optimized-quintic-v1 at the exact same source digest (no code changes):
    # must mint a fresh bound study ID rather than returning study1 for bispf-v1.
    req_v2 = _make_req("bispf-optimized-quintic-v1", "rev-2222")
    study2 = rfr._ensure_ordinary_requal_study(
        tmp_path,
        "bispf",
        "bispf-optimized-quintic-v1",
        req_v2,
        camp2,
        cuda_model="NVIDIA RTX A6000",
        through_tier=4,
        q_state={},
        concluded_records=[],
    )
    assert study2["id"] != study1["id"]
    assert study2["id"].startswith(f"requal-{src12}-bispf-")
    assert study2["candidate"] == "bispf-optimized-quintic-v1"
    assert study2["campaign"]["id"] == camp2["id"]


def test_check_selected_family_readouts_enforces_compat_key_cuda_model_and_tier3_unlocking(
    tmp_path: Path, monkeypatch
):
    rev = "efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd"
    fake_req = {
        "candidate_revision": rev,
        "view": {
            "assignments": [
                {"task": "gaussian1d_smoke", "qualification_tier": 1, "importance": "required"},
                {"task": "mode_hold", "qualification_tier": 2, "importance": "required"},
                {"task": "ring_hold", "qualification_tier": 3, "importance": "required"},
                {"task": "grid100", "qualification_tier": 2, "importance": "held"},
                {"task": "grid100_14k", "qualification_tier": 3, "importance": "held"},
            ]
        },
        "tasks": {
            "gaussian1d_smoke": {"preflight_blockers": [], "dependencies": []},
            "mode_hold": {"preflight_blockers": [], "dependencies": []},
            "ring_hold": {"preflight_blockers": [], "dependencies": []},
            "grid100": {"preflight_blockers": [], "dependencies": []},
            "grid100_14k": {
                "preflight_blockers": [],
                "dependencies": [{"kind": "checkpoint", "task": "grid100"}],
            },
        },
        "jobs": [
            {"task_id": "gaussian1d_smoke", "task_ids": ["gaussian1d_smoke"], "compatibility_key": "ck-t1-new"},
            {"task_id": "mode_hold", "task_ids": ["mode_hold"], "compatibility_key": "ck-t2-mode"},
            {"task_id": "grid100", "task_ids": ["grid100"], "compatibility_key": "ck-t2-grid"},
            {"task_id": "ring_hold", "task_ids": ["ring_hold"], "compatibility_key": "ck-t3-ring"},
            {"task_id": "grid100_14k", "task_ids": ["grid100_14k"], "compatibility_key": "ck-t3-grid14k"},
        ],
    }
    monkeypatch.setattr(
        rfr,
        "current_family_candidates",
        lambda root: [{"family": "atlas", "candidate_id": "atlas-particle-normal-v4"}],
    )
    monkeypatch.setattr(rfr.planning, "resolve_idea", lambda *args, **kwargs: deepcopy(fake_req))

    readout_path = tmp_path / "reports/forge/records/readout-atlas.json"

    # 1. Mismatched compatibility_key or wrong cuda_model is ignored -> Tier 1 remains unmeasured.
    atomic_json(
        readout_path,
        {
            "lifecycle": "concluded",
            "evidence_scope": "current",
            "candidate_id": "atlas-particle-normal-v4",
            "candidate_revision": rev,
            "goal": "discriminator_stability",
            "study": {
                "scope": {
                    "view": "discriminator_stability",
                    "execution_backend": "cuda",
                    "cuda_model": "NVIDIA RTX A6000",
                }
            },
            "task_results": [
                {"task_id": "gaussian1d_smoke", "compatibility_key": "ck-t1-STALE", "gate_status": "PASS"}
            ],
        },
    )
    stale_ck = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert stale_ck["current"] is False
    assert stale_ck["missing_families"][0]["unmeasured_tier1"] == ["gaussian1d_smoke"]

    # 2. Tier 1 required task PASS unlocks Tier 2 tasks (mode_hold and grid100).
    atomic_json(
        readout_path,
        {
            "lifecycle": "concluded",
            "evidence_scope": "current",
            "candidate_id": "atlas-particle-normal-v4",
            "candidate_revision": rev,
            "goal": "discriminator_stability",
            "study": {
                "scope": {
                    "view": "discriminator_stability",
                    "execution_backend": "cuda",
                    "cuda_model": "NVIDIA RTX A6000",
                }
            },
            "task_results": [
                {"task_id": "gaussian1d_smoke", "compatibility_key": "ck-t1-new", "gate_status": "PASS"}
            ],
        },
    )
    t2_unlocked = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert t2_unlocked["current"] is False
    assert t2_unlocked["missing_families"][0]["unmeasured_tier1"] == []
    assert sorted(t2_unlocked["missing_families"][0]["unmeasured_tier2"]) == ["grid100", "mode_hold"]

    # 3. Tier 1 + Tier 2 required tasks PASS (while held grid100 FAILs) unlocks Tier 3 ring_hold,
    #    while checkpoint-dependent grid100_14k is recognized as dependency-blocked.
    atomic_json(
        readout_path,
        {
            "lifecycle": "concluded",
            "evidence_scope": "current",
            "candidate_id": "atlas-particle-normal-v4",
            "candidate_revision": rev,
            "goal": "discriminator_stability",
            "study": {
                "scope": {
                    "view": "discriminator_stability",
                    "execution_backend": "cuda",
                    "cuda_model": "NVIDIA RTX A6000",
                }
            },
            "task_results": [
                {"task_id": "gaussian1d_smoke", "compatibility_key": "ck-t1-new", "gate_status": "PASS"},
                {"task_id": "mode_hold", "compatibility_key": "ck-t2-mode", "gate_status": "PASS"},
                {"task_id": "grid100", "compatibility_key": "ck-t2-grid", "gate_status": "FAIL"},
            ],
        },
    )
    t3_unlocked = rfr.check_selected_family_readouts(tmp_path, cuda_model="NVIDIA RTX A6000")
    assert t3_unlocked["current"] is False
    assert t3_unlocked["missing_families"][0]["unmeasured_by_tier"] == {"3": ["ring_hold"]}


def test_scaling_campaign_reuse_guard_and_revision_bound_readout_conclusion(
    tmp_path: Path, monkeypatch
):
    # 1. Existing scaling registration with already-published receipts and empty queue state is NOT overwritten.
    monkeypatch.setattr(
        rfr,
        "current_family_candidates",
        lambda root: [{"family": "atlas", "candidate_id": "atlas-particle-normal-v4"}],
    )
    monkeypatch.setattr(
        rfr,
        "load_view",
        lambda root, name: {"assignments": [{"task": "dcgan_performance_r32"}]},
    )
    live_sha = {"particlegan/__init__.py": "sha-new"}
    monkeypatch.setattr(rfr, "scaling_source_files_sha256", lambda root: live_sha)
    monkeypatch.setattr(
        rfr,
        "catalog_source_manifest",
        lambda root, idea=None: {"digest": "feedface1234567890abcdef"},
    )

    perf_dir = tmp_path / "configs/forge/performance"
    perf_dir.mkdir(parents=True, exist_ok=True)
    (perf_dir / "dcgan-performance-cloud-v3.json").write_text("{}\n")
    old_manifest = {
        "id": "dcgan-performance-cloud-v3",
        "view": "dcgan_performance",
        "execution_backend": "cuda",
        "cuda_model": "NVIDIA RTX A6000",
        "task_ids": ["dcgan_performance_r32"],
        "family_selections": [{"family": "atlas", "candidate_id": "atlas-particle-normal-v4"}],
        "campaign": {"id": "dcgan-performance-cloud-v3"},
        "studies": ["dcgan-performance-cloud-v3-atlas"],
    }
    atomic_json(
        tmp_path / "reports/forge/scaling-receipts/dcgan-performance-cloud-v3/cell-01.json",
        {"result": {"identity": {"source": {"files_sha256": live_sha}}}},
    )

    registered_ids: list[str] = []

    def _fake_load(root: Path, reg_id: str):
        if reg_id == "dcgan-performance-cloud-v3":
            return deepcopy(old_manifest)
        return {**deepcopy(old_manifest), "id": reg_id, "campaign": {"id": reg_id}}

    monkeypatch.setattr(rfr.scaling, "_load", _fake_load)
    monkeypatch.setattr(
        rfr.scaling,
        "register",
        lambda root, reg_id, **kwargs: registered_ids.append(reg_id),
    )

    chosen = rfr._select_or_register_scaling_campaign(
        tmp_path, tmp_path / "runs/forge", cuda_model="NVIDIA RTX A6000", json_mode=True
    )
    assert chosen["id"] == "dcgan-performance-cloud-feedface1234"
    assert registered_ids == ["dcgan-performance-cloud-feedface1234"]

    # 2. _conclude_queue_submissions binds candidate_ref as '<cid>@<candidate_revision>'.
    cand_rev = "efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd"
    q_root = tmp_path / "runs/forge"
    atomic_json(
        q_root / "queue/state.json",
        {
            "jobs": {
                "job-1": {"result": {"task_results": [{"task_id": "dcgan_performance_r32", "gate_status": "PASS"}]}}
            },
            "submissions": {
                "sub-1": {
                    "lifecycle": "awaiting_readout",
                    "request": {
                        "campaign_id": "camp-1",
                        "candidate": {"id": "atlas-particle-normal-v4"},
                        "candidate_revision": cand_rev,
                        "source": {"digest": "feedface1234567890abcdef"},
                        "view": {"id": "dcgan_performance"},
                        "jobs": [{"compatibility_key": "job-1"}],
                    },
                }
            },
        },
    )
    seen_refs: list[str] = []
    monkeypatch.setattr(
        rfr.knowledge,
        "readout",
        lambda root, ref, **kwargs: (seen_refs.append(ref) or {"record_id": "readout-new"}),
    )
    recs = rfr._conclude_queue_submissions(tmp_path, q_root, campaign_id="camp-1", json_mode=True)
    assert recs == ["readout-new"]
    assert seen_refs == [f"atlas-particle-normal-v4@{cand_rev}"]


def test_frozen_checkout_missing_commit_and_empty_gpu_flags_raise_code_change_required(
    tmp_path: Path, monkeypatch
):
    # 1. Missing frozen git commit raises CodeChangeRequiredError and leaves no partial checkout directory.
    missing_commit = "0000000000000000000000000000000000000000"
    checkout_root = tmp_path / "frozen-checkouts"
    with pytest.raises(rfr.CodeChangeRequiredError, match="git archive failed"):
        rfr._ensure_frozen_checkout(ROOT, missing_commit, checkout_root)
    assert not (checkout_root / missing_commit).exists()

    # 2. Empty --gpus list raises CodeChangeRequiredError when GPU stages are stale instead of defaulting to ['0','1'].
    monkeypatch.setattr(
        rfr,
        "check_status",
        lambda *args, **kwargs: {
            "status": "NEEDS_REFRESH",
            "source_digest": "2e8a97a0aa84",
            "stale_stages": ["scaling_performance"],
            "stages": {
                "queue": {"current": True},
                "scaling_performance": {"current": False},
                "attempt_coverage": {"current": True},
                "selected_family_readouts": {"current": True},
                "compiled_reports": {"current": True},
            },
        },
    )
    with pytest.raises(rfr.CodeChangeRequiredError, match="at least one GPU index is required"):
        rfr.refresh_all(tmp_path, gpus=[], json_mode=True)
    with pytest.raises(rfr.CodeChangeRequiredError, match="non-empty --scaling-gpu"):
        rfr.refresh_scaling_performance(
            tmp_path, tmp_path / "runs/forge", scaling_gpu="  ", cuda_model="NVIDIA RTX A6000", json_mode=True
        )


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

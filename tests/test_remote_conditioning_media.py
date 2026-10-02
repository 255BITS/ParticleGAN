"""Synthetic publication controls only; no training, archives or Git history."""

import hashlib
import json
from copy import deepcopy

import pytest
from PIL import Image

from benchmarks.toy_audit import remote_conditioning_media as media


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def contrast(total, error):
    return {
        "pair_MSE": total,
        "midpoint_loss": total - error,
        "conditional_error": error,
        "conditional_error_over_V": error,
        "alpha": 0.0,
        "predicted_contrast_power_over_V": 0.0,
        "target_contrast_power_over_V": 1.0,
        "decomposition_identity_error": 0.0,
        "marker_learned": error <= 0.5,
        "prediction_sha256": "a" * 64,
        "public_native_and_caller_replay_exact": True,
    }


def software_fixture(root, monkeypatch):
    """Numerical/byte fixtures are software controls and grant no science credit."""
    runs, decomposition = root / "raw", root / "diagnostic"
    runs.mkdir(parents=True)
    decomposition.mkdir()
    protocol = {
        "external_steps_per_arm": 200,
        "external_total_seconds": 60,
        "threads": 1,
        "arms": {
            "G16": {"D_batch": 16, "G_batch": 16},
            "G64": {"D_batch": 16, "G_batch": 64},
        },
        "source_hashes": {"software-source.py": "a" * 64},
        "package_git_sha": "d" * 40,
    }
    write(runs / "protocol.json", protocol)
    (runs / "source.py").write_text("# synthetic source; no models or trainer\n")
    arms = {}
    for name, values in (("G16", (10.0, 2.0, 1.0)), ("G64", (10.0, 2.02, 0.99))):
        points = []
        for step, value in zip(media.MEDIA_STEPS, values):
            point = {
                "live_mask_error": value,
                "live_full_error": value,
                "served_mask_error": value,
                "served_source": "fast",
                "codeblind_mask_lower_bound": 1.0,
                "metric": "software control only",
            }
            if step == 200:
                point["paired_decomposition"] = contrast(
                    value, 1.0 if name == "G16" else 0.98
                )
            else:
                point["paired_decomposition"] = contrast(value, 1.0)
            points.append(point)
        arms[name] = {
            "steps": 200,
            "initial": points[0],
            "evaluations": {"100": points[1], "200": points[2]},
            "health": {
                "finite_steps": 200,
                "min_dense_rows": 128,
                "ka2_applied_calls": 200,
                "ownership": True,
                "frozen_host": True,
                "critic_score_active_gradients": True,
                "generator_live_gradients": True,
            },
            "control": {
                "rows": {"counters": {"updates": 200}},
                "counters": {"evals": 1, "probes": 1},
            },
        }
        rows = [
            {
                "step": step,
                "d_batch": 16,
                "g_batch": int(name[1:]),
                "dense_rows": 128,
                "loss_g": 1.0,
                "loss_d": 1.0,
                "penalty": 0.1,
                "sigma": 1.3,
                "quadratic_weights": [-0.01, -0.02],
                "caller_panel_sha256": hashlib.sha256(str(step).encode()).hexdigest(),
            }
            for step in range(1, 201)
        ]
        (runs / (name + ".jsonl")).write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    checks = media.metric_checks(arms, 1.0)
    checks.update(
        {
            key: True
            for key in (
                "source_exact",
                "teacher_qualified",
                "initial_owners_match",
                "caller_streams_match",
                "fixed400_complete",
                "within60s",
                "G16_native_health",
                "G64_native_health",
            )
        }
    )
    result = {
        "failure": None,
        "status": "failed",
        "committed_updates": {"G16": 200, "G64": 200},
        "elapsed_seconds": 1.0,
        "teacher_nonlocal_variance": 1.0,
        "teacher_hash": "a" * 64,
        "fixture_hash": "b" * 64,
        "calibration": {
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
            "law": "software control only",
        },
        "identity": {
            "checkout_git_sha": "c" * 40,
            "reference_package_git_sha": "d" * 40,
            "identity_guard": media.IDENTITY_GUARD,
            "verified_source_files": len(protocol["source_hashes"]),
        },
        "arms": arms,
        "checks": checks,
    }
    write(runs / "result.json", result)
    (decomposition / "decompose.py").write_text(
        "# synthetic saved observation algebra\n"
    )
    write(decomposition / "protocol.json", {"software_fixture": True})
    diagnostic = {
        "status": "failed",
        "arms": {
            name: deepcopy(arm["evaluations"]["200"]["paired_decomposition"])
            for name, arm in arms.items()
        },
        "native_updates": 0,
        "optimizer_steps": 0,
        "autograd_calls": 0,
        "predictor_forwards": 9,
        "scientific_training": False,
        "V": 1.0,
        "elapsed_seconds": 0.1,
        "source_sha256": media.sha(decomposition / "decompose.py"),
        "protocol_sha256": media.sha(decomposition / "protocol.json"),
    }
    write(decomposition / "result.json", diagnostic)
    card = {
        key: deepcopy(result[key])
        for key in (
            "teacher_nonlocal_variance",
            "fixture_hash",
            "teacher_hash",
            "calibration",
        )
    }
    card.update(
        arms={
            name: {key: deepcopy(arm[key]) for key in ("initial", "evaluations")}
            for name, arm in arms.items()
        },
        saved_endpoint_paired_decomposition=deepcopy(diagnostic),
        provenance={
            "original_scientific_protocol_sha256": media.sha(runs / "protocol.json"),
            "original_scientific_driver_sha256": media.sha(runs / "source.py"),
            "original_result_sha256": media.sha(runs / "result.json"),
            "saved_endpoint_decomposition_result_sha256": media.sha(
                decomposition / "result.json"
            ),
        },
    )
    # The production source checker is strict; synthetic fixtures supply a fixed
    # oracle so no historical commit, library or retained local archive is needed.
    monkeypatch.setattr(
        media,
        "active_sources",
        lambda: {
            "protocol_sha256": media.sha(runs / "protocol.json"),
            "source_hashes": {media.DRIVER: media.sha(runs / "source.py")},
        },
    )
    return runs, decomposition, card, result


def test_retained_points_remain_distinct_from_endpoint_diagnostic(
    tmp_path, monkeypatch
):
    runs, decomposition, card, _ = software_fixture(tmp_path, monkeypatch)
    inputs = media.Inputs()
    data = media.verify(runs, card, inputs, decomposition=decomposition)
    assert data["scientific_status"] == "FAIL" and data["actual_updates"] == [
        0,
        100,
        200,
    ]
    assert set(data["failed_original_checks"]) == {
        "G64_better_at100",
        "G64_better_at200",
        "G16_beats_codeblind_bound",
        "G64_beats_codeblind_bound",
    }
    assert data["contrast_available_steps"] == [200] and inputs.unchanged()
    assert all(
        item["attestation"] == "original_public_receipt"
        for path, item in inputs.files.items()
        if not path.endswith(".jsonl")
    )
    assert all(
        item["attestation"] == "review_time_only"
        for path, item in inputs.files.items()
        if path.endswith(".jsonl")
    )


@pytest.mark.parametrize(
    "control,match",
    [
        ("partial_budget", "partial"),
        ("partial_trace", "coverage"),
        ("wrong_batch", "coverage"),
        ("missing_endpoint", "checkpoints"),
        ("nonfinite_loss", "nonfinite"),
        ("nonfinite_health", "nonfinite"),
        ("missing_weights", "channel weights"),
        ("different_caller", "caller histories"),
        ("changed_teacher", "teacher"),
        ("wrong_variance", "evaluation units"),
        ("faked_pass", "verdict"),
        ("nonfinite_contrast", "nonfinite"),
        ("false_contrast_ratio", "contrast arithmetic"),
        ("changed_budget", "protocol"),
        ("missing_identity", "source identity"),
        ("changed_package_identity", "source identity"),
        ("changed_source_count", "source identity"),
        ("invalid_checkout_identity", "source identity"),
        ("changed_identity_guard", "source identity"),
    ],
)
def test_fresh_controls_reject_without_immutable_hash_masking(
    tmp_path, monkeypatch, control, match
):
    runs, _, card, result = software_fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (runs / "G64.jsonl").read_text().splitlines()]
    if control == "partial_budget":
        result["committed_updates"]["G64"] = 199
    elif control == "partial_trace":
        rows.pop()
    elif control == "wrong_batch":
        rows[0]["g_batch"] = 16
    elif control == "missing_endpoint":
        result["arms"]["G64"]["evaluations"].pop("200")
    elif control == "nonfinite_loss":
        rows[0]["loss_g"] = float("nan")
    elif control == "nonfinite_health":
        result["arms"]["G64"]["health"]["energy"] = float("nan")
    elif control == "missing_weights":
        rows[0]["quadratic_weights"] = []
    elif control == "different_caller":
        rows[0]["caller_panel_sha256"] = "f" * 64
    elif control == "changed_teacher":
        result["teacher_hash"] = "f" * 64
    elif control == "wrong_variance":
        result["arms"]["G64"]["initial"]["codeblind_mask_lower_bound"] = 2.0
    elif control == "faked_pass":
        result["status"] = "passed"
    elif control == "nonfinite_contrast":
        result["arms"]["G64"]["evaluations"]["200"]["paired_decomposition"]["alpha"] = (
            float("nan")
        )
    elif control == "false_contrast_ratio":
        result["arms"]["G64"]["evaluations"]["200"]["paired_decomposition"][
            "conditional_error_over_V"
        ] = 0.1
    elif control == "changed_budget":
        protocol = json.loads((runs / "protocol.json").read_text())
        protocol["external_steps_per_arm"] = 199
        write(runs / "protocol.json", protocol)
    elif control == "missing_identity":
        result["identity"] = {}
    elif control == "changed_package_identity":
        result["identity"]["reference_package_git_sha"] = "f" * 40
    elif control == "changed_source_count":
        result["identity"]["verified_source_files"] = 0
    elif control == "invalid_checkout_identity":
        result["identity"]["checkout_git_sha"] = ""
    elif control == "changed_identity_guard":
        result["identity"]["identity_guard"] = "unverified"
    write(runs / "result.json", result)
    (runs / "G64.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match=match):
        media.verify(runs, card, media.Inputs(), fresh=True)


def test_fresh_rejects_borrowed_original_diagnostic(tmp_path, monkeypatch):
    runs, decomposition, card, _ = software_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="cannot borrow"):
        media.verify(
            runs, card, media.Inputs(), fresh=True, decomposition=decomposition
        )


def test_independent_positive_metric_control(tmp_path, monkeypatch):
    runs, _, card, result = software_fixture(tmp_path, monkeypatch)
    for name, arm in result["arms"].items():
        value = 0.4 if name == "G16" else 0.3
        for point in arm["evaluations"].values():
            point.update(
                live_mask_error=value,
                live_full_error=value,
                served_mask_error=value,
                paired_decomposition=contrast(value, value),
            )
    result["checks"].update(media.metric_checks(result["arms"], 1.0))
    result["status"] = "passed"
    write(runs / "result.json", result)
    data = media.verify(runs, card, media.Inputs(), fresh=True)
    assert data["scientific_status"] == "PASS" and not data["failed_original_checks"]
    assert data["contrast_available_steps"] == [0, 100, 200]


def test_export_leaves_all_raw_bytes_and_metrics_unchanged(tmp_path, monkeypatch):
    runs, decomposition, card, _ = software_fixture(tmp_path / "inputs", monkeypatch)
    fake_root = tmp_path / "repo"
    write(fake_root / media.CARD, card)
    monkeypatch.setattr(media, "ROOT", fake_root)
    monkeypatch.setattr(media, "CARD_SHA", media.sha(fake_root / media.CARD))
    monkeypatch.setattr(media, "load_card", lambda: deepcopy(card))
    monkeypatch.setattr(
        media,
        "exporter_source",
        lambda: {"commit": "e" * 40, "files_sha256": {"software.py": "a" * 64}},
    )
    before = {
        str(path): media.sha(path)
        for path in (tmp_path / "inputs").rglob("*")
        if path.is_file()
    }
    out = tmp_path / "media"
    receipt = media.export(runs, out, decomposition=decomposition)
    assert (
        receipt["scientific_status"] == "FAIL" and receipt["original_inputs_unchanged"]
    )
    assert (
        receipt["training_updates"]
        == receipt["model_forwards"]
        == receipt["new_draws"]
        == 0
    )
    assert (
        not receipt["metric_rescoring"] and not receipt["model_image_arrays_available"]
    )
    assert before == {
        str(path): media.sha(path)
        for path in (tmp_path / "inputs").rglob("*")
        if path.is_file()
    }
    with Image.open(out / "goal.gif") as gif:
        assert gif.n_frames == 3 and gif.size == (1150, 800)
    assert receipt["goal_gif"]["actual_updates"] == [0, 100, 200]
    with pytest.raises(ValueError, match="new output"):
        media.export(runs, out, decomposition=decomposition)


def test_partial_export_creates_no_media_directory(tmp_path, monkeypatch):
    runs, _, card, result = software_fixture(tmp_path / "inputs", monkeypatch)
    result["committed_updates"]["G64"] = 199
    write(runs / "result.json", result)
    fake_root = tmp_path / "repo"
    write(fake_root / media.CARD, card)
    monkeypatch.setattr(media, "ROOT", fake_root)
    monkeypatch.setattr(media, "CARD_SHA", media.sha(fake_root / media.CARD))
    monkeypatch.setattr(media, "load_card", lambda: deepcopy(card))
    monkeypatch.setattr(media, "exporter_source", lambda: {"commit": "e" * 40})
    out = tmp_path / "media"
    with pytest.raises(ValueError, match="partial"):
        media.export(runs, out, fresh=True)
    assert not out.exists()


def test_source_manifest_oracle_checks_actual_file_bytes(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "examples").mkdir(parents=True)
    source = root / media.DRIVER
    source.write_text("# public caller software control\n")
    protocol = {"source_hashes": {media.DRIVER: media.sha(source)}}
    write(root / media.PROTOCOL, protocol)
    monkeypatch.setattr(media, "ROOT", root)
    monkeypatch.setattr(media, "PROTOCOL_SHA", media.sha(root / media.PROTOCOL))
    assert media.active_sources()["source_hashes"][media.DRIVER] == media.sha(source)
    source.write_text("# drift\n")
    with pytest.raises(ValueError, match="source drift"):
        media.active_sources()


def test_exporter_commit_binding_uses_exact_current_file_bytes(monkeypatch):
    def git(args, **kwargs):
        if args[1] == "rev-parse":
            return "e" * 40 + "\n"
        assert args[:2] == ["git", "show"] and args[2].startswith("e" * 40 + ":")
        return (media.ROOT / args[2].split(":", 1)[1]).read_bytes()

    monkeypatch.setattr(media.subprocess, "check_output", git)
    proof = media.exporter_source()
    assert len(proof["files_sha256"]) == 2 and proof["commit"] == "e" * 40
    monkeypatch.setattr(
        media.subprocess,
        "check_output",
        lambda args, **kwargs: (
            "e" * 40 + "\n" if args[1] == "rev-parse" else b"wrong blob"
        ),
    )
    with pytest.raises(ValueError, match="exact exporter"):
        media.exporter_source()

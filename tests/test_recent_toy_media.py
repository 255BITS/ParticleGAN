"""Synthetic media/protocol controls: no historical Git objects or training."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from benchmarks.toy_audit import recent_toy_media as media


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    media.write(path, value)


def film_archive(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    names = ["test_only_additive_time_columns_change_and_code_path_remains_live[4]",
             "test_only_additive_time_columns_change_and_code_path_remains_live[16]",
             "test_code_preserved_checkpoint_replays_exactly_and_refuses_whole_zero_owner"]
    (root / "contracts.xml").write_text('<testsuites><testsuite>' + ''.join(
        f'<testcase classname="tests.test_routed_conditioning_code_preserved" name="{name}"/>'
        for name in names) + '</testsuite></testsuites>')
    save(root / "campaign-protocol.json", {"scope": "synthetic software control"})
    save(root / "readout.json", {"scope": "synthetic software control"})
    card = {"protocol_sha256": media.sha(root / "campaign-protocol.json"),
            "readout_sha256": media.sha(root / "readout.json"),
            "contracts": {"junit_sha256": media.sha(root / "contracts.xml")}, "source_sha256": {}, "widths": {}}
    for width in (4, 16):
        folder = root / f"width{width}"
        folder.mkdir()
        (folder / "spatial_damping.py").write_text('# Synthetic software archive, not a trained experiment.\n')
        h = media.sha(folder / "spatial_damping.py")
        card["source_sha256"]["benchmarks/routed_conditioning/spatial_damping.py"] = h
        protocol = {"profiles": list(media.PROFILES), "external_update_cap": 1200,
                    "width": width, "blocks": 1, "source_sha256": {"spatial_damping.py": h}}
        save(folder / "protocol.json", protocol)
        profiles = {}
        card["widths"][str(width)] = {"profiles": {}}
        for index, arm in enumerate(media.PROFILES):
            evaluations = [{"step": step, "live_mse": 1 / (1 + step) + .001 * index,
                            "served_mse": 1 / (1 + step) + .001 * index} for step in range(0, 1201, 100)]
            profiles[arm] = {"completed_steps": 1200, "live_mse": evaluations[-1]["live_mse"],
                "served_mse": evaluations[-1]["served_mse"], "served_source": "fast", "evaluations": evaluations}
            metadata = {"initial_code_jacobian_frobenius": (2, 1, 2)[index],
                "initial_hashes": {r: "shared-synthetic" for r in ("critic", "prior", "encoder", "router", "table")},
                "data_hashes": "synthetic-law", "recipe": {"name": "software control"},
                "additive_initialization": "zero time columns and bias; retain code columns" if index == 2 else "synthetic"}
            save(folder / f"{arm}-metadata.json", metadata)
            rows = [{"evaluation": evaluations[0]}]
            for step in range(1, 1201):
                row = {"step": step, "dense_gradient_rows": 128, "loss_g": .7, "loss_d": .7, "penalty": .1}
                if step % 100 == 0: row["evaluation"] = evaluations[step // 100]
                rows.append(row)
            (folder / f"{arm}.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (folder / f"{arm}-final.pt").write_bytes(b'Synthetic checkpoint identity, never training')
            card["widths"][str(width)]["profiles"][arm] = {"clean_curve": {str(r["step"]): r["live_mse"] for r in evaluations}}
        save(folder / "summary.json", {"protocol": protocol, "profiles": profiles})
    return root, card


def variance_archive(tmp_path):
    path = tmp_path / "probe.json"
    stats = {component: {role: {"ratio": ratio, "single_variance": 1., "antithetic_variance": ratio,
             "batch_statistics": [{"ratio": ratio, "single_variance": 1., "antithetic_variance": ratio}] * 2}
            for role in media.ROLES} for component, ratio in zip(media.COMPONENTS, (.98, .03))}
    save(path, {"native_updates": 0, "pairs_per_batch": 16, "batches": 2, "sigma": .125,
               "fixture": "routed-dv12", "ownership_checks": "PASS", "state_unchanged_before_restore": True,
               "statistics": stats, "pass": False, "variance_result": "FAIL"})
    return path, {"portable_fixed_fixture": {"first_receipt_sha256": media.sha(path)}}


def clean_archive(tmp_path):
    root = tmp_path / "clean-raw"
    root.mkdir()
    curves = {arm: {} for arm in ("native", "G_clean")}
    for arm in curves:
        for step in range(0, 513, 64):
            game = 2. - step * .001 - (step * .0006 / 512 if arm == "G_clean" else 0)
            frame = {c: {j: game for j in media.JUDGES} for c in ("clean", "DV12")}
            if step == 512: frame["zero_code"] = {j: game + .02 for j in media.JUDGES}
            curves[arm][str(step)] = frame
    final = curves["G_clean"]["512"]
    gate = {"pass": True, "threshold": -1e-4, "code_threshold": 1e-6,
            "metric_max_clean_minus_native_game": final["clean"][media.JUDGES[0]] - curves["native"]["512"]["clean"][media.JUDGES[0]],
            "minimum_code_gain": final["zero_code"][media.JUDGES[0]] - final["clean"][media.JUDGES[0]],
            **{key: True for key in ("calibrated", "bank_live", "query_live", "C_live")}}
    (root / "retained.pt").write_bytes(b'Synthetic tensor identity, never trained data')
    rows = [{"step": step, "batch_indices": [0, 1, 2, 3], "paired_base_digest": "same",
             "data_rng": "same", "paired_rng": "same", "dv12_rng": "same"} for step in range(1, 513)]
    for arm in curves: (root / f"{arm}.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = {"complete": True, "scientific_status": "PASS", "fixed_updates_each": 512,
              "quality_updates_total": 1024, "G_clean_extra_no_grad_forwards": 512,
              "source_sha256": "synthetic-source", "native_python_sha256": "synthetic-native",
              "matched_data_paired_and_DV12_schedules": True, "learned_owners_gradients_and_optimizer_moments_finite": True,
              "retained_sha256": media.sha(root / "retained.pt"), "gate": gate, "curves": curves,
              "references": {j: [1., 1.1, 1.2, 1.3] for j in media.JUDGES},
              "coverage": {"G_clean": {"bank": 511, "query": 511, "C_norms": [.1, .2]}}}
    save(root / "report.json", report)
    save(root / "completion.json", {"complete": True, "scientific_status": "PASS", "report_sha256": media.sha(root / "report.json")})
    bindings = {"source_sha256": "synthetic-source", "native_python_sha256": "synthetic-native",
                "retained_sha256": media.sha(root / "retained.pt")}
    for name, key in (("report.json", "report_sha256"), ("completion.json", "completion_sha256"),
                      ("native.jsonl", "native_trace_sha256"), ("G_clean.jsonl", "G_clean_trace_sha256")):
        bindings[key] = media.sha(root / name)
    return root, {"bindings": bindings, "gate": gate}


def file_bytes(path):
    return {str(p): p.read_bytes() for p in (path.rglob('*') if path.is_dir() else [path]) if p.is_file()}


def test_structural_gif_keeps_all_original_bytes_numeric_rows_and_no_learned_gate(tmp_path, monkeypatch):
    root, card = film_archive(tmp_path)
    monkeypatch.setattr(media, "load_card", lambda pr: card)
    monkeypatch.setattr(media, "reproduce", lambda *args: pytest.fail("export must not train"))
    before = file_bytes(root)
    review = media.export(233, root, tmp_path / "review")
    assert file_bytes(root) == before and review["raw_files_unchanged"]
    assert review["data"]["learned_scientific_status"] == "NO_FROZEN_GATE"
    assert review["data"]["structural_full_protocol_passed"] is True
    assert review["frozen_176_campaign_changed"] is False
    assert len(review["media"]) == 2
    for artifact in review["media"]:
        with Image.open(tmp_path / "review" / artifact["file"]) as gif: assert gif.n_frames == 13
        assert artifact["actual_updates"] == list(range(0, 1201, 100))


@pytest.mark.parametrize("change", ["partial", "missing_curve", "wrong_owner", "zero_code", "failed_control"])
def test_partial_or_failed_structural_protocol_cannot_pass_export(tmp_path, monkeypatch, change):
    root, card = film_archive(tmp_path)
    monkeypatch.setattr(media, "load_card", lambda pr: card)
    folder = root / "width4"
    if change in ("partial", "missing_curve"):
        summary = json.loads((folder / "summary.json").read_text())
        row = summary["profiles"][media.PROFILES[2]]
        if change == "partial": row["completed_steps"] = 1199
        else: row["evaluations"].pop(3)
        save(folder / "summary.json", summary)
    elif change in ("wrong_owner", "zero_code"):
        path = folder / f"{media.PROFILES[2]}-metadata.json"
        meta = json.loads(path.read_text())
        if change == "wrong_owner": meta["initial_hashes"]["critic"] = "changed"
        else: meta["initial_code_jacobian_frobenius"] = 0
        save(path, meta)
    else:
        path = root / "contracts.xml"
        path.write_text(path.read_text().replace('/>', '><failure/></testcase>', 1))
        card["contracts"]["junit_sha256"] = media.sha(path)
    assert media.main(["--pr", "233", "--input", str(root), "--output", str(tmp_path / "review")]) == 2
    assert not (tmp_path / "review").exists()


def test_zero_update_variance_fail_is_preserved_and_never_looks_like_training(tmp_path, monkeypatch):
    path, card = variance_archive(tmp_path)
    monkeypatch.setattr(media, "load_card", lambda pr: card)
    before = path.read_bytes()
    output = tmp_path / "review"
    assert media.main(["--pr", "234", "--input", str(path), "--output", str(output)]) == 1
    review = json.loads((output / "receipt.json").read_text())
    assert path.read_bytes() == before
    assert review["data"]["scientific_status"] == "FAIL" and review["data"]["native_updates"] == 0
    assert review["media"][0]["actual_updates"] == [0, 0, 0]
    assert review["media"][0]["variance_observation_phases"] == [0, 1, 2]
    with Image.open(output / review["media"][0]["file"]) as gif: assert gif.n_frames == 3


def test_forged_variance_pass_and_mutating_renderer_are_rejected(tmp_path, monkeypatch):
    path, card = variance_archive(tmp_path)
    value = json.loads(path.read_text()); value["pass"] = True; value["variance_result"] = "PASS"
    save(path, value);card["portable_fixed_fixture"]["first_receipt_sha256"] = media.sha(path)
    with pytest.raises(ValueError, match="verdict"):
        media.variance_data(path, card, media.Inputs())
    value["pass"] = False;value["variance_result"] = "FAIL";save(path, value)
    card["portable_fixed_fixture"]["first_receipt_sha256"] = media.sha(path)
    monkeypatch.setattr(media, "load_card", lambda pr: card)
    def corrupt(pr, data, out):
        data["scientific_status"] = "PASS"
        return []
    monkeypatch.setattr(media, "render", corrupt)
    with pytest.raises(ValueError, match="metrics"):
        media.export(234, path, tmp_path / "review")
    assert not (tmp_path / "review" / "receipt.json").exists()


def test_explicit_reproduction_uses_public_callers_and_keeps_caps(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "active_sources", lambda *args: {"synthetic_source": "fixed"})
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(media.subprocess, "run", run)
    raw = tmp_path / "new-run"
    assert media.reproduce(233, raw, {}) == raw
    assert len(calls) == 3 and [kwargs["timeout"] for _, kwargs in calls] == [120, 900, 900]
    assert all("benchmarks.routed_conditioning.spatial_damping" in cmd for cmd, _ in calls[1:])
    assert [cmd[cmd.index("--width") + 1] for cmd, _ in calls[1:]] == ["4", "16"]
    assert all(kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "" for _, kwargs in calls)
    with pytest.raises(ValueError, match="new artifact"):
        media.reproduce(233, raw, {})


def test_clean_game_gif_keeps_actual_four_judge_gate_and_full_budget(tmp_path, monkeypatch):
    root, card = clean_archive(tmp_path)
    before = file_bytes(root)
    monkeypatch.setattr(media, "load_card", lambda pr: card)
    result = media.export(235, root, tmp_path / "review")
    assert result["data"]["scientific_status"] == "PASS" and result["data"]["gate"] == card["gate"]
    assert result["data"]["native_updates_per_arm"] == 512 and result["data"]["arms"] == 2
    assert result["data"]["ordinary_toy_comparison"] == "UNAVAILABLE"
    assert result["media"][0]["actual_updates"] == list(range(0, 513, 64))
    with Image.open(tmp_path / "review" / result["media"][0]["file"]) as gif: assert gif.n_frames == 9
    assert file_bytes(root) == before


@pytest.mark.parametrize("change", ["missing_step", "judge", "nan", "uncalibrated", "fake_pass"])
def test_clean_game_rejects_incomplete_or_false_four_judge_evidence(tmp_path, change):
    root, card = clean_archive(tmp_path)
    report = json.loads((root / "report.json").read_text())
    if change == "missing_step":
        trace = root / "native.jsonl"
        trace.write_text('\n'.join(trace.read_text().splitlines()[1:]) + '\n')
        card["bindings"]["native_trace_sha256"] = media.sha(trace)
    elif change == "judge": report["curves"]["G_clean"]["128"]["clean"].pop(media.JUDGES[0])
    elif change == "nan":
        report["curves"]["G_clean"]["128"]["clean"][media.JUDGES[0]] = float("inf")
    elif change == "uncalibrated": report["references"][media.JUDGES[0]] = [1., .9, .8, .7]
    else: report["curves"]["G_clean"]["512"]["clean"][media.JUDGES[0]] += 1
    if change != "missing_step":
        if change == "nan": (root / "report.json").write_text(json.dumps(report))
        else: save(root / "report.json", report)
        card["bindings"]["report_sha256"] = media.sha(root / "report.json")
        save(root / "completion.json", {"complete": True, "scientific_status": "PASS", "report_sha256": media.sha(root / "report.json")})
        card["bindings"]["completion_sha256"] = media.sha(root / "completion.json")
    with pytest.raises(ValueError): media.clean_data(root, card, media.Inputs())

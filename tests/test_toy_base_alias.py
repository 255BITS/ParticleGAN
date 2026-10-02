"""Duplicate execution may share evidence, never manufacture a second run."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from benchmarks.toy_audit import base
from benchmarks.transfer_suite import image_tasks, suite, vector_tasks


ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_PROOF_LOADER = base.load_alias_proof


@pytest.fixture
def frozen(monkeypatch):
    archived = ARCHIVED_PROOF_LOADER(ROOT)
    assert archived is not None
    # Exercise dispatch and rejection controls with an in-memory software
    # contract. Later native changes must not requalify the archived capture.
    # The production loader and its immutable receipt are left untouched.
    proof = deepcopy(archived)
    current_sources = set(proof["source_sha256"]) | {
        str(p.relative_to(ROOT)) for p in (ROOT / "particlegan").glob("*.py")
    }
    proof["source_sha256"] = {name: base._sha256(ROOT / name)
                              for name in sorted(current_sources)}
    proof["driver_source_sha256"] = base.driver_sha256(ROOT)
    proof["purpose"] = "Synthetic software control; no training qualification"
    monkeypatch.setattr(base, "load_alias_proof", lambda root: deepcopy(proof))
    monkeypatch.setattr(base, "runtime_identity", lambda: deepcopy(proof["runtime"]))
    declared = {s["name"]:s for s in suite.manifest()["tasks"]}
    jobs = {j["spec"]["name"]:j for j in base.plan()}
    return proof, deepcopy(jobs["img_bars4"]["spec"]), deepcopy(declared["img_residual_bars4"])


def software_capture(output, spec, proof):
    """Analytic software fixture, explicitly not a training observation."""
    case = output / "episode-00"
    case.mkdir(parents=True)
    templates = image_tasks.templates(spec)
    cloud = templates.repeat(spec["particles"] // spec["modes"], 1, 1, 1)
    metrics = image_tasks.image_metrics(cloud, templates, spec["thresholds"])
    steps = image_tasks.evaluation_steps(spec)
    result = dict(spec=deepcopy(spec), protocol=deepcopy(proof["observed_protocol"]),
                  policy=vector_tasks.fixed_policy(), ablation="none", fixed=True,
                  live=metrics, ema=metrics, observations=[dict(step=step, seconds=0., **metrics,
                                                              ema=metrics) for step in steps])
    base.write(case / "result.json", result)
    np.savez_compressed(case / "observations.npz", templates=templates.numpy(), steps=steps,
                        live=np.repeat(cloud.numpy()[None], len(steps), axis=0),
                        ema=np.repeat(cloud.numpy()[None], len(steps), axis=0))
    return dict(name=spec["name"], spec=deepcopy(spec), kind="image", artifact=str(case),
                verdict=base.test_verdict(spec, result), frames=len(steps), seconds=1.,
                sampling="software fixture: exact clean enumeration", live=metrics, ema=metrics,
                capture_sha256=base._sha256(case / "observations.npz"),
                result_sha256=base._sha256(case / "result.json"))


def alias(spec, canonical, **kwargs):
    return base.shared_frozen_summary(spec, [canonical], root=ROOT, reference="frozen", **kwargs)


def test_alias_preserves_observed_source_and_both_served_clouds(tmp_path, frozen):
    proof, spec, residual = frozen
    canonical = software_capture(tmp_path / "canonical", spec, proof)
    shared = alias(residual, canonical)
    assert shared is not None
    assert shared["name"] == "img_residual_bars4" and shared["spec"] == residual
    assert shared["execution_status"] == "ALIAS"
    assert not shared["independently_trained"] and not shared["independent_qualification_evidence"]
    assert shared["seconds"] == 0 and shared["alias"]["source_training_seconds"] == 1
    assert shared["alias"]["observed_spec"] == spec
    assert shared["alias"]["source_artifact"] == canonical["artifact"] == shared["artifact"]
    assert shared["capture_sha256"] == canonical["capture_sha256"]
    assert shared["result_sha256"] == canonical["result_sha256"]
    assert shared["live"] == canonical["live"] and shared["ema"] == canonical["ema"]
    with np.load(shared["alias"]["source_capture"]) as capture:
        assert capture["live"].shape == capture["ema"].shape == (24, 32, 1, 8, 8)
    assert not (tmp_path / "img_residual_bars4").exists()


@pytest.mark.parametrize("changes", [
    dict(architecture="transpose"), dict(pattern="blobs4"), dict(steps=601),
    dict(width=24), dict(prior_learnable=False), dict(prior_lr_multiplier=2.),
    dict(noise_std=.02), dict(ema_decay=.98), dict(adam_betas=[.5,.99]),
    dict(thresholds=dict(hq_min=.8, min_mode_fraction=.125, minimum_stable_checks=5,
                         modes=4, observations=24, quality_rmse=.1)),
    dict(serving="noisy"), dict(seed=1),
])
def test_changed_or_unknown_recipe_fields_require_execution(tmp_path, frozen, changes):
    proof, spec, residual = frozen
    canonical = software_capture(tmp_path, spec, proof)
    assert alias(residual | changes, canonical) is None
    # Matching edits to both jobs still lack a frozen equality proof.
    canonical["spec"].update(changes)
    assert alias(residual | changes, canonical) is None


@pytest.mark.parametrize("change", ["permutation", "pixel", "cadence"])
def test_ordered_pixels_and_cadence_are_part_of_execution(tmp_path, frozen, monkeypatch, change):
    proof, spec, residual = frozen
    canonical = software_capture(tmp_path, spec, proof)
    original = image_tasks.templates
    if change == "cadence":
        monkeypatch.setattr(image_tasks, "evaluation_steps", lambda spec: list(range(1,25)))
    else:
        def altered(spec):
            value = original(spec)
            if change == "permutation":
                return value.flip(0)
            value[0,0,0,0] = .5
            return value
        monkeypatch.setattr(image_tasks, "templates", altered)
    assert alias(residual, canonical) is None


def test_source_runtime_and_proof_mutations_disable_reuse(tmp_path, frozen, monkeypatch):
    proof, _, _ = frozen
    for name in [base.ALIAS_PROOF, "benchmarks/toy_audit/base.py", *proof["source_sha256"]]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    assert base.source_runtime_matches(tmp_path, proof)
    changed_runtime = proof["runtime"] | dict(default_dtype="torch.float64")
    with monkeypatch.context() as patcher:
        patcher.setattr(base, "runtime_identity", lambda: changed_runtime)
        assert not base.source_runtime_matches(tmp_path, proof)
    driver = tmp_path / "benchmarks/toy_audit/base.py"
    driver.write_text(driver.read_text() + "\n# unreviewed dispatcher change\n")
    assert not base.source_runtime_matches(tmp_path, proof)
    shutil.copyfile(ROOT / "benchmarks/toy_audit/base.py", driver)
    target = tmp_path / "benchmarks/legacy/gan_loss.py"
    target.write_text(target.read_text() + "\n# unreviewed source change\n")
    assert not base.source_runtime_matches(tmp_path, proof)
    shutil.copyfile(ROOT / "benchmarks/legacy/gan_loss.py", target)
    (tmp_path / "particlegan/new_unknown_semantics.py").write_text("\n")
    assert not base.source_runtime_matches(tmp_path, proof)
    assert ARCHIVED_PROOF_LOADER(tmp_path) is not None
    (tmp_path / base.ALIAS_PROOF).write_text("{}")
    assert ARCHIVED_PROOF_LOADER(tmp_path) is None


def test_archived_capture_cannot_gain_current_source_qualification(tmp_path, frozen, monkeypatch):
    software_proof, spec, residual = frozen
    archived = ARCHIVED_PROOF_LOADER(ROOT)
    assert archived is not None
    changed = software_proof["source_sha256"] != archived["source_sha256"]
    changed |= software_proof["driver_source_sha256"] != archived["driver_source_sha256"]
    # This assertion is independent of the native package version: it binds
    # acceptance to actual bytes, and verifies fail-closed reuse after drift.
    assert base.source_runtime_matches(ROOT, archived) is (not changed)
    if changed:
        canonical = software_capture(tmp_path, spec, archived)
        monkeypatch.setattr(base, "load_alias_proof", ARCHIVED_PROOF_LOADER)
        assert alias(residual, canonical) is None


@pytest.mark.parametrize("change", ["hash", "incomplete", "policy", "protocol", "missing_ema"])
def test_invalid_or_incomplete_evidence_cannot_be_shared(tmp_path, frozen, change):
    proof, spec, residual = frozen
    canonical = software_capture(tmp_path, spec, proof)
    assert alias(residual, canonical) is not None
    path = Path(canonical["artifact"])
    result = json.loads((path / "result.json").read_text())
    if change == "hash":
        canonical["capture_sha256"] = "0" * 64
    elif change == "missing_ema":
        with np.load(path / "observations.npz") as arrays:
            remaining = {k:arrays[k] for k in arrays.files if k != "ema"}
        np.savez(path / "observations.npz", **remaining)
        canonical["capture_sha256"] = base._sha256(path / "observations.npz")
    else:
        if change == "incomplete":
            result["observations"] = result["observations"][:-1]
        elif change == "policy":
            result["policy"]["schedule"] = "constant"
        else:
            result["protocol"]["seed"] = 1
        base.write(path / "result.json", result)
        canonical["result_sha256"] = base._sha256(path / "result.json")
    assert alias(residual, canonical) is None


@pytest.mark.parametrize("reference,separate,executions", [
    ("frozen", False, 1), ("frozen", True, 2), ("public_v2", False, 2),
])
def test_driver_skips_only_the_proved_frozen_duplicate(tmp_path, frozen, monkeypatch,
                                                      reference, separate, executions):
    proof, _, residual = frozen
    declared = {s["name"]:s for s in suite.manifest()["tasks"]}
    tasks = [declared["img_bars4"], residual]
    monkeypatch.setattr(suite, "manifest", lambda: dict(tasks=deepcopy(tasks)))
    calls, active = [], {}

    class SoftwareCapture:
        def __init__(self, output):
            self.output, self.records = output, []

        @contextmanager
        def installed(self):
            active["capture"] = self
            yield

    def episode(spec, *args, **kwargs):
        calls.append(spec["name"])
        capture = active["capture"]
        capture.records.append(software_capture(capture.output, spec, proof))

    monkeypatch.setattr(base, "Capture", SoftwareCapture)
    monkeypatch.setattr(suite, "run_episode", episode)
    output = tmp_path / "run"
    argv = ["base", "--output", str(output), "--reference", reference]
    if separate:
        argv.append("--separate-execution")
    monkeypatch.setattr("sys.argv", argv)
    base.main()
    assert len(calls) == executions
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["tasks"] == tasks
    records = json.loads((output / "index.json").read_text())
    assert [s["name"] for s in records] == [s["name"] for s in tasks]
    summary_dir = output / "img_residual_bars4"
    if executions == 1:
        assert records[-1]["execution_status"] == "ALIAS"
        assert list(summary_dir.iterdir()) == [summary_dir / "summary.json"]
        assert records[-1]["alias"]["observed_spec"] == records[0]["spec"]
        assert records[-1]["artifact"] == records[0]["artifact"]
    else:
        assert all(s["execution_status"] == "EXECUTED" for s in records)
        assert (summary_dir / "episode-00" / "observations.npz").exists()

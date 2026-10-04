"""Synthetic-only observer guards: no model, sampler, scoring or updates."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest


from benchmarks.toy_audit import api_run, gaussian2d_observer as observer

PROPOSAL = observer.declaration()


def test_committed_protocol_matches_actual_case_recipe_cadence_and_budget():
    root = Path(__file__).resolve().parents[1]
    packet = json.loads((root / "reports/forge/gaussian2d-current-c6-20261004/protocol.json").read_text())
    assert {key: packet[key] for key in PROPOSAL} == PROPOSAL
    assert packet["runtime_source_binding"]["old_inspection_pins_authorize_current_execution"] is False
    assert packet["runtime_source_binding"]["no_current_source_commit_or_digest_asserted"] is True


class Parameter:
    shape = (20000, 2)


class Module:
    def __init__(self, parameter):
        self.parameter, self.training = parameter, True

    def parameters(self):
        return iter([self.parameter])

    def named_modules(self):
        return [("", self)]


class Trainer:
    pass


class Fixture:
    """Counter-only fake public host, not a torch.nn.Module or learned model."""
    def __init__(self):
        self.metadata = deepcopy(PROPOSAL["case"])
        self.recipe = SimpleNamespace(to_dict=lambda: deepcopy(PROPOSAL["resolved_recipe"]))
        self.case_id = self.metadata["id"]
        self.seed, self.execution_steps, self.completed_steps = 24002, 1000, 0
        self.device = "cuda:0"  # declaration only; no CUDA call or allocation
        self.trainer = Trainer()
        t = self.trainer
        t.max_steps, t.completed_steps = 1000, 0
        t.G, t.D = Module(Parameter()), Module(Parameter())
        t.prior, t.ema_G, t.ema_prior, t.ema_D = Module(Parameter()), Module(Parameter()), Module(Parameter()), Module(Parameter())
        t.prior.z = t.prior.parameter
        t.opt_g = SimpleNamespace(state={})
        t.opt_d = SimpleNamespace(state={})
        t.policy = SimpleNamespace(completed_steps=0, row_semantics="independent", prior=t.prior, table=t.prior.z)
        t.policy.G, t.policy.D, t.policy.table_optimizer = t.G, t.D, t.opt_g
        t.policy.controller = SimpleNamespace(variant="dv12")
        t.served_snapshot = lambda: {"source": "fast", "counter": t.completed_steps}
        self.calls = {"step": 0, "observe": []}
        self.data_state = "synthetic-unchanged-data-stream"
        self.record = {"metrics": {"synthetic_metric": 1.}, "passed": False,
                       "failed_bounds": ["synthetic_only"], "views": [{"kind": "scatter", "title": "synthetic",
                       "target": np.zeros((4096, 2)), "samples": np.zeros((4096, 2))}]}
        self.api_components = ("synthetic.CounterOnlyFakeFixture",)

    def step(self):
        self.calls["step"] += 1
        self.completed_steps += 1
        self.trainer.completed_steps += 1
        self.trainer.policy.completed_steps += 1
        for p in (self.trainer.G.parameter, self.trainer.prior.z):
            self.trainer.opt_g.state[p] = {"step": self.completed_steps}
        self.trainer.opt_d.state[self.trainer.D.parameter] = {"step": self.completed_steps}
        return "unaltered-public-result"

    def observe(self, n, seed):
        self.calls["observe"].append((n, seed))
        return deepcopy(self.record)

    def state_dict(self):
        return {"case_id": self.case_id, "completed_steps": self.completed_steps,
                "trainer": {"completed_steps": self.trainer.completed_steps,
                            "policy_counter": self.trainer.policy.completed_steps},
                "data_rng": self.data_state}


def digest(value):
    return hashlib.sha256(pickle.dumps(value)).hexdigest()


def control(policy, steps):
    return {"requested": dict.fromkeys(observer.OWNERS, True),
            "enabled": dict.fromkeys(observer.OWNERS, True),
            "requested_owners_bound": True, "implementation_observed": True,
            "row_evidence_observations": steps,
            "completed_steps": steps, "served_source": "fast",
            "execution": {"model_devices": ["cuda:0"], "floating_dtypes": ["torch.float32"],
                          "autocast_enabled": False},
            "diagnostics": {"controller": {"variant": "synthetic-dv12"}, "backend_selection": None}}


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(observer, "GANTrainer", Trainer)
    monkeypatch.setattr(observer, "PolicyLifecycleAudit", lambda policy: "synthetic-audit")
    monkeypatch.setattr(observer, "controls_receipt", control)
    monkeypatch.setattr(observer, "finite_policy_state", lambda state: True)
    monkeypatch.setattr(observer, "typed_state_digest", digest)
    monkeypatch.setattr(observer, "global_state", lambda: {"python": random.getstate(), "numpy": np.random.get_state()})
    fixture = Fixture()
    calls = []
    guard = lambda: calls.append("source_checked")
    return fixture, observer.ObservedGaussianFixture(fixture, PROPOSAL, source_guard=guard), calls


def test_exact_delegate_and_receipt_preserve_numeric_failure(setup):
    fixture, wrapped, guards = setup
    assert wrapped.step() == "unaltered-public-result"
    record = wrapped.observe()
    assert fixture.calls == {"step": 1, "observe": [(4096, 34002)]}
    assert {key: value for key, value in record.items() if key not in ("policy_observation", "views")} == {
        key: value for key, value in fixture.record.items() if key != "views"}
    for role in ("target", "samples"):
        assert np.array_equal(record["views"][0][role], fixture.record["views"][0][role])
    assert record["passed"] is False
    evidence = record["policy_observation"]
    assert evidence["purity"]["pure"] and evidence["purity"]["allowed_changes"] == "none"
    assert evidence["actual_optimizer_updates"]["prior"]["minimum"] == 1
    assert guards == ["source_checked"] * 3


@pytest.mark.parametrize("field,value", [("seed", 0), ("execution_steps", 999), ("completed_steps", 1)])
def test_wrong_original_seed_horizon_clock_rejected(setup, field, value):
    fixture, _, _ = setup
    setattr(fixture, field, value)
    with pytest.raises(ValueError):
        observer.ObservedGaussianFixture(fixture, PROPOSAL, source_guard=lambda: None)


@pytest.mark.parametrize("field", ["thresholds", "law", "batch_size", "particles"])
def test_changed_goal_or_resource_rejected(setup, field):
    fixture, _, _ = setup
    fixture.metadata[field] = "changed"
    with pytest.raises(ValueError, match="canonical case"):
        observer.ObservedGaussianFixture(fixture, PROPOSAL, source_guard=lambda: None)


def test_changed_complete_recipe_rejected(setup):
    fixture, _, _ = setup
    changed = deepcopy(PROPOSAL["resolved_recipe"]); changed["serve_average"] = 0
    fixture.recipe.to_dict = lambda: changed
    with pytest.raises(ValueError, match="Recipe"):
        observer.ObservedGaussianFixture(fixture, PROPOSAL, source_guard=lambda: None)


@pytest.mark.parametrize("options", [{"n": 1024}, {"seed": 713}])
def test_changed_holdout_law_rejected_before_read(setup, options):
    fixture, wrapped, _ = setup
    with pytest.raises(ValueError, match="held-out"):
        wrapped.observe(**options)
    assert not fixture.calls["observe"]


def test_source_import_failure_propagates_before_read(setup):
    fixture, wrapped, _ = setup
    def bad_guard():
        raise ValueError("foreign imported module")
    wrapped.source_guard = bad_guard
    with pytest.raises(ValueError, match="foreign"):
        wrapped.observe()
    assert not fixture.calls["observe"]


@pytest.mark.parametrize("mode", ["state", "mode", "python_rng", "numpy_rng", "selected"])
def test_observer_training_rng_mode_selected_mutation_rejected(setup, mode):
    fixture, wrapped, _ = setup
    original = fixture.observe
    def mutated(n, seed):
        record = original(n, seed)
        if mode == "state": fixture.data_state = "synthetic-changed-data-stream"
        if mode == "mode": fixture.trainer.ema_G.training = False
        if mode == "python_rng": random.random()
        if mode == "numpy_rng": np.random.random()
        if mode == "selected": fixture.trainer.served_snapshot = lambda: {"source": "averaged", "counter": 0}
        return record
    fixture.observe = mutated
    with pytest.raises(RuntimeError, match="changed complete"):
        wrapped.observe()
    assert not wrapped.observations


@pytest.mark.parametrize("owner", sorted(observer.OWNERS))
def test_no_omitted_enabled_owner_can_pass(setup, monkeypatch, owner):
    _, wrapped, _ = setup
    def missing(policy, steps):
        value = control(policy, steps); value["enabled"][owner] = False
        return value
    monkeypatch.setattr(observer, "controls_receipt", missing)
    with pytest.raises(ValueError, match="omitted"):
        wrapped.observe()


@pytest.mark.parametrize("role", ["generator", "discriminator", "prior"])
def test_missing_actual_optimizer_updates_rejected(setup, role):
    fixture, wrapped, _ = setup
    wrapped.step()
    parameter = fixture.trainer.prior.z if role == "prior" else getattr(fixture.trainer, "G" if role == "generator" else "D").parameter
    optimizer = fixture.trainer.opt_d if role == "discriminator" else fixture.trainer.opt_g
    optimizer.state.pop(parameter)
    with pytest.raises(ValueError, match="actual optimizer"):
        wrapped.observe()


@pytest.mark.parametrize("changed", ["model_devices", "floating_dtypes", "autocast_enabled"])
def test_wrong_runtime_evidence_rejected(setup, monkeypatch, changed):
    _, wrapped, _ = setup
    def wrong(policy, steps):
        value = control(policy, steps)
        value["execution"][changed] = True if changed == "autocast_enabled" else ["changed"]
        return value
    monkeypatch.setattr(observer, "controls_receipt", wrong)
    with pytest.raises(ValueError, match="device/dtype/autocast"):
        wrapped.observe()


def receipt(wrapped, *, status="INCOMPLETE"):
    return {"case": deepcopy(PROPOSAL["case"]), "recipe": wrapped.recipe.to_dict(),
            "completed_updates": wrapped.completed_steps, "status": status, "verdict": "FAIL",
            "observations": [{"step": row["completed_steps"], "policy_observation": deepcopy(row)}
                             for row in wrapped.observations]}


def test_partial_prefix_preserves_incomplete_and_no_qualification(setup):
    _, wrapped, _ = setup
    wrapped.observe(); wrapped.step(); wrapped.observe()
    sidecar = wrapped.finalize(receipt(wrapped))
    assert sidecar["pre_export_numerical_status"] == "INCOMPLETE"
    assert sidecar["policy_protocol_complete"] is False
    assert sidecar["quality_from_owner_evidence"] is False
    assert sidecar["training_or_rescoring_added"] is False
    assert sidecar["finalizer_purity"]["pure"] and sidecar["finalizer_purity"]["extra_observations"] == 0


def test_final_attestation_may_not_mutate_owned_modes(setup, monkeypatch):
    fixture, wrapped, _ = setup
    wrapped.observe()
    def changed(policy, steps):
        result = control(policy, steps)
        fixture.trainer.ema_D.training = False
        return result
    monkeypatch.setattr(observer, "controls_receipt", changed)
    with pytest.raises(RuntimeError, match="final attestation changed"):
        wrapped.finalize(receipt(wrapped))


@pytest.mark.parametrize("tamper", ["missing", "source", "clock", "full"])
def test_forged_numerical_join_rejected(setup, tamper):
    _, wrapped, _ = setup
    wrapped.observe(); wrapped.step(); wrapped.observe()
    value = receipt(wrapped)
    if tamper == "missing": value["observations"].pop()
    if tamper == "source": value["observations"][0]["policy_observation"]["selected_source"] = "averaged"
    if tamper == "clock": value["completed_updates"] += 1
    if tamper == "full": value["status"] = "COMPLETE"
    with pytest.raises(ValueError):
        wrapped.finalize(value)


def test_zero_role_or_nonfinite_clock_rejects():
    with pytest.raises(ValueError, match="no parameters"):
        observer._counts(SimpleNamespace(state={}), [])
    p = Parameter()
    with pytest.raises(ValueError, match="update clock"):
        observer._counts(SimpleNamespace(state={p: {"step": float("nan")}}), [p])


def test_actual_public_recipe_json_container_boundary_without_model():
    from particlegan import get_recipe
    recipe = get_recipe("atlas", num_particles=20000, z_dim=2, batch_size=2048,
                        lr=.0053125, prior_lr_mult=1.5)
    assert observer.recipe_identity(recipe.to_dict()) == PROPOSAL["resolved_recipe"]
    changed = recipe.replace(d_lr_mult=2.)
    assert observer.recipe_identity(changed.to_dict()) != PROPOSAL["resolved_recipe"]


def test_stale_row_evidence_clock_rejected(setup, monkeypatch):
    _, wrapped, _ = setup
    wrapped.step()
    def wrong(policy, steps):
        value = control(policy, steps); value["row_evidence_observations"] = 0
        return value
    monkeypatch.setattr(observer, "controls_receipt", wrong)
    with pytest.raises(ValueError, match="row-evidence clock"):
        wrapped.observe()


def test_other_latent_controller_rejected(setup):
    fixture, wrapped, _ = setup
    fixture.trainer.policy.controller.variant = "other"
    with pytest.raises(ValueError, match="not DV12"):
        wrapped.observe()


def test_control_receipt_read_itself_must_preserve_complete_state(setup, monkeypatch):
    fixture, wrapped, _ = setup
    def mutated(policy, steps):
        value = control(policy, steps)
        fixture.trainer.ema_D.training = False
        return value
    monkeypatch.setattr(observer, "controls_receipt", mutated)
    with pytest.raises(RuntimeError, match="changed complete"):
        wrapped.observe()


def test_public_build_factory_changes_no_existing_global(monkeypatch):
    from benchmarks.toy_audit import api_contract
    original = api_contract.build
    factory = observer.build_factory(PROPOSAL, source_guard=lambda: None)
    assert callable(factory) and api_contract.build is original
    with pytest.raises(ValueError, match="factory options"):
        factory(PROPOSAL["case"], device="cpu", seed=24002, recipe_name="atlas", max_steps=1000,
                recipe_overrides={"lr": .0053125, "prior_lr_mult": 1.5})


@pytest.fixture
def fake_runner(monkeypatch):
    """Run real orchestration/export with fake counters, never a learned host."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(api_run, "source_identity", lambda: {
        "commit": "synthetic-only-not-scientific", "files_sha256": {}})
    monkeypatch.setattr(api_run.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(api_run.torch.cuda, "get_device_name", lambda device: "synthetic-no-GPU")
    media = []
    def export(case, records, path, **options):
        media.append({"steps": [row["step"] for row in records], "options": options})
        path.write_bytes(b"synthetic-placeholder-no-scientific-media")
    monkeypatch.setattr(api_run, "render_gif", export)
    return media


def run_fake(path, *, factory=None, finalizer=None):
    return api_run.run_case(PROPOSAL["case"], path, device="cuda:0", recipe_name="atlas",
                            steps=1000, eval_samples=4096, frames=9, seed=24002,
                            recipe_overrides=deepcopy(observer.OVERRIDES),
                            fixture_factory=factory, receipt_finalizer=finalizer)


@pytest.mark.parametrize("passing", [False, True])
def test_full_bound_callback_path_has_exact_counts_original_verdict_and_nine_states(
        tmp_path, setup, fake_runner, monkeypatch, passing):
    fixture, wrapped, _ = setup
    fixture.record.update(passed=passing, failed_bounds=[] if passing else ["synthetic_only"])
    monkeypatch.setattr(api_run.contract, "build", lambda *args, **kwargs: pytest.fail("explicit factory ignored"))
    def factory(case, **options):
        assert options == {"device": "cuda:0", "seed": 24002, "recipe_name": "atlas",
                           "max_steps": 1000, "recipe_overrides": observer.OVERRIDES}
        return wrapped
    counts = []
    def finalize(host, numeric):
        counts.append(deepcopy(host.fixture.calls))
        result = host.finalize(numeric)
        assert counts[-1] == host.fixture.calls
        return result
    result = run_fake(tmp_path / "full", factory=factory, finalizer=finalize)
    verdict = "PASS" if passing else "FAIL"
    assert result["status"] == "COMPLETE" and result["verdict"] == verdict
    assert result["metric_passed"] is passing and result["sustained_metric_passed"] is passing
    assert observer.result_exit_code(result) == (0 if passing else 1)
    assert fixture.calls == {"step": 1000, "observe": [(4096, 34002)] * 25}
    assert result["protocol"]["metric_evaluation_steps"] == PROPOSAL["execution"]["metric_steps"]
    assert [row["step"] for row in result["observations"]] == PROPOSAL["execution"]["metric_steps"]
    assert fake_runner[0]["steps"] == PROPOSAL["execution"]["media_steps"]
    assert fake_runner[0]["options"]["final_verdict"] == verdict
    assert result["gif_frames"] == 9 and result["default_protocol_complete"]
    assert result["policy_observer"]["policy_protocol_complete"] is True
    assert result["policy_observer"]["quality_from_owner_evidence"] is False
    assert "policy_observation" in result["observations"][-1]
    assert json.loads((tmp_path / "full/receipt.json").read_text())["verdict"] == verdict


def test_default_runner_path_uses_original_factory_with_identical_counts_and_grade(
        tmp_path, setup, fake_runner, monkeypatch):
    fixture, _, _ = setup
    built = []
    monkeypatch.setattr(api_run.contract, "build", lambda *args, **kwargs: (built.append(kwargs), fixture)[1])
    result = run_fake(tmp_path / "default")
    assert len(built) == 1
    assert fixture.calls == {"step": 1000, "observe": [(4096, 34002)] * 25}
    assert result["status"] == "COMPLETE" and result["verdict"] == "FAIL"
    assert result["completed_updates"] == 1000 and result["default_protocol_complete"]
    assert "policy_observer" not in result and "numerical_before_observer_error" not in result
    assert fake_runner[0]["steps"] == PROPOSAL["execution"]["media_steps"]


@pytest.mark.parametrize("passing", [False, True])
def test_finalizer_error_retains_raw_numeric_authority_and_actual_full_prefix(
        tmp_path, setup, fake_runner, passing):
    fixture, wrapped, _ = setup
    fixture.record.update(passed=passing, failed_bounds=[] if passing else ["synthetic_only"])
    def bad_finalizer(host, numeric):
        assert host.fixture.calls == {"step": 1000, "observe": [(4096, 34002)] * 25}
        raise RuntimeError("synthetic finalizer failed")
    result = run_fake(tmp_path / "error", factory=lambda *args, **kwargs: wrapped, finalizer=bad_finalizer)
    assert result["status"] == "ERROR" and result["verdict"] == "FAIL" and not result["passed"]
    assert observer.result_exit_code(result) == 2
    assert result["completed_updates"] == 1000 and len(result["observations"]) == 25
    assert result["numerical_before_observer_error"]["verdict"] == ("PASS" if passing else "FAIL")
    assert result["metric_passed"] is passing and result["sustained_metric_passed"] is passing
    if not passing:
        assert "synthetic_only" in result["failed_bounds"]
    assert "finalizer failed" in result["failed_bounds"][-1]
    assert fixture.calls == {"step": 1000, "observe": [(4096, 34002)] * 25}


def test_factory_error_before_updates_saves_error_receipt_without_finalizer(
        tmp_path, fake_runner):
    def bad_factory(*args, **kwargs):
        raise ValueError("synthetic factory refusal")
    result = run_fake(tmp_path / "factory", factory=bad_factory,
                      finalizer=lambda *args: pytest.fail("missing host must not finalize"))
    assert result["status"] == "ERROR" and observer.result_exit_code(result) == 2
    assert result["completed_updates"] == 0 and not result["observations"]
    assert result["gif_frames"] == 0 and not fake_runner
    assert "factory refusal" in result["failed_bounds"][0]
    assert (tmp_path / "factory/receipt.json").is_file()


@pytest.mark.parametrize("callback", [None, {}, [], True])
def test_invalid_finalizer_payload_is_recorded_error(tmp_path, setup, fake_runner, callback):
    _, wrapped, _ = setup
    result = run_fake(tmp_path / "bad_payload", factory=lambda *args, **kwargs: wrapped,
                      finalizer=lambda *args: callback)
    assert result["status"] == "ERROR" and observer.result_exit_code(result) == 2
    assert result["numerical_before_observer_error"]["verdict"] == "FAIL"


def test_finalizer_receipt_copy_cannot_rewrite_numeric_fail_or_bounds(tmp_path, setup, fake_runner):
    _, wrapped, _ = setup
    def attempted_upgrade(host, numeric):
        numeric.update(verdict="PASS", passed=True, failed_bounds=[])
        numeric["observations"][-1]["passed"] = True
        return {"synthetic_attempted_claim": "PASS"}
    result = run_fake(tmp_path / "cannot_upgrade", factory=lambda *args, **kwargs: wrapped,
                      finalizer=attempted_upgrade)
    assert result["verdict"] == "FAIL" and not result["passed"]
    assert result["observations"][-1]["passed"] is False
    assert "synthetic_only" in result["failed_bounds"]


def test_export_error_after_successful_attestation_keeps_final_error_authority(
        tmp_path, setup, fake_runner, monkeypatch):
    _, wrapped, _ = setup
    def bad_export(*args, **kwargs):
        raise ValueError("synthetic media export error")
    monkeypatch.setattr(api_run, "render_gif", bad_export)
    result = run_fake(tmp_path / "export_error", factory=lambda *args, **kwargs: wrapped,
                      finalizer=lambda host, numeric: host.finalize(numeric))
    assert result["status"] == "ERROR" and observer.result_exit_code(result) == 2
    assert result["policy_observer"]["pre_export_numerical_status"] == "COMPLETE"
    assert result["policy_observer"]["pre_export_numerical_verdict"] == "FAIL"
    assert "media export error" in result["failed_bounds"][-1]


@pytest.mark.parametrize("argument", ["fixture_factory", "receipt_finalizer"])
def test_noncallable_hook_rejects_before_fake_construction(tmp_path, fake_runner, argument):
    result = api_run.run_case(PROPOSAL["case"], tmp_path / argument, **{argument: True})
    assert result["status"] == "ERROR" and result["completed_updates"] == 0
    assert observer.result_exit_code(result) == 2
    assert not result["observations"] and not fake_runner


def test_source_closure_uses_public_snapshot_api_and_keeps_ring_catalog_examples(tmp_path):
    files = {path: "{}\n" for path in observer.EXTRA_SOURCE_PATHS}
    files.update({"particlegan/__init__.py": "# synthetic public source\n",
                  "benchmarks/toy_audit/gaussian2d_observer.py": "# synthetic observer\n",
                  "examples/e22_synthetic.py": "# deferred source binding\n",
                  "reports/synthetic_supervisor.py": "# explicit root driver\n"})
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest = observer.source_manifest(tmp_path, supervisor_source_paths=("reports/synthetic_supervisor.py",))
    assert set(manifest["files"]) == set(files)
    assert manifest["files"][observer.EXTRA_SOURCE_PATHS[0]] == hashlib.sha256(b"{}\n").hexdigest()
    (tmp_path / observer.EXTRA_SOURCE_PATHS[0]).unlink()
    with pytest.raises(ValueError, match="declared source"):
        observer.source_manifest(tmp_path)


def synthetic_manifest():
    from experiments.forge.contracts import stable_hash
    files = dict.fromkeys(observer.EXTRA_SOURCE_PATHS + (
        "benchmarks/toy_audit/gaussian2d_observer.py", "benchmarks/toy_audit/api_run.py"), "a" * 64)
    return {"schema_version": 1, "files": files, "digest": stable_hash(files), "origin_commit": "b" * 40}


@pytest.fixture
def bound_source(monkeypatch):
    # Deliberately synthetic: no current or historical scientific source credit.
    monkeypatch.setattr(observer, "source_manifest", lambda *args, **kwargs: synthetic_manifest())


def synthetic_admission():
    return {"status": "running", "device": "cuda:0", "physical_gpu": 0, "threads": 1,
            "memory_fraction": .2, "allowance_seconds": 180, "grace_seconds": 0,
            "lease_verified": True, "single_attempt": True}


@pytest.mark.parametrize("field,value", [
    ("physical_gpu", 1), ("device", "cpu"), ("threads", 2), ("memory_fraction", .3),
    ("allowance_seconds", 181), ("grace_seconds", 60), ("lease_verified", False),
    ("single_attempt", False), ("status", "queued")])
def test_root_bound_resource_mismatch_rejects_before_runner(tmp_path, monkeypatch, bound_source, field, value):
    monkeypatch.setattr(api_run, "run_case", lambda *args, **kwargs: pytest.fail("invalid admission reached runner"))
    admission = synthetic_admission(); admission[field] = value
    with pytest.raises(ValueError, match="admitted"):
        observer.run_bound_case(tmp_path / "none", proposal=PROPOSAL, frozen_source=synthetic_manifest(),
                                source_guard=lambda: None, admission_guard=lambda: admission)


@pytest.mark.parametrize("tamper", ["digest", "commit", "ring", "helper", "old_runner", "missing_runner"])
def test_root_bound_manifest_mismatch_rejects_before_admission(tmp_path, monkeypatch, bound_source, tamper):
    from experiments.forge.contracts import stable_hash
    manifest = synthetic_manifest()
    if tamper == "digest": manifest["digest"] = "0" * 64
    if tamper == "commit": manifest["origin_commit"] = "uncommitted"
    if tamper in {"ring", "helper"}:
        manifest["files"].pop(observer.EXTRA_SOURCE_PATHS[0] if tamper == "ring" else "benchmarks/toy_audit/gaussian2d_observer.py")
        manifest["digest"] = stable_hash(manifest["files"])
    if tamper in {"old_runner", "missing_runner"}:
        if tamper == "old_runner": manifest["files"]["benchmarks/toy_audit/api_run.py"] = "c" * 64
        else: manifest["files"].pop("benchmarks/toy_audit/api_run.py")
        manifest["digest"] = stable_hash(manifest["files"])
    with pytest.raises(ValueError, match="frozen source"):
        observer.run_bound_case(tmp_path / "none", proposal=PROPOSAL, frozen_source=manifest,
                                source_guard=lambda: None, admission_guard=lambda: pytest.fail("bad manifest reached admission"))


def test_root_bound_driver_delegates_once_full_fixed_protocol_and_retains_source_error(tmp_path, monkeypatch, bound_source):
    calls, guards = [], []
    output = tmp_path / "root_bound"
    def runner(case, destination, **options):
        calls.append(options)
        destination.mkdir()
        return {"status": "COMPLETE", "verdict": "FAIL", "passed": False, "failed_bounds": ["synthetic_only"],
                "default_protocol_complete": True}
    def guard():
        guards.append("guard")
        if len(guards) == 2:
            raise ValueError("synthetic foreign module after export")
    monkeypatch.setattr(api_run, "run_case", runner)
    result = observer.run_bound_case(output, proposal=PROPOSAL, frozen_source=synthetic_manifest(),
                                    source_guard=guard, admission_guard=synthetic_admission)
    assert len(calls) == 1 and callable(calls[0]["fixture_factory"]) and callable(calls[0]["receipt_finalizer"])
    assert {key: calls[0][key] for key in ("device", "recipe_name", "steps", "eval_samples", "frames", "seed", "recipe_overrides", "wall_cap_seconds")} == {
        "device": "cuda:0", "recipe_name": "atlas", "steps": 1000, "eval_samples": 4096, "frames": 9,
        "seed": 24002, "recipe_overrides": observer.OVERRIDES, "wall_cap_seconds": 180}
    assert result["status"] == "ERROR" and observer.result_exit_code(result) == 2
    assert "synthetic_only" in result["failed_bounds"]
    assert result["numerical_before_bound_source_error"]["verdict"] == "FAIL"
    assert result["gaussian_bound_protocol"]["final_raw_receipt_is_verdict_authority"] is True
    assert json.loads((output / "receipt.json").read_text())["status"] == "ERROR"

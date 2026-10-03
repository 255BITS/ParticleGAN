"""Fail-closed software controls; no campaign, GPU work or old evidence credit."""
from copy import deepcopy
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from benchmarks.toy_audit import api_contract, api_family_search, api_run
from experiments.forge.policy_execution import PolicyCoordinator

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("critic_balance_runner_tests", ROOT / "reports/forge/critic-balance-20261003/run_critic_balance.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def one_cpu_thread(monkeypatch):
    previous = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_num_threads(1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    yield
    torch.set_num_threads(previous)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32


@pytest.fixture
def cases():
    found = api_contract.discover()
    return {name: found[name] for name, _, _ in runner.CASE_ROWS}


def spec_for(tmp_path):
    path = tmp_path / "capacity.json"
    requested = [{"family": family, "case_id": name} for family in runner.FAMILIES for name, _, _ in runner.CASE_ROWS]
    path.write_text(json.dumps({"schema": "software-capacity", "claim_scope": "Synthetic fixture; no scientific evidence",
                               "status": "COMPLETE", "required_records": 16, "requested_cells": requested,
                               "ordinary_training_updates": 0, "fitting_updates": 0, "ordinary_qualification_credit": False,
                               "records": [{**item, "status": "SUPPORTED"} for item in requested]}))
    return runner.default_spec(str(path), api_run.file_hash(path), identifier="software-critic-contrast")


def planned(tmp_path, cases, monkeypatch):
    # Explicit software card, never recorded as capacity evidence.
    monkeypatch.setattr(runner, "verify_engineering_carryover", lambda: deepcopy(runner.ENGINEERING_ARTIFACTS))
    monkeypatch.setattr(runner, "capacity_module", lambda: SimpleNamespace(
        SHARED_OVERRIDES=runner.OVERRIDES, SCHEMA="software-capacity", CLAIM_SCOPE="Synthetic fixture; no scientific evidence",
        verify_packet=lambda packet: deepcopy(packet)))
    return runner.plan_study(spec_for(tmp_path), cases=cases)


def refresh_card(spec, edit):
    path = Path(spec["representation_card"]["path"])
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))
    spec["representation_card"]["sha256"] = api_run.file_hash(path)


def test_plan_retains_two_singletons_and_sixteen_cells_without_queue_writes(tmp_path, cases, monkeypatch):
    monkeypatch.setattr(PolicyCoordinator, "__init__", lambda *a, **k: pytest.fail("planning initialized queue"))
    packet = planned(tmp_path, cases, monkeypatch)
    assert len(packet["trials"]) == 2 and sum(len(t["cases"]) for t in packet["trials"]) == 16
    assert all(t["recipe_overrides"] == runner.OVERRIDES for t in packet["trials"])
    assert all(r["status"] == "UNKNOWN" for t in packet["trials"] for r in t["cases"])
    assert packet["round_worst_case_reservation_seconds"] == 15360
    assert packet["selection"]["speed_winner"] is None
    assert not (tmp_path / "queue").exists()


@pytest.mark.parametrize("field,value", [("lr", .00425), ("prior_lr_mult", 1.), ("d_lr_mult", 1.5),
                                       ("lr", True), ("d_lr_mult", float("nan"))])
def test_changed_singleton_knob_rejected(tmp_path, cases, field, value):
    spec = spec_for(tmp_path)
    spec["recipe_overrides"][field] = value
    with pytest.raises((ValueError, TypeError)):
        runner.validate_spec(spec, cases)


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(seed=24003), lambda s: s.update(frames=8),
    lambda s: s.update(export_grace_seconds=120.), lambda s: s.update(budget_seconds=15361.),
    lambda s: s["family_budget_seconds"].update(atlas=8000.),
    lambda s: s["cases"][2].update(timeout_seconds=2101.),
    lambda s: s["stability"].update(post_confirmation_hold_checks=4),
    lambda s: s["admission"].update(maximum_temperature_c=83.),
    lambda s: s.update(speed_ranking=True), lambda s: s.update(default_adoption=True),
])
def test_protocol_budget_gate_admission_changes_rejected(tmp_path, cases, mutate):
    spec = spec_for(tmp_path)
    mutate(spec)
    with pytest.raises(ValueError):
        runner.validate_spec(spec, cases)


@pytest.mark.parametrize("field,value", [("default_steps", 6999), ("eval_samples", 1024),
                                       ("thresholds", {"easy": 1}), ("sampling", "clean alternative"),
                                       ("terminal_observations", 4), ("protocol_seed", 2)])
def test_registered_gate_horizon_sampling_drift_rejected(tmp_path, cases, field, value):
    modified = deepcopy(cases)
    modified["api-grid100"][field] = value
    with pytest.raises(ValueError, match="metadata changed"):
        runner.validate_spec(spec_for(tmp_path), modified)


@pytest.mark.parametrize("edit", [lambda d: d["records"].pop(),
                                  lambda d: d["records"].__setitem__(1, deepcopy(d["records"][0])),
                                  lambda d: d["records"][0].update(case_id="unknown")])
def test_missing_duplicate_unknown_capacity_denominator_rejected(tmp_path, cases, monkeypatch, edit):
    spec = spec_for(tmp_path)
    refresh_card(spec, edit)
    monkeypatch.setattr(runner, "capacity_module", lambda: pytest.fail("invalid card invoked replay"))
    with pytest.raises(ValueError, match="sixteen"):
        runner.plan_study(spec, cases=cases)


def test_source_stale_capacity_failure_propagates_before_queue(tmp_path, cases, monkeypatch):
    def reject(*args):
        raise ValueError("capacity source drift")
    monkeypatch.setattr(runner, "capacity_module", lambda: SimpleNamespace(SHARED_OVERRIDES=runner.OVERRIDES,
        SCHEMA="software-capacity", CLAIM_SCOPE="Synthetic fixture; no scientific evidence", verify_packet=reject))
    monkeypatch.setattr(PolicyCoordinator, "__init__", lambda *a, **k: pytest.fail("invalid capacity initialized queue"))
    with pytest.raises(ValueError, match="source drift"):
        runner.run_study(spec_for(tmp_path), tmp_path / "run", family="atlas")


def test_negative_capacity_blocks_own_family_preserves_supported_sibling(tmp_path, cases, monkeypatch):
    planned(tmp_path, cases, monkeypatch)
    spec = spec_for(tmp_path)
    refresh_card(spec, lambda d: d["records"][0].update(status="UNRESOLVED", reason="actual finite gate fails"))
    packet = runner.plan_study(spec, cases=cases)
    atlas, e22 = packet["trials"]
    assert atlas["family"] == "atlas" and atlas["status"] == "BLOCKED"
    assert all(r["status"] == "BLOCKED" for r in atlas["cases"])
    assert e22["status"] == "UNKNOWN" and not e22["capacity_blocked_cases"]
    assert len(packet["capacity_preflight"]) == 16


@pytest.mark.parametrize("mutation", [lambda p: p["trials"].pop(),
                                     lambda p: p["trials"][0]["cases"].pop(),
                                     lambda p: p["trials"][1].update(id=p["trials"][0]["id"]),
                                     lambda p: p["trials"][0].update(status="PASS")])
def test_omitted_or_forged_whole_winner_rejected(tmp_path, cases, monkeypatch, mutation):
    packet = planned(tmp_path, cases, monkeypatch)
    mutation(packet)
    with pytest.raises(ValueError):
        runner.select_results(packet)


def test_quality_admission_requires_both_smoke_passes_and_preserves_unknowns(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    trial = packet["trials"][0]
    assert runner.allow_next(packet, trial)["tier"] == 1
    trial["cases"][0]["status"] = "PASS"
    assert runner.allow_next(packet, trial)["id"] == "api-vector-two-broad"
    trial["cases"][1]["status"] = "FAIL"
    assert runner.allow_next(packet, trial) is None
    trial["cases"][2]["status"] = "RUNNING"
    with pytest.raises(ValueError, match="downstream"):
        runner.select_results(packet)


def test_whole_pass_keeps_ties_and_contended_speed_unknown(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    for trial in packet["trials"]:
        trial["status"] = "PASS"
        for row in trial["cases"]:
            row["status"] = "PASS"
    result = runner.select_results(packet)
    assert len(result["fully_qualified_ids"]) == 2 and result["outcome"] == "scoped_fully_qualified"
    assert result["speed_winner"] is None and result["default_adoption"] is False


@pytest.mark.parametrize("flags,status", [([True] * 10, "PASS"), ([True] * 9, "INCOMPLETE"),
                                          ([True] * 5 + [False] + [True] * 12, "FAIL")])
def test_inherited_first_window_and_every_later_primary_rule(flags, status):
    receipt = {"protocol": {"metric_evaluation_steps": list(range(len(flags) + 1))},
               "observations": [{"step": i, "passed": flag, "elapsed_seconds": i / 10}
                                for i, flag in enumerate([True, *flags])]}
    assert api_family_search.acquisition_hold(receipt)["status"] == status


@pytest.mark.parametrize("telemetry", [{"physical_gpu": 0, "free_mib": 20000., "temperature_c": 70.},
                                      {"physical_gpu": 1, "free_mib": 12287., "temperature_c": 70.},
                                      {"physical_gpu": 1, "free_mib": 20000., "temperature_c": 82.1},
                                      {"physical_gpu": 1, "free_mib": float("nan"), "temperature_c": 70.},
                                      {"physical_gpu": 1, "free_mib": 20000., "temperature_c": None}])
def test_unsafe_or_unknown_lane_rejected(monkeypatch, telemetry):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError):
        runner.validate_lane("cuda:0", telemetry=telemetry)


@pytest.mark.parametrize("visible,device", [("0", "cuda:0"), ("0,1", "cuda:1"), ("1", "cuda:1"), ("", "cpu")])
def test_lane_visibility_is_exact_physical_gpu1(monkeypatch, visible, device):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError):
        runner.validate_lane(device, telemetry={"physical_gpu": 1, "free_mib": 30000., "temperature_c": 30.})


def test_safe_admission_boundary(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert runner.validate_lane("cuda:0", telemetry={"physical_gpu": 1, "free_mib": 12288., "temperature_c": 82.})


def test_generated_child_uses_actual_public_parser_without_updates(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    trial, row = packet["trials"][0], packet["trials"][0]["cases"][0]
    command = runner.child_command(packet["spec"], trial, row, tmp_path / row["id"], "cuda:0")
    assert "--seed" not in command and "--steps" not in command
    calls = []
    def execute(request):
        case, output, options = request
        calls.append((case, output, options))
        return {"id": case["id"], "verdict": "PASS", "status": "COMPLETE"}
    monkeypatch.setattr(api_run, "_execute_request", execute)
    assert api_run.main(command[4:]) == 0
    assert len(calls) == 1 and calls[0][2]["seed"] == 24002
    assert calls[0][2]["recipe_overrides"] == runner.OVERRIDES
    assert calls[0][2]["steps"] is None and calls[0][2]["eval_samples"] is None
    assert calls[0][2]["wall_cap_seconds"] == 180


def test_child_resource_bootstrap_before_unchanged_public_main(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    command = runner.child_command(packet["spec"], packet["trials"][0], packet["trials"][0]["cases"][0],
                                   tmp_path / packet["trials"][0]["cases"][0]["id"], "cuda:0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    seen = []
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: seen.append(("interop", n)))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, device: seen.append(("memory", fraction, device)))
    monkeypatch.setattr(api_run, "main", lambda args: seen.append(("api", os.environ["CUBLAS_WORKSPACE_CONFIG"])) or 0)
    assert runner.child_main(command[4:]) == 0
    assert seen == [("interop", 1), ("memory", .2, 0), ("api", ":4096:8")]


@pytest.mark.parametrize("family", ["atlas", "e22"])
@pytest.mark.parametrize("identifier", ["api-vector-two-broad", "image-develop-img_intensity2-source-transpose12"])
def test_three_fields_bind_actual_public_optimizers_before_two_updates(cases, family, identifier):
    fixture = api_contract.build(cases[identifier], device="cpu", recipe_name=family,
                                 max_steps=2, recipe_overrides=runner.OVERRIDES)
    assert fixture.recipe.lr == .0053125 and fixture.recipe.d_lr_mult == 2.25 and fixture.recipe.prior_lr_mult == 1.5
    assert fixture.trainer.opt_d.param_groups[0]["lr"] == .0053125 * 2.25
    prior_ids = {id(p) for p in fixture.trainer.prior.parameters() if p.requires_grad}
    groups = [g for g in fixture.trainer.opt_g.param_groups if prior_ids & {id(p) for p in g["params"]}]
    assert len(groups) == 1 and groups[0]["lr"] == .0053125 * 1.5
    before = [p.detach().clone() for p in fixture.trainer.D.parameters()]
    fixture.step(); fixture.step()
    assert fixture.trainer.completed_steps == 2
    assert any(not torch.equal(a, b) for a, b in zip(before, fixture.trainer.D.parameters()))
    assert api_run.json_value(fixture.recipe.to_dict()) == api_family_search.resolved_recipe(cases[identifier], family, runner.OVERRIDES)


def test_reservation_refuses_next_full_task_when_budget_used(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    trial = packet["trials"][0]
    trial["paid_wall_seconds"] = 7680 - 239.999
    assert runner.allow_next(packet, trial) is None


@pytest.mark.parametrize("value", [-1., float("nan"), True])
def test_negative_nonfinite_bool_cost_rejected(tmp_path, cases, monkeypatch, value):
    packet = planned(tmp_path, cases, monkeypatch)
    packet["trials"][0]["cases"][0]["paid_wall_seconds"] = value
    with pytest.raises(ValueError):
        runner.verify_costs(packet)


def durable_fixture(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    packet["executed_family"] = "atlas"
    packet["lane_runtime"] = {"device": "cuda:0", "torch_threads": 1}
    packet["execution_source"] = {"digest": "software-source", "snapshot_path": str(tmp_path / "snapshot")}
    packet["coordinator"] = {"queue_root": str(tmp_path / "queue")}
    trial, row = packet["trials"][0], packet["trials"][0]["cases"][0]
    key = PolicyCoordinator.attempt_key(None, packet, trial, row)
    row.update(attempt_key=key, command=["synthetic-software-child"], paid_wall_seconds=2., child_returncode=0)
    directory = tmp_path / "queue/policy/attempts" / key
    directory.mkdir(parents=True)
    (directory / "supervisor-request.json").write_text(json.dumps({"command": row["command"], "source": packet["execution_source"],
                                                                "token": "software-token", "started_monotonic": 1., "deadline_monotonic": 241.}))
    (directory / "supervisor-terminal.json").write_text(json.dumps({"token": "software-token", "paid_wall_seconds": 2., "child_returncode": 0,
                                                                 "attempt_status": "completed"}))
    return packet, trial, row


def test_coherently_reduced_cost_cannot_replace_durable_supervisor(tmp_path, cases, monkeypatch):
    packet, trial, row = durable_fixture(tmp_path, cases, monkeypatch)
    runner.durable_cost(packet, trial, row)
    row["paid_wall_seconds"] = 0.
    with pytest.raises(ValueError, match="durable original"):
        runner.durable_cost(packet, trial, row)


def test_enlarged_quota_and_wrong_interrupt_cost_rejected(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    packet["family_paid_budget_seconds"] = {"atlas": 9000., "e22": 7680.}
    with pytest.raises(ValueError, match="quotas"):
        runner.verify_costs(packet)
    packet["family_paid_budget_seconds"] = packet["spec"]["family_budget_seconds"]
    packet["trials"][0]["cases"][0]["unmeasured_interrupt_reserved_seconds"] = 1.
    packet["trials"][0]["paid_wall_seconds"] = packet["spent_seconds"] = packet["unmeasured_interrupt_reservation_seconds"] = 1.
    with pytest.raises(ValueError, match="interrupt"):
        runner.verify_costs(packet)


def test_original_pass_but_hold_incomplete_is_recertified_not_skipped(tmp_path, cases, monkeypatch):
    packet, trial, row = durable_fixture(tmp_path, cases, monkeypatch)
    path = tmp_path / "case/receipt.json"
    path.parent.mkdir(); path.write_text('{}')
    verified = {"status": "INCOMPLETE", "original_gate": "PASS", "study_gate": "INCOMPLETE",
                "full_protocol_complete": True, "elapsed_seconds": 1., "receipt_path": str(path),
                "receipt_sha256": api_run.file_hash(path)}
    row.update(verified)
    trial.update(status="INCOMPLETE", paid_wall_seconds=2.)
    packet.update(spent_seconds=2., measured_paid_seconds=2.)
    calls = []
    monkeypatch.setattr(runner, "outcome", lambda *a: calls.append(True) or deepcopy(verified))
    runner.recertify_archive(packet)
    assert calls == [True]
    row["full_protocol_complete"] = False
    with pytest.raises(ValueError, match="completion"):
        runner.recertify_archive(packet)


def test_partial_exit_zero_retains_incomplete_not_numeric_fail_or_pass(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    trial, row = packet["trials"][0], packet["trials"][0]["cases"][0]
    raw = partial_receipt(packet, row)
    (tmp_path / "receipt.json").write_text(json.dumps(raw))
    result = runner.outcome(tmp_path, packet, trial, row, 0)
    assert result["status"] == "INCOMPLETE" and result["original_gate"] is None
    assert result["reported_original_verdict"] == "FAIL" and not result["full_protocol_complete"]


def test_readout_pairs_original_gif_grade_with_added_hold(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    row = packet["trials"][0]["cases"][0]
    row.update(original_gate="PASS", study_gate="FAIL", status="FAIL",
               acquisition_hold={"acquired_step": 250, "hold_passed": 3, "hold_checks": 5},
               receipt_path=str(tmp_path / "receipt.json"), artifacts={"goal.gif": {"sha256": "software-media"}})
    text = runner.human_readout(packet)
    assert "PASS | FAIL; confirm step 250; later 3/5" in text
    assert "No independent 100,000-output gate" in text and "original goal GIF" in text


def test_direct_cli_outside_checkout_needs_no_pythonpath_and_does_no_science(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(ROOT / runner.RELATIVE), "--help"], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and "combine" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def partial_receipt(packet, row):
    case = packet["case_definitions"][row["id"]]
    packet["lane_runtime"] = {"device": "cuda:0", "torch_threads": 1}
    metric = api_contract.evaluation_steps(case["default_steps"], api_contract.metric_observations(case) + 1)
    media = api_contract.evaluation_steps(case["default_steps"], 9)
    return {"status": "INCOMPLETE", "passed": False, "default_protocol_complete": False,
            "verdict": "FAIL", "artifacts": {}, "case": case, "seed": 24002,
            "requested_recipe_overrides": runner.OVERRIDES, "source": packet["source"],
            "runtime": packet["lane_runtime"], "recipe": None, "completed_updates": 0, "observations": [],
            "protocol": {"updates": case["default_steps"], "default_updates": case["default_steps"],
                         "evaluation_samples": case["eval_samples"], "default_evaluation_samples": case["eval_samples"],
                         "evaluation_steps": sorted(set(metric) | set(media)), "metric_evaluation_steps": metric,
                         "media_steps": media, "metric_observations": api_contract.metric_observations(case),
                         "media_frames": 9, "terminal_observations": 5, "wall_cap_seconds": row["timeout_seconds"]}}


@pytest.mark.parametrize("edit", [lambda r: r.update(seed=24003),
                                  lambda r: r["requested_recipe_overrides"].update(d_lr_mult=1.5),
                                  lambda r: r["source"].update(commit="0" * 40),
                                  lambda r: r["protocol"].update(updates=16),
                                  lambda r: r["protocol"].update(evaluation_samples=32),
                                  lambda r: r["runtime"].update(device="cpu"),
                                  lambda r: r.update(completed_updates=601)])
def test_partial_candidate_source_protocol_drift_rejected(tmp_path, cases, monkeypatch, edit):
    packet = planned(tmp_path, cases, monkeypatch)
    trial, row = packet["trials"][0], packet["trials"][0]["cases"][0]
    raw = deepcopy(partial_receipt(packet, row))
    edit(raw)
    (tmp_path / "receipt.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="attribution"):
        runner.outcome(tmp_path, packet, trial, row, 1)


@pytest.mark.parametrize("initialized,visible", [(True, ""), (False, "1"), (False, None)])
def test_cuda_parent_capacity_recheck_uses_fresh_cpu_subprocess(tmp_path, cases, monkeypatch, initialized, visible):
    spec = spec_for(tmp_path)
    records = json.loads(Path(spec["representation_card"]["path"]).read_text())["records"]
    expected = {r["family"] + "/" + r["case_id"]: r for r in records}
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    calls = []
    def invoke(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(expected), stderr="")
    monkeypatch.setattr(subprocess, "run", invoke)
    monkeypatch.setattr(runner, "capacity_module", lambda: pytest.fail("CPU verifier invoked in CUDA parent"))
    assert runner.capacity_outcomes(spec, cases) == expected
    assert calls[0][0][2] == "--verify-capacity"
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert calls[0][1]["env"]["OMP_NUM_THREADS"] == "1"
    assert calls[0][1]["timeout"] == 300


def test_recovered_interruption_preserves_measured_and_central_reserved_cost(tmp_path, cases, monkeypatch):
    packet, trial, row = durable_fixture(tmp_path, cases, monkeypatch)
    admission = {"status": "interrupted", "terminal": {"paid_wall_seconds": 2., "child_returncode": None},
                 "charged_seconds": 240., "command": row["command"]}
    runner.retain_terminal(packet, trial, row, None, admission)
    assert row["paid_wall_seconds"] == 2. and row["unmeasured_interrupt_reserved_seconds"] == 238.
    trial["paid_wall_seconds"] = packet["spent_seconds"] = 240.
    packet["measured_paid_seconds"], packet["unmeasured_interrupt_reservation_seconds"] = 2., 238.
    runner.verify_costs(packet)
    row["unmeasured_interrupt_reserved_seconds"] = 0.
    trial["paid_wall_seconds"] = packet["spent_seconds"] = 2.
    packet["unmeasured_interrupt_reservation_seconds"] = 0.
    with pytest.raises(ValueError, match="conservative"):
        runner.verify_costs(packet)


def test_recovered_interruption_cannot_invent_shared_allowance(tmp_path, cases, monkeypatch):
    packet, trial, row = durable_fixture(tmp_path, cases, monkeypatch)
    admission = {"status": "interrupted", "terminal": None, "charged_seconds": 1.}
    with pytest.raises(ValueError, match="conservative allowance"):
        runner.retain_terminal(packet, trial, row, None, admission)


def test_fresh_supervisor_error_retains_engineering_error_and_full_conservative_charge(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    packet["executed_family"] = "atlas"
    packet["lane_runtime"] = {"device": "cuda:0", "torch_threads": 1}
    packet["execution_source"] = {"digest": "software-source", "snapshot_path": str(tmp_path / "snapshot")}
    packet["coordinator"] = {"queue_root": str(tmp_path / "queue")}
    completed = []
    class Coordinator:
        root = tmp_path / "queue"
        attempt_key = PolicyCoordinator.attempt_key
        @contextmanager
        def admit(self, key, packet, row, device):
            self.key = key
            yield {"status": "running"}, None
        def launch(self, command, packet, log, leases, allowance):
            directory = self.root / "policy/attempts" / self.key
            directory.mkdir(parents=True)
            (directory / "supervisor-request.json").write_text(json.dumps({"token": "software-token", "command": command,
                "source": packet["execution_source"], "started_monotonic": 1., "deadline_monotonic": 241.}))
            (directory / "supervisor-terminal.json").write_text(json.dumps({"token": "software-token", "attempt_status": "error",
                "paid_wall_seconds": 2., "child_returncode": None, "reason": "synthetic infrastructure failure"}))
            raise RuntimeError("synthetic infrastructure failure")
        def complete(self, key, result):
            completed.append(deepcopy(result))
    monkeypatch.setattr(runner, "validate_lane", lambda *a: {"physical_gpu": 1, "free_mib": 20000., "temperature_c": 60.})
    output = tmp_path / "run"
    output.mkdir()
    runner.run_owned(packet, output, "atlas", Coordinator(), None)
    row = packet["trials"][0]["cases"][0]
    assert row["status"] == packet["trials"][0]["status"] == "ERROR"
    assert row["paid_wall_seconds"] == 2. and row["unmeasured_interrupt_reserved_seconds"] == 238.
    assert packet["spent_seconds"] == 240. and len(completed) == 1
    assert all(r["status"] == "UNKNOWN" for r in packet["trials"][0]["cases"][1:])
    runner.durable_cost(packet, packet["trials"][0], row)


def test_shared_completed_scientific_result_recertified_before_progression(tmp_path, cases, monkeypatch):
    packet, trial, row = durable_fixture(tmp_path, cases, monkeypatch)
    path = tmp_path / "retained/receipt.json"
    path.parent.mkdir(); path.write_text('{}')
    retained = {**deepcopy(row), "status": "PASS", "full_protocol_complete": True,
                "receipt_path": str(path), "receipt_sha256": api_run.file_hash(path)}
    monkeypatch.setattr(runner, "outcome", lambda *a: {"status": "FAIL", "full_protocol_complete": True})
    with pytest.raises(ValueError, match="grade differs"):
        runner.retain_terminal(packet, trial, row, None, {"status": "completed", "result": retained})
    path.write_text('{"modified": true}')
    with pytest.raises(ValueError, match="receipt changed"):
        runner.retain_terminal(packet, trial, row, None, {"status": "completed", "result": retained})


def test_noncompleted_terminal_cannot_drop_interruption_reserve(tmp_path, cases, monkeypatch):
    packet, trial, row = durable_fixture(tmp_path, cases, monkeypatch)
    terminal = tmp_path / "queue/policy/attempts" / row["attempt_key"] / "supervisor-terminal.json"
    data = json.loads(terminal.read_text()); data["attempt_status"] = "error"
    terminal.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="conservative interruption"):
        runner.durable_cost(packet, trial, row)


@pytest.mark.parametrize("edit", [lambda p: p.update(status="CAPTURING"), lambda p: p.update(required_records=15),
                                  lambda p: p.update(ordinary_training_updates=1), lambda p: p.update(fitting_updates=1),
                                  lambda p: p.update(ordinary_qualification_credit=True), lambda p: p["requested_cells"].pop(),
                                  lambda p: p.update(claim_scope="trained qualification")])
def test_capacity_header_scope_and_complete_denominator_rejected(tmp_path, cases, monkeypatch, edit):
    planned(tmp_path, cases, monkeypatch)
    spec = spec_for(tmp_path)
    refresh_card(spec, edit)
    with pytest.raises(ValueError, match="zero-update capacity"):
        runner.plan_study(spec, cases=cases)


def test_real_snapshot_public_discovery_has_original_task_and_all_eight_definitions(tmp_path, cases):
    """Exercise the omitted dependency in an isolated copied source, without models."""
    from experiments.forge.sources import verify_snapshot
    execution = runner.freeze_execution_source(ROOT, tmp_path / "source-cache", runner.source(cases))
    snapshot = Path(execution["snapshot_path"])
    verify_snapshot(snapshot, execution)
    for relative, sha in runner.DISCOVERY_INPUTS.items():
        assert execution["files"][relative] == sha
        assert api_run.file_hash(snapshot / relative) == sha
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(snapshot), PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    code = ("import json, pathlib, torch; from benchmarks.toy_audit import api_contract; "
            "from experiments.forge.contracts import stable_hash; torch.set_num_threads(1); "
            "c=api_contract.discover(); "
            "print(json.dumps({'root':str(api_contract.ROOT),'cases':{k:stable_hash(v) for k,v in c.items()},"
            "'cuda_initialized':torch.cuda.is_initialized()}))")
    process = subprocess.run([sys.executable, "-B", "-c", code], cwd=snapshot, env=environment,
                             capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    actual = json.loads(process.stdout)
    assert Path(actual["root"]).resolve() == snapshot.resolve()
    assert actual["cuda_initialized"] is False
    assert "api-ring16-acquisition" in actual["cases"]
    for row, sha in zip(runner.CASE_ROWS, runner.CASE_SHA256):
        assert actual["cases"][row[0]] == sha
    # Losing the precise original dependency must invalidate this test snapshot.
    (snapshot / next(iter(runner.DISCOVERY_INPUTS))).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        verify_snapshot(snapshot, execution)


def test_snapshot_rejects_changed_discovery_pin_before_export(tmp_path, cases):
    expected = runner.source(cases)
    expected["discovery_inputs_sha256"][next(iter(runner.DISCOVERY_INPUTS))] = "0" * 64
    with pytest.raises(ValueError, match="dependency binding"):
        runner.freeze_execution_source(ROOT, tmp_path / "source-cache", expected)
    assert not (tmp_path / "source-cache").exists()


def test_prior_bootstrap_cost_debits_budget_without_a_case_grade(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    spec = packet["spec"]
    assert spec["family_budget_seconds"]["atlas"] + runner.ENGINEERING_PAID_SECONDS == 7680.
    assert spec["family_budget_seconds"]["e22"] == 7680.
    assert spec["budget_seconds"] + runner.ENGINEERING_PAID_SECONDS == 15360.
    assert packet["spent_seconds"] == 0.
    trial = packet["trials"][0]
    trial["paid_wall_seconds"] = spec["family_budget_seconds"]["atlas"] - 239.999
    assert runner.allow_next(packet, trial) is None
    spec["engineering_carryover"]["paid_seconds"] = 0.
    with pytest.raises(ValueError, match="fixed"):
        runner.validate_spec(spec, cases)


def engineering_fixture(tmp_path, monkeypatch):
    row = {"id": runner.CASE_ROWS[0][0], "status": "ERROR", "attempt_key": runner.ENGINEERING_ATTEMPT,
           "paid_wall_seconds": runner.ENGINEERING_PAID_SECONDS, "full_protocol_complete": False}
    packet = {"executed_family": "atlas", "source": {"commit": "92167813cb2b04af0c4e0395984c503b7fb7d7fe"},
              "trials": [{"family": "atlas", "status": "ERROR", "paid_wall_seconds": runner.ENGINEERING_PAID_SECONDS,
                          "cases": [row, *[{"status": "UNKNOWN"} for _ in range(7)]]},
                         {"family": "e22", "status": "UNKNOWN", "paid_wall_seconds": 0.,
                          "cases": [{"status": "UNKNOWN"} for _ in range(8)]}],
              "spent_seconds": runner.ENGINEERING_PAID_SECONDS,
              "measured_paid_seconds": runner.ENGINEERING_PAID_SECONDS,
              "unmeasured_interrupt_reservation_seconds": 0.}
    artifacts = {}
    for role in ("study", "request", "terminal", "log"):
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps(packet if role == "study" else {"software_fixture": True}))
        artifacts[role] = {"path": str(path), "sha256": api_run.file_hash(path), "bytes": path.stat().st_size}
    monkeypatch.setattr(runner, "ENGINEERING_ARTIFACTS", artifacts)
    calls = []
    monkeypatch.setattr(runner, "durable_cost", lambda p, t, r: calls.append((p, t, r)))
    return artifacts, calls


def test_prior_error_requires_hash_bound_artifacts_and_durable_cost(tmp_path, monkeypatch):
    artifacts, calls = engineering_fixture(tmp_path, monkeypatch)
    assert runner.verify_engineering_carryover() == artifacts
    assert len(calls) == 1 and calls[0][2]["status"] == "ERROR"


@pytest.mark.parametrize("role", ["study", "request", "terminal", "log"])
def test_prior_error_artifact_tamper_rejected(tmp_path, monkeypatch, role):
    artifacts, calls = engineering_fixture(tmp_path, monkeypatch)
    Path(artifacts[role]["path"]).write_text('{"changed":true}')
    with pytest.raises(ValueError, match="artifact unavailable or changed"):
        runner.verify_engineering_carryover()
    assert not calls

"""Private CPU software controls: no scientific fixture, GPU or shared queue."""
from copy import deepcopy
from contextlib import contextmanager, nullcontext
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "reports/forge/atlas-current-gpu-diagnostics-v1"
loader = importlib.util.spec_from_file_location("_atlas_gpu_diagnostics_controls", DIRECTORY / "run_diagnostics.py")
diagnostic = importlib.util.module_from_spec(loader)
loader.loader.exec_module(diagnostic)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    for key, value in diagnostic.ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture
def spec():
    return diagnostic.read(DIRECTORY / "protocol.json")


@pytest.fixture
def current_source_spec(spec):
    """Private software metadata for today's declarations, never old evidence.

    The historical protocol stays byte-for-byte intact. These controls exercise
    reconstruction against current source; its different hashes create a
    different contract digest and confer no scientific or qualification credit.
    """
    current = deepcopy(spec)
    for row in current["cases"]:
        row["sha256"] = diagnostic.file_hash(ROOT / row["definition"])
    diagnostic.validate_spec(current, ROOT)
    return current


def base_request(spec):
    tasks = {}
    jobs = {}
    assignments = []
    for row in spec["cases"]:
        task = diagnostic.read(ROOT / row["definition"])
        task["preflight_blockers"] = [] if row["executable"] else ["unsupported conditional/component owner"]
        tasks[row["id"]] = task
        assignments.append({"task": row["id"], "qualification_tier": row["tier"],
                            "order": row["order"], "importance": "required"})
        job = jobs.setdefault(row["execution_group"], {"task_id": row["id"], "task_ids": [],
            "execution_group": row["execution_group"], "budget_seconds": row["timeout_seconds"],
            "resources": {"backend": "cuda", "cpu_threads": 1},
            "science": {"execution": {}, "evaluation": {}, "runtime": {"python": "synthetic"}}})
        job["task_ids"].append(row["id"])
        job["science"]["execution"][row["id"]] = diagnostic.digest(task["execution"])
        job["science"]["evaluation"][row["id"]] = diagnostic.digest(task["evaluation"])
    candidate = {"id": spec["candidate_id"], "recipe_preset": "atlas", "recipe_overrides": diagnostic.OVERRIDES,
                 "task_cohort": spec["task_cohort"], "prior": {}, "resolved_recipe": {"synthetic": True}}
    return {"candidate": candidate, "protocol": {"seed": 0}, "execution_backend": "cuda",
            "tasks": tasks, "jobs": list(jobs.values()),
            "view": {"id": "discriminator_stability", "revision": 4, "assignments": assignments},
            "preflight_blockers": ["ordinary draft"], "decision_review": {"blockers": ["ordinary draft"]},
            "source": {"digest": "b" * 64, "files": {}}, "runtime": {"python": "synthetic"},
            "compute_profiles": {"cuda": {"backend": "cuda", "model": "NVIDIA RTX A6000", "threads": 1}}}


def packet_for(spec):
    source = {"digest": "a" * 64, "files": {}, "snapshot_path": "/synthetic/no-model-source"}
    request = diagnostic.request_from_base(base_request(spec), spec, source)
    return {"schema": diagnostic.SCHEMA, "spec": deepcopy(spec), "spec_sha256": diagnostic.digest(spec),
            "source": source, "execution_source": source, "request": request,
            "family_paid_budget_seconds": {"atlas": 34800},
            "case_definitions": diagnostic.case_definitions(request, spec)}


def dumped(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def test_current_26_definitions_gpu_gates_and_group_caps(current_source_spec):
    spec = current_source_spec
    diagnostic.validate_spec(spec, ROOT)
    packet = packet_for(spec)
    state = diagnostic.initial_state(packet)
    assert len(state["slots"]) == 26
    assert sum(s["diagnostic_status"] == "BLOCKED" for s in state["slots"].values()) == 8
    assert sum(s["diagnostic_status"] == "NOT_RUN" for s in state["slots"].values()) == 18
    jobs = packet["request"]["jobs"]
    assert len(jobs) == 25
    assert sum(j["budget_seconds"] for j in jobs) == 45300
    supported = [j for j in jobs if not packet["request"]["tasks"][j["task_id"]]["preflight_blockers"]]
    assert len(supported) == 17 and sum(j["budget_seconds"] for j in supported) == 34800
    ring = next(j for j in jobs if j["execution_group"] == "ring_endurance_policy_selected_cloud_v1")
    assert set(ring["task_ids"]) == {"ring_hold" + diagnostic.SUFFIX, "ring_extension" + diagnostic.SUFFIX}
    assert ring["budget_seconds"] == 3600
    for parent in ("grid100", "rotated100", "staggered100"):
        task = packet["request"]["tasks"][parent + diagnostic.SUFFIX]
        assert task["execution"]["steps"] == 7000
        assert task["evaluation"]["eval_samples"] == 20000
        assert task["evaluation"]["holdout_samples"] == 100000
    assert all(s["qualification_input"] is False for s in state["slots"].values())
    assert state["ordinary_qualified_tier"] == 0


def test_historical_source_contract_rejects_current_declaration_drift(spec, current_source_spec):
    original = diagnostic.read(DIRECTORY / "protocol.json")
    assert spec == original
    assert diagnostic.digest(current_source_spec) != diagnostic.digest(original)
    with pytest.raises(ValueError, match="frozen task declaration changed"):
        diagnostic.validate_spec(original, ROOT)
    assert diagnostic.read(DIRECTORY / "protocol.json") == original


@pytest.mark.parametrize("key,value", [("seed", 1), ("seed", False), ("physical_gpu", "0"),
    ("diagnostic_cap_seconds", 45300), ("export_grace_seconds", 60), ("ordinary_tier_credit", True),
    ("calibration_credit", True), ("speed_ranking", True), ("default_adoption", True),
    ("recipe_overrides", {"lr": .0053125, "prior_lr_mult": 2.})])
def test_fixed_protocol_rejects_changed_scope(spec, key, value):
    spec[key] = value
    with pytest.raises(ValueError):
        diagnostic.validate_spec(spec)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "unblocked", "split", "budget", "nan"])
def test_denominator_and_full_allowances_fail_closed(spec, change):
    if change == "missing": spec["cases"].pop()
    elif change == "duplicate": spec["cases"][1] = deepcopy(spec["cases"][0])
    elif change == "unknown": spec["cases"][0]["parent_id"] = "invented"
    elif change == "unblocked": spec["cases"][1]["executable"] = True
    elif change == "split": spec["cases"][-1]["execution_group"] = "separate"
    elif change == "budget": spec["cases"][0]["timeout_seconds"] += 1
    elif change == "nan": spec["cases"][0]["timeout_seconds"] = float("nan")
    with pytest.raises(ValueError): diagnostic.validate_spec(spec)


def test_candidate_review_is_retained_and_only_exact_decision_blockers_are_advisory(spec):
    base = base_request(spec); before = deepcopy(base)
    result = diagnostic.request_from_base(base, spec, {"digest": "a" * 64, "files": {}})
    assert base == before and result["decision_review"] == base["decision_review"]
    assert result["ordinary_request_sha256"] == diagnostic.digest(base)
    assert result["qualification_reuse"] is False
    assert all(j["science"]["evidence_use"] == "current_policy_task_diagnostic" for j in result["jobs"])
    base["preflight_blockers"].append("actual source corruption")
    with pytest.raises(ValueError, match="non-decision"):
        diagnostic.request_from_base(base, spec, {"digest": "a" * 64})


@pytest.mark.parametrize("change", [None, "recipe", "numeric_type", "runtime", "gate", "job", "source"])
def test_serialized_preparation_reconstructs_actual_recipe_metadata_and_rejects_drift(current_source_spec, tmp_path, monkeypatch, change):
    """Exercise the durable entry point with public Recipe tuples and a real snapshot.

    Only planner discovery/import routing is replaced by private software metadata;
    request construction, JSON serialization, snapshot hashes and packet guards run.
    """
    from particlegan import get_recipe
    from experiments.forge.contracts import stable_hash
    from experiments.forge import planning

    spec = current_source_spec
    snapshot = tmp_path / "source"
    files = {}
    for relative in [diagnostic.SELF] + [row["definition"] for row in spec["cases"]]:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
        files[relative] = diagnostic.file_hash(destination)
    metadata = {"schema_version": 1, "files": files, "digest": stable_hash(files), "origin_commit": None}
    dumped(snapshot / "forge-source.json", metadata)
    source = {**metadata, "snapshot_path": str(snapshot)}
    base = base_request(spec)
    base["source"] = metadata
    base["candidate"]["resolved_recipe"] = get_recipe("atlas").to_dict()
    assert isinstance(base["candidate"]["resolved_recipe"]["betas"], tuple)
    monkeypatch.setattr(planning, "resolve_idea", lambda *args, **kwargs: deepcopy(base))
    monkeypatch.setattr(diagnostic, "guard_imports", lambda *args, **kwargs: None)
    request = diagnostic.build_request(snapshot, spec, source)
    card = diagnostic.card_for(request, spec)
    card_path = dumped(tmp_path / "card.json", card)
    packet = {"schema": diagnostic.SCHEMA, "spec": {**spec, "representation_card": diagnostic.pin(card_path)},
              "spec_sha256": diagnostic.digest(spec), "request": request, "source": source,
              "execution_source": source, "case_definitions": diagnostic.case_definitions(request, spec),
              "capacity_preflight": card, "runtime_contract": request["runtime"],
              "family_paid_budget_seconds": {"atlas": 34800}}
    packet = diagnostic.read(dumped(tmp_path / "preparation.json", packet))
    assert packet["request"]["candidate"]["resolved_recipe"]["betas"] == [0., .999]
    assert base["candidate"]["resolved_recipe"]["betas"] == (0., .999)
    if change == "recipe": packet["request"]["candidate"]["resolved_recipe"]["alpha_bar"][1] = .8
    elif change == "numeric_type": packet["request"]["candidate"]["resolved_recipe"]["betas"][0] = False
    elif change == "runtime": packet["request"]["runtime"]["python"] = "different-runtime"
    elif change == "gate": packet["request"]["tasks"][spec["cases"][0]["id"]]["evaluation"]["thresholds"] = []
    elif change == "job": packet["request"]["jobs"][0]["compatibility_key"] = "foreign"
    elif change == "source": (snapshot / diagnostic.SELF).write_text("changed pinned wrapper\n")
    if change is None:
        assert diagnostic.verify_packet(packet) == spec
    else:
        with pytest.raises(ValueError): diagnostic.verify_packet(packet)


def test_unsupported_change_requires_new_contract_not_automatic_expansion(spec):
    base = base_request(spec)
    base["tasks"]["five_word_joint_acquisition" + diagnostic.SUFFIX]["preflight_blockers"] = []
    with pytest.raises(ValueError, match="18/8"):
        diagnostic.request_from_base(base, spec, {"digest": "a" * 64})


def test_group_attempt_identity_contains_both_ring_task_definitions(spec):
    packet = packet_for(spec)
    group = packet["case_definitions"]["ring_hold" + diagnostic.SUFFIX]
    assert set(group["execution_group_members"]) == {"ring_hold" + diagnostic.SUFFIX, "ring_extension" + diagnostic.SUFFIX}
    original = diagnostic.digest(group)
    group["execution_group_members"]["ring_extension" + diagnostic.SUFFIX]["evaluation"]["extension_steps"] = 299
    assert diagnostic.digest(group) != original


def test_resolved_refuses_blocked_foreign_device_and_checkpoint(spec, monkeypatch):
    packet = packet_for(spec)
    job = packet["request"]["jobs"][0]
    resolved = {"packet": packet, "packet_sha256": diagnostic.digest(packet), "request": packet["request"],
        "job": job, "worker": {"device": "1", "lane_runtime": diagnostic.lane_runtime(packet["request"])}, "prerequisites": {}}
    monkeypatch.setattr(diagnostic, "verify_packet", lambda p: spec)
    diagnostic.validate_resolved(resolved)
    resolved["worker"]["device"] = "0"
    with pytest.raises(ValueError, match="GPU/runtime"): diagnostic.validate_resolved(resolved)
    resolved["worker"]["device"] = "1"
    job["science"]["execution"][job["task_id"]] = "corrupt"
    resolved["packet_sha256"] = diagnostic.digest(packet)
    packet["request"]["tasks"][job["task_id"]]["dependencies"] = [{"kind": "checkpoint", "task": "missing"}]
    resolved["packet_sha256"] = diagnostic.digest(packet)
    with pytest.raises(ValueError, match="checkpoint"): diagnostic.validate_resolved(resolved)


@pytest.mark.parametrize("telemetry", ["1, NVIDIA RTX A6000, 12287, 60", "1, NVIDIA RTX A6000, 20000, 83",
    "0, NVIDIA RTX A6000, 20000, 60", "1, wrong, 20000, 60", "1, NVIDIA RTX A6000, nan, 60", ""])
def test_unsafe_lane_never_reaches_reservation(spec, telemetry):
    with pytest.raises(ValueError): diagnostic.gpu_readiness(spec, query=lambda command: telemetry)


def test_safe_telemetry_is_metadata_only(spec):
    result = diagnostic.gpu_readiness(spec, query=lambda command: "1, NVIDIA RTX A6000, 12288, 82")
    assert result["physical_gpu"] == "1"


def cost_row(tmp_path, job, terminal_status="completed", paid=5.):
    terminal = dumped(tmp_path / (job["task_id"] + ".json"),
                      {"attempt_status": terminal_status, "paid_wall_seconds": paid, "token": "owned"})
    return {"compatibility_key": job["compatibility_key"], "task_ids": job["task_ids"], "token": "owned",
            "terminal": diagnostic.pin(terminal), **diagnostic.charge(paid, diagnostic.read(terminal), job["budget_seconds"])}


def totals(state):
    state["measured_paid_seconds"] = sum(r["paid_wall_seconds"] for r in state["jobs"])
    state["unmeasured_interrupt_reserved_seconds"] = sum(r["unmeasured_interrupt_reserved_seconds"] for r in state["jobs"])
    state["spent_seconds"] = state["measured_paid_seconds"] + state["unmeasured_interrupt_reserved_seconds"]


@pytest.mark.parametrize("terminal", ["timeout", "error", "cancelled"])
def test_interruption_preserves_measured_and_conservative_cost(spec, tmp_path, terminal):
    state = diagnostic.initial_state(packet_for(spec)); job = state["request"]["jobs"][0]
    state["jobs"] = [cost_row(tmp_path, job, terminal)]
    totals(state); diagnostic.verify_costs(state)
    assert state["measured_paid_seconds"] == 5.
    assert state["spent_seconds"] == job["budget_seconds"]
    assert state["unmeasured_interrupt_reserved_seconds"] == job["budget_seconds"] - 5.


def test_coherent_zero_cost_quota_and_duplicate_tampering_rejected(spec, tmp_path):
    original = diagnostic.initial_state(packet_for(spec)); job = original["request"]["jobs"][0]
    original["jobs"] = [cost_row(tmp_path, job)]; totals(original); diagnostic.verify_costs(original)
    zero = deepcopy(original)
    zero["jobs"][0].update(paid_wall_seconds=0., charged_seconds=0.); totals(zero)
    with pytest.raises(ValueError, match="durable measured"): diagnostic.verify_costs(zero)
    duplicate = deepcopy(original); duplicate["jobs"].append(deepcopy(duplicate["jobs"][0])); totals(duplicate)
    with pytest.raises(ValueError, match="duplicate"): diagnostic.verify_costs(duplicate)
    quota = deepcopy(original); quota["family_paid_budget_seconds"]["atlas"] = 45300
    with pytest.raises(ValueError, match="quota"): diagnostic.verify_costs(quota)
    qualifying = deepcopy(original); qualifying["ordinary_qualified_tier"] = 1
    with pytest.raises(ValueError, match="scope"): diagnostic.verify_costs(qualifying)


def test_real_main_parser_child_arguments_not_help(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(diagnostic, "child", lambda path, fd: observed.append((path, fd)) or 0)
    assert diagnostic.main(["--child", str(tmp_path / "resolved.json"), "--lease-fd", "17"]) == 0
    assert observed == [(tmp_path / "resolved.json", 17)]
    with pytest.raises(SystemExit): diagnostic.main(["--child", "x", "--lease-fd", "17", "--seed", "0"])
    with pytest.raises(ValueError): diagnostic.main(["--child", "x"])


def test_prepare_cli_is_queue_free_and_keeps_output_fresh(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(diagnostic, "prepare", lambda spec, output: called.append(output) or {
        "spec_sha256": "a" * 64, "source": {"digest": "b" * 64}})
    monkeypatch.setattr(diagnostic, "run", lambda *args: pytest.fail("prepare must not reserve a queue"))
    assert diagnostic.main(["--output", str(tmp_path / "fresh"), "--prepare-only"]) == 0
    assert called == [tmp_path / "fresh"] and not (tmp_path / "fresh").exists()
    assert diagnostic.preparation_path(tmp_path / "fresh").parent == tmp_path


def test_actual_coordinator_uses_the_supplied_shared_queue_not_the_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "private-checkout"; checkout.mkdir()
    shared = tmp_path / "private-shared-queue"
    monkeypatch.setattr(diagnostic, "ROOT", checkout)
    coordinator = diagnostic.coordinator_for(shared)
    assert coordinator.queue.root == shared.resolve()
    assert coordinator.root == shared.resolve()
    assert not (checkout / "queue").exists()
    assert not (checkout / "state.json").exists()


def test_unadmitted_stage_refuses_before_runtime_or_gpu(monkeypatch, tmp_path):
    path = dumped(tmp_path / "resolved.json", {})
    monkeypatch.setattr(diagnostic, "validate_resolved", lambda r: {"source": {}})
    monkeypatch.setattr(diagnostic, "guard_imports", lambda s: None)
    monkeypatch.delenv("FORGE_LEASE_FD", raising=False)
    with pytest.raises(KeyError, match="FORGE_LEASE_FD"):
        diagnostic.stage(path, execute=True)


def test_durable_cpu_supervisor_success_and_source_drift(tmp_path):
    from experiments.forge.policy_execution import supervise
    from experiments.forge.contracts import stable_hash
    import time
    snapshot = tmp_path / "source"; snapshot.mkdir()
    payload = snapshot / "payload.py"; payload.write_text("print('private CPU control')\n")
    files = {"payload.py": diagnostic.file_hash(payload)}
    source = {"schema_version": 1, "files": files, "digest": stable_hash(files), "origin_commit": None}
    dumped(snapshot / "forge-source.json", source)
    source["snapshot_path"] = str(snapshot)
    request = {"token": "control", "source": source, "command": [sys.executable, str(payload)],
               "log_path": str(tmp_path / "control.log"), "lease_fds": [],
               "started_monotonic": time.monotonic(), "deadline_monotonic": time.monotonic() + 5.}
    path = dumped(tmp_path / "supervisor-request.json", request)
    assert supervise(path) == 0
    terminal = diagnostic.read(tmp_path / "supervisor-terminal.json")
    assert terminal["attempt_status"] == "completed" and terminal["child_returncode"] == 0
    assert terminal["paid_wall_seconds"] >= 0
    payload.write_text("raise RuntimeError('changed source must never execute')\n")
    request.update(started_monotonic=time.monotonic(), deadline_monotonic=time.monotonic() + 5.)
    dumped(path, request)
    assert supervise(path) == 1
    assert diagnostic.read(tmp_path / "supervisor-terminal.json")["attempt_status"] == "error"


def test_actual_child_separates_gpu_runtime_and_cpu_grade(monkeypatch, tmp_path):
    dumped(tmp_path / "resolved.json", {})
    dumped(tmp_path / "graded-result.json", {"grades": {"case": {"gate_status": "FAIL"}}})
    monkeypatch.setattr(diagnostic, "validate_resolved", lambda r: {})
    monkeypatch.setattr(diagnostic, "child_environment", lambda fd, r: None)
    calls = []
    def runner(command, **kwargs):
        calls.append((command, kwargs)); return subprocess.CompletedProcess(command, 0)
    assert diagnostic.child(tmp_path / "resolved.json", 7, runner=runner) == 0
    assert calls[0][0][-2] == "--execute" and calls[1][0][-2] == "--evaluate"
    assert calls[0][1]["pass_fds"] == calls[1][1]["pass_fds"] == (7,)
    assert calls[1][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    dumped(tmp_path / "graded-result.json", {"grades": {"case": {"gate_status": "INVALID"}}})
    assert diagnostic.child(tmp_path / "resolved.json", 7, runner=runner) == 2


def test_nominal_seed_is_not_cli_overridable(spec):
    assert spec["seed"] == 0
    args = diagnostic.parser().parse_args(["--output", "/unused", "--queue-root", "/unused-queue", "--gpus", "1"])
    assert not hasattr(args, "seed") and not hasattr(args, "recipe")


def test_protocol_snapshot_and_import_origin_fail_closed(tmp_path):
    snapshot = tmp_path / "source"; (snapshot / "lib").mkdir(parents=True)
    member = snapshot / "lib/example.py"; member.write_text("VALUE = 1\n")
    source = {"snapshot_path": str(snapshot), "files": {"lib/example.py": diagnostic.file_hash(member)}}
    namespace = SimpleNamespace(__file__=None, __path__=[str(snapshot / "lib")])
    child = SimpleNamespace(__file__=str(member))
    modules = {"lib": namespace, "lib.example": child}
    diagnostic.guard_imports(source, modules=modules)
    namespace.__path__.append("/wrong/lib")
    with pytest.raises(ValueError, match="namespace"): diagnostic.guard_imports(source, modules=modules)
    namespace.__path__ = [str(snapshot / "lib")]
    member.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="unpinned"): diagnostic.guard_imports(source, modules=modules)


def test_parent_reconstruction_must_use_the_same_pinned_implementation(tmp_path):
    parent = tmp_path / "parent"; (parent / "experiments").mkdir(parents=True)
    module = parent / "experiments/example.py"; module.write_text("VALUE = 1\n")
    source = {"snapshot_path": str(tmp_path / "snapshot"), "files": {"experiments/example.py": diagnostic.file_hash(module)}}
    modules = {"experiments.example": SimpleNamespace(__file__=str(module))}
    diagnostic.guard_imports(source, execution_root=parent, modules=modules)
    with pytest.raises(ValueError, match="foreign"):
        diagnostic.guard_imports(source, modules=modules)
    module.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="unpinned"):
        diagnostic.guard_imports(source, execution_root=parent, modules=modules)


@pytest.mark.parametrize("change", ["token", "source", "descriptor", "path"])
def test_real_descriptor_is_bound_to_its_durable_source_and_token(tmp_path, change):
    lease_path = tmp_path / "execution.lock"; lease_path.write_text("")
    source = {"digest": "a" * 64}
    with lease_path.open() as lease:
        resolved = {"worker": {"lease_path": str(lease_path), "token": "owned"}, "packet": {"execution_source": source}}
        receipt = {"token": "owned", "source": source, "lease_fds": [lease.fileno()]}
        dumped(tmp_path / "supervisor-request.json", receipt)
        diagnostic.verify_lease(lease.fileno(), resolved)
        if change == "token": receipt["token"] = "foreign"
        elif change == "source": receipt["source"] = {"digest": "b" * 64}
        elif change == "descriptor": receipt["lease_fds"] = []
        else: resolved["worker"]["lease_path"] = str(tmp_path / "wrong.lock")
        dumped(tmp_path / "supervisor-request.json", receipt)
        with pytest.raises(ValueError): diagnostic.verify_lease(lease.fileno(), resolved)


def test_known_parent_interruption_cost_is_not_lost_when_terminal_is_missing(spec, tmp_path):
    state = diagnostic.initial_state(packet_for(spec)); job = state["request"]["jobs"][0]
    error = dumped(tmp_path / "error.json", {"token": "owned", "source_digest": state["source"]["digest"],
                                             "paid_wall_seconds": 3.5})
    row = {"compatibility_key": job["compatibility_key"], "task_ids": job["task_ids"], "token": "owned",
           "launch_error": diagnostic.pin(error), **diagnostic.charge(3.5, None, job["budget_seconds"])}
    state["jobs"] = [row]; totals(state); diagnostic.verify_costs(state)
    assert state["measured_paid_seconds"] == 3.5 and state["spent_seconds"] == job["budget_seconds"]
    state["jobs"][0].update(paid_wall_seconds=0., charged_seconds=job["budget_seconds"],
                            unmeasured_interrupt_reserved_seconds=job["budget_seconds"])
    totals(state)
    with pytest.raises(ValueError, match="interruption charge"): diagnostic.verify_costs(state)


def test_complete_allowance_boundary_and_nonfinite_costs():
    assert diagnostic.fits_allowance(31200., 3600.)
    assert not diagnostic.fits_allowance(31200.001, 3600.)
    for value in (float("nan"), float("inf"), -1., True):
        with pytest.raises(ValueError): diagnostic.fits_allowance(value, 3600.)


@pytest.mark.parametrize("change", ["slot", "prefix", "incomplete_full", "missing_grade", "default"])
def test_status_denominator_and_order_cannot_be_forged(spec, tmp_path, change):
    state = diagnostic.initial_state(packet_for(spec)); job = state["request"]["jobs"][0]
    row = {**cost_row(tmp_path, job), "status": "COMPLETE", "outcome": {
        "statuses": {job["task_id"]: "FAIL"}, "qualification_input": False}}
    state["jobs"] = [row]; state["slots"][job["task_id"]]["diagnostic_status"] = "FAIL"; totals(state)
    diagnostic.verify_state(state)
    if change == "slot": state["slots"][job["task_id"]]["diagnostic_status"] = "PASS"
    elif change == "prefix":
        next_job = state["request"]["jobs"][3]
        row.update(compatibility_key=next_job["compatibility_key"], task_ids=next_job["task_ids"])
    elif change == "incomplete_full": state["status"] = "COMPLETE_DIAGNOSTIC"
    elif change == "missing_grade": row["outcome"]["statuses"] = {}
    else: state["default_adoption"] = True
    with pytest.raises(ValueError): diagnostic.verify_state(state)


@pytest.mark.parametrize("change", [None, "step", "foreign_input", "renderer", "frame_count", "grade", "artifact"])
def test_complete_numerical_fail_media_and_artifacts_remain_bound(spec, tmp_path, monkeypatch, change):
    from PIL import Image
    from experiments.forge.artifacts import manifest_artifacts
    packet = packet_for(spec); packet["source"]["files"][diagnostic.SELF] = diagnostic.file_hash(DIRECTORY / "run_diagnostics.py")
    job = packet["request"]["jobs"][0]; name = job["task_id"]
    artifact_root = tmp_path / "artifacts"; artifact_root.mkdir()
    points = [{"step": 0}, {"step": 80}]
    for point in points: dumped(artifact_root / f"observations/step_{point['step']:06d}.npz", {"private_software_fixture": True})
    evidence = {"artifact_root": str(artifact_root), "artifact_manifest": manifest_artifacts(artifact_root), "observations": points}
    resolved_path = dumped(tmp_path / "resolved.json", {"packet": packet, "request": packet["request"], "job": job})
    raw = {"evidence": evidence}; dumped(tmp_path / "raw-result.json", raw)
    grade = {"gate_status": "FAIL", "reason": "software-control numerical failure"}
    dumped(tmp_path / "graded-result.json", {"raw_hash": diagnostic.digest(raw), "source_digest": packet["source"]["digest"],
                                            "grades": {name: grade}})
    gif = tmp_path / (name + "-goal.gif")
    Image.new("RGB", (4, 4), "black").save(gif, save_all=True, append_images=[Image.new("RGB", (4, 4), "white")], duration=100)
    media = {"schema": diagnostic.SCHEMA + "_media", "task": name, "source_digest": packet["source"]["digest"],
             "original_gate": "FAIL", "qualification_input": False, "draws": 0, "optimizer_updates": 0,
             "renderer_sha256": packet["source"]["files"][diagnostic.SELF], "actual_steps": [0, 80],
             "inputs": [diagnostic.pin(artifact_root / f"observations/step_{p['step']:06d}.npz") for p in points],
             "gif": diagnostic.pin(gif)}
    if change == "step": media["actual_steps"] = [0, 79]
    elif change == "foreign_input": media["inputs"][0] = diagnostic.pin(dumped(tmp_path / "foreign.json", {}))
    elif change == "renderer": media["renderer_sha256"] = "b" * 64
    elif change == "frame_count": Image.new("RGB", (4, 4), "black").save(gif); media["gif"] = diagnostic.pin(gif)
    elif change == "grade": media["original_gate"] = "PASS"
    elif change == "artifact": dumped(artifact_root / "observations/step_000080.npz", {"changed": True})
    dumped(tmp_path / (name + "-media.json"), media)
    monkeypatch.setattr(diagnostic, "validate_resolved", lambda r: packet)
    invoke = lambda: diagnostic.certified_outcome(resolved_path, grader=lambda task, member: grade, policy_guard=lambda task, e: None)
    if change is None:
        assert invoke()["statuses"] == {name: "FAIL"}
    else:
        with pytest.raises(ValueError): invoke()


@pytest.mark.parametrize("invalid_at", [None, 0])
def test_real_dispatch_continues_numeric_fail_but_halts_invalid_and_never_admits_blockers(spec, tmp_path, monkeypatch, invalid_at):
    packet = packet_for(spec); output = tmp_path / "output"
    dumped(diagnostic.preparation_path(output), packet)
    admitted = []
    class FakeCoordinator:
        def __init__(self, root, report_root): pass
        def register(self, packet, output, family, runtime):
            packet["coordinator"] = {"study_key": "private"}; output.mkdir()
            return "private", output
        def study_lease(self, key): return nullcontext(SimpleNamespace(name="private-study"))
        def attempt_key(self, packet, trial, row): return row["id"]
        @contextmanager
        def admit(self, attempt, packet, row, device):
            assert (output / "study.json").exists(), "registration state must exist before first attempt"
            admitted.append(row["id"])
            lease_path = tmp_path / "durables" / attempt / "execution.lock"
            lease_path.parent.mkdir(parents=True, exist_ok=True); lease_path.write_text("")
            with lease_path.open() as lease:
                yield {"status": "running", "token": attempt, "lease_path": str(lease_path)}, lease
        def launch(self, command, packet, log, leases, allowance):
            directory = Path(leases[-1].name).parent
            dumped(directory / "supervisor-terminal.json", {"token": directory.name, "attempt_status": "completed",
                   "paid_wall_seconds": 1., "child_returncode": 1 if len(admitted) - 1 == invalid_at else 0})
        def complete(self, attempt, result): pass
    original_forge = diagnostic.forge
    monkeypatch.setattr(diagnostic, "forge", lambda name: SimpleNamespace(PolicyCoordinator=FakeCoordinator)
                        if name == "policy_execution" else original_forge(name))
    monkeypatch.setattr(diagnostic, "verify_packet", lambda p: spec)
    monkeypatch.setattr(diagnostic, "gpu_readiness", lambda s: {"physical_gpu": "1"})
    def outcome(path):
        resolved = diagnostic.read(path)
        return {"resolved": diagnostic.pin(path), "statuses": {m: "FAIL" for m in resolved["job"]["task_ids"]},
                "qualification_input": False}
    monkeypatch.setattr(diagnostic, "certified_outcome", outcome)
    state = diagnostic.run(output, tmp_path / "private-no-real-queue")
    assert not any(s["parent_id"] in diagnostic.BLOCKED for s in (state["slots"][i] for i in admitted))
    if invalid_at is None:
        assert len(admitted) == 17 and state["status"] == "COMPLETE_DIAGNOSTIC"
        assert sum(s["diagnostic_status"] == "FAIL" for s in state["slots"].values()) == 18
        assert state["spent_seconds"] == 17. and state["ordinary_qualified_tier"] == 0
        ring = [j for j in state["jobs"] if len(j["task_ids"]) == 2]
        assert len(ring) == 1 and ring[0]["charged_seconds"] == 1.
    else:
        assert len(admitted) == 1 and state["status"] == "INVALID"
        assert sum(s["diagnostic_status"] == "NOT_RUN" for s in state["slots"].values()) == 17


def test_no_cuda_context_was_initialized_by_controls():
    import torch
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not torch.cuda.is_initialized()

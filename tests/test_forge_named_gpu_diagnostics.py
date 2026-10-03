"""CPU-only private orchestration controls; no learned qualification or queue."""
from copy import deepcopy
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "reports/forge/atlas-named-gpu-diagnostics-v1"
loader = importlib.util.spec_from_file_location("_named_gpu_diagnostic_controls", DIRECTORY / "run_diagnostics.py")
diagnostic = importlib.util.module_from_spec(loader)
loader.loader.exec_module(diagnostic)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name, value in diagnostic.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def spec():
    return diagnostic.read(DIRECTORY / "protocol.json")


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True))
    return path


def base_for(spec, family):
    info = diagnostic.FAMILIES[family]
    mapping = {p: p + "_" + info["cohort"] for p in info["parents"]}
    tasks, assignments, jobs = {}, [], []
    for index, parent in enumerate(diagnostic.legacy().PARENTS):
        name = mapping.get(parent, parent)
        path = (ROOT / "configs/forge/task-variants" / info["cohort"] / (name + ".json")) if parent in mapping else ROOT / "configs/forge/tasks" / (name + ".json")
        task = diagnostic.read(path); task["preflight_blockers"] = [] if parent in mapping else ["outside this diagnostic subset"]
        tasks[name] = task
        tier = 1 if index < 5 else 2 if index < 24 else 3
        assignments.append({"task": name, "qualification_tier": tier, "importance": "required", "order": index - (0 if tier == 1 else 5 if tier == 2 else 24)})
        jobs.append({"task_id": name, "task_ids": [name], "execution_group": name,
                     "budget_seconds": task["resources"]["timeout_seconds"], "resources": {"backend": "cuda", "cpu_threads": 1},
                     "science": {"execution": {name: diagnostic.digest(task["execution"])}, "evaluation": {name: diagnostic.digest(task["evaluation"])}}})
    return {"candidate": {**diagnostic.declaration(ROOT, spec, family), "id": family + "-c6-named-gpu-diagnostic-v1", "trainer_family": family,
                "recipe_preset": "atlas", "recipe_overrides": deepcopy(diagnostic.OVERRIDES), "execution_path": "public_trainer",
                "task_cohort": info["cohort"], "resolved_recipe": {"software_fixture": True}, "prior": {}},
            "protocol": {"seed": 0}, "execution_backend": "cuda", "tasks": tasks, "jobs": jobs,
            "view": {"id": "discriminator_stability", "revision": 4, "policy_family": family, "assignments": assignments},
            "preflight_blockers": [diagnostic.LEGACY_ADMISSION_BLOCKER],
            "runtime": {"python": "private_software_fixture"},
            "compute_profiles": {"cuda": {"backend": "cuda", "model": "NVIDIA RTX A6000", "availability": "declared", "threads": 1}}}


def packet_for(spec):
    source = {"digest": "a" * 64, "files": {}, "snapshot_path": "/private/software/no-model"}
    requests = {f: diagnostic.request_from_base(base_for(spec, f), spec, f, source) for f in diagnostic.FAMILIES}
    return {"schema": diagnostic.SCHEMA, "spec": deepcopy(spec), "spec_sha256": diagnostic.digest(spec), "source": source,
            "execution_source": source, "requests": requests, "runtime_contract": requests["atlas_conditional"]["runtime"],
            "case_definitions": diagnostic.case_definitions(requests, spec),
            "engineering_carryover": diagnostic.engineering_carryover(spec, ROOT),
            "continuation_carryover": diagnostic.continuation_carryover(spec, ROOT),
            "family_paid_budget_seconds": {f: v["cap"] for f, v in diagnostic.FAMILIES.items()}}


def test_three_remaining_cases_and_five_complete_separate_denominators(spec):
    diagnostic.validate_spec(spec, ROOT)
    packet = packet_for(spec)
    assert sum(i["cap"] for i in diagnostic.FAMILIES.values()) == 10500
    assert {g: sum(i["cap"] for i in diagnostic.FAMILIES.values() if i["gpu"] == g) for g in ("0", "1")} == {"0": 7500, "1": 3000}
    for family in diagnostic.ACTIVE_FAMILIES:
        info = diagnostic.FAMILIES[family]
        p = diagnostic.family_packet(packet, family); state = diagnostic.initial_state(p)
        assert len(state["slots"]) == 26 and all(v["diagnostic_status"] == "NOT_RUN" for v in state["slots"].values())
        assert [sum(a["tier"] == tier for a in state["slots"].values()) for tier in (1, 2, 3)] == [5, 19, 2]
        assert sum(v["in_diagnostic_batch"] for v in state["slots"].values()) == len(info["parents"])
        assert state["historical_original_word"]["status"] == "BLOCKED" and state["qualification_input"] is False
        diagnostic.verify_state(state)


@pytest.mark.parametrize("key,value", [("seed", False), ("seed", 1), ("diagnostic_cap_seconds", 10501),
    ("lane_cap_seconds", {"0": 3000, "1": 7500}), ("export_grace_seconds", 60), ("qualification_input", True),
    ("ordinary_tier_credit", True), ("cross_cohort_pooling", True), ("default_adoption", True), ("speed_ranking", True),
    ("recipe_overrides", {"lr": .0053125, "prior_lr_mult": 2.})])
def test_fixed_protocol_scope_fails_closed(spec, key, value):
    spec[key] = value
    with pytest.raises(ValueError): diagnostic.validate_spec(spec)


@pytest.mark.parametrize("change", ["missing", "duplicate", "family", "gpu", "steps", "cap", "source", "gate"])
def test_no_borrowed_case_or_changed_full_protocol(spec, change):
    if change == "missing": spec["cases"].pop()
    elif change == "duplicate": spec["cases"][1] = deepcopy(spec["cases"][0])
    else:
        key, value = {"family": ("family", "atlas"), "gpu": ("gpu", "0"), "steps": ("steps", 199),
                      "cap": ("timeout_seconds", 1799), "source": ("sha256", "0" * 64), "gate": ("evaluation_sha256", "0" * 64)}[change]
        spec["cases"][0][key] = value
    with pytest.raises(ValueError): diagnostic.validate_spec(spec, ROOT)


def test_real_metadata_planner_and_all_eight_actual_contexts_need_no_model(spec, monkeypatch):
    """Actual planner/context; only GPU metadata and Git-backed scan are synthetic."""
    import torch
    from experiments.forge import planning
    from experiments.forge.api import task_formulation_context
    from particlegan import Recipe
    def metadata_source(root, extras):
        files = {p: diagnostic.file_hash(Path(root) / p) for p in extras}
        return {"schema_version": 1, "files": files, "digest": diagnostic.digest(files), "origin_commit": None}
    monkeypatch.setattr(planning, "inspect_source", metadata_source)
    monkeypatch.setattr(planning, "compute_profile", lambda backend, model=None: {"backend": backend, "availability": "declared", "model": model if backend == "cuda" else "private CPU", "threads": 1})
    monkeypatch.setattr(Recipe, "make_prior", lambda *a, **k: pytest.fail("metadata constructed a prior"))
    before = torch.cuda.is_initialized()
    for family in diagnostic.FAMILIES:
        declared = diagnostic.declaration(ROOT, spec, family)
        assert "particle_cloud" not in declared["requires_capabilities"]
        base = planning.resolve_idea(ROOT, declared["id"], declaration=declared, execution_backend="cuda", cuda_model=spec["cuda_model"])
        result = diagnostic.request_from_base(base, spec, family, base["source"])
        assert len(result["tasks"]) == 26 and result["candidate"]["trainer_family"] == family
        assert base["preflight_blockers"] == [diagnostic.LEGACY_ADMISSION_BLOCKER]
        assert result["ordinary_admission"]["status"] == "BLOCKED" and result["qualification_reuse"] is False
        assert result["candidate"]["ordinary_decision_reference"]["decision_contract"] == planning.load_idea(ROOT, "atlas-c6-observed-policy-current-v1")["decision_contract"]
        for row in diagnostic.active_rows(spec, family):
            task = result["tasks"][row["id"]]
            assert task["preflight_blockers"] == []
            context = task_formulation_context(result["candidate"], task, result["protocol"], root=ROOT)
            assert context.execution_path == "public_components"
            assert context.recipe.lr == .0053125 and context.recipe.prior_lr_mult == 1.5 and context.recipe.total_steps is None
            if family == "atlas_ae_routed":
                assert context.prior_config["kind"] == "mog" and context.prior_config["sigma"] == .025
                assert context.capabilities()["ae_encoder"] and context.recipe.row_policy == "routed_paired"
            if family == "atlas_word_joint_min11":
                assert context.recipe.num_particles == 11 and context.recipe.encoder_mode == "none"
            assert row["definition"] in base["source"]["files"]
    assert torch.cuda.is_initialized() == before is False


@pytest.mark.parametrize("change", ["family", "recipe", "seed", "roster", "blocker", "missing_refusal", "similar_refusal", "active_decision", "reference_ready"])
def test_request_reconstruction_preserves_family_and_full_roster(spec, change):
    family = "atlas_conditional"; base = base_for(spec, family)
    if change == "family": base["candidate"]["trainer_family"] = "atlas"
    elif change == "recipe": base["candidate"]["recipe_overrides"]["prior_lr_mult"] = 2.
    elif change == "seed": base["protocol"]["seed"] = 1
    elif change == "roster": base["view"]["assignments"].pop()
    elif change == "blocker": base["preflight_blockers"].append("actual source drift")
    elif change == "missing_refusal": base["preflight_blockers"] = []
    elif change == "similar_refusal": base["preflight_blockers"] = [diagnostic.LEGACY_ADMISSION_BLOCKER + ": altered"]
    elif change == "active_decision": base["candidate"]["decision_contract"] = {"status": "draft"}
    elif change == "reference_ready": base["candidate"]["ordinary_decision_reference"]["decision_contract"]["status"] = "ready"
    with pytest.raises(ValueError): diagnostic.request_from_base(base, spec, family, {"digest": "a" * 64, "files": {}})


@pytest.mark.parametrize("physical", ["1"])
def test_correct_physical_lane_telemetry_and_environment(spec, monkeypatch, physical):
    commands = []
    def query(command):
        commands.append(command); return f"{physical}, NVIDIA RTX A6000, 12288, 82\n"
    assert diagnostic.gpu_readiness(spec, physical, query)["physical_gpu"] == physical
    assert "--id=" + physical in commands[0]
    diagnostic.configure_environment(physical)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == physical
    diagnostic.configure_environment()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("text", ["1, NVIDIA RTX A6000, 12287, 50", "1, NVIDIA RTX A6000, 20000, 83",
    "0, NVIDIA RTX A6000, 20000, 50", "1, other GPU, 20000, 50", "1, NVIDIA RTX A6000, nan, 50", ""])
def test_unsafe_lane_has_no_admission(spec, text):
    with pytest.raises(ValueError): diagnostic.gpu_readiness(spec, "1", lambda command: text)


def result_row(state, job, tmp_path, status="COMPLETE", grade="FAIL", terminal_status="completed"):
    terminal = dump(tmp_path / (job["task_id"] + ".json"), {"token": job["task_id"], "attempt_status": terminal_status, "paid_wall_seconds": 1., "child_returncode": 0 if status == "COMPLETE" else 1})
    result = {"compatibility_key": job["compatibility_key"], "task_ids": job["task_ids"], "token": job["task_id"],
              "status": status, "terminal": diagnostic.pin(terminal), **diagnostic.legacy().charge(1., diagnostic.read(terminal), job["budget_seconds"])}
    if status == "COMPLETE": result["outcome"] = {"statuses": {job["task_id"]: grade}, "qualification_input": False}
    state["jobs"].append(result)
    state["slots"][job["task_id"]]["diagnostic_status"] = grade if status == "COMPLETE" else status
    state["measured_paid_seconds"] += result["paid_wall_seconds"]
    state["unmeasured_interrupt_reserved_seconds"] += result["unmeasured_interrupt_reserved_seconds"]
    state["spent_seconds"] = state["measured_paid_seconds"] + state["unmeasured_interrupt_reserved_seconds"]
    state["lane_accounting"] = diagnostic.lane_accounting(state, require_ready=True)
    return result


def test_numerical_fail_completes_family_while_unknown_denominator_stays(spec, tmp_path):
    packet = diagnostic.family_packet(packet_for(spec), "atlas_routed"); state = diagnostic.initial_state(packet)
    for job in diagnostic.selected_jobs(packet): result_row(state, job, tmp_path, grade="FAIL")
    state["status"] = "COMPLETE_DIAGNOSTIC"; diagnostic.verify_state(state)
    assert sum(v["diagnostic_status"] == "FAIL" for v in state["slots"].values()) == 1
    assert sum(v["diagnostic_status"] == "NOT_RUN" for v in state["slots"].values()) == 25


def test_infrastructure_halts_prefix_and_conservatively_charges_original_allowance(spec, tmp_path):
    packet = diagnostic.family_packet(packet_for(spec), "atlas_routed"); state = diagnostic.initial_state(packet)
    jobs = diagnostic.selected_jobs(packet)
    result_row(state, jobs[0], tmp_path, status="INVALID", terminal_status="error")
    assert state["jobs"][0]["charged_seconds"] == 300 and state["jobs"][0]["unmeasured_interrupt_reserved_seconds"] == 299
    diagnostic.verify_state(state)
    result_row(state, jobs[0], tmp_path)
    with pytest.raises(ValueError, match="extra/duplicate"): diagnostic.verify_state(state)


@pytest.mark.parametrize("change", ["paid", "quota", "unknown_pass", "ordinary", "early_complete", "duplicate"])
def test_cost_denominator_and_credit_tamper_rejected(spec, tmp_path, change):
    packet = diagnostic.family_packet(packet_for(spec), "atlas_routed"); state = diagnostic.initial_state(packet)
    job = diagnostic.selected_jobs(packet)[0]; row = result_row(state, job, tmp_path)
    if change == "paid":
        row.update(paid_wall_seconds=0., charged_seconds=0.); state.update(spent_seconds=0., measured_paid_seconds=0.)
    elif change == "quota": state["family_paid_budget_seconds"]["atlas_conditional"] += 1
    elif change == "unknown_pass": state["slots"]["vector_two_broad"]["diagnostic_status"] = "PASS"
    elif change == "ordinary": state["ordinary_qualified_tier"] = 1
    elif change == "early_complete":
        state["jobs"].clear(); state.update(status="COMPLETE_DIAGNOSTIC", spent_seconds=0., measured_paid_seconds=0.)
        state["slots"][job["task_id"]]["diagnostic_status"] = "NOT_RUN"
        state["lane_accounting"] = diagnostic.lane_accounting(state)
    else: state["jobs"].append(deepcopy(row)); state.update(spent_seconds=2., measured_paid_seconds=2.)
    with pytest.raises(ValueError): diagnostic.verify_state(state)


def test_blocked_active_host_has_zero_allocation_and_no_borrowed_parent_pass(spec):
    packet = packet_for(spec); name = diagnostic.active_rows(spec, "atlas_routed")[0]["id"]
    packet["requests"]["atlas_routed"]["tasks"][name]["preflight_blockers"] = ["explicit source/API blocker"]
    state = diagnostic.initial_state(diagnostic.family_packet(packet, "atlas_routed"))
    assert state["slots"][name]["diagnostic_status"] == "BLOCKED" and not state["jobs"] and state["spent_seconds"] == 0
    assert name not in [j["task_id"] for j in diagnostic.executable_jobs(state)]
    diagnostic.verify_state(state)


def test_actual_coordinator_uses_private_supplied_queue_not_checkout(tmp_path):
    from experiments.forge.policy_execution import PolicyCoordinator
    queue = tmp_path / "private-existing-queue"
    coordinator = diagnostic.coordinator_for(queue)
    assert isinstance(coordinator, PolicyCoordinator) and coordinator.queue.root == queue.resolve()
    assert coordinator.root == queue.resolve()
    assert not (ROOT / "queue").exists()


def test_real_parser_accepts_only_gpu1_and_exact_generated_stage_commands(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(diagnostic, "stage", lambda path, execute: calls.append((path, execute)) or 0)
    for flag, execute in (("--execute", True), ("--evaluate", False)):
        assert diagnostic.main([flag, str(tmp_path / "resolved.json")]) == 0
        assert calls[-1][1] is execute
    for gpu in ("1",):
        args = diagnostic.parser().parse_args(["--output", str(tmp_path), "--gpus", gpu, "--queue-root", str(tmp_path / "queue")])
        assert args.gpus == gpu
    with pytest.raises(SystemExit): diagnostic.parser().parse_args(["--gpus", "0,1"])
    with pytest.raises(SystemExit): diagnostic.parser().parse_args(["--gpus", "0"])


def test_child_executes_actual_stages_then_fresh_cpu_evaluator_and_keeps_numeric_fail(monkeypatch, tmp_path):
    path = dump(tmp_path / "resolved.json", {"software_fixture": True})
    monkeypatch.setattr(diagnostic, "validate_resolved", lambda resolved: {})
    monkeypatch.setattr(diagnostic, "child_environment", lambda fd, resolved: None)
    dump(tmp_path / "graded-result.json", {"grades": {"named": {"gate_status": "FAIL"}}})
    commands = []
    def run(command, **kwargs):
        commands.append((command, kwargs)); return SimpleNamespace(returncode=0)
    assert diagnostic.child(path, 19, run) == 0
    assert commands[0][0][-2:] == ["--execute", str(path)]
    assert commands[1][0][-2:] == ["--evaluate", str(path)]
    assert commands[0][1]["pass_fds"] == (19,) and commands[1][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    commands.clear()
    assert diagnostic.child(path, 19, lambda command, **kwargs: SimpleNamespace(returncode=2)) == 2
    dump(tmp_path / "graded-result.json", {"grades": {"named": {"gate_status": "INCOMPLETE"}}})
    assert diagnostic.child(path, 19, run) == 2


def test_real_inherited_descriptor_is_bound_to_exact_physical_family_source(spec, tmp_path, monkeypatch):
    packet = diagnostic.family_packet(packet_for(spec), "atlas_routed")
    path = tmp_path / "attempt" / "execution.lock"; path.parent.mkdir(); path.touch()
    with path.open("r") as lease:
        fd = lease.fileno(); resolved = {"packet": packet, "worker": {"lease_path": str(path), "token": "this-attempt"}}
        dump(path.parent / "supervisor-request.json", {"token": "this-attempt", "source": packet["execution_source"], "lease_fds": [fd]})
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        diagnostic.child_environment(fd, resolved)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        with pytest.raises(ValueError, match="visibility"): diagnostic.child_environment(fd, resolved)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        resolved["worker"]["token"] = "foreign"
        with pytest.raises(ValueError, match="bound"): diagnostic.child_environment(fd, resolved)


@pytest.mark.parametrize("parent", list(diagnostic.FIXED_HOSTS))
def test_goal_views_use_actual_recorded_inputs_and_full_requested_coordinates(spec, parent):
    family, info = next((f, i) for f, i in diagnostic.FAMILIES.items() if parent in i["parents"])
    row = {"family": family, "id": parent + "_" + info["cohort"]}
    task = diagnostic.read(ROOT / "configs/forge/task-variants" / info["cohort"] / (row["id"] + ".json"))
    if row["family"] == "atlas_conditional":
        shapes = (3, 16) if parent in {"trajectory", "residual_student"} else (3, 4)
        target = np.arange(np.prod(shapes), dtype=np.float32).reshape(shapes)
        arrays = {"given_context": target / 2 if shapes[1] == 16 else np.array([[-1.], [0.], [1.]]), "target": target, "samples": target + 1}
    elif row["family"] == "atlas_routed": arrays = {"neu": np.array([[0., 0.], [1., 0.]]), "embeds": np.array([[.1, .2], [1., 1.]]), "concept_target": np.array([1., 1.])}
    elif row["family"] == "atlas_multibank": arrays = {"neu": np.arange(4), "targets": np.arange(8).reshape(2, 4), "residual": np.arange(8).reshape(2, 4) / 2}
    elif row["family"] == "atlas_ae_routed": arrays = {"target": np.arange(6).reshape(3, 2), "reconstruction": np.arange(6).reshape(3, 2) + 1, "samples": np.arange(8).reshape(4, 2), "anchors": np.array([[-1.5, 0.], [1.5, 0.]])}
    else:
        values = np.eye(28, dtype=np.float32)[np.arange(6)][None].transpose(0, 2, 1)
        arrays = {"target": values, "generated": np.repeat(values, 8, 0), "reconstruction": values.copy()}
    before = {k: v.copy() for k, v in arrays.items()}
    views = diagnostic.media_views(task, arrays)
    assert all(np.array_equal(arrays[k], value) for k, value in before.items())
    if parent in {"trajectory", "residual_student"}:
        assert views[0]["target"].shape == (3, 8, 2) and views[1]["samples"].shape == (3, 8, 2)
        assert np.array_equal(views[0]["target"].reshape(3, 16), arrays["given_context"]) and np.array_equal(views[1]["samples"].reshape(3, 16), arrays["samples"])
        assert views[1]["kind"] == "line"
    if row["family"] == "atlas_routed": assert np.array_equal(views[1]["samples"], arrays["embeds"])
    if row["family"] == "atlas_multibank": assert np.array_equal(views[0]["target"], arrays["targets"] - arrays["neu"])
    if row["family"] == "atlas_ae_routed": assert np.array_equal(views[0]["samples"], arrays["reconstruction"]) and np.array_equal(views[1]["samples"], arrays["samples"])
    if row["family"] == "atlas_word_joint_min11":
        assert views[0]["sample_labels"] == ["abcdef"] * 8
        assert np.array_equal(views[2]["samples"].reshape(arrays["reconstruction"].shape), arrays["reconstruction"])
        assert "padding" in views[2]["caption"] and "No reconstruction loss" in views[1]["caption"]


def test_durable_timeout_teardown_overshoot_is_saved_and_never_reset(spec, tmp_path):
    packet = diagnostic.family_packet(packet_for(spec), "atlas_routed"); state = diagnostic.initial_state(packet)
    job = diagnostic.selected_jobs(packet)[0]; row = result_row(state, job, tmp_path, status="INCOMPLETE", terminal_status="timeout")
    terminal = diagnostic.read(row["terminal"]["path"]); terminal["paid_wall_seconds"] = 301.
    path = dump(Path(row["terminal"]["path"]), terminal)
    row.update(terminal=diagnostic.pin(path), **diagnostic.legacy().charge(301., terminal, 300))
    state.update(measured_paid_seconds=301., unmeasured_interrupt_reserved_seconds=0., spent_seconds=301.)
    state["lane_accounting"] = diagnostic.lane_accounting(state, require_ready=True)
    with pytest.raises(ValueError, match="BUDGET_EXCEEDED"): diagnostic.verify_state(state)
    output = tmp_path / "archive"; output.mkdir()
    diagnostic.save_state(output, state)
    retained = diagnostic.read(output / "study.json")
    assert retained["status"] == "BUDGET_EXCEEDED" and retained["measured_paid_seconds"] == 301.
    assert retained["jobs"][0]["terminal"] == row["terminal"] and retained["slots"][job["task_id"]]["diagnostic_status"] == "INCOMPLETE"
    diagnostic.verify_state(retained)


@pytest.mark.parametrize("status", ["PASS", "FAIL"])
def test_recorded_signed_media_has_one_full_range_and_literal_original_badge(spec, tmp_path, monkeypatch, status):
    """Renderer inputs only: no model, draw or scientific grader is invoked."""
    from benchmarks.toy_audit import api_run
    row = {"id": "unipolar_conditional_policy_selected_cloud_v1"}
    task = diagnostic.read(ROOT / "configs/forge/task-variants/conditional_policy_selected_cloud_v1" / (row["id"] + ".json"))
    resolved = {"job": {"task_id": row["id"]}, "request": {"tasks": {row["id"]: task}, "source": {"digest": "a" * 64}}}
    dump(tmp_path / "raw-result.json", {"evidence": {"artifact_root": str(tmp_path), "artifact_manifest": {}}})
    points = [{"step": i + 1} for i in range(24)]; selected = [0, 3, 6, 9, 12, 15, 18, 21, 23]
    paths = []
    for index in range(9):
        path = tmp_path / f"recorded-{index}.npz"
        np.savez(path, given_context=np.array([[-1.], [1.]]), target=np.array([[-3., 2.], [1., -2.]]), samples=np.array([[float(index), -1.], [2., 3.]]))
        paths.append(path)
    monkeypatch.setattr(diagnostic.forge("artifacts"), "verify_artifacts", lambda *args: None)
    monkeypatch.setattr(diagnostic.legacy(), "media_selection", lambda *args: (points, selected, paths, {}))
    monkeypatch.setattr(diagnostic.legacy(), "point_pass", lambda *args: True)
    captured = {}
    def render(case, records, gif, **kwargs):
        captured.update(case=case, records=records, kwargs=kwargs)
        from PIL import Image
        Image.new("RGB", (2, 2)).save(gif, format="GIF")
    monkeypatch.setattr(api_run, "render_gif", render)
    before = {p: diagnostic.file_hash(p) for p in paths}
    diagnostic.render_media(resolved, tmp_path, {"grades": {row["id"]: {"gate_status": status}}})
    assert captured["kwargs"]["final_verdict"] == status
    assert "named diagnostic only" in captured["case"]["goal"]
    assert [r["step"] for r in captured["records"]] == [points[i]["step"] for i in selected]
    for index, record in enumerate(captured["records"]):
        view = record["views"][0]
        assert (view["vmin"], view["vmax"]) == (-3., 8.)
        assert view["target"][0, 0] == -3. and view["samples"][0, 0] == index
        assert "signed values are preserved" in view["caption"]
    assert {p: diagnostic.file_hash(p) for p in paths} == before


def completed_predecessor(spec, family, tmp_path, *, paid_each=1., predecessor_paths=()):
    """Private metadata/terminal fixtures, not scientific evidence."""
    packet = diagnostic.family_packet(packet_for(spec), family, predecessor_paths)
    state = diagnostic.initial_state(packet); folder = tmp_path / family; folder.mkdir(exist_ok=True)
    for job in diagnostic.executable_jobs(packet):
        row = result_row(state, job, folder, grade="FAIL")
        terminal_path = Path(row["terminal"]["path"]); terminal = diagnostic.read(terminal_path); terminal["paid_wall_seconds"] = paid_each
        dump(terminal_path, terminal)
        row.update(terminal=diagnostic.pin(terminal_path), **diagnostic.legacy().charge(paid_each, terminal, job["budget_seconds"]))
        attempt = folder / job["task_id"]; attempt.mkdir()
        raw = {"private_software_fixture": True}
        resolved = {"packet": packet, "packet_sha256": diagnostic.digest(packet), "request": state["request"], "job": job, "worker": {"token": row["token"]}}
        grade = {"raw_hash": diagnostic.digest(raw), "source_digest": state["source"]["digest"], "grades": {job["task_id"]: {"gate_status": "FAIL"}}}
        gif = attempt / "goal.gif"; gif.write_bytes(b"private retained software media")
        media = {"family": family, "task": job["task_id"], "source_digest": state["source"]["digest"], "original_gate": "FAIL", "qualification_input": False, "gif": diagnostic.pin(gif), "inputs": []}
        row["outcome"].update(resolved=diagnostic.pin(dump(attempt / "resolved.json", resolved)), raw=diagnostic.pin(dump(attempt / "raw.json", raw)),
                              grading=diagnostic.pin(dump(attempt / "grade.json", grade)), media={job["task_id"]: {"receipt": diagnostic.pin(dump(attempt / "media.json", media)), "gif": diagnostic.pin(gif)}})
    state["status"] = "COMPLETE_DIAGNOSTIC"
    diagnostic.save_state(folder, state)
    return folder / "study.json"


def test_old_engineering_reference_preserves_cost_only_and_fixed_family_caps(spec):
    proof = diagnostic.engineering_carryover(spec, ROOT)
    assert proof["reference"]["paid_seconds_by_lane"] == {"0": 12.873334385920316, "1": 12.449620655039325}
    assert [row["status"] for row in proof["attempts"]] == ["INVALID", "INVALID"]
    assert proof["qualification_input"] is False
    assert [diagnostic.FAMILIES[f]["cap"] for f in diagnostic.FAMILIES] == [7200, 300, 300, 1800, 900]


@pytest.mark.parametrize("change", ["missing", "reset", "swapped", "summary", "source", "credit", "repeat"])
def test_carryover_cannot_reset_change_or_supply_science(spec, change):
    if change == "missing": spec.pop("engineering_carryover")
    elif change == "reset": spec["engineering_carryover"]["paid_seconds_by_lane"]["0"] = 0.
    elif change == "swapped": spec["engineering_carryover"]["paid_seconds_by_lane"] = dict(reversed(list(diagnostic.PRIOR_DEBITS.items()))) | {"0": diagnostic.PRIOR_DEBITS["1"], "1": diagnostic.PRIOR_DEBITS["0"]}
    elif change == "summary": spec["engineering_carryover"]["summary"]["sha256"] = "0" * 64
    elif change == "source": spec["engineering_carryover"]["source"]["digest"] = "0" * 64
    elif change == "credit": spec["engineering_carryover"]["qualification_input"] = True
    else: spec["engineering_carryover"]["authorized_successors"] = 2
    with pytest.raises(ValueError): diagnostic.validate_spec(spec)


def test_completed_gpu0_families_are_never_executable_in_this_source(spec, tmp_path):
    packet = packet_for(spec)
    assert len(packet["continuation_carryover"]["preserved_outcomes"]) == 5
    for family in diagnostic.PRESERVED_FAMILIES:
        with pytest.raises(ValueError, match="cannot rerun"):
            diagnostic.family_packet(packet, family)
        with pytest.raises(ValueError, match="not executable"):
            diagnostic.run_family(tmp_path / family, {**packet, "family": family}, tmp_path / "queue")
    with pytest.raises(ValueError, match="re-executed"):
        diagnostic.run_lane(tmp_path, tmp_path / "queue", "0")


def test_cheap_unused_completion_admits_original_full_cover_allowance(spec, tmp_path):
    prior = completed_predecessor(spec, "atlas_routed", tmp_path, paid_each=1.)
    packet = diagnostic.family_packet(packet_for(spec), "atlas_multibank", [("atlas_routed", prior)])
    state = diagnostic.initial_state(packet)
    assert diagnostic.full_allowance_fits(state, diagnostic.selected_jobs(packet)[0])
    assert state["lane_accounting"]["historical_engineering_debit_seconds"] == diagnostic.PRIOR_DEBITS["1"]
    assert state["lane_accounting"]["historical_v3_paid_seconds"] == diagnostic.V3_DEBITS["1"]
    assert state["lane_accounting"]["predecessor_paid_seconds"] == 1.


def test_unused_and_cover_full_charges_refuse_word_without_borrowing_gpu0(spec, tmp_path):
    unused = completed_predecessor(spec, "atlas_routed", tmp_path, paid_each=300.)
    cover = completed_predecessor(spec, "atlas_multibank", tmp_path, paid_each=1800., predecessor_paths=[("atlas_routed", unused)])
    packet = diagnostic.family_packet(packet_for(spec), "atlas_word_joint_min11", [("atlas_routed", unused), ("atlas_multibank", cover)])
    state = diagnostic.initial_state(packet)
    assert state["lane_accounting"]["predecessor_charged_seconds"] == 2100.
    assert not diagnostic.full_allowance_fits(state, diagnostic.selected_jobs(packet)[0])
    assert state["lane_accounting"]["lane_cap_seconds"] == 3000


@pytest.mark.parametrize("change", ["missing", "foreign_family", "duplicate", "source", "unfinished", "unrecorded_terminal", "cost", "gate", "bytes"])
def test_lane_prefix_forgery_stale_cost_and_unrecorded_crash_refuse_admission(spec, tmp_path, change):
    prior = completed_predecessor(spec, "atlas_routed", tmp_path)
    paths = [("atlas_routed", prior)]
    if change == "missing": paths = []
    elif change == "foreign_family": paths = [("atlas_conditional", prior)]
    elif change == "duplicate": paths *= 2
    elif change == "bytes":
        packet = diagnostic.family_packet(packet_for(spec), "atlas_multibank", paths)
        prior.write_text(prior.read_text() + " ")
        with pytest.raises(ValueError): diagnostic.lane_accounting(packet, require_ready=True)
        return
    else:
        state = diagnostic.read(prior)
        if change == "source": state["source"]["digest"] = "b" * 64
        elif change == "unfinished": state["status"] = "RUNNING"
        elif change == "unrecorded_terminal":
            # Durable terminal exists, but completion is not yet recorded.
            state["jobs"].pop(); state["status"] = "RUNNING"
        elif change == "cost":
            for row in state["jobs"]: row.update(paid_wall_seconds=0., charged_seconds=0.)
            state.update(spent_seconds=0., measured_paid_seconds=0.)
        elif change == "gate":
            row = state["jobs"][0]; name = row["task_ids"][0]; row["outcome"]["statuses"][name] = "PASS"; state["slots"][name]["diagnostic_status"] = "PASS"
        dump(prior, state)
    packet = diagnostic.family_packet(packet_for(spec), "atlas_multibank", paths)
    with pytest.raises(ValueError): diagnostic.lane_accounting(packet, require_ready=True)


def test_inclusive_ledger_and_debit_cannot_be_reset_on_resume(spec, tmp_path):
    state = diagnostic.initial_state(diagnostic.family_packet(packet_for(spec), "atlas_routed"))
    result_row(state, diagnostic.selected_jobs(state)[0], tmp_path)
    diagnostic.verify_state(state)
    state["lane_accounting"]["historical_engineering_debit_seconds"] = 0.
    state["lane_accounting"]["inclusive_lane_charged_seconds"] = state["spent_seconds"]
    with pytest.raises(ValueError, match="lane"): diagnostic.verify_state(state)


def test_missing_supervisor_reservation_remains_charged_once_and_halts_lane(spec, tmp_path):
    state = diagnostic.initial_state(diagnostic.family_packet(packet_for(spec), "atlas_routed"))
    job = diagnostic.selected_jobs(state)[0]
    row = {"compatibility_key": job["compatibility_key"], "task_ids": job["task_ids"], "token": "private-software-attempt", "status": "INCOMPLETE", **diagnostic.legacy().charge(0., None, 300)}
    state["jobs"].append(row); state["slots"][job["task_id"]]["diagnostic_status"] = "INCOMPLETE"; state["status"] = "INCOMPLETE"
    folder = tmp_path / "atlas_routed"; folder.mkdir(); diagnostic.save_state(folder, state)
    assert state["measured_paid_seconds"] == 0. and state["unmeasured_interrupt_reserved_seconds"] == state["spent_seconds"] == 300.
    assert state["lane_accounting"]["inclusive_lane_charged_seconds"] == 300. + diagnostic.HISTORICAL_LANE_DEBITS["1"]
    packet = diagnostic.family_packet(packet_for(spec), "atlas_multibank", [("atlas_routed", folder / "study.json")])
    with pytest.raises(ValueError, match="unfinished"): diagnostic.lane_accounting(packet, require_ready=True)


def test_v3_cost_and_five_passes_remain_separate_from_all_current_slots(spec):
    proof = diagnostic.continuation_carryover(spec, ROOT)
    assert len(proof["attempts"]) == 6 and len(proof["preserved_outcomes"]) == 5
    assert sum(row["paid_seconds"] for row in proof["attempts"]) == sum(diagnostic.V3_DEBITS.values())
    assert all(row["current_slots_reused"] is False and row["reexecution_authorized"] is False
               and row["qualification_input"] is False for row in proof["preserved_outcomes"])
    packet = packet_for(spec)
    for family in diagnostic.ACTIVE_FAMILIES:
        state = diagnostic.initial_state(diagnostic.family_packet(packet, family))
        assert len(state["slots"]) == 26 and not any(row["diagnostic_status"] == "PASS" for row in state["slots"].values())
        assert state["lane_accounting"]["historical_lane_debit_seconds"] == 26.204209604067728
        assert state["lane_accounting"]["historical_v3_paid_seconds"] == 13.754588949028403
        assert state["spent_seconds"] == state["unmeasured_interrupt_reserved_seconds"] == 0.
    assert diagnostic.HISTORICAL_LANE_DEBITS == {"0": 234.82608077581972, "1": 26.204209604067728}


@pytest.mark.parametrize("change", ["missing", "reset", "source", "publication", "lane", "old_pass_credit", "rerun"])
def test_v4_history_declaration_rejects_reset_source_and_outcome_reuse(spec, change):
    if change == "missing": spec.pop("continuation_carryover")
    elif change == "reset": spec["continuation_carryover"]["current_paid_seconds_by_lane"]["1"] = 0.
    elif change == "source": spec["continuation_carryover"]["source"]["digest"] = "0" * 64
    elif change == "publication": spec["continuation_carryover"]["publication"]["sha256"] = "0" * 64
    elif change == "lane": spec["executable_physical_gpus"] = ["0", "1"]
    elif change == "old_pass_credit": spec["continuation_carryover"]["outcomes_reused"] = True
    else: spec["cases"].append({"family": "atlas_conditional", "parent_id": "trajectory"})
    with pytest.raises(ValueError): diagnostic.validate_spec(spec)


def test_v3_debit_is_counted_once_when_new_full_allowances_fit_or_fail(spec, tmp_path):
    packet = diagnostic.family_packet(packet_for(spec), "atlas_routed")
    state = diagnostic.initial_state(packet)
    assert diagnostic.full_allowance_fits(state, diagnostic.selected_jobs(state)[0])
    unused = completed_predecessor(spec, "atlas_routed", tmp_path, paid_each=300.)
    cover = completed_predecessor(spec, "atlas_multibank", tmp_path, paid_each=1773., predecessor_paths=[("atlas_routed", unused)])
    state = diagnostic.initial_state(diagnostic.family_packet(packet_for(spec), "atlas_word_joint_min11",
                                [("atlas_routed", unused), ("atlas_multibank", cover)]))
    assert state["lane_accounting"]["inclusive_lane_charged_seconds"] == 2073. + 26.204209604067728
    assert diagnostic.full_allowance_fits(state, diagnostic.selected_jobs(state)[0])
    assert state["spent_seconds"] == 0. and not state["jobs"]
    # The existing full300+1800 case refuses the final900 allowance. No
    # historical debit or reserve is subtracted twice from a family cap.


def software_history(spec, tmp_path, monkeypatch):
    """Self-contained historical metadata/byte fixtures, never trained evidence.

    Replace only the fixture's immutable publication/source identity oracle.
    Real file hashing, source-manifest joins, terminal/supervisor validation,
    exact paid amounts, old outcomes and all denominator guards remain active.
    """
    previous = diagnostic.read(ROOT / diagnostic.PRIOR_PUBLICATION)
    snapshot = tmp_path / "software-only-old-source"; snapshot.mkdir()
    source_file = snapshot / "source.py"; source_file.write_text("# software-only source binding\n")
    files = {"source.py": diagnostic.file_hash(source_file)}
    expected_source = {"commit": diagnostic.V3_SOURCE["commit"], "digest": diagnostic.digest(files)}
    source = {"schema_version": 1, "origin_commit": expected_source["commit"], "digest": expected_source["digest"],
              "files": files, "snapshot_path": str(snapshot)}
    dump(snapshot / "forge-source.json", {k: v for k, v in source.items() if k != "snapshot_path"})
    previous["source"] = {"origin_commit": expected_source["commit"], "digest": expected_source["digest"]}
    previous["source_file_count"] = len(files)
    for family in (*diagnostic.PRESERVED_FAMILIES, "atlas_routed"):
        old = previous["families"][family]
        jobs = [{"task_id": item["task_ids"][0], "task_ids": item["task_ids"],
                 "budget_seconds": item["allowance_seconds"], "compatibility_key": item["compatibility_key"]}
                for item in old["attempts"]]
        request = {"jobs": jobs, "software_only": True}
        packet = {"source": source, "family": family, "lane_predecessors": []}
        state = {"source": source, "execution_source": source, "family": family, "status": old["status"],
                 "spec_sha256": previous["spec_sha256"], "spec": {"id": "atlas-named-hosts-current-gpu-diagnostics-v3"},
                 "measured_paid_seconds": old["paid_seconds"], "spent_seconds": old["paid_seconds"],
                 "unmeasured_interrupt_reserved_seconds": 0., "qualification_input": False,
                 "slots": deepcopy(old["slots"]), "lane_runtime": deepcopy(old["runtime"]), "jobs": [],
                 "request": request, "lane_predecessors": []}
        for item, job in zip(old["attempts"], jobs):
            directory = tmp_path / "software-attempts" / job["task_id"]; directory.mkdir(parents=True)
            token = "software-only-" + job["task_id"]
            terminal = {"token": token, "attempt_status": "completed", "child_returncode": item["child_returncode"],
                        "paid_wall_seconds": item["paid_seconds"]}
            item["terminal"] = diagnostic.pin(dump(directory / "supervisor-terminal.json", terminal))
            item["token_sha256"] = diagnostic.hashlib.sha256(token.encode()).hexdigest()
            dump(directory / "supervisor-request.json", {"token": token, "source": source})
            row = {"task_ids": item["task_ids"], "status": item["status"], "attempt_key": item["attempt_key"],
                   "compatibility_key": item["compatibility_key"], "terminal": item["terminal"], "token": token,
                   "paid_wall_seconds": item["paid_seconds"], "charged_seconds": item["paid_seconds"],
                   "unmeasured_interrupt_reserved_seconds": 0.}
            raw = {"software_only": True}; raw_pin = diagnostic.pin(dump(directory / "raw.json", raw))
            if family in diagnostic.PRESERVED_FAMILIES:
                name = job["task_id"]
                grading = {"raw_hash": diagnostic.digest(raw), "source_digest": source["digest"],
                           "grades": {name: {"gate_status": "PASS"}}}
                grade_pin = diagnostic.pin(dump(directory / "grade.json", grading))
                gif = directory / "software-only.gif"; gif.write_bytes(b"software-only media binding; no generated image")
                media = {"family": family, "task": name, "source_digest": source["digest"], "original_gate": "PASS",
                         "qualification_input": False, "gif": diagnostic.pin(gif), "inputs": []}
                media_input = {"receipt": diagnostic.pin(dump(directory / "media.json", media)), "gif": diagnostic.pin(gif)}
                resolved = {"packet": packet, "packet_sha256": diagnostic.digest(packet), "request": request,
                            "job": job, "worker": {"token": token}}
                row["outcome"] = {"statuses": {name: "PASS"}, "qualification_input": False, "raw": raw_pin,
                    "grading": grade_pin, "resolved": diagnostic.pin(dump(directory / "resolved.json", resolved)),
                    "media": {name: media_input}}
                item["outcome"].update(raw_input=raw_pin, grading_input=grade_pin, media_input=media_input)
            else:
                item["raw_error_input"] = raw_pin
            state["jobs"].append(row)
        old["study_input"] = diagnostic.pin(dump(tmp_path / (family + "-study.json"), state))
    previous["trusted_cut"] = diagnostic.pin(dump(tmp_path / "software-only-cut.json", {"software_only": True}))
    publication = tmp_path / diagnostic.PRIOR_PUBLICATION
    publication.parent.mkdir(parents=True)
    entries = [diagnostic.pin(p) for p in sorted(tmp_path.rglob("*")) if p.is_file()]
    index_path = dump(publication.parent / "input-index.json", {"file_count": len(entries), "files": entries, "raw_files_changed": False})
    previous["input_index"] = {"path": "input-index.json", "sha256": diagnostic.file_hash(index_path),
                               "bytes": index_path.stat().st_size, "file_count": len(entries)}
    reference = deepcopy(diagnostic.CONTINUATION_REFERENCE); reference["source"] = expected_source
    dump(publication, previous)
    reference["publication"].update(sha256=diagnostic.file_hash(publication), bytes=publication.stat().st_size)
    monkeypatch.setattr(diagnostic, "V3_SOURCE", expected_source)
    monkeypatch.setattr(diagnostic, "CONTINUATION_REFERENCE", reference)
    spec = deepcopy(spec); spec["continuation_carryover"] = deepcopy(reference)
    return spec, publication, previous


def rebind_software_publication(spec, publication, previous, monkeypatch):
    dump(publication, previous)
    reference = deepcopy(diagnostic.CONTINUATION_REFERENCE)
    reference["publication"].update(sha256=diagnostic.file_hash(publication), bytes=publication.stat().st_size)
    monkeypatch.setattr(diagnostic, "CONTINUATION_REFERENCE", reference)
    spec["continuation_carryover"] = deepcopy(reference)


def test_complete_old_cost_source_and_terminal_proof_is_self_contained_software_only(spec, tmp_path, monkeypatch):
    spec, publication, previous = software_history(spec, tmp_path, monkeypatch)
    before = {p: diagnostic.file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    proof = diagnostic.continuation_carryover(spec, tmp_path, durable=True)
    assert len(proof["attempts"]) == 6 and len(proof["preserved_outcomes"]) == 5
    assert {p: diagnostic.file_hash(p) for p in before} == before
    assert proof["outcomes_reused"] is False and proof["qualification_input"] is False


@pytest.mark.parametrize("change", ["cost", "lane", "scope", "family_cap", "missing_attempt", "source", "old_grade", "slot", "terminal_bytes", "supervisor_source", "missing_source_index"])
def test_coherent_old_history_and_durable_tamper_is_rejected_without_hash_masking(spec, tmp_path, monkeypatch, change):
    spec, publication, previous = software_history(spec, tmp_path, monkeypatch)
    if change == "cost": previous["cost"]["current_paid_seconds"] = 0.
    elif change == "lane": previous["cost"]["lanes"]["1"]["current_paid_seconds"] = 0.
    elif change == "scope": previous["qualification_input"] = True
    elif change == "family_cap": previous["families"]["atlas_routed"]["family_cap_seconds"] = 301
    elif change == "missing_attempt": previous["families"]["atlas_conditional"]["attempts"].pop()
    elif change == "source": previous["source"]["origin_commit"] = "0" * 40
    elif change == "old_grade": previous["families"]["atlas_conditional"]["attempts"][0]["outcome"]["status"] = "FAIL"
    elif change == "slot": previous["families"]["atlas_multibank"]["slots"]["vector_two_broad"]["diagnostic_status"] = "PASS"
    elif change == "terminal_bytes":
        path = Path(previous["families"]["atlas_routed"]["attempts"][0]["terminal"]["path"])
        path.write_text(path.read_text() + " ")
    elif change == "supervisor_source":
        path = Path(previous["families"]["atlas_routed"]["attempts"][0]["terminal"]["path"]).with_name("supervisor-request.json")
        data = diagnostic.read(path); data["source"]["digest"] = "0" * 64; dump(path, data)
        # Rebind the index/publication too, so the source attribution check,
        # rather than the byte guard, must reject this coherent forgery.
        index_path = publication.parent / "input-index.json"; index = diagnostic.read(index_path)
        index["files"] = [diagnostic.pin(path) if item["path"] == str(path) else item for item in index["files"]]
        dump(index_path, index); previous["input_index"].update(sha256=diagnostic.file_hash(index_path), bytes=index_path.stat().st_size)
    else:
        index_path = publication.parent / "input-index.json"; index = diagnostic.read(index_path)
        index["files"] = [item for item in index["files"] if not item["path"].endswith("source.py")]
        index["file_count"] = len(index["files"]); dump(index_path, index)
        previous["input_index"].update(sha256=diagnostic.file_hash(index_path), bytes=index_path.stat().st_size, file_count=len(index["files"]))
    rebind_software_publication(spec, publication, previous, monkeypatch)
    with pytest.raises(ValueError): diagnostic.continuation_carryover(spec, tmp_path, durable=True)


def test_lane_continues_numeric_fail_in_declared_order_and_never_calls_preserved_gpu0(spec, tmp_path, monkeypatch):
    packet = packet_for(spec); dump(diagnostic.preparation_path(tmp_path / "output"), packet)
    monkeypatch.setattr(diagnostic, "verify_packet", lambda value: None)
    calls = []
    def run_family(output, family_packet, queue_root):
        family = family_packet["family"]; calls.append(family)
        assert family_packet["lane_predecessors"] == [
            {"family": name, "study": diagnostic.pin(tmp_path / "output" / name / "study.json")} for name in calls[:-1]]
        state = {"status": "COMPLETE_DIAGNOSTIC", "measured_paid_seconds": 1.,
                 "unmeasured_interrupt_reserved_seconds": 0., "lane_accounting": {"software_only": True},
                 "slots": {name: {"diagnostic_status": "FAIL" if name in {row["id"] for row in diagnostic.active_rows(spec, family)}
                                   else "NOT_RUN"} for name in family_packet["request"]["tasks"]}}
        dump(output / "study.json", state)
        return state
    monkeypatch.setattr(diagnostic, "run_family", run_family)
    assert len(diagnostic.run_lane(tmp_path / "output", tmp_path / "private-queue", "1")) == 3
    assert calls == list(diagnostic.ACTIVE_FAMILIES)
    with pytest.raises(ValueError): diagnostic.run_lane(tmp_path / "output", tmp_path / "private-queue", "0")
    assert calls == list(diagnostic.ACTIVE_FAMILIES)

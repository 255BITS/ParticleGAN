"""Self-contained publication controls; synthetic bytes, no model operations."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from PIL import Image
import pytest

MODULE = Path(__file__).with_name("publish_critic_balance.py")
if not MODULE.is_file():
    MODULE = Path(__file__).resolve().parents[1] / "reports/forge/critic-balance-20261003/publish_critic_balance.py"
SPEC = importlib.util.spec_from_file_location("critic_balance_external_publisher_tests", MODULE)
pub = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pub)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return pub.binding(path)


def gif(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.new("RGB", (8, 8), (i * 20, 0, 0)) for i in range(9)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return pub.binding(path)


class SoftwareOracle:
    """Stub only scientific certifier; file/copy/accounting checks stay real."""
    def select(self, packet):
        assert packet["spec"]["recipe_overrides"] == {"lr": .0053125, "prior_lr_mult": 1.5, "d_lr_mult": 2.25}
        if len(packet["trials"]) != 2 or any(len(t["cases"]) != 8 for t in packet["trials"]):
            raise ValueError("required denominator")
        return {"required_configurations": 2, "required_cells": 16,
                "required_cases_per_config": 8, "fully_qualified_ids": [],
                "speed_winner": None, "default_adoption": False}

    def costs(self, packet):
        for trial in packet["trials"]:
            pub.equal_cost(trial["paid_wall_seconds"], sum(
                r.get("paid_wall_seconds", 0.) + r.get("unmeasured_interrupt_reserved_seconds", 0.)
                for r in trial["cases"]), "trial")
        pub.equal_cost(packet["spent_seconds"], sum(t["paid_wall_seconds"] for t in packet["trials"]), "packet")

    def durable(self, packet, trial, row):
        directory = Path(packet["coordinator"]["queue_root"]) / "policy/attempts" / row["attempt_key"]
        terminal = pub.read(directory / "supervisor-terminal.json")
        request = pub.read(directory / "supervisor-request.json")
        if terminal["token"] != request["token"] or request["source"] != packet["execution_source"]:
            raise ValueError("durable source/token")
        pub.equal_cost(row.get("paid_wall_seconds", 0.), terminal["paid_wall_seconds"], "durable paid")

    def complete(self, directory):
        raw = pub.read(Path(directory) / "receipt.json")
        sustained = all(row["passed"] for row in raw["observations"][-5:])
        if raw["passed"] != sustained or raw["verdict"] != ("PASS" if sustained else "FAIL"):
            raise ValueError("original numeric grade")
        return raw

    def hold(self, raw):
        rows = raw["observations"][1:]
        streak = 0
        for index, row in enumerate(rows):
            streak = streak + 1 if row["passed"] else 0
            if streak == 5:
                later = rows[index + 1:]
                status = "FAIL" if not all(r["passed"] for r in later) else "INCOMPLETE" if len(later) < 5 else "PASS"
                return {"status": status, "acquired_step": row["step"], "hold_passed": sum(r["passed"] for r in later), "hold_checks": len(later)}
        return {"status": "FAIL", "reason": "no acquisition"}

    def partial(self, path, packet, trial, row):
        return {"status": pub.read(Path(path) / "receipt.json")["status"],
                "original_gate": None, "study_gate": None, "full_protocol_complete": False}


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    root = tmp_path / "science"
    root.mkdir()
    source_file = root / "benchmarks/pure.py"
    source_file.parent.mkdir()
    source_file.write_text("# Synthetic source; no science\n")
    source = {"commit": "a" * 40, "files_sha256": {"benchmarks/pure.py": pub.file_sha(source_file)}}
    manifest = {"schema_version": 1, "origin_commit": source["commit"],
                "files": dict(source["files_sha256"]), "digest": pub.digest(source["files_sha256"])}
    frozen = tmp_path / "snapshot"
    (frozen / "benchmarks").mkdir(parents=True)
    (frozen / "benchmarks/pure.py").write_bytes(source_file.read_bytes())
    write(frozen / "forge-source.json", manifest)
    execution = {**manifest, "snapshot_path": str(frozen)}
    queue = tmp_path / "queue"
    case_ids = [f"api-software-{i}" for i in range(8)]
    cases = {name: {"id": name, "goal": f"Synthetic question {i}", "title": name,
                    "default_steps": 24, "eval_samples": 8, "batch_size": 4,
                    "thresholds": {"quality_min": .9}, "sampling": {"law": "synthetic"}}
             for i, name in enumerate(case_ids)}
    overrides = {"lr": .0053125, "prior_lr_mult": 1.5, "d_lr_mult": 2.25}
    declared = [{"id": name, "tier": 1 if i < 2 else 2, "timeout_seconds": 180.,
                 "case_sha256": pub.digest(cases[name])} for i, name in enumerate(case_ids)]
    records = []
    for family in pub.FAMILIES:
        for name in case_ids:
            artifact = tmp_path / "capacity" / family / name / "state.pt"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"Synthetic capacity state; never deserialized")
            samples = artifact.with_name("samples.npz")
            samples.write_bytes(b"Synthetic supplied arrays; never scored")
            records.append({"family": family, "case_id": name, "status": "SUPPORTED",
                            "ordinary_training_updates": 0, "fitting_updates": 0, "ordinary_qualification_credit": False,
                            "bindings": {"case_sha256": pub.digest(cases[name])},
                            "resolved_recipe": {"name": family, **overrides}, "artifacts": {"state": pub.binding(artifact), "samples": pub.binding(samples)},
                            "observations": [{"completed_steps": 0, "samples": 8, "evaluation_seed": 34002,
                                              "passed": True, "failed_bounds": [], "metrics": {"quality": 1.}}]})
    capacity = {"schema": "particlegan_critic_balance_capacity_v1", "status": "COMPLETE", "required_records": 16,
                "requested_cells": [{"family": f, "case_id": n} for f in pub.FAMILIES for n in case_ids],
                "ordinary_training_updates": 0, "fitting_updates": 0, "ordinary_qualification_credit": False, "records": records}
    capacity_pin = write(tmp_path / "capacity.json", capacity)
    spec = {"id": "software-critic-contrast", "recipe_overrides": overrides, "seed": 24002,
            "cases": declared, "representation_card": capacity_pin,
            "candidate_budget_seconds": 7680., "family_budget_seconds": {"atlas": 7680., "e22": 7680.},
            "budget_seconds": 15360., "export_grace_seconds": 60., "frames": 9}
    runtime = {"device": "cuda:0", "torch_threads": 1, "cuda_device_model": "Synthetic GPU"}
    trials = []
    for family in pub.FAMILIES:
        recipe = {"name": family, **overrides}
        rows = [{**item, "status": "UNKNOWN", "original_gate": None, "study_gate": None,
                 "full_protocol_complete": False, "resolved_recipe_sha256": pub.digest(recipe)} for item in declared]
        directory = tmp_path / family / case_ids[0]
        media = gif(directory / "goal.gif")
        state = directory / "final-state.pt"
        state.write_bytes(b"Synthetic final state; no model restored")
        arr = directory / "observations.npz"
        arr.write_bytes(b"Synthetic arrays; no model scored")
        pins = {name: {key: val for key, val in pub.binding(directory / name).items() if key != "path"}
                for name in ("goal.gif", "final-state.pt", "observations.npz")}
        observations = [{"step": i, "elapsed_seconds": i / 24, "passed": i != 7,
                         "failed_bounds": [] if i != 7 else ["quality"], "metrics": {"quality": 1. if i != 7 else .5},
                         "views": [{"kind": "image", "title": "synthetic"}]} for i in range(25)]
        raw = {"status": "COMPLETE", "case": cases[case_ids[0]], "seed": 24002,
               "requested_recipe_overrides": overrides, "source": source, "recipe": recipe, "runtime": runtime,
               "protocol": {"updates": 24, "evaluation_samples": 8, "wall_cap_seconds": 180., "media_frames": 9},
               "passed": True, "verdict": "PASS", "elapsed_seconds": 1., "completed_updates": 24,
               "observations": observations, "artifacts": pins, "gif_frames": media["frames"] if "frames" in media else 9}
        receipt_pin = write(directory / "receipt.json", raw)
        hold = SoftwareOracle().hold(raw)
        rows[0].update(status="FAIL", original_gate="PASS", study_gate="FAIL", full_protocol_complete=True,
                       acquisition_hold=hold, recipe=recipe, runtime=runtime, elapsed_seconds=1.,
                       final_metrics=observations[-1]["metrics"], artifacts=pins, receipt_path=receipt_pin["path"],
                       receipt_sha256=receipt_pin["sha256"], child_returncode=0, paid_wall_seconds=2., attempt_key=family + "-attempt")
        attempt = queue / "policy/attempts" / rows[0]["attempt_key"]
        write(attempt / "supervisor-request.json", {"token": family, "source": execution})
        write(attempt / "supervisor-terminal.json", {"token": family, "paid_wall_seconds": 2., "attempt_status": "completed"})
        trials.append({"id": family + "-synthetic", "family": family, "recipe_overrides": overrides,
                       "status": "FAIL", "paid_wall_seconds": 2., "cases": rows})
    packet = {"study_id": spec["id"], "goal": "software-contrast", "spec": spec, "spec_sha256": pub.digest(spec),
              "source": source, "execution_source": execution, "case_definitions": cases,
              "capacity_preflight": {r["family"] + "/" + r["case_id"]: r for r in records}, "runtime_contract": {"torch_threads": 1},
              "trials": trials, "family_paid_budget_seconds": spec["family_budget_seconds"], "spent_seconds": 4.,
              "measured_paid_seconds": 4., "unmeasured_interrupt_reservation_seconds": 0.,
              "paid_cost_scope": "Synthetic durable child time", "scope": "Synthetic software fixture; no science",
              "native_scope": "24 primary20k checks; no100k", "selection": SoftwareOracle().select({"spec": spec, "trials": trials}),
              "coordinator": {"queue_root": str(queue)}}
    archive_paths = []
    for family in pub.FAMILIES:
        archive = deepcopy(packet)
        archive["executed_family"] = family
        archive["lane_runtime"] = runtime
        for trial in archive["trials"]:
            if trial["family"] != family:
                trial["paid_wall_seconds"] = 0.
                trial["status"] = "UNKNOWN"
                trial["cases"] = [{**item, "status": "UNKNOWN", "original_gate": None, "study_gate": None,
                                   "full_protocol_complete": False, "resolved_recipe_sha256": pub.digest({"name": trial["family"], **overrides})}
                                  for item in declared]
        archive.update(spent_seconds=2., measured_paid_seconds=2.)
        pin = write(tmp_path / family / "study.json", archive)
        archive_paths.append({"family": family, **pin, "runtime": runtime})
    packet["family_archives"] = archive_paths
    combined = tmp_path / "combined.json"
    write(combined, packet)
    card = tmp_path / "certification.json"
    write(card, pub.bind_certification(combined, root))
    monkeypatch.setattr(pub, "git_blob", lambda checkout, commit, path: pub.file_sha(MODULE) if path == pub.SELF else pub.file_sha(Path(checkout) / path))
    (root / pub.SELF).parent.mkdir(parents=True)
    (root / pub.SELF).write_bytes(MODULE.read_bytes())
    monkeypatch.setattr(pub.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="b" * 40 + "\n", returncode=0))
    return SimpleNamespace(root=root, combined=combined, card=card, output=tmp_path / "publication",
                           packet=packet, oracle=SoftwareOracle(), capacity=tmp_path / "capacity.json", frozen=frozen)


def export(c):
    return pub.publish(c.card, pub.file_sha(c.card), c.output, publisher_root=c.root, oracle=c.oracle)


def refresh(c, edit):
    packet = pub.read(c.combined)
    edit(packet)
    write(c.combined, packet)
    write(c.card, pub.bind_certification(c.combined, c.root))


def test_retains_all_denominators_original_pass_added_fail_and_unknown_without_draws(cohort, monkeypatch):
    monkeypatch.setattr(pub, "Oracle", lambda *a: pytest.fail("scientific namespace not needed in software controls"))
    before = {p: pub.file_sha(p) for p in cohort.output.parent.rglob("*") if p.is_file()}
    result = export(cohort)
    assert len(result["cases"]) == 16
    assert result["counts"]["execution"] == {"FAIL": 2, "UNKNOWN": 14}
    assert result["counts"]["original_gates"] == {"PASS": 2, "UNAVAILABLE": 14}
    assert result["counts"]["goal_gifs"] == 2
    assert result["verification"]["publication_draws"] == 0
    assert result["speed_winner"] is None and not result["default_adoption"]
    assert "PASS" in (cohort.output / "README.md").read_text()
    assert "first-window" in result["cases"][0]["media"]["caption"]
    for row in result["cases"]:
        if row["media"]:
            assert pub.file_sha(cohort.output / row["media"]["path"]) == row["media"]["sha256"]
    assert all(pub.file_sha(p) == sha for p, sha in before.items())


def test_card_trust_anchor_cannot_be_replaced_by_new_self_hash(cohort):
    old_sha = pub.file_sha(cohort.card)
    card = pub.read(cohort.card)
    card["verification"]["capacity_sampler_replay"] = False
    write(cohort.card, card)
    with pytest.raises(ValueError, match="drift"):
        pub.publish(cohort.card, old_sha, cohort.output, publisher_root=cohort.root, oracle=cohort.oracle)


@pytest.mark.parametrize("mutate", [
    lambda p: p["trials"].pop(),
    lambda p: p["trials"][0]["cases"].pop(),
    lambda p: p["selection"].update(fully_qualified_ids=["forged"]),
    lambda p: p["trials"][0]["cases"][0].update(status="PASS"),
    lambda p: p["trials"][0].update(status="RUNNING"),
    lambda p: p.update(measured_paid_seconds=0.),
    lambda p: p["family_paid_budget_seconds"].update(atlas=1.),
])
def test_forged_denominator_grade_runtime_cost_or_live_study_rejected(cohort, mutate):
    refresh(cohort, mutate)
    with pytest.raises((ValueError, AssertionError)):
        export(cohort)
    assert not cohort.output.exists()


@pytest.mark.parametrize("artifact", ["goal.gif", "final-state.pt", "observations.npz"])
def test_missing_or_modified_public_artifact_rejected_before_output(cohort, artifact):
    row = cohort.packet["trials"][0]["cases"][0]
    path = Path(row["receipt_path"]).parent / artifact
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="drift"):
        export(cohort)
    assert not cohort.output.exists()


def test_capacity_negative_is_separate_from_convergence_failure(cohort):
    packet = pub.read(cohort.capacity)
    packet["records"][-1]["status"] = "UNRESOLVED"
    pin = write(cohort.capacity, packet)
    value = deepcopy(cohort.packet)
    value["spec"]["representation_card"] = pin
    value["capacity_preflight"]["e22/api-software-7"] = packet["records"][-1]
    projected = pub.capacity(pub.Inputs(), value, cohort.root)
    assert projected["e22/api-software-7"]["status"] == "UNRESOLVED"


@pytest.mark.parametrize("mutate", [lambda p: p["records"].pop(),
                                   lambda p: p.update(ordinary_training_updates=1),
                                   lambda p: p["records"][0].update(ordinary_qualification_credit=True),
                                   lambda p: p.update(status="INCOMPLETE")])
def test_capacity_partial_or_learning_credit_forgery_rejected(cohort, mutate):
    packet = pub.read(cohort.capacity)
    mutate(packet)
    pin = write(cohort.capacity, packet)
    value = deepcopy(cohort.packet)
    value["spec"]["representation_card"] = pin
    with pytest.raises(ValueError):
        pub.capacity(pub.Inputs(), value, cohort.root)


def test_source_commit_blob_is_independent_of_manifest(cohort, monkeypatch):
    monkeypatch.setattr(pub, "git_blob", lambda *a: "0" * 64)
    with pytest.raises(ValueError, match="committed|contain"):
        export(cohort)


def test_snapshot_file_drift_rejected(cohort):
    (cohort.frozen / "benchmarks/pure.py").write_text("# drift\n")
    with pytest.raises(ValueError, match="drift"):
        export(cohort)


def test_differing_family_runtime_cannot_be_pooled(cohort):
    packet = deepcopy(cohort.packet)
    pin = packet["family_archives"][1]
    raw = pub.read(pin["path"])
    raw["lane_runtime"]["torch"] = "different"
    pin["runtime"] = deepcopy(raw["lane_runtime"])
    pin.update(write(Path(pin["path"]), raw))
    with pytest.raises(ValueError, match="different execution runtimes"):
        pub.check_archives(pub.Inputs(), packet)


@pytest.mark.parametrize("mutate", [lambda p: p.update(origin_commit="c" * 40),
                                   lambda p: p.update(digest="0" * 64),
                                   lambda p: p["files"].update({"../escape.py": "0" * 64})])
def test_snapshot_wrong_commit_digest_or_escaping_path_rejected(cohort, mutate):
    manifest = deepcopy(cohort.packet["execution_source"])
    mutate(manifest)
    with pytest.raises(ValueError):
        pub.snapshot(pub.Inputs(), manifest, cohort.packet["source"])


def test_gif_frame_count_not_trusted_from_json(cohort):
    row = cohort.packet["trials"][0]["cases"][0]
    receipt = pub.read(row["receipt_path"])
    receipt["gif_frames"] = 8
    row = deepcopy(row)
    pin = write(Path(row["receipt_path"]), receipt)
    row["receipt_sha256"] = pin["sha256"]
    archive = pub.read(cohort.packet["family_archives"][0]["path"])
    with pytest.raises(ValueError, match="frame"):
        pub.receipt_projection(pub.Inputs(), cohort.oracle, cohort.packet, archive, cohort.packet["trials"][0], row)


def test_numerical_grade_forgery_rejected_even_with_rebound_raw_hash(cohort):
    trial = deepcopy(cohort.packet["trials"][0])
    row = trial["cases"][0]
    raw = pub.read(row["receipt_path"])
    raw["observations"][-1].update(passed=False, failed_bounds=["quality"])
    pin = write(Path(row["receipt_path"]), raw)
    row["receipt_sha256"] = pin["sha256"]
    archive = pub.read(cohort.packet["family_archives"][0]["path"])
    with pytest.raises(ValueError, match="numeric grade"):
        pub.receipt_projection(pub.Inputs(), cohort.oracle, cohort.packet, archive, trial, row)


def test_coherent_paid_zeroing_rejected_by_durable_supervisor(cohort):
    trial = deepcopy(cohort.packet["trials"][0])
    trial["cases"][0]["paid_wall_seconds"] = 0.
    with pytest.raises(ValueError, match="durable paid"):
        pub.supervised(pub.Inputs(), cohort.oracle, cohort.packet, trial, trial["cases"][0])


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf"), True])
def test_nonfinite_or_negative_cost_is_never_publication_credit(value):
    with pytest.raises(ValueError):
        pub.number(value, "cost")


def test_bind_command_is_attestation_binding_and_never_runs_the_certifier(cohort, monkeypatch):
    monkeypatch.setattr(pub, "Oracle", lambda *a: pytest.fail("binding executed certifier"))
    output = cohort.output.parent / "new-card.json"
    assert pub.main(["bind-certification", "--combined", str(cohort.combined), "--scientific-root", str(cohort.root), "--output", str(output)]) == 0
    assert pub.read(output)["combined"]["sha256"] == pub.file_sha(cohort.combined)


def test_existing_output_refused(cohort):
    cohort.output.mkdir()
    with pytest.raises(ValueError, match="new output"):
        export(cohort)


def test_output_cannot_write_into_scientific_checkout(cohort):
    cohort.output = cohort.root / "new-publication"
    with pytest.raises(ValueError, match="outside scientific"):
        export(cohort)


def test_undeclared_snapshot_source_is_rejected(cohort):
    (cohort.frozen / "injected.py").write_text("# undeclared\n")
    with pytest.raises(ValueError, match="undeclared"):
        export(cohort)


def test_capacity_state_or_array_missing_is_rejected(cohort):
    value = deepcopy(cohort.packet)
    raw = pub.read(cohort.capacity)
    del raw["records"][0]["artifacts"]["samples"]
    value["capacity_preflight"]["atlas/api-software-0"] = raw["records"][0]
    value["spec"]["representation_card"] = write(cohort.capacity, raw)
    with pytest.raises(ValueError, match="state and sampled"):
        pub.capacity(pub.Inputs(), value, cohort.root)


def test_setup_overhead_is_debited_once_across_both_family_archives(cohort):
    overhead = deepcopy(cohort.packet)
    overhead["study_id"] = "software-original-startup"
    overhead["executed_family"] = "atlas"
    for trial in overhead["trials"]:
        trial["cases"] = [{**row, "status": "UNKNOWN", "original_gate": None, "study_gate": None,
                           "full_protocol_complete": False, "resolved_recipe_sha256": pub.digest({"name": trial["family"], **trial["recipe_overrides"]})}
                          for row in overhead["spec"]["cases"]]
        trial["status"] = "ERROR" if trial["family"] == "atlas" else "UNKNOWN"
        trial["paid_wall_seconds"] = 4.75 if trial["family"] == "atlas" else 0.
    row = overhead["trials"][0]["cases"][0]
    row.update(status="ERROR", paid_wall_seconds=4.75, attempt_key="old-engineering")
    overhead.update(spent_seconds=4.75, measured_paid_seconds=4.75)
    directory = Path(overhead["coordinator"]["queue_root"]) / "policy/attempts/old-engineering"
    request = write(directory / "supervisor-request.json", {"token": "old", "source": overhead["execution_source"]})
    terminal = write(directory / "supervisor-terminal.json", {"token": "old", "paid_wall_seconds": 4.75,
                                                             "attempt_status": "completed", "child_returncode": 1})
    log = directory / "startup.log"
    log.write_text("Synthetic failure before fixture construction\n")
    old_study = write(cohort.output.parent / "old-study.json", overhead)
    carryover = {"family": "atlas", "paid_seconds": 4.75, "claim": "Explicit synthetic pre-fixture startup ERROR; zero updates",
                 "artifacts": {"study": old_study, "request": request, "terminal": terminal, "log": pub.binding(log)}}
    packet = pub.read(cohort.combined)
    packet["spec"]["engineering_carryover"] = carryover
    packet["spec"]["family_budget_seconds"]["atlas"] -= 4.75
    packet["spec"]["budget_seconds"] -= 4.75
    packet["spec_sha256"] = pub.digest(packet["spec"])
    packet["family_paid_budget_seconds"] = deepcopy(packet["spec"]["family_budget_seconds"])
    for pin in packet["family_archives"]:
        raw = pub.read(pin["path"])
        raw["spec"] = deepcopy(packet["spec"])
        raw["spec_sha256"] = packet["spec_sha256"]
        raw["family_paid_budget_seconds"] = deepcopy(packet["family_paid_budget_seconds"])
        pin.update(write(Path(pin["path"]), raw))
    write(cohort.combined, packet)
    write(cohort.card, pub.bind_certification(cohort.combined, cohort.root))
    result = export(cohort)
    assert result["costs"]["engineering_paid_seconds"] == 4.75
    assert result["costs"]["combined_charged_seconds"] == 8.75
    assert len(result["engineering_cohorts"]) == 1
    assert len(result["cases"]) == 16 and result["counts"]["execution"] == {"FAIL": 2, "UNKNOWN": 14}


def test_missing_receipt_is_engineering_error_not_scientific_fail(cohort):
    trial = deepcopy(cohort.packet["trials"][0])
    row = trial["cases"][0]
    for name in ["receipt_path", "receipt_sha256", "artifacts", "elapsed_seconds", "recipe", "runtime", "final_metrics", "acquisition_hold"]:
        row.pop(name, None)
    row.update(status="ERROR", original_gate=None, study_gate=None, full_protocol_complete=False)
    value = pub.receipt_projection(pub.Inputs(), cohort.oracle, cohort.packet, {}, trial, row)
    assert value == {"completed_updates": None, "final_metrics": None, "media": None}


def test_partial_failure_keeps_actual_execution_status_without_a_scientific_grade(cohort):
    trial = deepcopy(cohort.packet["trials"][0])
    row = trial["cases"][0]
    raw = pub.read(row["receipt_path"])
    raw.update(status="INCOMPLETE", passed=False, default_protocol_complete=False, completed_updates=10)
    row.update(status="INCOMPLETE", original_gate=None, study_gate=None, full_protocol_complete=False)
    row["receipt_sha256"] = write(Path(row["receipt_path"]), raw)["sha256"]
    value = pub.receipt_projection(pub.Inputs(), cohort.oracle, cohort.packet, {}, trial, row)
    assert value["completed_updates"] == 10 and value["media"] is not None


def test_real_git_blob_binding_needs_no_historical_checkout(tmp_path, monkeypatch):
    # A real shallow-compatible local commit oracle; no fake Git hash here.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "source.py").write_text("answer = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Software control", "-c", "user.email=software@example.invalid", "commit", "-qm", "source"], check=True)
    commit = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    assert pub.git_blob(tmp_path, commit, "source.py") == pub.file_sha(tmp_path / "source.py")
    (tmp_path / "source.py").write_text("answer = 2\n")
    assert pub.git_blob(tmp_path, commit, "source.py") != pub.file_sha(tmp_path / "source.py")

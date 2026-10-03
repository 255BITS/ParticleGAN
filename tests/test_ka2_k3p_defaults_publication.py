"""Self-contained publication controls; synthetic bytes, no model operations."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from PIL import Image
import pytest

MODULE = Path(__file__).with_name("publish_ka2_k3p_defaults.py")
if not MODULE.is_file():
    MODULE = Path(__file__).resolve().parents[1] / "reports/forge/ka2-k3p-defaults-20261003/publish_ka2_k3p_defaults.py"
SPEC = importlib.util.spec_from_file_location("ka2_k3p_external_publisher_tests", MODULE)
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
        assert packet["spec"]["recipe_overrides"] == pub.OVERRIDES
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

    def recipe(self, case, family):
        return {"name": family, **pub.OVERRIDES, "total_steps": case["default_steps"]}

    def law(self, case, family):
        return {"family": family, "served_model": "fast_only", "continuous_policy": None,
                "schedule_horizon": case["default_steps"], "output_noise_std": .029,
                "output_noise_warmup_fraction": .2, "amsgrad": False,
                "primary_law": "fast_without_output_noise", "actual_read_standardize": False,
                "old_policy_credit": False}


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    root = tmp_path / "science"
    root.mkdir()
    source_file = root / "benchmarks/pure.py"
    source_file.parent.mkdir()
    source_file.write_text("# Synthetic source; no science\n")
    source_files = {"benchmarks/pure.py": pub.file_sha(source_file)}
    for relative in (pub.RUNNER, pub.BINDER, pub.PROTOCOL, pub.PREVIOUS, pub.DELEGATED):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Synthetic protected helper; never imported\n")
        source_files[relative] = pub.file_sha(target)
    discovery_path = "configs/forge/tasks/ring16_acquisition.json"
    discovery_file = root / discovery_path
    discovery_file.parent.mkdir(parents=True)
    discovery_file.write_text('{"software_fixture": true}')
    # Synthetic source oracle, including its own discovery bytes, never science.
    monkeypatch.setattr(pub, "DISCOVERY", {discovery_path: pub.file_sha(discovery_file)})
    source = {"commit": "a" * 40, "files_sha256": source_files,
              "discovery_inputs_sha256": dict(pub.DISCOVERY)}
    manifest = {"schema_version": 1, "origin_commit": source["commit"],
                "files": {**source["files_sha256"], **source["discovery_inputs_sha256"]}}
    manifest["digest"] = pub.digest(manifest["files"])
    frozen = tmp_path / "snapshot"
    for name in manifest["files"]:
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    write(frozen / "forge-source.json", manifest)
    execution = {**manifest, "snapshot_path": str(frozen)}
    queue = tmp_path / "queue"
    case_ids = [f"api-software-{i}" for i in range(8)]
    cases = {name: {"id": name, "goal": f"Synthetic question {i}", "title": name,
                    "default_steps": 600, "eval_samples": 8, "batch_size": 4,
                    "thresholds": {"quality_min": .9}, "sampling": {"law": "synthetic"}}
             for i, name in enumerate(case_ids)}
    overrides = dict(pub.OVERRIDES)
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
            records.append({"schema": pub.CAPACITY_SCHEMA, "family": family, "case_id": name,
                            "case": deepcopy(cases[name]), "seed": 24002, "status": "SUPPORTED",
                            "ordinary_training_updates": 0, "fitting_updates": 0, "ordinary_qualification_credit": False,
                            "bindings": {"case_sha256": pub.digest(cases[name])},
                            "recipe_overrides": deepcopy(overrides), "requested_recipe_overrides": deepcopy(overrides),
                            "resolved_recipe": SoftwareOracle().recipe(cases[name], family),
                            "family_law": SoftwareOracle().law(cases[name], family),
                            "source": {"reproducer_commit": source["commit"], "files_sha256": dict(source_files),
                                       "new_test_files_sha256": {pub.PROTOCOL: source_files[pub.PROTOCOL]},
                                       "binder_sha256": source_files[pub.BINDER]},
                            "artifacts": {"state": pub.binding(artifact), "samples": pub.binding(samples)},
                            "observations": [{"completed_steps": 0, "samples": 8, "evaluation_seed": 34002,
                                              "passed": True, "failed_bounds": [], "metrics": {"quality": 1.}}]})
    capacity = {"schema": pub.CAPACITY_SCHEMA, "status": "COMPLETE", "required_records": 16,
                "requested_cells": [{"family": f, "case_id": n} for f in pub.FAMILIES for n in case_ids],
                "ordinary_training_updates": 0, "fitting_updates": 0, "ordinary_qualification_credit": False, "records": records}
    capacity_pin = write(tmp_path / "capacity.json", capacity)
    spec = {"id": "software-named-defaults-contrast", "recipe_overrides": overrides, "seed": 24002,
            "cases": declared, "representation_card": capacity_pin,
            "candidate_budget_seconds": 7680., "family_budget_seconds": dict(pub.FAMILY_REMAINING),
            "budget_seconds": pub.PAIR_REMAINING, "campaign_cap_seconds": pub.CAMPAIGN_CAP, "export_grace_seconds": 60., "frames": 9}
    runtime = {"device": "cuda:0", "torch_threads": 1, "cuda_device_model": "Synthetic GPU"}
    trials = []
    for family in pub.FAMILIES:
        recipe = SoftwareOracle().recipe(cases[case_ids[0]], family)
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
        observations = [{"step": i * 25, "elapsed_seconds": i / 24, "passed": i != 7,
                         "failed_bounds": [] if i != 7 else ["quality"], "metrics": {"quality": 1. if i != 7 else .5},
                         "views": [{"kind": "image", "title": "synthetic"}]} for i in range(25)]
        raw = {"status": "COMPLETE", "case": cases[case_ids[0]], "seed": 24002,
               "requested_recipe_overrides": overrides, "source": source, "recipe": recipe, "runtime": runtime,
               "protocol": {"updates": 600, "evaluation_samples": 8, "wall_cap_seconds": 180., "media_frames": 9},
               "passed": True, "verdict": "PASS", "elapsed_seconds": 1., "completed_updates": 600,
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
    prior_info = add_prior(tmp_path, root, packet)
    spec["prior_carryover"] = prior_info.carry
    packet["spec_sha256"] = pub.digest(spec)
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
                                   "full_protocol_complete": False,
                                   "resolved_recipe_sha256": pub.digest(SoftwareOracle().recipe(cases[item["id"]], trial["family"]))}
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
    (root / pub.SELF).parent.mkdir(parents=True, exist_ok=True)
    (root / pub.SELF).write_bytes(MODULE.read_bytes())
    monkeypatch.setattr(pub.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="b" * 40 + "\n", returncode=0))
    return SimpleNamespace(root=root, combined=combined, card=card, output=tmp_path / "publication",
                           packet=packet, oracle=SoftwareOracle(), capacity=tmp_path / "capacity.json", frozen=frozen, prior=prior_info)


def export(c):
    return pub.publish(c.card, pub.file_sha(c.card), c.output, publisher_root=c.root, oracle=c.oracle)


def refresh(c, edit):
    packet = pub.read(c.combined)
    edit(packet)
    write(c.combined, packet)
    write(c.card, pub.bind_certification(c.combined, c.root))


def rebind_archives(c, packet):
    """Rebind explicit synthetic certification after a software-only fixture edit."""
    packet["spec_sha256"] = pub.digest(packet["spec"])
    archives = []
    for family in pub.FAMILIES:
        previous_pin = next(pin for pin in packet["family_archives"] if pin["family"] == family)
        archive = deepcopy(packet)
        archive.pop("family_archives", None)
        archive.update(executed_family=family)
        own_trial = next(trial for trial in archive["trials"] if trial["family"] == family)
        runtime = None if own_trial["status"] == "BLOCKED" else deepcopy(c.packet["family_archives"][0]["runtime"])
        archive.update(lane_runtime=runtime, spent_seconds=own_trial["paid_wall_seconds"],
                       measured_paid_seconds=sum(row.get("paid_wall_seconds", 0.) for row in own_trial["cases"]),
                       unmeasured_interrupt_reservation_seconds=0.)
        if runtime is None:
            archive.pop("execution_source", None)
        for trial in archive["trials"]:
            if trial["family"] != family:
                trial.update(status="UNKNOWN", paid_wall_seconds=0.)
                trial["cases"] = [{**item, "status": "UNKNOWN", "original_gate": None, "study_gate": None,
                    "full_protocol_complete": False,
                    "resolved_recipe_sha256": pub.digest(c.oracle.recipe(packet["case_definitions"][item["id"]], trial["family"]))}
                    for item in packet["spec"]["cases"]]
        archives.append({"family": family, **write(Path(previous_pin["path"]), archive), "runtime": runtime})
    packet["family_archives"] = archives
    packet["selection"] = c.oracle.select(packet)
    write(c.combined, packet)
    write(c.card, pub.bind_certification(c.combined, c.root))


def negative_capacity(c, status, families):
    raw = pub.read(c.capacity)
    packet = deepcopy(c.packet)
    for record in raw["records"]:
        if record["family"] not in families:
            continue
        record["status"] = status
        if status == "UNRESOLVED":
            record["observations"][0].update(passed=False, failed_bounds=["quality"], metrics={"quality": .5})
        else:
            error = {"stage": "capacity_preparation", "type": "SyntheticPreparationError",
                     "message": "Bound software preparation failure", "traceback": "Synthetic traceback; no model called"}
            pin = write(c.capacity.parent / "errors" / record["family"] / (record["case_id"] + ".json"), error)
            record.update(observations=[], artifacts={}, capacity_credit=False, error=error, error_artifact=pin)
        packet["capacity_preflight"][record["family"] + "/" + record["case_id"]] = deepcopy(record)
    packet["spec"]["representation_card"] = write(c.capacity, raw)
    return packet


def block_families(c, packet, families):
    for trial in packet["trials"]:
        if trial["family"] not in families:
            continue
        trial.update(status="BLOCKED", paid_wall_seconds=0.)
        trial["cases"] = [{**item, "status": "BLOCKED", "original_gate": None, "study_gate": None,
            "full_protocol_complete": False,
            "resolved_recipe_sha256": pub.digest(c.oracle.recipe(packet["case_definitions"][item["id"]], trial["family"]))}
            for item in packet["spec"]["cases"]]
    packet.update(spent_seconds=sum(trial["paid_wall_seconds"] for trial in packet["trials"]),
                  measured_paid_seconds=sum(row.get("paid_wall_seconds", 0.) for trial in packet["trials"] for row in trial["cases"]))
    if set(families) == set(pub.FAMILIES):
        packet.pop("execution_source", None)
    rebind_archives(c, packet)


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
    lambda p: p["family_paid_budget_seconds"].update(ka2=1.),
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
    packet["records"][-1]["observations"][0].update(passed=False, failed_bounds=["quality"], metrics={"quality": .5})
    pin = write(cohort.capacity, packet)
    value = deepcopy(cohort.packet)
    value["spec"]["representation_card"] = pin
    value["capacity_preflight"]["k3p/api-software-7"] = packet["records"][-1]
    projected = pub.capacity(pub.Inputs(), value, cohort.root, cohort.oracle)
    assert projected["k3p/api-software-7"]["status"] == "UNRESOLVED"


@pytest.mark.parametrize("status", ["UNRESOLVED", "BLOCKED"])
def test_all_sixteen_negative_capacity_outcomes_publish_without_learning_credit(cohort, status):
    packet = negative_capacity(cohort, status, pub.FAMILIES)
    block_families(cohort, packet, pub.FAMILIES)
    result = export(cohort)
    assert result["counts"]["capacity"] == {status: 16}
    assert result["counts"]["execution"] == {"BLOCKED": 16}
    assert result["counts"]["original_gates"] == {"UNAVAILABLE": 16}
    assert result["counts"]["study_gates"] == {"UNAVAILABLE": 16}
    assert result["counts"]["goal_gifs"] == 0 and result["selection"]["fully_qualified_ids"] == []
    assert result["costs"]["ordinary_paid_seconds"] == 0.
    assert result["costs"]["prior_total_paid_seconds"] == pub.PRIOR_PAID
    assert all(row["completed_updates"] is None for row in result["cases"])


def test_one_negative_capacity_family_keeps_supported_sibling_observation_separate(cohort):
    packet = negative_capacity(cohort, "UNRESOLVED", ("ka2",))
    block_families(cohort, packet, ("ka2",))
    result = export(cohort)
    assert result["counts"]["execution"] == {"BLOCKED": 8, "FAIL": 1, "UNKNOWN": 7}
    assert result["counts"]["capacity"] == {"UNRESOLVED": 8, "SUPPORTED": 8}
    assert result["counts"]["goal_gifs"] == 1 and result["costs"]["ordinary_paid_seconds"] == 2.


def test_negative_capacity_cannot_authenticate_an_ordinary_training_row(cohort):
    packet = negative_capacity(cohort, "UNRESOLVED", ("ka2",))
    rebind_archives(cohort, packet)
    with pytest.raises(ValueError, match="negative capacity family"):
        export(cohort)
    assert not cohort.output.exists()


@pytest.mark.parametrize("mutate", [
    lambda r: r["resolved_recipe"].update(total_steps=None),
    lambda r: r["requested_recipe_overrides"].update(prior_lr_mult=True),
    lambda r: r["family_law"].update(continuous_policy="dv12"),
    lambda r: r["source"]["new_test_files_sha256"].update({pub.PROTOCOL: "0" * 64}),
    lambda r: r["observations"][0].update(samples=4),
    lambda r: r["observations"][0].update(passed=False),
])
def test_capacity_named_recipe_source_law_and_original_boolean_guards(cohort, mutate):
    raw = pub.read(cohort.capacity)
    mutate(raw["records"][0])
    packet = deepcopy(cohort.packet)
    packet["capacity_preflight"]["ka2/api-software-0"] = deepcopy(raw["records"][0])
    packet["spec"]["representation_card"] = write(cohort.capacity, raw)
    with pytest.raises(ValueError):
        pub.capacity(pub.Inputs(), packet, cohort.root, cohort.oracle)


def test_blocked_capacity_error_artifact_is_required_and_bound(cohort):
    packet = negative_capacity(cohort, "BLOCKED", pub.FAMILIES)
    error = packet["capacity_preflight"]["ka2/api-software-0"]["error_artifact"]
    Path(error["path"]).write_text("modified preparation evidence")
    with pytest.raises(ValueError, match="drift"):
        pub.capacity(pub.Inputs(), packet, cohort.root, cohort.oracle)


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
        pub.capacity(pub.Inputs(), value, cohort.root, cohort.oracle)


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
    value["capacity_preflight"]["ka2/api-software-0"] = raw["records"][0]
    value["spec"]["representation_card"] = write(cohort.capacity, raw)
    with pytest.raises(ValueError, match="state and sampled"):
        pub.capacity(pub.Inputs(), value, cohort.root, cohort.oracle)


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


def test_git_blob_reader_compares_committed_bytes_without_checkout_mutation(tmp_path, monkeypatch):
    original = b"answer = 1\n"
    monkeypatch.setattr(pub.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=original, returncode=0))
    path = tmp_path / "source.py"
    path.write_bytes(original)
    assert pub.git_blob(tmp_path, "a" * 40, "source.py") == pub.file_sha(path)
    path.write_bytes(b"answer = 2\n")
    assert pub.git_blob(tmp_path, "a" * 40, "source.py") != pub.file_sha(path)
    monkeypatch.setattr(pub.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=b"", returncode=1))
    with pytest.raises(ValueError, match="blob unavailable"):
        pub.git_blob(tmp_path, "a" * 40, "source.py")


def add_prior(tmp_path, root, template):
    """Four negative synthetic runs and a distinct ERROR; no model operations."""
    old_root = tmp_path / "previous-science"
    files = {}
    for name in ("benchmarks/pure.py", pub.DELEGATED, pub.PREVIOUS, *pub.DISCOVERY):
        target = old_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
        if name not in pub.DISCOVERY:
            files[name] = pub.file_sha(target)
    queue = Path(template["coordinator"]["queue_root"])

    def frozen_source(source, directory):
        manifest = {"schema_version": 1, "origin_commit": source["commit"],
                    "files": {**source["files_sha256"], **source.get("discovery_inputs_sha256", {})}}
        manifest["digest"] = pub.digest(manifest["files"])
        for name in manifest["files"]:
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((old_root / name).read_bytes())
        write(directory / "forge-source.json", manifest)
        return {**manifest, "snapshot_path": str(directory)}

    def original_template():
        packet = deepcopy(template)
        packet.pop("family_archives", None)
        for trial, family in zip(packet["trials"], pub.OLD_FAMILIES):
            trial["family"] = family
        return packet

    setup = original_template()
    setup_source = {"commit": "e" * 40, "files_sha256": files,
                    "discovery_inputs_sha256": dict(pub.DISCOVERY)}
    setup_execution = frozen_source(setup_source, tmp_path / "setup-snapshot")
    setup.update(source=setup_source, execution_source=setup_execution,
                 study_id="prior-engineering", executed_family="atlas",
                 spent_seconds=pub.PRIOR_ENGINEERING, measured_paid_seconds=pub.PRIOR_ENGINEERING)
    for trial in setup["trials"]:
        trial["status"] = "ERROR" if trial["family"] == "atlas" else "UNKNOWN"
        trial["paid_wall_seconds"] = pub.PRIOR_ENGINEERING if trial["family"] == "atlas" else 0.
        trial["cases"] = [{**row, "status": "UNKNOWN", "original_gate": None,
                           "study_gate": None, "full_protocol_complete": False}
                          for row in setup["spec"]["cases"]]
    setup["trials"][0]["cases"][0].update(status="ERROR", paid_wall_seconds=pub.PRIOR_ENGINEERING,
                                         attempt_key="previous-engineering")
    durable = queue / "policy/attempts/previous-engineering"
    request = write(durable / "supervisor-request.json", {"token": "setup", "source": setup_execution})
    terminal = write(durable / "supervisor-terminal.json", {
        "token": "setup", "attempt_status": "completed", "child_returncode": 1,
        "paid_wall_seconds": pub.PRIOR_ENGINEERING})
    log = durable / "startup.log"
    log.write_text("Synthetic startup ERROR before model construction; no science\n")
    setup_pin = write(tmp_path / "previous-engineering/study.json", setup)
    engineering = {"family": "atlas", "paid_seconds": pub.PRIOR_ENGINEERING,
        "claim": "Original bootstrap ERROR before fixture construction; no scientific credit",
        "artifacts": {"study": setup_pin, "request": request, "terminal": terminal, "log": pub.binding(log)}}

    def closed_pair(label, commit, knobs, scientific_paid, previous_carry=None):
        directory = tmp_path / label
        source = {"commit": commit * 40, "files_sha256": dict(files),
                  "discovery_inputs_sha256": dict(pub.DISCOVERY)}
        execution = frozen_source(source, tmp_path / (label + "-snapshot"))
        old = original_template()
        old.update(source=source, execution_source=execution, study_id=label,
                   measured_paid_seconds=sum(scientific_paid.values()), spent_seconds=sum(scientific_paid.values()))
        old["spec"].update(id=label, recipe_overrides=knobs,
                           family_budget_seconds={"atlas": 7680. - pub.PRIOR_ENGINEERING, "e22": 7680.},
                           budget_seconds=15360. - pub.PRIOR_ENGINEERING)
        if previous_carry is None:
            old["spec"]["engineering_carryover"] = deepcopy(engineering)
        else:
            old["spec"]["prior_carryover"] = deepcopy(previous_carry)
        old["family_paid_budget_seconds"] = deepcopy(old["spec"]["family_budget_seconds"])
        old["spec_sha256"] = pub.digest(old["spec"])
        for trial in old["trials"]:
            family = trial["family"]
            trial.update(recipe_overrides=knobs, paid_wall_seconds=scientific_paid[family])
            row = trial["cases"][0]
            original = pub.read(row["receipt_path"])
            target = directory / family / row["id"]
            target.mkdir(parents=True, exist_ok=True)
            for name in original["artifacts"]:
                (target / name).write_bytes((Path(row["receipt_path"]).parent / name).read_bytes())
            recipe = {"name": family, **knobs}
            original.update(source=source, requested_recipe_overrides=knobs, recipe=recipe,
                            passed=False, verdict="FAIL")
            for observation in original["observations"]:
                observation.update(passed=False, failed_bounds=["quality"], metrics={"quality": .5})
            receipt_pin = write(target / "receipt.json", original)
            row.update(status="FAIL", original_gate="FAIL", study_gate="FAIL", acquisition_hold=SoftwareOracle().hold(original),
                recipe=recipe, resolved_recipe_sha256=pub.digest(recipe), child_returncode=1,
                paid_wall_seconds=scientific_paid[family], final_metrics={"quality": .5},
                receipt_path=receipt_pin["path"], receipt_sha256=receipt_pin["sha256"], attempt_key=label + "-" + family)
            attempt = queue / "policy/attempts" / row["attempt_key"]
            write(attempt / "supervisor-request.json", {"token": family, "source": execution})
            write(attempt / "supervisor-terminal.json", {
                "token": family, "attempt_status": "completed", "child_returncode": 1,
                "paid_wall_seconds": scientific_paid[family]})
        archive_pins = []
        for family in pub.OLD_FAMILIES:
            archive = deepcopy(old)
            archive.update(executed_family=family, lane_runtime=old["trials"][0]["cases"][0]["runtime"],
                           spent_seconds=scientific_paid[family], measured_paid_seconds=scientific_paid[family])
            for trial in archive["trials"]:
                if trial["family"] != family:
                    trial.update(paid_wall_seconds=0., status="UNKNOWN")
                    trial["cases"] = [{**row, "status": "UNKNOWN", "original_gate": None,
                                       "study_gate": None, "full_protocol_complete": False}
                                      for row in archive["spec"]["cases"]]
            archive_pins.append({"family": family, **write(directory / family / "study.json", archive),
                                 "runtime": archive["lane_runtime"]})
        old["family_archives"] = archive_pins
        combined_pin = write(directory / "combined.json", old)
        card = {"schema": "particlegan_" + label + "_publication_certification_v1", "combined": combined_pin,
                "certifier": {"root": str(old_root), **source, "function": "combine_studies"},
                "verification": dict(pub.VERIFICATION)}
        costs = {"ordinary_paid_seconds": sum(scientific_paid.values()), "ordinary_reserved_seconds": 0.}
        if previous_carry is None:
            card["engineering_carryover"] = deepcopy(engineering)
            costs.update(engineering_paid_seconds=pub.PRIOR_ENGINEERING, combined_charged_seconds=pub.EARLIER_TOTAL)
        else:
            card["prior_carryover"] = deepcopy(previous_carry)
            costs.update(prior_scientific_paid_seconds=sum(pub.EARLIER_SCIENTIFIC.values()),
                         prior_engineering_paid_seconds=pub.PRIOR_ENGINEERING,
                         prior_total_paid_seconds=pub.EARLIER_TOTAL, combined_charged_seconds=pub.PRIOR_PAID)
        card_pin = write(directory / "certification.json", card)
        report_pin = write(directory / "publication-v1/results.json", {
            "schema": "particlegan_" + label + "_publication_v1", "combined": combined_pin,
            "certification": card_pin, "costs": costs})
        pins = {"combined.json": combined_pin, "certification.json": card_pin, "publication-v1/results.json": report_pin,
                **{pin["family"] + "/study.json": {key: pin[key] for key in ("path", "sha256", "bytes")} for pin in archive_pins}}
        return SimpleNamespace(packet=old, pins=pins, root=old_root, combined=Path(combined_pin["path"]),
                               card=Path(card_pin["path"]), report=Path(report_pin["path"]))

    first = closed_pair("critic_balance", "c", {"lr": .0053125, "prior_lr_mult": 1.5, "d_lr_mult": 2.25},
                        pub.EARLIER_SCIENTIFIC)
    earlier_carry = {"schema": "fixed_contrast_previous_debit_v1", "engineering": deepcopy(engineering),
                     "scientific_paid_seconds": dict(pub.EARLIER_SCIENTIFIC), "total_paid_seconds": pub.EARLIER_TOTAL,
                     "artifacts": deepcopy(first.pins)}
    latest = closed_pair("generator_step", "d", {"lr": .00265625, "prior_lr_mult": 3., "d_lr_mult": 4.5},
                         pub.PREVIOUS_SCIENTIFIC, earlier_carry)
    latest.carry = {"schema": "named_family_previous_debit_v1", "engineering_paid_seconds": pub.PRIOR_ENGINEERING,
                    "scientific_paid_seconds": dict(pub.PRIOR_SCIENTIFIC), "total_paid_seconds": pub.PRIOR_PAID,
                    "slot_predecessor": dict(pub.SLOT_PREDECESSOR), "slot_prior_paid_seconds": dict(pub.SLOT_PRIOR),
                    "artifacts": deepcopy(latest.pins)}
    latest.setup = Path(setup_pin["path"])
    latest.first = first
    return latest


def test_prior_science_and_engineering_debited_once_without_new_cell_credit(cohort):
    result = export(cohort)
    costs = result["costs"]
    assert costs["prior_scientific_paid_seconds"] == 109.23634317959659
    assert costs["prior_engineering_paid_seconds"] == pub.PRIOR_ENGINEERING
    assert costs["prior_total_paid_seconds"] == pub.PRIOR_PAID
    assert costs["combined_charged_seconds"] == 4. + pub.PRIOR_PAID
    assert costs["combined_cap_seconds"] == 15360.
    previous = result["prior_cohorts"]
    assert len(previous["scientific"]) == 4 and len(previous["engineering"]) == 1
    assert all(row["status"] == "FAIL" and not row["new_cell_credit"] for row in previous["scientific"])
    assert previous["engineering"][0]["status"] == "ERROR"
    assert len(result["cases"]) == 16 and result["counts"]["goal_gifs"] == 2
    text = (cohort.output / "README.md").read_text()
    assert "prior science counted once" in text and "prior engineering counted once" in text


@pytest.mark.parametrize("mutation", [
    lambda p: p["spec"]["recipe_overrides"].update(d_lr_mult=2.25),
    lambda p: p["spec"]["recipe_overrides"].update(prior_lr_mult=1.5),
    lambda p: p["spec"].update(budget_seconds=15360.),
    lambda p: p["spec"]["family_budget_seconds"].update(ka2=7680.),
    lambda p: p["spec"]["prior_carryover"].update(total_paid_seconds=0.),
    lambda p: p["spec"]["prior_carryover"]["scientific_paid_seconds"].update(atlas=0.),
    lambda p: p["spec"]["prior_carryover"]["slot_predecessor"].update(ka2="e22"),
    lambda p: p["spec"]["prior_carryover"]["slot_prior_paid_seconds"].update(k3p=0.),
    lambda p: p["source"]["files_sha256"].pop(pub.DELEGATED),
    lambda p: p["source"]["files_sha256"].pop(pub.BINDER),
    lambda p: p["source"]["files_sha256"].pop(pub.PROTOCOL),
    lambda p: p["source"]["files_sha256"].pop(pub.PREVIOUS),
    lambda p: p["source"]["discovery_inputs_sha256"].clear(),
])
def test_fixed_new_profile_prior_debit_and_delegated_discovery_source_guards(cohort, mutation):
    refresh(cohort, mutation)
    with pytest.raises((ValueError, AssertionError)):
        export(cohort)
    assert not cohort.output.exists()


@pytest.mark.parametrize("name", ["combined.json", "certification.json", "publication-v1/results.json", "atlas/study.json", "e22/study.json"])
def test_each_prior_artifact_hash_is_required_without_recertifying_models(cohort, name):
    path = Path(cohort.packet["spec"]["prior_carryover"]["artifacts"][name]["path"])
    path.write_bytes(b"modified prior evidence")
    with pytest.raises(ValueError, match="drift"):
        export(cohort)
    assert not cohort.output.exists()


@pytest.mark.parametrize("name", ["combined.json", "certification.json", "publication-v1/results.json", "atlas/study.json", "e22/study.json"])
def test_nested_earlier_artifact_is_bound_without_counting_its_cost_twice(cohort, name):
    path = Path(cohort.prior.packet["spec"]["prior_carryover"]["artifacts"][name]["path"])
    path.write_bytes(b"modified nested prior evidence")
    with pytest.raises(ValueError, match="drift"):
        export(cohort)
    assert not cohort.output.exists()


def test_current_named_recipe_hash_requires_full_host_schedule(cohort):
    packet = deepcopy(cohort.packet)
    trial = packet["trials"][0]
    recipe = cohort.oracle.recipe(packet["case_definitions"][trial["cases"][0]["id"]], trial["family"])
    recipe["total_steps"] = None
    trial["cases"][0]["resolved_recipe_sha256"] = pub.digest(recipe)
    rebind_archives(cohort, packet)
    with pytest.raises(ValueError, match="exact named factory horizon"):
        export(cohort)


def test_prior_paid_counter_cannot_hide_durable_scientific_interval(cohort):
    row = cohort.prior.packet["trials"][0]["cases"][0]
    path = Path(cohort.packet["coordinator"]["queue_root"]) / "policy/attempts" / row["attempt_key"] / "supervisor-terminal.json"
    raw = pub.read(path)
    raw["paid_wall_seconds"] = 0.
    write(path, raw)
    with pytest.raises(ValueError, match="durable paid"):
        export(cohort)


def test_prior_source_cannot_be_pooled_with_current_trio(cohort):
    value, paid = pub.prior(pub.Inputs(), cohort.oracle, cohort.packet, pub.read(cohort.card))
    assert paid == pub.PRIOR_PAID
    assert value["source"]["commit"] != cohort.packet["source"]["commit"]
    assert value["source"]["root"] == str(cohort.prior.root)
    cohort.output = cohort.prior.root / "new-publication"
    with pytest.raises(ValueError, match="outside scientific"):
        export(cohort)


def test_original_discovery_and_fixed_profile_constants_are_not_software_aliases():
    assert pub.OVERRIDES == {"lr": .006375, "prior_lr_mult": 1., "d_lr_mult": 1.}
    assert pub.DISCOVERY == {"configs/forge/tasks/ring16_acquisition.json":
                             "e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1"}
    assert pub.FAMILIES == ("ka2", "k3p")
    assert pub.SLOT_PREDECESSOR == {"ka2": "atlas", "k3p": "e22"}


def test_construction_summary_and_nested_inputs_are_hash_bound(tmp_path):
    state = tmp_path / "reference.pt"
    state.write_bytes(b"Synthetic reference parameters; never loaded")
    summary = write(tmp_path / "summary.json", {"reference": "synthetic"})
    child = tmp_path / "samples.npz"
    child.write_bytes(b"Synthetic retained numeric artifact")
    inputs = pub.Inputs()
    value = {**pub.binding(state), "summary_path": summary["path"], "summary_sha256": summary["sha256"],
             "extra": {"child": pub.binding(child)}}
    pub.retained_bindings(inputs, value)
    assert set(inputs.files) == {str(state), str(child), summary["path"]}
    summary_path = Path(summary["path"])
    summary_path.write_text("modified bound summary")
    with pytest.raises(ValueError, match="drift"):
        pub.retained_bindings(pub.Inputs(), value)

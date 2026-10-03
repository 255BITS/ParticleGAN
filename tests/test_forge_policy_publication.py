"""Self-contained publication controls; no historical Git objects or models."""
from copy import deepcopy
import json
from pathlib import Path

from PIL import Image
import pytest


from experiments.forge import policy_publication as publisher


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False))


def fixture(root, name="software-cohort", commit="a" * 40, *, all_late=False):
    """Boolean/byte fixtures exercise projection; they are not toy evidence."""
    root.mkdir(parents=True, exist_ok=True)
    source = {"commit": commit, "files_sha256": {"software-fixture.py": "b" * 64}}
    cases = [{"id": f"software-case{i}", "title": f"Software fixture {i}", "goal": "Publication software control only",
              "default_steps": 24, "batch_size": 2, "eval_samples": 2, "evaluation_observations": 24,
              "sampling": {"law": "synthetic software bytes; no samples generated"},
              "thresholds": {"observations": 24}} for i in range(8)]
    protocol = {"id": name, "families": ["atlas", "e22"], "seed": 24002,
                "grid": {"lr": [.006375, .0085], "prior_lr_mult": [1., 2.]}, "frames": 2,
                "cases": [{"id": case["id"], "tier": 1 if i < 2 else 2, "timeout_seconds": 120}
                          for i, case in enumerate(cases)], "speed_ranking": False, "default_adoption": False,
                "stability": {"confirmation_checks": 5, "post_confirmation_hold_checks": 5,
                              "first_window_only": True, "all_subsequent_primary_checks": True}}
    trials = []
    for family in ("atlas", "e22"):
        runtime = {"device": "cuda:0", "cuda_device_model": "software fixture; no GPU used"}
        for lr in protocol["grid"]["lr"]:
            for rate in protocol["grid"]["prior_lr_mult"]:
                knobs = {"lr": lr, "prior_lr_mult": rate}
                identity = family + "--" + publisher.digest({"family": family, "overrides": knobs})
                recipe = {"family": family, **knobs}
                rows = [{**declared, "status": "UNKNOWN", "case_sha256": publisher.digest(case),
                         "resolved_recipe_sha256": publisher.digest(recipe), "original_gate": None,
                         "study_gate": None, "full_protocol_complete": False}
                        for declared, case in zip(protocol["cases"], cases)]
                trial = {"id": identity, "family": family, "recipe_overrides": knobs, "status": "UNKNOWN",
                         "paid_wall_seconds": 0., "cases": rows}
                # One source/config acquires too late; another recovers after a
                # first-window break. Neither can borrow another case's PASS.
                if lr == .006375:
                    count = 1 if rate == 1 or all_late else 2
                    for index in range(count):
                        late = rate == 1 or all_late
                        passing = (lambda step: step >= 16) if late else (
                            (lambda step: step > 0) if index == 0 else (lambda step: step > 0 and step != 6))
                        observations = [{"step": step, "elapsed_seconds": float(step), "passed": passing(step),
                                         "failed_bounds": [] if passing(step) else ["software shape bound"],
                                         "metrics": {"quality": float(passing(step))}}
                                        for step in range(25)]
                        directory = root / family / identity / cases[index]["id"]
                        directory.mkdir(parents=True, exist_ok=True)
                        first, final = Image.new("RGB", (4, 4), "black"), Image.new("RGB", (4, 4), "white")
                        first.save(directory / "goal.gif", save_all=True, append_images=[final], duration=100, loop=0)
                        (directory / "observations.npz").write_bytes(b"software-only fixture bytes; no observations")
                        (directory / "final-state.pt").write_bytes(b"software-only fixture bytes; no model")
                        artifacts = {filename: {"sha256": publisher.file_hash(directory / filename),
                                                "bytes": (directory / filename).stat().st_size}
                                     for filename in ("goal.gif", "observations.npz", "final-state.pt")}
                        receipt = {"status": "COMPLETE", "default_protocol_complete": True, "verdict": "PASS",
                                   "passed": True, "sustained_metric_passed": True, "source_unchanged": True,
                                   "source": source, "case": cases[index], "seed": 24002,
                                   "requested_recipe_overrides": knobs, "recipe": recipe, "runtime": runtime,
                                   "elapsed_seconds": 24., "artifacts": artifacts, "failed_bounds": [], "gif_frames": 2,
                                   "protocol": {"metric_evaluation_steps": list(range(25)), "terminal_observations": 5,
                                                "updates": 24, "default_updates": 24, "evaluation_samples": 2,
                                                "default_evaluation_samples": 2, "wall_cap_seconds": 120,
                                                "media_frames": 2, "media_steps": [0, 24]}, "observations": observations}
                        write(directory / "receipt.json", receipt)
                        acquired = 20 if late else 5
                        hold = {"status": "INCOMPLETE" if late else "PASS" if index == 0 else "FAIL",
                                "reason": "software control", "acquired_step": acquired,
                                "acquired_seconds": float(acquired), "hold_checks": 4 if late else 19,
                                "hold_passed": 4 if late else 19 if index == 0 else 18, "speed_eligible": False}
                        rows[index].update(status=hold["status"], original_gate="PASS", study_gate=hold["status"],
                                           full_protocol_complete=True, acquisition_hold=hold, paid_wall_seconds=30.,
                                           elapsed_seconds=24., receipt_path=str(directory / "receipt.json"),
                                           receipt_sha256=publisher.file_hash(directory / "receipt.json"),
                                           recipe=recipe, runtime=runtime, artifacts=artifacts,
                                           final_metrics=observations[-1]["metrics"], child_returncode=0)
                    trial.update(status="INCOMPLETE" if rate == 1 or all_late else "FAIL", paid_wall_seconds=30. * count)
                trials.append(trial)
    packet = {"schema": publisher.SCHEMA, "spec": protocol, "spec_sha256": publisher.digest(protocol),
              "source": source, "case_definitions": {case["id"]: case for case in cases},
              "capacity_preflight": {"software_fixture": True}, "runtime_contract": {"fixture": True},
              "measured_paid_seconds": sum(trial["paid_wall_seconds"] for trial in trials),
              "unmeasured_interrupt_reservation_seconds": 0.,
              "selection": {"fully_qualified_ids": [], "outcome": "pending"}, "trials": trials}
    seal(root, packet)
    return root / "combined.json", packet


def seal(root, packet):
    """Rebind software mutations so the intended field guard is exercised."""
    archives = []
    for family in ("atlas", "e22"):
        runtime = {"device": "cuda:0", "cuda_device_model": "software fixture; no GPU used"}
        lane = deepcopy(packet)
        lane.pop("family_archives", None)
        lane.update(executed_family=family, lane_runtime=runtime)
        for trial in lane["trials"]:
            if trial["family"] != family:
                trial["status"] = "UNKNOWN"
                for row in trial["cases"]:
                    row.update(status="UNKNOWN", original_gate=None, study_gate=None, full_protocol_complete=False)
                    for key in ("receipt_path", "receipt_sha256", "artifacts", "runtime", "final_metrics", "recipe", "acquisition_hold"):
                        row.pop(key, None)
        path = root / family / "study.json"
        write(path, lane)
        archives.append({"family": family, "path": str(path), "sha256": publisher.file_hash(path), "runtime": runtime})
    packet["family_archives"] = archives
    write(root / "combined.json", packet)


def test_preserves_original_pass_late_hold_and_all_denominators(tmp_path):
    path, _ = fixture(tmp_path / "raw")
    cohort = publisher.load_combined(path)
    assert len(cohort["trials"]) == 8
    assert sum(len(trial["cases"]) for trial in cohort["trials"]) == 64
    late = [row for trial in cohort["trials"] for row in trial["cases"] if row["study_gate"] == "INCOMPLETE"]
    assert len(late) == 2 and all(row["original_gate"] == "PASS" for row in late)
    assert all(row["diagnostic"]["terminal_passing_suffix"] == 9 for row in late)
    recovered = [row for trial in cohort["trials"] for row in trial["cases"]
                 if row["study_gate"] == "FAIL" and row["original_gate"] == "PASS"]
    assert len(recovered) == 2
    assert all(row["diagnostic"]["first_post_acquisition_failure"]["step"] == 6 for row in recovered)
    assert not cohort["selection"]["fully_qualified_ids"]


@pytest.mark.parametrize("control,match", [
    ("drop_unknown", "roster|denominator"), ("paper_pass", "lacks.*receipt"),
    ("late_to_pass", "retention gate changed"), ("paid_underclaim", "paid cost"),
    ("changed_recipe_hash", "Recipe"), ("changed_exit", "child exit"),
    ("source_pooling", "cohorts cannot be pooled"), ("source_receipt_changed", "source-bound protocol"),
    ("changed_stability", "unknown.*contract"), ("drop_config", "denominator"),
    ("changed_runtime", "original evidence"), ("borrow_case_other_config", "Recipe"),
    ("changed_media_bytes", "artifact changed"), ("changed_parent_archive", "archive changed")])
def test_rebound_negative_controls(tmp_path, control, match):
    path, packet = fixture(tmp_path / "raw")
    trial = next(trial for trial in packet["trials"] if trial["family"] == "atlas"
                 and trial["recipe_overrides"] == {"lr": .006375, "prior_lr_mult": 1.})
    row = trial["cases"][0]
    if control == "drop_unknown": trial["cases"].pop()
    elif control == "paper_pass": trial["cases"][2].update(status="PASS", original_gate="PASS", study_gate="PASS")
    elif control == "late_to_pass": row.update(status="PASS", study_gate="PASS", acquisition_hold={**row["acquisition_hold"], "status": "PASS"})
    elif control == "paid_underclaim": row["paid_wall_seconds"] = 0.
    elif control == "changed_recipe_hash": row["resolved_recipe_sha256"] = "0" * 64
    elif control == "changed_exit": row["child_returncode"] = 1
    elif control == "source_receipt_changed": packet["source"]["commit"] = "f" * 40
    elif control == "changed_stability":
        packet["spec"]["stability"]["post_confirmation_hold_checks"] = 4
        packet["spec_sha256"] = publisher.digest(packet["spec"])
    elif control == "drop_config": packet["trials"].pop()
    elif control == "changed_runtime": row["runtime"] = {**row["runtime"], "device": "cuda:9"}
    elif control == "borrow_case_other_config":
        other = next(t for t in packet["trials"] if t["family"] == "atlas"
                     and t["recipe_overrides"] == {"lr": .006375, "prior_lr_mult": 2.})
        trial["cases"][0] = deepcopy(other["cases"][0])
    seal(path.parent, packet)
    if control == "source_pooling":
        archive = packet["family_archives"][0]
        lane = json.loads(Path(archive["path"]).read_text())
        lane["source"]["commit"] = "f" * 40
        write(Path(archive["path"]), lane)
        archive["sha256"] = publisher.file_hash(archive["path"])
        write(path, packet)
    elif control == "changed_media_bytes":
        raw = Path(row["receipt_path"]).parent / "goal.gif"
        raw.write_bytes(raw.read_bytes() + b"software corruption")
    elif control == "changed_parent_archive":
        raw = Path(packet["family_archives"][0]["path"])
        raw.write_text(raw.read_text() + " ")
    with pytest.raises(ValueError, match=match): publisher.load_combined(path)


def test_cohorts_remain_separate_and_raw_unchanged(tmp_path):
    first, _ = fixture(tmp_path / "first", name="old-source", commit="a" * 40)
    second, _ = fixture(tmp_path / "second", name="new-source", commit="c" * 40)
    before = {str(path): publisher.file_hash(path) for parent in (first.parent, second.parent)
              for path in parent.rglob("*") if path.is_file()}
    board = publisher.publish([first, second], tmp_path / "publication")
    assert len(board["cohorts"]) == 2
    assert board["current_cohort_id"].startswith("new-source")
    assert board["speed_winner"] is None and board["default_adoption"] is False
    assert board["cross_cohort_case_pooling"] is False
    assert all(not cohort["selection"]["fully_qualified_ids"] for cohort in board["cohorts"])
    assert before == {str(path): publisher.file_hash(path) for parent in (first.parent, second.parent)
                      for path in parent.rglob("*") if path.is_file()}


def test_all_media_copies_every_observed_run_without_changing_raw(tmp_path):
    path, _ = fixture(tmp_path / "raw")
    before = {str(raw): publisher.file_hash(raw) for raw in path.parent.rglob("*") if raw.is_file()}
    output = tmp_path / "publication"
    board = publisher.publish([path], output, all_media=True)
    cohort = board["cohorts"][0]
    observed = [row for trial in cohort["trials"] for row in trial["cases"] if row["actual_gif"]]
    assert len(observed) == len(cohort["representative_media"]) == 6
    assert sum(media["representative"] for media in cohort["representative_media"]) == 4
    assert cohort["published_media_scope"] == "all actual observed runs"
    for row in observed:
        published = row["published_gif"]
        assert publisher.file_hash(output / published["relative_path"]) == row["actual_gif"]["sha256"]
        assert published["relative_path"] in (output / "policy-family-inventory.md").read_text()
    assert before == {str(raw): publisher.file_hash(raw) for raw in path.parent.rglob("*") if raw.is_file()}


def test_pass_count_ties_are_explicit_and_exclude_unmeasured_configs(tmp_path):
    path, _ = fixture(tmp_path / "raw", all_late=True)
    board = publisher.publish([path], tmp_path / "publication")
    cohort = board["cohorts"][0]
    for family, ties in cohort["per_family_best_observed_ties"].items():
        assert len(ties) == 2 and ties[0] == cohort["per_family_best_observed"][family]
        assert all(next(trial for trial in cohort["trials"] if trial["id"] == identity)["measured_cases"] == 1
                   for identity in ties)
    text = (tmp_path / "publication/policy-family-inventory.md").read_text()
    assert "Pass-count ties:" in text and "no measured quality advantage" in text


def frozen_modules(monkeypatch):
    """Load preserved reproduction bytes without changing global import paths."""
    import importlib.util
    import sys
    directory = Path(__file__).resolve().parents[1] / "reports/forge/family-winner-round1"
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, directory / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    readout = load("_frozen_policy_family_readout", "policy_family_readout.py")
    # Only the original dated publisher uses a bare sibling import.
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "policy_family_readout", readout)
        publication = load("_frozen_policy_publication", "publish_policy_results.py")
    return publication, readout


def test_package_migration_preserves_original_projection_and_publication_bytes(tmp_path, monkeypatch):
    from experiments.forge import policy_family_readout as maintained_readout
    frozen_publication, frozen_readout = frozen_modules(monkeypatch)
    combined, _ = fixture(tmp_path / "originals")
    assert publisher.load_combined(combined) == frozen_publication.load_combined(combined)
    for family in ("atlas", "e22"):
        study = combined.parent / family / "study.json"
        retained, maintained = frozen_readout.project(study), maintained_readout.project(study)
        assert maintained == retained
        assert maintained_readout.markdown(maintained) == frozen_readout.markdown(retained)
    output = tmp_path / "publication"
    retained = frozen_publication.publish([combined], output, all_media=True)
    frozen_markdown = (output / "policy-family-inventory.md").read_bytes()
    maintained = publisher.publish([combined], output, all_media=True)
    # The publisher implementation identity changes deliberately; scientific
    # source/spec/runtime/receipt/artifact identities and rendered bytes do not.
    assert retained.pop("generated_by")["path"] == "reports/forge/family-winner-round1/publish_policy_results.py"
    assert maintained.pop("generated_by")["path"] == "experiments/forge/policy_publication.py"
    assert maintained == retained
    assert (output / "policy-family-inventory.md").read_bytes() == frozen_markdown


def test_maintained_publication_is_byte_deterministic_and_preserves_input_tree(tmp_path):
    combined, _ = fixture(tmp_path / "originals")
    original_hashes = {p.relative_to(combined.parent): publisher.file_hash(p)
                       for p in combined.parent.rglob("*") if p.is_file()}
    output = tmp_path / "publication"
    publisher.publish([combined], output, all_media=True)
    first = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    publisher.publish([combined], output, all_media=True)
    assert first == {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert original_hashes == {p.relative_to(combined.parent): publisher.file_hash(p)
                              for p in combined.parent.rglob("*") if p.is_file()}


def test_package_readout_cli_preserves_frozen_bytes(tmp_path, monkeypatch, capsys):
    from experiments.forge import policy_family_readout as maintained_readout
    _, frozen_readout = frozen_modules(monkeypatch)
    combined, _ = fixture(tmp_path / "originals")
    study = combined.parent / "atlas/study.json"
    frozen_readout.main([str(study), "--output", str(tmp_path / "frozen")])
    frozen_stdout = capsys.readouterr().out
    maintained_readout.main([str(study), "--output", str(tmp_path / "maintained")])
    assert capsys.readouterr().out == frozen_stdout
    for filename in ("README.md", "readout.json"):
        assert (tmp_path / "maintained" / filename).read_bytes() == (tmp_path / "frozen" / filename).read_bytes()


def test_frozen_reproduction_sources_keep_exact_git_and_byte_identities():
    import hashlib
    import subprocess
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / "reports/forge/publication-migration-2026-10-02.json").read_text())
    assert receipt["qualification_changed"] is False
    for original in receipt["originals_preserved"]:
        path = root / original["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original["sha256"]
        blob = subprocess.check_output(["git", "hash-object", str(path)], cwd=root, text=True).strip()
        assert blob == original["git_blob"]
        assert (root / original["maintained_path"]).is_file()


def test_maintained_publisher_provenance_binds_package_sources():
    provenance = publisher.publisher_source()
    expected = {"experiments/forge/policy_publication.py", "experiments/forge/policy_family_readout.py"}
    assert set(provenance["files_sha256"]) == expected
    assert provenance["path"] == "experiments/forge/policy_publication.py"
    assert all(publisher.file_hash(publisher.ROOT / name) == digest
               for name, digest in provenance["files_sha256"].items())

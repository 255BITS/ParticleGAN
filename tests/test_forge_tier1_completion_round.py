"""Frozen full-roster admission controls; no worker or training is launched."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest

from experiments.forge.configuration_search import materialize_search
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.queue import Queue
from experiments.forge.sources import inspect_source

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tier1_completion_driver", ROOT / "reports/forge/tier1-completion/run.py")
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


def test_singleton_registrations_retain_all_existing_cards_and_budgets(tmp_path):
    shutil.copytree(ROOT / "configs/forge", tmp_path / "configs/forge")
    definition = read_json(tmp_path / DRIVER.ROUND)
    before = {p.name: p.read_bytes() for p in (tmp_path / "configs/forge/configurations").glob("*.json")}
    specs = DRIVER.registration_specs(tmp_path, definition)
    assert len(specs) == 5
    assert {spec["base_candidate"] for spec in specs} == {
        row["candidate_id"] for row in definition["candidate_roster"] if "--" in row["candidate_id"]}
    for spec in specs:
        assert spec["campaign"] == definition["campaign"]
        assert spec["campaign"]["budget_seconds"] == 55440
        assert spec["campaign"]["candidate_budget_seconds"] == 5040
        assert len(spec["grid"]) == 1 and len(next(iter(spec["grid"].values()))) == 1
        assert spec["tuning_through_tier"] == 1
        assert [p.stem for p in materialize_search(tmp_path, spec)] == [spec["base_candidate"]]
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "configs/forge/configurations").glob("*.json")}
    release = next(spec for spec in specs if "release07" in spec["id"])
    assert release["trainer_family"] == "release07-gan-v3-mog"


@pytest.mark.parametrize("change", ["missing", "duplicate", "grid", "campaign"])
def test_registration_drift_rejected_before_queue_admission(tmp_path, change):
    shutil.copytree(ROOT / "configs/forge", tmp_path / "configs/forge")
    definition = read_json(tmp_path / DRIVER.ROUND)
    if change == "missing":
        definition["configuration_searches"].pop()
    elif change == "duplicate":
        definition["configuration_searches"][-1] = deepcopy(definition["configuration_searches"][0])
    else:
        reference = definition["configuration_searches"][0]
        spec = read_json(tmp_path / reference["spec"])
        if change == "grid":
            next(iter(spec["grid"].values())).append(1.1)
        else:
            spec["campaign"]["budget_seconds"] += 1
        # A rewritten reference hash cannot authorize a changed frozen recipe or budget.
        reference["spec_sha256"] = stable_hash(spec)
        atomic_json(tmp_path / reference["spec"], spec)
    with pytest.raises(ValueError, match="registration"):
        DRIVER.registration_specs(tmp_path, definition)
    assert not (tmp_path / "runs").exists()


def test_actual_full_roster_submit_registration_and_resume_without_training(tmp_path):
    root = tmp_path / "checkout"
    queue_root = tmp_path / "queue"
    # A compact real checkout includes all frozen evaluator/host source bytes and
    # ordinary admission manifests, without copying existing research logs.
    for relative in inspect_source(ROOT)["files"]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    shutil.copytree(ROOT / "configs/forge", root / "configs/forge", dirs_exist_ok=True)
    # Freeze a software-only fixture against this checkout. The historical
    # campaign cannot be resubmitted after task/source-pin changes; its committed
    # definition and outcomes remain immutable.
    for path in (root / "configs/forge/searches").glob("tier1-completion-v1-*.json"):
        path.unlink()  # Test-local fresh registrations; never edit the saved study.
    (root / DRIVER.ROUND).unlink()
    (root / DRIVER.CAMPAIGN).unlink()
    definition = DRIVER.prepare(root)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Forge controls", "-c", "user.email=forge@example.invalid",
                    "commit", "-qm", "Freeze admission software fixture"], cwd=root, check=True)
    _, requests = DRIVER.requests(root, queue_root, freeze=True)
    queue = Queue(queue_root, report_root=root / "reports/forge")
    configuration = next(req for _, req in requests if "configuration_id" in req["candidate"])
    with pytest.raises(ValueError, match="exact bounded search registration"):
        queue.submit(configuration, definition["campaign"])
    assert queue.inspect()["submissions"] == {}

    queue = DRIVER.enqueue(root, queue_root)
    state = queue.inspect()
    roster = read_json(queue_root / DRIVER.ID / "roster.json")
    assert len(roster) == len(state["submissions"]) == 11
    assert {r["candidate_id"] for r in roster} == {r["candidate_id"] for r in definition["candidate_roster"]}
    assert {entry["request"]["source"]["digest"] for entry in state["submissions"].values()} == {
        requests[0][1]["source"]["digest"]}
    assert state["campaigns"][DRIVER.ID]["definition"] == definition["campaign"]
    assert state["campaigns"][DRIVER.ID]["spent_seconds"] == 0
    assert not state["charges"] and all(not job["attempts"] for job in state["jobs"].values())
    for entry in state["submissions"].values():
        eligible, reason, running = queue._eligible(state, entry)
        assert eligible and reason is None and not running
        candidate = entry["request"]["candidate"]["id"]
        # The separate scheduled audit refuses KA2's unreviewed delayed switch;
        # policy candidates retain their explicitly unsupported host cells.
        expected = 4 if candidate in {"atlas", "e22"} else 7 if candidate.startswith("ka2--") else 8
        assert len(eligible) == expected
    registrations = list((root / "reports/forge/configuration-search").glob("*.json"))
    assert len(registrations) == 5
    assert all(read_json(p)["submitted_count"] == 1 for p in registrations)

    DRIVER.enqueue(root, queue_root)
    resumed = queue.inspect()
    assert set(resumed["submissions"]) == set(state["submissions"])
    assert set(resumed["jobs"]) == set(state["jobs"])
    assert not resumed["charges"] and all(not job["attempts"] for job in resumed["jobs"].values())

    DRIVER.summarize(root, queue_root)
    external = tmp_path / "external-volume" / "artifacts.tar.gz"
    receipt = DRIVER.archive(root, queue_root, external)
    assert external.is_file() and receipt["archive"]["path"] == str(external)
    assert not (root / "artifacts").exists()
    assert len(receipt["source_digests"]) == 1
    with pytest.raises(ValueError, match="archive already exists"):
        DRIVER.archive(root, queue_root, external)


def test_historical_round_rejects_current_task_revisions_before_admission(tmp_path):
    with pytest.raises(ValueError, match="view changed|task definition changed"):
        DRIVER.requests(ROOT, tmp_path / "queue")
    assert not (tmp_path / "queue").exists()

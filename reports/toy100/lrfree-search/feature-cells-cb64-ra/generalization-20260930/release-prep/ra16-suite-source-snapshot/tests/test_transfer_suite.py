"""The central runner must preserve the predeclared split and freeze ordering."""
import json
import pytest

from benchmarks.transfer_suite import suite
from benchmarks.transfer_suite.protocol import digest


def test_changed_fitting_source_blocks_transfer():
    with pytest.raises(RuntimeError, match="source changed during fitting"):
        suite.verify_source({"source_sha256": {"benchmarks/transfer_suite/protocol.py": "incorrect"}})


def task(name, tier, phase):
    return dict(name=name, family=name, tier=tier, phase=phase, runner="fixture",
                split="reserved" if phase == "reserved" else "development", steps=24,
                thresholds=[["quality", ">=", .9]], importance_reason="Fixture relevance.",
                limitations="Fixture only.")


def test_reserved_access_requires_freeze_and_diagnostics_do_not_disqualify(tmp_path, monkeypatch):
    output = tmp_path / "study"
    declared = dict(tasks=[task("core", "required", "fit"), task("fit", "ranking", "fit"),
                           task("image", "ranking", "validation"), task("stress", "diagnostic", "fit"),
                           task("reserved", "ranking", "reserved")], selection="fixture selection",
                    search=dict(proposal_rng=2731, initial_lr_weight_std=.008))
    calls = []
    def episode(spec, card, **options):
        if spec["split"] == "reserved":
            assert options["allow_reserved"]
            assert (output / "frozen.json").exists()
        calls.append((spec["name"], options))
        quality = 0. if spec["tier"] == "diagnostic" else 1.
        observations = [dict(step=i, quality=quality) for i in range(1, 25)]
        return dict(live=observations[-1], observations=observations, actions=[{"role": "g"}],
                    seconds=1., ema={"quality": 1.})
    monkeypatch.setattr(suite, "manifest", lambda *args: declared)
    monkeypatch.setattr(suite, "snapshot", lambda _: {"source_sha256": {"fixture.py": "fixed"}})
    monkeypatch.setattr(suite, "verify_source", lambda _: None)
    monkeypatch.setattr(suite, "run_episode", episode)
    report = suite.run(output, generations=1, population=4)
    frozen = json.loads((output / "frozen.json").read_text())
    assert frozen["manifest_sha256"] == digest(declared)
    assert len([c for c in calls if c[0] == "reserved"]) == 3
    before_ablation = [r for r in report["rows"] if r["phase"] != "post-freeze ablation"]
    assert frozen["development_results_sha256"] == digest(before_ablation)
    assert all(row["results"]["reserved"]["action_records"] == 1 for row in report["transfer"])
    assert all((output / r["results"]["reserved"]["artifact"]).is_file() for r in report["transfer"])

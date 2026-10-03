"""Keep the selected research bundle tied to its retained qualification evidence.

These checks read saved results; they do not rerun GPU training or qualify the
public GANTrainer as K3P.
"""
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/toy100"
GAP = REPORT / "gap-fill-20260925"
OLD = REPORT / "overnight-20260925"


def read_json(path):
    return json.loads(path.read_text())


def snapshot(directory, record):
    raw = gzip.decompress((directory / record["snapshot"]).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == record["artifact_sha256"]
    return json.loads(raw)


def test_selected_bundle_matches_executed_sources():
    selected = read_json(REPORT / "current-research-base.json")
    assert selected["candidate"] == "k3p"
    roles = (
        "config", "mechanism", "latent", "response", "probe", "native_probe",
        "hold_probe", "shift_probe", "frozen_shift_probe", "checkpoint",
        "convergence_gate",
    )
    manifest = read_json(GAP / "manifest.json")
    executed = {row["saved"]: row["sha256"] for row in manifest["sources"]}
    for role in roles:
        path = ROOT / selected[role]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == selected[role + "_sha256"]
        assert digest == executed[str(path.relative_to(GAP))]
    assert all(selected["requires_" + role] for role in ("mechanism", "latent", "response"))


def test_selected_22_gates_match_raw_results():
    selected = read_json(REPORT / "current-research-base.json")
    qualification = read_json(ROOT / selected["qualification"])["candidates"]["k3p"]
    gates = qualification["gates"]
    tasks = selected["first_gates"] + selected["remaining_regression_gates"]
    assert len(tasks) == len(set(tasks)) == len(gates) == 22
    assert {gate["task"] for gate in gates} == set(tasks)
    assert qualification["statuses"] == {"PASS": 22}
    assert selected["measured_gpu_passes"] == 22
    config = read_json(ROOT / selected["config"])
    for gate in gates:
        result = snapshot(GAP, gate)
        assert result["task"] == gate["task"]
        assert result["status"] == gate["status"] == "PASS"
        if gate["kind"] == "native":
            assert gate["seed"] == 1234
            assert result["steps"] == 7000
            assert result["coverage"]["status"] == result["accuracy"]["status"] == "PASS"
            assert result["driver_sha256"] == selected["native_probe_sha256"]
            for role in ("config", "mechanism", "latent", "response"):
                name = Path(selected[role]).name
                assert result["candidate_hashes"][name] == selected[role + "_sha256"]
        else:
            assert result["worker_sha256"] == selected["probe_sha256"]
            assert {k: v for k, v in result["config"].items() if k != "device"} == {
                k: v for k, v in config.items() if k != "device"
            }
            assert result["verdict"]["passed"]
            convergence = result["verdict"]["convergence"]
            assert convergence["complete"]
            assert convergence["passing_suffix"] >= convergence["minimum_stable_checks"]


def test_hold_extension_and_recovery_claims_match_raw_results():
    selected = read_json(REPORT / "current-research-base.json")
    evidence = read_json(OLD / "evidence.json")
    records = {r["kind"]: r for r in evidence["records"]
               if r["candidate"] == "k3p" and r["kind"].startswith("ring_")}
    hold = snapshot(OLD, records["ring_hold"])
    assert hold["gate"]["status"] == selected["ring_hold"]["status"] == "PASS"
    assert hold["gate"]["hold_checks"] == selected["ring_hold"]["checks"] == 1200
    assert hold["hold_window"]["min_hq"] == selected["ring_hold"]["min_hq"]
    end = hold["gate"]["converged_step"] + hold["gate"]["hold_budget"]
    extension = [row for row in hold["dense"] if row["step"] > end]
    assert len(extension) == selected["ring_extension"]["checks"] == 300
    assert all(row["modes"] == 8 and row["hq"] >= .9 for row in extension)
    assert min(row["hq"] for row in extension) == selected["ring_extension"]["min_hq"]
    shift = snapshot(OLD, records["ring_shift"])
    assert shift["status"] == selected["shift_recovery"]["status"] == "FAIL"
    recovery = shift["shift_recovery"]
    assert not recovery["deadline_pass"]
    assert recovery["deadline_window"]["passing_checks"] == selected["shift_recovery"]["deadline_passing_checks"] == 28
    assert recovery["deadline_window"]["checks"] == selected["shift_recovery"]["deadline_checks"] == 81
    assert recovery["delay_updates"] == selected["shift_recovery"]["sustained_recovery_delay_updates"] == 1130

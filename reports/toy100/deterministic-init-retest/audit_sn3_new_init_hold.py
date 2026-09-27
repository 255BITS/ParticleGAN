#!/usr/bin/env python3
"""Audit SN3's uninterrupted new-initialization long hold from retained files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
PREPARED = HERE / "research-screen-queue/prepared/sn3-2f595f84"
PROOF = HERE / "research-screen-queue/reviews/sn3-2f595f84/cpu-constructor-proof.json"
QUICK = HERE / "research-evidence/research-sn3-2f595f84-new-init/result.json.gz"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def good(row):
    return row["modes"] == 8 and row["hq"] >= .9


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads((PREPARED / "source-plan.json").read_text())
    seal = json.loads((PREPARED / "manifest.json").read_text())
    proof = json.loads(PROOF.read_text())
    result = json.loads((a.source / "result.json").read_text())
    init = json.loads((a.source / "initialization-receipt.json").read_text())
    quick = json.load(gzip.open(QUICK, "rt"))
    assert proof["status"] == "PASS" and init["cpu_proof_sha256"] == sha(PROOF)
    assert init["source_plan_sha256"] == sha(PREPARED / "source-plan.json")
    assert init["initializer_commit"] == plan["initializer_commit"]
    assert init["initial_material"] == proof["all_initial_material"]
    assert init["old_parameter_fixture_loaded"] is False
    assert result["worker_sha256"] == plan["candidate_files"]["hold.py"]
    assert result["initialization_fixture_sha256"] is None
    assert result["torch"] == plan["runtime_expected"]["torch"]
    assert result["backend"] == "cuda" and result["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert result["schedule"] == {"network_lr_horizon_cap": 1600, "network_lr_floor": 1.0,
                                   "prior_lr_floor": 1.0, "lr_anneal_start": 0.0, "frozen_budget": 1200}
    config = json.loads((PREPARED / "candidate-source/config.json").read_text())
    config.update(device="cuda:0", network_lr_floor=1.0, lr_floor=1.0, lr_anneal_start=0.0)
    assert result["config"] == config
    with zipfile.ZipFile(a.source / "prepared-source.zip") as archive:
        assert set(archive.namelist()) == set(seal["files"]) | {"manifest.json", "reviewed-cpu-proof.json"}
        for rel, digest in seal["files"].items():
            assert hashlib.sha256(archive.read(rel)).hexdigest() == digest == sha(PREPARED / rel), rel
        assert hashlib.sha256(archive.read("manifest.json")).hexdigest() == sha(PREPARED / "manifest.json")
        assert hashlib.sha256(archive.read("reviewed-cpu-proof.json")).hexdigest() == sha(PROOF)
    coarse = result["coarse"]
    reference = quick["result"]["observations"]
    assert [(r["step"], r["modes"], r["hq"]) for r in coarse] == [
        (r["step"], r["modes"], r["hq"]) for r in reference]
    dense = result["dense"]
    assert [r["step"] for r in dense] == list(range(1201, dense[-1]["step"] + 1))
    gate = result["gate"]
    assert gate["status"] == result["status"] == "POST_CONVERGENCE_FAIL"
    arrival = gate["converged_step"]
    failure = gate["first_hold_failure"]
    assert len([r for r in dense if arrival - 199 <= r["step"] <= arrival and good(r)]) == 200
    assert all(good(r) for r in dense if arrival < r["step"] < failure)
    assert not good(dense[failure - 1201])
    assert gate["hold_checks"] == failure - arrival
    assert result["good_hold_updates"] == failure - arrival - 1
    assert result["good_hold_updates"] < gate["hold_budget"] == 1200
    assert result["stopped"] == "gate_done_plus_post_window"
    assert dense[-1]["step"] == failure + 300
    post = [r for r in dense if r["step"] >= failure]
    assert result["post_failure_diagnostic"] == {
        "checks": len(post), "passing": sum(map(good, post)),
        "min_modes": min(r["modes"] for r in post),
        "min_hq": min(r["hq"] for r in post), "final": post[-1]}
    assert len(result["proof"]["optimizers"]) == 2
    assert all(row["calls"] == dense[-1]["step"] and row["device"] == "cuda:0"
               for row in result["proof"]["optimizers"].values())
    assert result["proof"]["adam_calls"] == 2 * dense[-1]["step"]
    assert len(result["randomness"]["sha256"]) == 64
    files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in a.source.iterdir() if p.is_file()}
    audit = {"status": "PASS", "quality_status": result["status"], "scope": "Original research host, uninterrupted long hold with reviewed merged initialization; not public API qualification",
             "source": str(a.source.resolve()), "artifacts": files,
             "runner_source_sha256": sha(HERE / "run_sn3_new_init_hold.py"),
             "audit_source_sha256": sha(Path(__file__)),
             "quick_prefix_matches": True, "initializer_commit": plan["initializer_commit"],
             "converged_step": arrival, "good_hold_updates": result["good_hold_updates"],
             "required_hold_updates": gate["hold_budget"], "first_hold_failure": failure,
             "failure_modes": dense[failure - 1201]["modes"],
             "failure_hq": dense[failure - 1201]["hq"],
             "post_failure_passing": result["post_failure_diagnostic"]["passing"],
             "post_failure_checks": result["post_failure_diagnostic"]["checks"],
             "final": post[-1], "seconds": result["seconds"],
             "run_forever_qualified": False}
    a.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("status", "quality_status", "converged_step", "good_hold_updates", "first_hold_failure", "failure_modes", "failure_hq")}))


if __name__ == "__main__":
    main()

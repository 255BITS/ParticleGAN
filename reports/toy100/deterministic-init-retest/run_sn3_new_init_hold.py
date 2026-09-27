#!/usr/bin/env python3
"""Run SN3's original uninterrupted long hold with reviewed new initialization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "research-screen-queue/prepared/sn3-2f595f84"
PROOF = HERE / "research-screen-queue/reviews/sn3-2f595f84/cpu-constructor-proof.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), "Use a fresh output directory"
    plan = json.loads((PREPARED / "source-plan.json").read_text())
    seal = json.loads((PREPARED / "manifest.json").read_text())
    proof = json.loads(PROOF.read_text())
    assert proof["status"] == "PASS"
    assert proof["source_plan_sha256"] == sha(PREPARED / "source-plan.json")
    assert proof["bridge_sha256"] == sha(PREPARED / "initialization_bridge.py")
    eligibility = json.loads((HERE / "research-eligibility-audits/sn3-2f595f84-current-configuration.json").read_text())
    assert sha(PREPARED / "candidate-source/hold.py") == eligibility["minimal_existing_followup"]["runner_sha256"]
    for rel, digest in seal["files"].items():
        assert sha(PREPARED / rel) == digest, rel
    source = PREPARED / "candidate-source"
    for rel, digest in plan["candidate_files"].items():
        assert sha(source / rel) == digest, rel
    runtime = Path(plan["historical_runtime"])
    for rel, digest in plan["historical_runtime_files"].items():
        assert sha(runtime / rel) == digest, rel
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    sys.path[:0] = [str(source), str(runtime), str(PREPARED)]
    import torch
    from initialization_bridge import bind_mode_hold, load_initializer
    from benchmarks.locked_shared import mode_hold
    assert str(torch.__version__) == plan["runtime_expected"]["torch"]
    assert torch.version.cuda == plan["runtime_expected"]["cuda"]
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == plan["runtime_expected"]["gpu"]
    assert sha(Path(mode_hold.__file__)) == plan["frozen_host_sha256"]
    public = load_initializer(PREPARED / "initializer-authority", plan["initializer_package"])
    captures = []

    def capture(generator, critic, prior, stream):
        def material(model):
            return {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
                    for key, value in model.state_dict().items()}
        actual = {role: material(model) for role, model in
                  (("generator", generator), ("critic", critic), ("prior", prior))}
        assert actual == proof["all_initial_material"], "Initialization differs from reviewed public constructor"
        assert all(str(v.device) == "cuda:0" for m in (generator, critic, prior) for v in m.parameters())
        assert str(stream.device) == "cuda:0"
        captures.append(actual)
        (a.output / "initialization-receipt.json").write_text(json.dumps({
            "initializer_commit": plan["initializer_commit"], "source_plan_sha256": sha(PREPARED / "source-plan.json"),
            "cpu_proof_sha256": sha(PROOF), "initial_material": actual,
            "old_parameter_fixture_loaded": False}, indent=2) + "\n")

    old_argv = sys.argv[:]
    sys.argv = [str(source / "hold.py"), "--repo", str(runtime), "--config", str(source / "config.json"),
                "--task", "mode_hold", "--backend", "cuda", "--output", str(a.output),
                "--network-floor", "1", "--prior-floor", "1", "--anneal-start", "0",
                "--steps", "7500", "--post-window", "300"]
    try:
        with bind_mode_hold(mode_hold, public, plan["train_mode_hold_sha256"], capture):
            with torch.autograd.set_multithreading_enabled(False):
                try:
                    runpy.run_path(str(source / "hold.py"), run_name="__main__")
                except SystemExit as result:
                    code = int(result.code or 0)
    finally:
        sys.argv = old_argv
        if a.output.exists():
            with zipfile.ZipFile(a.output / "prepared-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for rel in sorted(seal["files"]):
                    archive.write(PREPARED / rel, rel)
                archive.write(PREPARED / "manifest.json", "manifest.json")
                archive.write(PROOF, "reviewed-cpu-proof.json")
    assert len(captures) == 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()

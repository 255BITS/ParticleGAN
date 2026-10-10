"""Reclassify PR227's unchanged saved endpoints with the signed follow-up.

No training updates, rewritten receipt bindings, or original artifact writes.
The proposal must have the signed patch applied in an isolated source checkout.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("preserve the prior verification receipt and select a new output")
    root = args.source.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "examples"))
    from examples import e22_routed_convergence_neutral as neutral
    from examples import evaluate_e22_routed_convergence_neutral as recovery
    from examples import run_e22_routed_convergence as common
    import particlegan
    assert Path(particlegan.__file__).resolve().parent == root / "particlegan"
    base = neutral.baseline
    torch.set_num_threads(1)
    path = root / "tests/test_e22_routed_convergence_long.py"
    spec = importlib.util.spec_from_file_location("signed_routed_regression", path)
    regression = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regression)
    parent_receipt = json.loads((args.parent / "receipt.json").read_text())
    candidate_receipt = json.loads((args.candidate / "recovered-evaluation-receipt.json").read_text())
    failed = json.loads((args.candidate / "training-receipt-error.json").read_text())
    assert parent_receipt["status"] == "complete"
    assert candidate_receipt["status"] == "complete_recovered_evaluation"
    assert common.native_source_hash() == parent_receipt["bindings"]["native_source_hash"]
    assert common.native_source_hash() == failed["bindings"]["native_source_hash"]
    for name, expected in parent_receipt["bindings"]["source_hashes"].items():
        assert sha(root / name) == expected, name
    for name, expected in failed["bindings"]["source_hashes"].items():
        if name != "examples/e22_routed_convergence_neutral.py":
            assert sha(root / name) == expected, name
    archive = args.candidate / "training-source-504df385.py"
    proof = recovery.observational_source_repair(archive.read_text(), Path(neutral.__file__).read_text())
    artifacts = [args.parent / "receipt.json", args.candidate / "recovered-evaluation-receipt.json",
                 args.candidate / "training-receipt-error.json", archive,
                 args.candidate / "step-6400.pt"]
    artifacts.extend(args.parent / arm / f"step-{step:04d}.pt"
                     for arm in base.ARMS for step in ((800, 6400) if arm != "ordinary_mse_reference" else (6400,)))
    before = {str(path): sha(path) for path in artifacts}
    rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        scores = regression._assert_registered_game_regression(args.parent, args.candidate)
        original_game = regression._game
        harmful_controls = {}
        for name in scores:
            current = [None]
            original_load = regression._load

            def observe_load(path):
                current[0] = f"{path.parent.name}@{int(path.stem.split('-')[1])}"
                return original_load(path)

            def harmful_game(loop, judge, panels, *, ablate=False):
                if ablate and current[0] == name:
                    return original_game(loop, judge, panels, ablate=False) - .01
                return original_game(loop, judge, panels, ablate=ablate)

            regression._load, regression._game = observe_load, harmful_game
            try:
                regression._assert_registered_game_regression(args.parent, args.candidate)
            except AssertionError as error:
                arm, step, values = error.args[0]
                assert f"{arm}@{step}" == name and values["zero_code_minus_live"] < 0
                harmful_controls[name] = {"rejected": True,
                                           "synthetic_zero_code_minus_live": values["zero_code_minus_live"]}
            else:
                raise AssertionError("harmful particle use passed the signed regression: " + name)
            finally:
                regression._load, regression._game = original_load, original_game
    assert torch.equal(torch.get_rng_state(), rng)
    assert {str(path): sha(path) for path in artifacts} == before
    source_paths = ("examples/e22_routed_convergence_neutral.py",
                    "examples/evaluate_e22_routed_convergence_neutral.py",
                    "tests/test_e22_routed_convergence_long.py",
                    "tests/test_e22_routed_convergence_signed_gate.py",
                    "docs/e22_routed_convergence_signed_gate.md")
    receipt = {"format": "pr227_beneficial_signed_gate_reclassification_v2",
               "original_proposal_head": "b0e4f420856d7607baa98cbef488e569d20a3f31",
               "gate": "zero_code_minus_live > 1e-6 under all four mandatory judges",
               "saved_tensor_assertion": "PASS", "scores": scores,
               "harmful_direction_controls": harmful_controls,
               "synthetic_control_is_training_evidence": False,
               "source_proof": proof, "patched_source_sha256": {name: sha(root / name) for name in source_paths},
               "native_source_hash": common.native_source_hash(),
               "original_artifact_sha256": before, "original_artifacts_unchanged": True,
               "original_training_bindings_unchanged": True, "owned_state_and_RNG_unchanged": True,
               "global_RNG_isolation": "new loop constructors and scoring run inside fork_rng",
               "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu"},
               "new_training_updates": 0, "qualification_credit": "none"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"saved_tensor_assertion": "PASS", "harmful_controls_rejected": len(harmful_controls),
                      "new_training_updates": 0, "receipt": str(args.out)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

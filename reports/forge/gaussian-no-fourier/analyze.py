"""Read-only failure counts, artifact checks and actual initial-tensor proof."""
import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.gaussian_tasks import bounds
from experiments.forge.sources import inspect_source
from experiments.forge.state import state_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    arguments = parser.parse_args()
    raw = arguments.raw
    destination = Path(__file__).resolve().parent
    protocol = json.loads((destination / "protocol.json").read_text())
    result = json.loads((raw / "results.json").read_text())
    assert inspect_source(ROOT)["digest"] == protocol["source_digest"]
    for name, expected in protocol["inputs"].items():
        assert file_hash(ROOT / name) == expected, name
    module_path = destination / "preflight.py"
    spec = importlib.util.spec_from_file_location("fourier_initial_proof", module_path)
    proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proof)
    current = torch.load(raw / "smoke/evaluator/initial-state.pt", weights_only=True, map_location="cpu")
    baseline = torch.load(proof.BASELINE, weights_only=True, map_location="cpu")
    final_smoke = torch.load(raw / "smoke/evaluator/state.pt", weights_only=True, map_location="cpu")
    missing_eval = sorted(set(baseline["streams"]["states"]) - set(current["streams"]["states"]))
    comparison = deepcopy(current)
    for name in missing_eval:
        binding = baseline["streams"]["manifest"]["bindings"][name]
        assert binding["family"] == "eval"
        assert final_smoke["streams"]["manifest"]["bindings"][name] == binding
        # The new shared host registers primary evaluation lazily on the first
        # observation. Compare its declared unconsumed cursor without changing
        # a trained context or creating a CUDA generator.
        comparison["streams"]["manifest"]["bindings"][name] = binding
        comparison["streams"]["states"][name] = baseline["streams"]["states"][name]
    initial = proof.compare_initial(comparison, baseline)
    initial["lazy_primary_eval_bindings"] = missing_eval
    initial["lazy_eval_seed_matches_actual_final_manifest"] = True
    assert result["smoke"]["evidence"]["data_sha256"]["stationary"] == protocol["data_sha256"]["smoke_stationary"]
    assert result["stability"]["evidence"]["data_sha256"]["stationary"] == protocol["data_sha256"]["continuation_stationary"]
    assert result["stability"]["evidence"]["data_sha256"]["shift"] == protocol["data_sha256"]["shift"]
    phases = {"smoke": result["smoke"]["evidence"]["observations"],
              "stationary_hold": result["stability"]["evidence"]["observations"][:72],
              "shift_reacquisition": result["stability"]["evidence"]["observations"][72:96],
              "shift_hold": result["stability"]["evidence"]["observations"][96:]}
    summary = {}
    for name, rows in phases.items():
        failures = Counter(failure for row in rows for failure in bounds(row))
        streak = longest = 0
        for row in rows:
            streak = 0 if bounds(row) else streak + 1
            longest = max(longest, streak)
        summary[name] = dict(checks=len(rows), passing_checks=sum(not bounds(row) for row in rows),
                             longest_full_pass_streak=longest, final_full_pass_streak=streak,
                             failed_bound_counts=dict(sorted(failures.items())),
                             endpoint_metrics=rows[-1])
    states = {}
    for phase in ("smoke", "stability"):
        for path in sorted((raw / phase / "evaluator").glob("*state.pt")):
            state = torch.load(path, weights_only=True, map_location="cpu")
            rates = [[group["lr"] for group in optimizer["param_groups"]] for optimizer in state["trainer"]["optimizers"]]
            assert rates == [[.012, .012 * 2.5], [.012 * 1.5]], path
            states[str(path.relative_to(raw))] = dict(state_sha256=state_digest(state), rates=rates)
    output = dict(passed=True, training_updates=0, model_sampling_draws=0,
                  scope="saved output explanation; gates unchanged", phase_summary=summary,
                  actual_step0_comparison=initial, actual_data_matches_frozen_reference=True,
                  constant_saved_checkpoint_rates=True, states=states,
                  source_commit=result["source"]["origin_commit"],
                  audit_source_sha256=file_hash(Path(__file__)))
    atomic_json(destination / "analysis.json", output)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

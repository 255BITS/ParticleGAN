"""Compare completed CUDA diagnostic receipts using saved CPU tensors only."""
import argparse
import json
from pathlib import Path

import torch

from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.state import state_digest


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def gradients(a, b):
    return {name: {"bit_exact": exact(value, b[name]),
                   "max_absolute_difference": float((value.double() - b[name].double()).abs().max())}
            for name, value in a.items()}


def topology(nodes):
    return [{key: value for key, value in row.items() if key != "sequence_number"} for row in nodes]


def order(nodes):
    # AccumulateGrad has the maximum uint64 number. Remove absolute offsets
    # while retaining equal sequence numbers and ordering among graph nodes.
    values = sorted({row["sequence_number"] for row in nodes})
    ranks = {value: rank for rank, value in enumerate(values)}
    return [ranks[row["sequence_number"]] for row in nodes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    arms = ("fresh_graph", "fresh_serial401", "restored_graph", "restored_serial401")
    rows = {arm: torch.load(args.root / arm / "trace401.pt", weights_only=True, map_location="cpu") for arm in arms}
    states = {arm: torch.load(args.root / arm / "state401.pt", weights_only=True, map_location="cpu") for arm in arms}
    receipts = {arm: json.loads((args.root / arm / "receipt.json").read_text()) for arm in arms}
    assert all(r["prefix_bit_exact"] and r["completed_updates"] == 401 for r in receipts.values())
    assert sum(r["new_updates"] for r in receipts.values()) == 804
    historical = {arm: torch.load(args.prior_root / arm / "trace.pt", weights_only=True, map_location="cpu")[0]
                  for arm in ("live", "restart")}
    inputs = {str(p): file_hash(p) for arm in arms for p in
              (args.root / arm / "receipt.json", args.root / arm / "trace401.pt", args.root / arm / "state401.pt")}
    result = {"scope": "cuda_runtime_boundary_diagnostic_readout", "qualification_input": False,
              "inputs": inputs, "new_updates": 804, "receipts": receipts, "comparisons": {}}
    for name, left, right in (("ordinary_live_vs_restored", "fresh_graph", "restored_graph"),
                              ("serialized_live_vs_restored", "fresh_serial401", "restored_serial401"),
                              ("live_serial_effect", "fresh_graph", "fresh_serial401"),
                              ("restored_serial_effect", "restored_graph", "restored_serial401")):
        a, b = rows[left], rows[right]
        result["comparisons"][name] = {
            "real_batch_bit_exact": exact(a["real"], b["real"]),
            "first_six_forward_outputs_bit_exact": [exact(x["output"], y["output"])
                                                   for x, y in zip(a["forwards"][:6], b["forwards"][:6])],
            "critic_gradient": gradients(a["optimizer_gradients"]["D"], b["optimizer_gradients"]["D"]),
            "critic_graph_topology_exact": topology(a["backward_graphs"][0]["nodes"]) == topology(b["backward_graphs"][0]["nodes"]),
            "critic_graph_relative_sequence_order_exact": order(a["backward_graphs"][0]["nodes"]) == order(b["backward_graphs"][0]["nodes"]),
            "all_named_stream_states_exact": state_digest(states[left]["streams"]) == state_digest(states[right]["streams"]),
            "whole_context401_exact": state_digest(states[left]) == state_digest(states[right])}
    result["instrumentation_cohort_parity"] = {
        arm: gradients(rows[arm]["optimizer_gradients"]["D"],
                       {"D." + key: value for key, value in historical[old]["optimizers"]["D"]["gradients"].items()})
        for arm, old in (("fresh_graph", "live"), ("restored_graph", "restart"))}
    atomic_json(args.output, result)
    print(json.dumps({"event": "boundary_readout_complete", "output": str(args.output), "new_analysis_model_calls": 0}), flush=True)


if __name__ == "__main__":
    main()

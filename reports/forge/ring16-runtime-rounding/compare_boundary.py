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


def sequence_relation_changes(a, b):
    """Count changed pairwise priorities, including changed equality ties."""
    def sign(value):
        return (value > 0) - (value < 0)
    changed = []
    for i in range(len(a)):
        for j in range(i):
            if sign(a[i]["sequence_number"] - a[j]["sequence_number"]) != sign(
                    b[i]["sequence_number"] - b[j]["sequence_number"]):
                changed.append({"node_ids": [i, j], "types": [a[i]["type"], a[j]["type"]],
                                "left_sequence": [a[i]["sequence_number"], a[j]["sequence_number"]],
                                "right_sequence": [b[i]["sequence_number"], b[j]["sequence_number"]]})
    return {"compared_node_pairs": len(a) * (len(a) - 1) // 2,
            "changed_relative_relations": len(changed), "examples": changed[:6]}


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
    assert all(len(trace["forwards"]) == 8 and exact(trace["polar"][1]["input"],
               trace["optimizer_gradients"]["D"]["D.net.2.weight"]) for trace in rows.values())
    historical = {arm: torch.load(args.prior_root / arm / "trace.pt", weights_only=True, map_location="cpu")[0]
                  for arm in ("live", "restart")}
    inputs = {str(p): file_hash(p) for arm in arms for p in
              (args.root / arm / "receipt.json", args.root / arm / "trace401.pt", args.root / arm / "state401.pt")}
    result = {"scope": "cuda_runtime_boundary_diagnostic_readout", "qualification_input": False,
              "inputs": inputs, "new_updates": 804, "receipts": receipts, "comparisons": {}}
    result["analysis_model_calls"] = 0
    result["state401_digests"] = {arm: state_digest(state) for arm, state in states.items()}
    for name, left, right in (("ordinary_live_vs_restored", "fresh_graph", "restored_graph"),
                              ("serialized_live_vs_restored", "fresh_serial401", "restored_serial401"),
                              ("live_serial_effect", "fresh_graph", "fresh_serial401"),
                              ("restored_serial_effect", "restored_graph", "restored_serial401")):
        a, b = rows[left], rows[right]
        result["comparisons"][name] = {
            "real_batch_bit_exact": exact(a["real"], b["real"]),
            "first_six_forward_outputs_bit_exact": [exact(x["output"], y["output"])
                                                   for x, y in zip(a["forwards"][:6], b["forwards"][:6])],
            "first_six_forward_inputs_bit_exact": [exact(x["input"], y["input"])
                                                  for x, y in zip(a["forwards"][:6], b["forwards"][:6])],
            "critic_gradient": gradients(a["optimizer_gradients"]["D"], b["optimizer_gradients"]["D"]),
            "critic_graph_topology_exact": topology(a["backward_graphs"][0]["nodes"]) == topology(b["backward_graphs"][0]["nodes"]),
            "critic_graph_relative_sequence_order_exact": order(a["backward_graphs"][0]["nodes"]) == order(b["backward_graphs"][0]["nodes"]),
            "critic_graph_sequence_relations": sequence_relation_changes(
                a["backward_graphs"][0]["nodes"], b["backward_graphs"][0]["nodes"]),
            "hidden_critic_polar": {
                "bit_exact": exact(a["polar"][1]["output"], b["polar"][1]["output"]),
                "max_absolute_difference": float((a["polar"][1]["output"].double() -
                                                   b["polar"][1]["output"].double()).abs().max()),
                "relative_frobenius_difference": float((a["polar"][1]["output"].double() -
                                                          b["polar"][1]["output"].double()).norm() /
                                                         a["polar"][1]["output"].double().norm())},
            "all_named_stream_states_exact": state_digest(states[left]["streams"]) == state_digest(states[right]["streams"]),
            "whole_context401_exact": state_digest(states[left]) == state_digest(states[right])}
    result["critic_graph_summary"] = {}
    for arm, trace in rows.items():
        nodes = trace["backward_graphs"][0]["nodes"]
        sequences = [node["sequence_number"] for node in nodes if node["sequence_number"] != 2 ** 64 - 1]
        result["critic_graph_summary"][arm] = {
            "nodes": len(nodes), "accumulator_nodes": len(nodes) - len(sequences),
            "minimum_nonaccumulator_sequence": min(sequences),
            "maximum_nonaccumulator_sequence": max(sequences),
            "nonaccumulator_nodes_below_20000": sum(x < 20000 for x in sequences),
            "nonaccumulator_nodes_above_50000": sum(x > 50000 for x in sequences),
            "autograd_multithreading_enabled": trace["backward_graphs"][0]["autograd_multithreading_enabled"]}
    result["instrumentation_cohort_parity"] = {
        arm: gradients(rows[arm]["optimizer_gradients"]["D"],
                       {"D." + key: value for key, value in historical[old]["optimizers"]["D"]["gradients"].items()})
        for arm, old in (("fresh_graph", "live"), ("restored_graph", "restart"))}
    atomic_json(args.output, result)
    print(json.dumps({"event": "boundary_readout_complete", "output": str(args.output), "new_analysis_model_calls": 0}), flush=True)


if __name__ == "__main__":
    main()

"""Read-only fixed-readout feasibility at existing cold and passing states."""

import argparse
import gzip
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from reports.toy100.anchor_drift_assessment import fixed_readout_chart, sha, verify_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-state", type=Path, required=True)
    parser.add_argument("--prefix-receipt", type=Path, required=True)
    parser.add_argument("--original-states", type=Path, required=True)
    parser.add_argument("--original-states-sha256", required=True)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prefix_receipt_raw = args.prefix_receipt.read_bytes()
    prefix_receipt = json.loads(prefix_receipt_raw)
    prefix_raw = gzip.decompress(args.prefix_state.read_bytes())
    if sha(prefix_raw) != prefix_receipt["state_file_sha256"]:
        raise RuntimeError("cold prefix snapshot differs from its archived receipt")
    prefix = torch.load(io.BytesIO(prefix_raw), weights_only=True, map_location="cpu")
    original_raw = verify_bytes(args.original_states, args.original_states_sha256)
    original = torch.load(io.BytesIO(original_raw), weights_only=True, map_location="cpu")
    rows = {
        "cold1_pre_step": fixed_readout_chart(prefix["selected"][1]["pre_step"]),
        "cold100_post_bounded_g": fixed_readout_chart(prefix["selected"][100]["post_bounded_g"]),
        "original_passing1324_pre_step": fixed_readout_chart(original[1324]["pre_step"]),
    }
    inputs = dict(cold_prefix_state_sha256=sha(prefix_raw),
        cold_prefix_receipt_sha256=sha(prefix_receipt_raw), original_states_sha256=sha(original_raw))
    for directory in args.run:
        raw = (directory/"forks/candidate.json").read_bytes()
        receipt = json.loads(raw)["dynamics_receipt"]
        state_raw = verify_bytes(directory/receipt["final_state_file"], receipt["final_state_file_sha256"])
        state = torch.load(io.BytesIO(state_raw), weights_only=True, map_location="cpu")
        row = fixed_readout_chart(state)
        if row["state_sha256"] != receipt["final_snapshot_sha256"]:
            raise RuntimeError("completed snapshot contents differ from the declared full state")
        rows[directory.name] = row
        inputs[directory.name] = dict(candidate_receipt_sha256=sha(raw), state_file_sha256=sha(state_raw))
    source = {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in (
        Path(__file__), ROOT/"reports/toy100/anchor_drift_assessment.py",
        ROOT/"benchmarks/locked_shared/mlp.py", ROOT/"benchmarks/locked_shared/mode_hold.py",
        ROOT/"reports/toy100/pr84_critic_refinement_capture.py")}
    result = dict(scope="read-only fixed 12-code last-layer chart; no candidate or training", shared_gate_eligible=False,
        source=source, inputs=inputs, rows=rows,
        limitations=["features and latent codes are frozen in this algebraic chart",
                     "full row rank says nothing about acquisition by the current joint optimizer",
                     "large inverse norm can require large readout motion",
                     "this chart does not bound the critic or prove infinite-time sampled-group correctness"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({name: {key: row[key] for key in (
        "rank", "condition", "right_inverse_operator_norm", "reconstruction_max_abs_error")}
        for name, row in rows.items()}), flush=True)


if __name__ == "__main__":
    main()

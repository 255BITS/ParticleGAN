"""Reproduce the local scale-search timing and invariant-operation counts."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--physical-uuid")
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    if args.device != "cpu" and (not args.physical_uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != args.physical_uuid):
        parser.error("CUDA requires the coordinated physical UUID in CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(1)
    if args.device != "cpu":
        assert torch.cuda.device_count() == 1
        torch.cuda.set_per_process_memory_fraction((2048 * 1024 ** 2) / torch.cuda.get_device_properties(0).total_memory)
    checkout = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(checkout))
    spec = importlib.util.spec_from_file_location("scale_reference_checks", checkout / "tests/test_row_evidence_scale.py")
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    results = []
    for n in (5, 25, 128):
        t2 = torch.logspace(-2, 5, n, dtype=torch.float64, device=args.device)
        n_eff = torch.linspace(6, 99, n, dtype=torch.float64, device=args.device)
        ok = torch.ones(n, dtype=torch.bool, device=args.device)
        evidence = {case: cls(torch.zeros(n, 2, device=args.device), null="scaled")
                    for case, cls in (("baseline", checks.OriginalScale), ("optimized", checks.RowEvidence))}
        checks.same_state(evidence["baseline"]._scale(t2, n_eff, ok), evidence["optimized"]._scale(t2, n_eff, ok))
        counts = {}
        for case, owner in evidence.items():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                owner._scale(t2, n_eff, ok)
            counts[case] = {event.key: event.count for event in profile.key_averages()
                            if event.key in ("aten::sub", "aten::neg", "aten::div", "aten::clamp_min")}
        windows = []
        for case in ("baseline", "optimized", "optimized", "baseline"):
            owner = evidence[case]
            for _ in range(8):
                owner._scale(t2, n_eff, ok)
            started = time.perf_counter()
            for _ in range(args.iterations):
                owner._scale(t2, n_eff, ok)
            # _scale's exported host scalar already synchronizes each call.
            elapsed = time.perf_counter() - started
            windows.append({"case": case, "calls": args.iterations, "us_per_call": elapsed * 1e6 / args.iterations})
        checks.same_state(evidence["baseline"].state_dict(), evidence["optimized"].state_dict())
        means = {case: statistics.mean(w["us_per_call"] for w in windows if w["case"] == case) for case in evidence}
        results.append({"rows": n, "counts": counts, "windows": windows, "mean_us": means,
                        "observed_time_reduction_percent": 100 * (1 - means["optimized"] / means["baseline"])})
    result = {"device": args.device, "physical_uuid": args.physical_uuid,
              "torch": torch.__version__, "rows": results,
              "scope": "local operation timing; shared-GPU load does not establish clean whole-training throughput"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

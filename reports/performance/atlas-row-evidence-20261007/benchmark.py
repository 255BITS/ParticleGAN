"""Time an implementation delta on the immutable matched-noise Atlas host.

No task/recipe/source binding is rewritten. The archived adapter constructs
each fixture, then only RowEvidence._scale is transplanted from this checkout.
Bulk profiler output and final checkpoints belong outside Git.
"""
from __future__ import annotations

import argparse
import ast
import cProfile
from copy import deepcopy
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch


def pin(path):
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return deepcopy(value)


def differences(left, right, path=""):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            return [path + ":type/shape"]
        # Compare storage bytes, including signed zero and NaN payloads.
        a, b = [value.contiguous().reshape(-1).view(torch.uint8) for value in (left, right)]
        return [] if torch.equal(a, b) else [path + ":tensor bytes"]
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return [path + ":keys"]
        return [item for key in left for item in differences(left[key], right[key], path + "/" + str(key))]
    if isinstance(left, (tuple, list)):
        if type(left) != type(right) or len(left) != len(right):
            return [path + ":sequence"]
        return [item for index, (a, b) in enumerate(zip(left, right))
                for item in differences(a, b, path + "/" + str(index))]
    if isinstance(left, float) and math.isnan(left):
        return [] if isinstance(right, float) and math.isnan(right) else [path]
    return [] if left == right else [path]


def gpu_view():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                             "--format=csv"], text=True, capture_output=True, timeout=5)
    return {"exit_code": result.returncode, "rows": result.stdout.strip().splitlines()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--stability-steps", type=int, default=0)
    parser.add_argument("--physical-uuid")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.windows < 1 or args.warmup < 0 or args.windows + args.warmup > 20001:
        parser.error("timing updates must fit the original 20001-update execution budget")
    if not 0 <= args.stability_steps <= 20001:
        parser.error("stability updates must fit the original execution budget")
    if args.device != "cpu" and (not args.physical_uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != args.physical_uuid):
        parser.error("CUDA requires the coordinated physical UUID in CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device != "cpu":
        assert torch.cuda.device_count() == 1
        torch.cuda.set_per_process_memory_fraction((2048 * 1024 ** 2) / torch.cuda.get_device_properties(0).total_memory)
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.owner_root.resolve()))
    spec = importlib.util.spec_from_file_location("matched_atlas_timing_adapter", args.adapter)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    binding = json.loads(args.binding.read_text())
    loaded = adapter.load_adapter(args.owner_root, binding)
    from particlegan.row_evidence import RowEvidence
    import particlegan.row_evidence as row_module
    baseline = RowEvidence._scale
    checkout = Path(__file__).resolve().parents[3]
    optimized_path = checkout / "particlegan/row_evidence.py"
    tree = ast.parse(optimized_path.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RowEvidence")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_scale")
    namespace = dict(vars(row_module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(optimized_path), "exec"), namespace)
    optimized = namespace["_scale"]
    result = {"scope": "implementation timing and correctness only; no scientific qualification",
              "owner_root": str(args.owner_root), "adapter": pin(args.adapter), "binding": pin(args.binding),
              "baseline_module": pin(row_module.__file__), "optimized_module": pin(optimized_path),
              "python": sys.version, "torch": torch.__version__, "device": args.device,
              "physical_uuid": args.physical_uuid, "cpu_threads": 1,
              "schedule_contract": "original full recipe and 20001 execution limit retained in every fixture",
              "windows": [], "equivalence": {}, "gpu_before": gpu_view() if args.device != "cpu" else None}
    global_cpu = torch.get_rng_state().clone()
    global_cuda = torch.cuda.get_rng_state().clone() if args.device != "cpu" else None

    def sync():
        if args.device != "cpu":
            torch.cuda.synchronize()

    def fixture(case):
        torch.set_rng_state(global_cpu)
        if global_cuda is not None:
            torch.cuda.set_rng_state(global_cuda)
        RowEvidence._scale = baseline
        ctx = loaded["word_context"]({"candidate": binding["candidate"], "protocol": {"seed": 0}},
                                     binding["task"], args.device, root=args.owner_root)
        f = loaded["WordFixture"](device=args.device, seed=0, recipe_name=None, max_steps=20001, components=ctx)
        assert f.policy.row_evidence is not None and f.policy.birth_death is not None
        RowEvidence._scale = baseline if case == "baseline" else optimized
        return f, ctx

    def snapshot(f, ctx, outputs):
        return cpu_tree({"fixture": f.state_dict(), "streams": ctx.streams.state_dict(), "outputs": outputs,
                         "gradients": {role: {name: param.grad for name, param in module.named_parameters()}
                                       for role, module in (("G", f.G), ("E", f.E), ("D", f.D), ("prior", f.prior))}})

    snapshots = {}
    for index, case in enumerate(("baseline", "optimized", "optimized", "baseline")):
        f, ctx = fixture(case)
        for _ in range(args.warmup):
            f.step()
        sync()
        before = gpu_view() if args.device != "cpu" else None
        if args.device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        outputs = []
        profile = cProfile.Profile() if args.profile else None
        if profile:
            profile.enable()
        started = time.perf_counter()
        checkpoints = {}
        for count in range(1, args.windows + 1):
            outputs.append(f.step())
            if count in (32, 64, 128, 256, args.windows):
                sync()
                checkpoints[str(count)] = time.perf_counter() - started
        sync()
        elapsed = time.perf_counter() - started
        if profile:
            profile.disable()
            profile.dump_stats(str(args.output / f"{index}-{case}.pstats"))
        row = {"index": index, "case": case, "warmup": args.warmup, "updates": args.windows,
               "seconds": elapsed, "ms_per_update": elapsed * 1000 / args.windows,
               "prefix_seconds": checkpoints, "gpu_before": before,
               "gpu_after": gpu_view() if args.device != "cpu" else None,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device != "cpu" else None,
               "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device != "cpu" else None}
        snapshots[(index, case)] = snapshot(f, ctx, outputs)
        result["windows"].append(row)
        print(json.dumps(row), flush=True)
        (args.output / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
        del f, ctx, outputs
        gc.collect()
        if args.device != "cpu":
            torch.cuda.empty_cache()
    control = snapshots[(0, "baseline")]
    for key, state in snapshots.items():
        delta = differences(control, state)
        result["equivalence"][str(key)] = {"tensor_bytes_and_values_equal": not delta, "differences": delta[:20]}
        assert not delta, delta[:20]
    del snapshots, control
    if args.stability_steps:
        final = {}
        for case in ("baseline", "optimized"):
            f, ctx = fixture(case)
            outputs = []
            started = time.perf_counter()
            for step in range(1, args.stability_steps + 1):
                outputs.append(f.step())
                if step % 200 == 0:
                    print(json.dumps({"stability": case, "step": step, "elapsed": time.perf_counter()-started}), flush=True)
            sync()
            final[case] = snapshot(f, ctx, outputs)
            torch.save(final[case], args.output / f"{case}-stability.pt")
            del f, ctx, outputs
            gc.collect()
            if args.device != "cpu":
                torch.cuda.empty_cache()
        delta = differences(final["baseline"], final["optimized"])
        result["stability"] = {"updates_per_case": args.stability_steps, "tensor_bytes_and_values_equal": not delta,
                               "differences": delta[:20]}
        assert not delta, delta[:20]
    means = {case: statistics.mean(row["ms_per_update"] for row in result["windows"] if row["case"] == case)
             for case in ("baseline", "optimized")}
    result["summary"] = {**means, "observed_time_reduction_percent": 100 * (1 - means["optimized"] / means["baseline"])}
    result["gpu_after"] = gpu_view() if args.device != "cpu" else None
    result["contention_limit"] = "Shared device; external load snapshots do not establish exclusive or stationary GPU throughput."
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    RowEvidence._scale = baseline
    print(json.dumps({"summary": result["summary"], "equivalence": result["equivalence"], "stability": result.get("stability")}), flush=True)


if __name__ == "__main__":
    main()

"""Verify and summarize the saved CUDA restart diagnostic, without execution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from experiments.forge.state import state_digest


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_tensors(a, b, prefix=""):
    differences = []
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            differences.append({"path": prefix, "shape": list(a.shape),
                "max_absolute_difference": float((a - b).abs().max())})
    elif isinstance(a, dict):
        if set(a) != set(b):
            raise ValueError("Different tree keys: " + prefix)
        for k in a:
            differences += compare_tensors(a[k], b[k], prefix + "/" + str(k))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("Different tree lengths: " + prefix)
        for i, (x, y) in enumerate(zip(a, b)):
            differences += compare_tensors(x, y, prefix + "/" + str(i))
    return differences


def equal_without_cap(a, b):
    a, b = dict(a), dict(b)
    a["trainer"], b["trainer"] = dict(a["trainer"]), dict(b["trainer"])
    a["trainer"].pop("max_steps", None)
    b["trainer"].pop("max_steps", None)
    return state_digest(a) == state_digest(b)


def summarize(raw, prior_root, inventory_root):
    inputs = {}

    def recorded(path):
        inputs[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return path

    def load(path):
        return torch.load(recorded(path), map_location="cpu", weights_only=True)

    def json_input(path):
        return read(recorded(path))

    parent = prior_root / "runs/api/tier1-prior-smoke-v1/mog100-n256/ring16_acquisition"
    historic = prior_root / "runs/api/tier1-prior-duration-v1/mog100-n256-ring16_acquisition"
    old_parent = load(parent / "state.pt")
    prefix = load(raw / "live/prefix-state.pt")
    if state_digest(old_parent) != state_digest(prefix):
        raise ValueError("Fresh prefix differs from the archived full400 state")
    interrupted = json_input(raw / "live/interruption-receipt.json")
    receipts = {p.parent.name: json_input(p) for p in raw.glob("*/receipt.json")}
    if set(receipts) != {"restart", "archive-replay", "restored_untraced", "restored_stats", "restored_twice", "restored_runtime"}:
        raise ValueError("Incomplete declared attempt set")
    for arm, receipt in receipts.items():
        for name, expected in receipt["artifacts"].items():
            path = raw / arm / name
            if sha(path) != expected:
                raise ValueError("Changed raw artifact: " + str(path))
    for name, expected in interrupted["artifacts"].items():
        if sha(raw / "live" / name) != expected:
            raise ValueError("Changed interrupted original artifact: " + name)
    old = [p for p in load(historic / "observations.pt") if p["step"] > 0]
    source_manifest = json_input(raw / "restart/source.json")
    protocol = json_input(raw / "frozen-protocol.json")
    full = {}
    for arm in ("restart", "archive-replay"):
        saved = load(raw / arm / "observations.pt")
        equal = [p["step"] for p, q in zip(old, saved) if torch.equal(p["samples"], q["samples"])]
        if len(saved) != 96 or len(equal) != 96:
            raise ValueError("Historical passing output reproduction failed")
        if receipts[arm]["full_verdict"] != "PASS" or receipts[arm]["terminal_suffix"] != 6:
            raise ValueError("Historical grade reproduction failed")
        full[arm] = {"full_verdict": "PASS", "terminal_suffix": 6, "exact_historical_observations": 96,
                     "new_updates": 1200, "elapsed_seconds": receipts[arm]["elapsed_seconds"],
                     "final_metrics": receipts[arm]["final_metrics"]}
    logs = json_input(ROOT / "reports/forge/ring16-failure/restart-interruption.json")
    if logs["stdout"]["sha256"] != sha(ROOT / logs["stdout"]["path"]):
        raise ValueError("Live original stdout changed")
    recorded(ROOT / logs["stdout"]["path"])
    observations = []
    for line in (ROOT / logs["stdout"]["path"]).read_text().splitlines():
        try:
            point = json.loads(line)
        except json.JSONDecodeError:
            continue
        if point.get("event") == "observation" and point.get("arm") == "live":
            observations.append(point)
    previous = json_input(inventory_root / "reports/forge/attempts/b2d55fb3d5434002aa2b443522a83b79/result.json")["task_results"][0]
    curve = previous["evidence"]["observations"]
    if len(observations) != 96 or any(p["step"] != q["step"] or p["covariance"] != q["component_covariance_error"]
          or p["hq"] != q["hq"] for p, q in zip(observations, curve)):
        raise ValueError("Live printed metric reproduction differs")
    full["live"] = {"execution_status": interrupted["status"], "original_receipt_status": "INCOMPLETE",
        "printed_full_quality_verdict": "FAIL", "passing_observations": 0, "terminal_suffix": 0,
        "new_updates": 1600, "matched_prior_covariance_and_hq_observations": 96,
        "final_printed_metrics": observations[-1], "conservative_charged_seconds": 300,
        "limitations": interrupted["missing"]}
    live, restart = load(raw / "live/trace.pt"), load(raw / "restart/trace.pt")
    if [p["step"] for p in live] != list(range(401, 417)) or [p["step"] for p in restart] != list(range(401, 417)):
        raise ValueError("Incomplete boundary traces")
    before = load(raw / "restart/before-state.pt")
    if state_digest(prefix) != state_digest(before):
        raise ValueError("Restored before-state differs")
    live_runtime = load(raw / "live/prefix-runtime.pt")
    restored_runtime = load(raw / "restart/before-runtime.pt")
    runtime_boundary = {
        "live_gradient_buffer_counts": {role: sum(x is not None for x in values.values())
                                        for role, values in live_runtime["gradients"].items()},
        "restored_gradient_buffer_counts": {role: sum(x is not None for x in values.values())
                                            for role, values in restored_runtime["gradients"].items()},
        "module_mode_changes": {role: {name: {"live": value, "restored": restored_runtime["module_modes"][role][name]}
             for name, value in modes.items() if value != restored_runtime["module_modes"][role][name]}
             for role, modes in live_runtime["module_modes"].items()},
        "parameter_version_counters": {role: {name: {"live": value["version"],
             "restored": restored_runtime["parameter_layout"][role][name]["version"]}
             for name, value in layouts.items()} for role, layouts in live_runtime["parameter_layout"].items()},
        "prior_noise_enabled_equal": live_runtime["prior_noise_enabled"] == restored_runtime["prior_noise_enabled"],
        "penalty_collect_stats_equal": live_runtime["penalty_collect_stats"] == restored_runtime["penalty_collect_stats"]}
    states416 = {arm: load(raw / arm / ("state416.pt" if arm in ("live", "restart") else "state.pt"))
                 for arm in ("live", "restart", "restored_untraced", "restored_stats", "restored_twice", "restored_runtime")}
    controls = {}
    restart_final = load(raw / "restart/state.pt")
    if state_digest(restart_final) != state_digest(load(raw / "archive-replay/state.pt")):
        raise ValueError("The two full restored final contexts differ")
    for arm in ("restored_untraced", "restored_stats", "restored_twice", "restored_runtime"):
        equal = equal_without_cap(states416["restart"], states416[arm])
        if not equal:
            raise ValueError("Short restore control did not agree: " + arm)
        controls[arm] = {"completed_updates": 416, "new_updates": 16,
                         "exact_restored_state_except_external_cap": True,
                         "elapsed_seconds": receipts[arm]["elapsed_seconds"]}
    first = []
    for a, b in zip(live, restart):
        batch_equal = torch.equal(a["real"], b["real"])
        indices_equal = all(torch.equal(x["indices"], y["indices"]) for x, y in zip(a["latents"], b["latents"]))
        stream_equal = state_digest(a["named_streams"]) == state_digest(b["named_streams"])
        if not batch_equal or not indices_equal or not stream_equal or not torch.equal(
                a["global_cpu_rng"], b["global_cpu_rng"]) or not torch.equal(a["global_cuda_rng"], b["global_cuda_rng"]):
            raise ValueError("Matched input/stream contract changed")
        tensors = compare_tensors(a, b)
        first.append({"step": a["step"], "real_equal": True, "sampled_indices_equal": True,
            "all_named_streams_equal": True,
            "ambient_cpu_rng_equal": torch.equal(a["global_cpu_rng"], b["global_cpu_rng"]),
            "ambient_cuda_rng_equal": torch.equal(a["global_cuda_rng"], b["global_cuda_rng"]),
            "differing_tensor_count": len(tensors), "first_differing_tensor": tensors[0] if tensors else None})
    a, b = live[0], restart[0]
    forward = [{"call": i, "role": x["role"], "input_equal": torch.equal(x["input"], y["input"]),
                "output_equal": torch.equal(x["output"], y["output"]),
                "output_max_difference": float((x["output"] - y["output"]).abs().max())}
               for i, (x, y) in enumerate(zip(a["forwards"], b["forwards"]))]
    polar = [{"index": i, "shape": list(x["input"].shape),
              "input_max_difference": float((x["input"] - y["input"]).abs().max()),
              "factor_max_difference": float((x["output"] - y["output"]).abs().max()),
              "input_layout_equal": x["input_layout"] == y["input_layout"]}
             for i, (x, y) in enumerate(zip(a["polar"], b["polar"]))]
    gradients = []
    for k, x in a["optimizers"]["D"]["gradients"].items():
        y = b["optimizers"]["D"]["gradients"][k]
        gradients.append({"parameter": k, "exact": torch.equal(x, y), "max_difference": float((x-y).abs().max())})
    params_before_equal = state_digest(a["optimizers"]["D"]["before"]) == state_digest(b["optimizers"]["D"]["before"])
    if not params_before_equal or not all(p["output_equal"] for p in forward[:6]):
        raise ValueError("Divergence preceded the expected critic backward")
    if not all(torch.equal(x["values"], y["values"]) for x, y in zip(a["latents"], b["latents"])):
        raise ValueError("First-update latent codes differ")
    svd = json_input(raw / "svd-probe.json")
    updates = interrupted["new_updates"] + sum(r["new_updates"] for r in receipts.values())
    charged = 300 + sum(r["elapsed_seconds"] for r in receipts.values())
    if updates != 4064 or charged > protocol["max_reserved_seconds"]:
        raise ValueError("Budget/debit mismatch")
    result = {"schema_version": 1, "scope": "cuda_checkpoint_boundary_reproduction", "qualification_input": False,
        "qualification_reuse": False, "seed": 0, "source_commits": sorted({interrupted["source_commit"], source_manifest["origin_commit"]}),
        "parent400_full_state_exact": True, "parent400_state_sha256": state_digest(prefix),
        "restored_full_final_contexts_exact": True,
        "runtime_boundary": runtime_boundary,
        "restored_before400_full_state_exact": True, "first_differing_update": 401,
        "first_differing_stage": "critic backward weight gradients; forward outputs and all consumed random inputs agree",
        "first_update": {"critic_before_parameters_equal": params_before_equal, "both_latent_batches_equal": True,
                         "forward_calls": forward, "critic_gradients": gradients, "polar_updates": polar,
                         "critic_hidden_weight_max_difference": float((a["optimizers"]["D"]["after"]["net.2.weight"] -
                              b["optimizers"]["D"]["after"]["net.2.weight"]).abs().max())},
        "trace_summary": first, "full_runs": full, "controls": controls, "svd": svd,
        "cost": {"new_host_updates": updates, "attempts": 7, "conservative_charged_seconds": charged,
                 "reserved_ceiling_seconds": protocol["max_reserved_seconds"], "scientific_retries": 0,
                 "note": "Live1600 lost exact elapsed receipt after bookkeeping error; charge its full300-second reservation. Remaining six arms retain measured elapsed time. Saved-gradient SVD analysis has zero training updates."},
        "inputs": inputs, "limitations": ["No seed variation was tested; seed sensitivity is not established.",
            "The exact CUDA/autograd reduction or accumulation mechanism creating the first rounding difference is not isolated.",
            "One reproduced restart PASS does not establish periodic restarts as a general training technique.",
            "The live final model and post400 full scored tensors were lost at the postexecution receipt error; its preserved metrics and16-update boundary trace are used without qualification credit."]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "runs/api/ring16-restart-diagnostic-v1")
    parser.add_argument("--prior-root", type=Path, default=Path("/home/martyn/dev/ParticleGAN-tier1-prior-smoke"))
    parser.add_argument("--inventory-root", type=Path, default=Path("/home/martyn/dev/ParticleGAN-gaussian-smoke-inventory"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("reproduction-results.json"))
    args = parser.parse_args()
    result = summarize(args.raw.resolve(), args.prior_root.resolve(), args.inventory_root.resolve())
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"event": "restart_analysis_complete", "first_difference": 401,
                      "new_updates": result["cost"]["new_host_updates"], "charged_seconds": result["cost"]["conservative_charged_seconds"]}))

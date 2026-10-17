"""Pinned original-device endpoint diagnostic; preflight imports no ML package.

No training/update path is called. Run mode is reserved for the parent-owned,
idle physical GPU1 lane after explicit source/protocol review.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
SCHEMA = "pg_grid100_public_output_kernel_endpoint_v2"
TASK = "grid100_policy_selected_cloud_v1"
COMMIT = "9563dea57bb150f2a0275bbe8d785bf76210fca3"
DIGEST = "db5492df4aa5ef60ce9e7d5869b2b6be8d492f19191a2889967274555f8d0037"
CHECKPOINT_SHA = "963fbf0e0789c4289559c95f252a4838d08e348889bc46dd201f65312f26dbbc"
STATE_SHA = "895e5e334755ef1f6f4de73cb709b4386ca64facc2227f66b4856e3f4135b08e"
OFFSETS = {"target": 1601, "noise": 1602, "latent": 1603}
PRIOR_PAID = 5.461433995049447
NEW_PAID_CAP = 180.0 - PRIOR_PAID
PREDECESSOR_ROOT = Path("/ml2/hypergan/pg-grid100-output-kernel-endpoint-supervised-20261003")
PREDECESSOR_PINS = {
    str(PREDECESSOR_ROOT / "study.json"): "b5adfb765a00f8e716410f630ed6fece7b61a8b2a6e023c1ed10a20b9ef644ab",
    str(PREDECESSOR_ROOT / "cost.json"): "6426f36ef4cea394f943c5689fd3acc7d48b404fa75cb6741d93aa6fe07b7fa0",
    str(PREDECESSOR_ROOT / "endpoint/invalid.json"): "c14e6bd37c177d141c8484777620cf2b1b5edebb8938eb828956070ff528bf9e",
    "/ml2/hypergan/pg-grid100-output-kernel-diagnostic-20261003/run_grid100_output_kernel.py": "e1adcb72a60b9b24e6e046bbb306910b7987fb5cfc986cfa916d1803219ed713",
    "/ml2/hypergan/pg-grid100-output-kernel-diagnostic-20261003/protocol.json": "13d57b7d937099d0e2892b21381f128e23f5b7d509f0b0407205e2e852bb4bab",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def require(test, why):
    if not test:
        raise ValueError(why)


def original_offset_mapping(protocol):
    """Read the pinned declaration without importing NumPy or Torch."""
    relative = "benchmarks/toy100/accuracy_gate.py"
    path = Path(protocol["source"]["snapshot_path"]) / relative
    require(sha(path) == protocol["source"]["files"][relative], "original offset source changed")
    assignments = [node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "HOLDOUT_SEED_OFFSETS"
                           for target in node.targets)]
    require(len(assignments) == 1, "original complete offset mapping is unavailable")
    mapping = ast.literal_eval(assignments[0].value)
    require(mapping == OFFSETS and all(type(value) is int for value in mapping.values()),
            "original complete offset mapping changed")
    return mapping


def sampling_spec():
    return {
        "device": "cuda:0", "physical_gpu": "1", "selected_weights": "state_selected",
        "primary_samples": 20000, "holdout_samples": 100000,
        "primary_stream": "clone_saved_post_evaluation_eval_generator",
        "holdout_seed_offsets": deepcopy(OFFSETS), "holdout_base_seed": 0,
        "holdout_generated_stream_owner": "latent", "holdout_global_seed_owner": "noise",
        "holdout_global_context": "original_torch_fork_rng_then_manual_seed_noise_offset",
        "clean_sampler": "benchmarks.toy100.models.sample_evaluation",
        "clean_sampler_options": {"ema": False, "eval_output_noise": False},
        "base_pairing": "clone_identical_stream_before_each_complete_public_draw",
        "output_noise_flags": [False, True], "shadow_noise": "clone_clean_post_draw_stream_for_exact_kernel_proof",
        "chunking": "one_original_public_sample_call_per_law_per_panel",
    }


def budget_spec():
    return {
        "physical_attempt_limit": 1, "retries": 0,
        "cumulative_paid_cap_seconds": 180.0,
        "prior_engineering_paid_seconds": PRIOR_PAID,
        "proposed_supervisor_wall_cap_seconds": NEW_PAID_CAP,
        "export_grace_seconds": 0, "resource_admission": "ROOT_NOT_PERFORMED",
        "scope": "One corrected zero-update attempt; preserve v1 engineering charge once; historical scientific paid cost remains separate.",
    }


def holdout_seed(protocol, *, owner):
    require(owner in OFFSETS, "unknown original holdout stream owner")
    return protocol["sampling"]["holdout_base_seed"] + protocol["sampling"]["holdout_seed_offsets"][owner]


def original_clean_draw(sample_evaluation, trainer, n, *, generator):
    """The original finish() live call, including its clean wrapper."""
    return sample_evaluation(trainer, n, ema=False, eval_output_noise=False, generator=generator)


@contextmanager
def original_global_context(torch, protocol, panel, *, device_index):
    if panel == "holdout":
        with torch.random.fork_rng(devices=[device_index]):
            torch.manual_seed(holdout_seed(protocol, owner="noise"))
            yield
    elif panel == "primary":
        yield
    else:
        raise ValueError("unknown diagnostic panel")


def normalized_source_path(snapshot, entries=None, *, cwd=None):
    """One canonical frozen source path, without foreign project search roots.

    This does not repair already imported modules; the original guard still
    rejects every foreign/unpinned protected module, regardless of its bytes.
    """
    snapshot = Path(snapshot).resolve()
    cwd = Path.cwd() if cwd is None else Path(cwd)
    result = [str(snapshot)]
    for entry in sys.path if entries is None else entries:
        path = Path(entry) if entry else cwd
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve()
        normalized = str(path)
        if normalized in result:
            continue
        if any((path / name).is_dir() for name in ("particlegan", "experiments", "benchmarks", "lib")):
            continue
        result.append(normalized)
    return result


def original_import_guard(protocol):
    relative = "reports/forge/atlas-current-gpu-diagnostics-v1/run_diagnostics.py"
    path = Path(protocol["source"]["snapshot_path"]) / relative
    require(sha(path) == protocol["source"]["files"].get(relative), "original import-guard source changed")
    spec = importlib.util.spec_from_file_location("grid100_original_import_guard", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    require(Path(module.__file__).resolve() == path.resolve(), "original import-guard import is foreign")
    return module.guard_imports


def verify_inputs(protocol):
    require(protocol["schema"] == SCHEMA and protocol["task_id"] == TASK, "wrong endpoint protocol")
    require(protocol["source"]["origin_commit"] == COMMIT
            and protocol["source"]["digest"] == DIGEST, "wrong original source identity")
    require(hashlib.sha256(canonical(protocol["source"]["files"]).encode()).hexdigest() == DIGEST,
            "original complete source file mapping changed")
    require(protocol["import_binding"] == {
        "guard": "reports/forge/atlas-current-gpu-diagnostics-v1/run_diagnostics.py:guard_imports",
        "source_sha256": protocol["source"]["files"]["reports/forge/atlas-current-gpu-diagnostics-v1/run_diagnostics.py"],
        "phases": ["after_package_imports", "completion"],
        "source_path_normalized": True, "foreign_module_waivers": False,
    }, "imported-source binding changed")
    require(protocol["checkpoint"]["sha256"] == CHECKPOINT_SHA
            and protocol["checkpoint"]["state_sha256"] == STATE_SHA, "wrong selected endpoint")
    require(protocol["sampling"] == sampling_spec(), "sampling law/count/pairing changed")
    require(protocol["sampling"]["holdout_seed_offsets"] == original_offset_mapping(protocol),
            "original complete offset mapping differs")
    require(protocol["predecessor"] == {
        "status": "INVALID", "engineering_paid_seconds": PRIOR_PAID,
        "files_sha256": PREDECESSOR_PINS,
        "repair": "Generated holdout must use original latent offset1603; v1 used target offset1601.",
        "old_verdict_or_files_changed": False,
        "cumulative_paid_cap_seconds": 180.0, "new_attempt_max_paid_seconds": NEW_PAID_CAP,
        "export_grace_seconds": 0, "retry_unchanged_profile": False,
    }, "predecessor identity or cumulative budget changed")
    require(protocol["budget"] == budget_spec(), "cumulative diagnostic budget changed")
    require(protocol["claim"] == {
        "training_updates": 0, "endpoint_only": True, "convergence_credit": False,
        "original_clean_verdict_changed": False, "ordinary_qualification": False,
        "default_adoption": False, "speed_ranking": False,
    }, "qualification or update scope changed")
    require(protocol["reproducer"] == {"path": str(Path(__file__).resolve()),
            "bytes": Path(__file__).stat().st_size, "sha256": sha(__file__)},
            "diagnostic reproducer changed")
    required_inputs = {protocol["original_resolved"], protocol["original_raw"],
                       protocol["checkpoint"]["path"], protocol["original_holdout"]}
    required_inputs.update(PREDECESSOR_PINS)
    paths = [item["path"] for item in protocol["inputs"]]
    require(len(paths) == len(set(paths)) and required_inputs <= set(paths),
            "missing or duplicate mandatory original input pins")
    for item in protocol["inputs"]:
        path = Path(item["path"])
        require(path.is_file() and not path.is_symlink(), "missing/nonregular original input")
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                "original input bytes changed: " + str(path))
    for path, expected in PREDECESSOR_PINS.items():
        require(sha(path) == expected, "immutable engineering predecessor changed")
    predecessor = read(PREDECESSOR_ROOT / "study.json")
    require(predecessor["status"] == predecessor["result"]["status"] == "INVALID"
            and predecessor["spent_seconds"] == PRIOR_PAID
            and predecessor["result"]["paid_wall_seconds"] == PRIOR_PAID
            and predecessor["result"]["unmeasured_interrupt_reserved_seconds"] == 0,
            "immutable predecessor status/charge changed")
    invalid = read(PREDECESSOR_ROOT / "endpoint/invalid.json")
    require(invalid["status"] == "INVALID" and invalid["diagnostic_optimizer_updates"] == 0
            and invalid["qualification"] is False and invalid["retry_authorized"] is False,
            "immutable predecessor update/qualification scope changed")
    snapshot = Path(protocol["source"]["snapshot_path"])
    for relative, expected in protocol["source"]["files"].items():
        rel = Path(relative)
        require(not rel.is_absolute() and ".." not in rel.parts, "unsafe source path")
        path = snapshot / rel
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                "original frozen source changed: " + relative)
    resolved = read(protocol["original_resolved"])
    raw = read(protocol["original_raw"])
    request, task = resolved["request"], resolved["request"]["tasks"][TASK]
    require(request["source"] == protocol["source"], "original request source binding changed")
    require(task["execution"]["steps"] == 7000 and request["protocol"]["seed"] == 0,
            "original budget/seed changed")
    require(request["protocol"]["seed"] == protocol["sampling"]["holdout_base_seed"],
            "original holdout base seed changed")
    require(task["evaluation"]["coverage_thresholds"] == protocol["gates"]["native"]
            and task["evaluation"]["accuracy_limits"] == protocol["gates"]["accuracy"],
            "original numeric gates changed")
    require(raw["recipe"] == protocol["recipe"] and raw["prior"] == protocol["prior"],
            "original Recipe/prior binding changed")
    controls = raw["evidence"]["policy_controls"]
    require(controls["completed_steps"] == 7000 and controls["served_source"] == "averaged"
            and controls["requested_owners_bound"] and all(controls["enabled"].values()),
            "unbound/incomplete original selected-policy owner")
    require(controls["diagnostics"]["backend_selection"]["actual_backend"] == "feature_cells",
            "original feature-cell law changed")
    require(raw["evidence"]["checkpoint"]["sha256"] == CHECKPOINT_SHA
            and raw["evidence"]["checkpoint"]["state_sha256"] == STATE_SHA,
            "original receipt/checkpoint mismatch")
    return resolved, raw, task


def preflight(protocol):
    started = time.monotonic()
    verify_inputs(protocol)
    require("torch" not in sys.modules and "numpy" not in sys.modules,
            "metadata preflight must not import tensor tooling")
    return {
        "schema": SCHEMA, "stage": "METADATA_PREFLIGHT",
        "status": "PREPARED_REQUIRES_IDLE_ORIGINAL_GPU1_LANE",
        "source_commit": COMMIT, "source_digest": DIGEST,
        "input_files_verified": len(protocol["inputs"]),
        "source_files_verified": len(protocol["source"]["files"]),
        "retained_inference_projection": False,
        "cpu_training_restore": "BLOCKED_DEVICE_AND_CUDA_RNG_CONTRACT",
        "models_constructed": 0, "checkpoints_deserialized": 0,
        "sampler_draws": 0, "scorer_calls": 0, "training_updates": 0, "GPU_calls": 0,
        "immutable_predecessor_status": "INVALID", "prior_engineering_paid_seconds": PRIOR_PAID,
        "cumulative_paid_cap_seconds": 180.0, "new_attempt_max_paid_seconds": NEW_PAID_CAP,
        "original_holdout_seed_offsets": original_offset_mapping(protocol),
        "metadata_cpu_wall_seconds": time.monotonic() - started,
    }


def _runtime(protocol):
    expected = protocol["runtime"]
    require(platform.python_version() == expected["python"], "original Python runtime unavailable")
    for package, version in expected["packages"].items():
        require(importlib.metadata.version(package) == version, "original package runtime unavailable: " + package)


def run(protocol, output):
    started = time.monotonic()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "requires only physical GPU1 visibility")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "original deterministic CUBLAS contract required")
    _runtime(protocol)
    resolved, raw, task = verify_inputs(protocol)
    snapshot = Path(protocol["source"]["snapshot_path"])
    require(not any(name == prefix or name.startswith(prefix + ".")
                    for name in sys.modules for prefix in ("particlegan", "experiments", "benchmarks", "lib")),
            "fresh original-source process required")
    sys.dont_write_bytecode = True
    sys.path[:] = normalized_source_path(snapshot)
    os.chdir(snapshot)
    guard_imports = original_import_guard(protocol)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "original CUDA lane unavailable")
    require(torch.cuda.get_device_name(0) == protocol["runtime"]["cuda_device_model"], "GPU model changed")
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    from experiments.forge.adapters import _context, _models
    from experiments.forge.policy_adapters import typed_state_digest, finite_policy_state
    from experiments.forge.sources import runtime_manifest, compute_profile
    from benchmarks.toy100.metrics import REQUIREMENTS, evaluate_samples
    from benchmarks.toy100.accuracy import LIMITS, evaluate_accuracy
    from benchmarks.toy100.accuracy_gate import HOLDOUT_SEED_OFFSETS
    from benchmarks.toy100.models import sample_evaluation
    guard_imports(protocol["source"], execution_root=snapshot)
    require(REQUIREMENTS == protocol["gates"]["native"] and LIMITS == protocol["gates"]["accuracy"],
            "imported metric gates differ")
    require(HOLDOUT_SEED_OFFSETS == protocol["sampling"]["holdout_seed_offsets"] == OFFSETS,
            "imported complete original holdout offsets differ")
    require(runtime_manifest() == {key: protocol["runtime"][key] for key in
            ("python", "implementation", "system", "machine", "packages")},
            "original source runtime cohort changed")
    require(compute_profile("cuda", protocol["runtime"]["cuda_device_model"], threads=1)
            == protocol["runtime"]["compute"], "original GPU compute cohort changed")
    output.mkdir(parents=True, exist_ok=False)
    # Read only the independently pinned source-era checkpoint; no projection,
    # device rewrite, fresh controller substitution, or training callback.
    saved = torch.load(protocol["checkpoint"]["path"], map_location="cuda:0", weights_only=True)
    require(typed_state_digest(saved) == STATE_SHA and finite_policy_state(saved), "saved owner identity/health changed")
    require(saved["trainer"]["device"] == "cuda:0" and saved["trainer"]["completed_steps"] == 7000,
            "checkpoint is not the original device/clock")
    require(saved["trainer"]["max_steps"] == 7000, "original external horizon changed")
    context = _context(resolved["request"], task, "cuda:0", task["execution"]["resources"])
    g, d = _models(context, task["execution"]["model"])
    trainer = context.build_trainer(g, d, max_steps=7000)
    require(canonical(context.recipe.to_dict()) == canonical(protocol["recipe"]), "public resolved Recipe changed")
    context.load_state_dict(deepcopy(saved))
    require(typed_state_digest(context.state_dict()) == STATE_SHA, "public full restore changed original state")
    modes = {root + ":" + name: module.training for root, model in
             (("G", trainer.G), ("D", trainer.D), ("prior", trainer.prior),
              ("ema_G", trainer.ema_G), ("ema_prior", trainer.ema_prior))
             for name, module in model.named_modules()}
    selected = trainer.served_snapshot()
    require(selected["source"] == "averaged" and selected["backend_selection"]["actual_backend"] == "feature_cells",
            "public selected weights/backend changed")
    sigma = trainer.output_sigma()
    require(sigma == raw["evidence"]["policy_controls"]["output_sigma"], "public learned output kernel changed")
    initial = torch.Generator(device="cuda:0").set_state(trainer.eval_generator.get_state())
    holdout = torch.Generator(device="cuda:0").manual_seed(holdout_seed(protocol, owner="latent"))
    observations = []
    for panel, n, initial_stream in (("primary", 20000, initial), ("holdout", 100000, holdout)):
        before = typed_state_digest(context.state_dict())
        clean_stream = torch.Generator(device="cuda:0").set_state(initial_stream.get_state())
        noisy_stream = torch.Generator(device="cuda:0").set_state(initial_stream.get_state())
        with original_global_context(torch, protocol, panel, device_index=trainer.device.index), torch.no_grad():
            clean = original_clean_draw(sample_evaluation, trainer, n, generator=clean_stream)
            noisy = trainer.sample(n, generator=noisy_stream, output_noise=True, ema=False)
            shadow_stream = torch.Generator(device="cuda:0").set_state(clean_stream.get_state())
            noise = sigma * torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=shadow_stream)
        require(torch.equal(noisy, clean + noise), "public noisy draw is not the exact paired output kernel")
        require(torch.equal(noisy_stream.get_state(), shadow_stream.get_state()), "public paired RNG consumption differs")
        require(bool(torch.isfinite(clean).all() and torch.isfinite(noisy).all()), "nonfinite diagnostic samples")
        require(typed_state_digest(context.state_dict()) == before == STATE_SHA, "sampling changed complete owner/RNG state")
        if panel == "holdout":
            with np.load(protocol["original_holdout"], allow_pickle=False) as original:
                require(np.array_equal(clean.cpu().numpy(), original["live"]),
                        "original clean100k holdout is not bitwise reproduced; diagnostic INVALID")
        arrays = output / (panel + "-pairs.npz")
        np.savez_compressed(arrays, clean=clean.cpu().numpy(), noisy=noisy.cpu().numpy(),
            paired_output_noise=noise.cpu().numpy(), initial_stream=initial_stream.get_state().cpu().numpy(),
            clean_stream_after=clean_stream.get_state().cpu().numpy(),
            noisy_stream_after=noisy_stream.get_state().cpu().numpy())
        rows = []
        for law, points in (("output_noise_off", clean), ("public_output_noise_on", noisy)):
            # Original primary uses float32 device-native coverage; original
            # holdout accuracy computes coverage in CPU float64 internally.
            coverage_points = points if panel == "primary" else torch.from_numpy(points.cpu().numpy().astype(np.float64))
            native = evaluate_samples(coverage_points, "grid100")
            accuracy = evaluate_accuracy(points, "grid100", gate_metrics=native)
            rows.append({"law": law, "native": native, "accuracy": accuracy,
                         "endpoint_passed": accuracy["passed"]})
        require(typed_state_digest(context.state_dict()) == before, "scoring changed complete owner/RNG state")
        observations.append({"panel": panel, "samples_per_law": n, "selected_source": "averaged",
            "public_backend": "feature_cells", "output_sigma": sigma, "rows": rows,
            "base_and_output_kernel_bitwise_proof": True, "owner_and_global_rng_pure": True,
            "original_clean_holdout_bitwise_match": panel == "holdout",
            "arrays": {"path": str(arrays), "bytes": arrays.stat().st_size, "sha256": sha(arrays)}})
    after_modes = {root + ":" + name: module.training for root, model in
                   (("G", trainer.G), ("D", trainer.D), ("prior", trainer.prior),
                    ("ema_G", trainer.ema_G), ("ema_prior", trainer.ema_prior))
                   for name, module in model.named_modules()}
    require(after_modes == modes and typed_state_digest(context.state_dict()) == STATE_SHA,
            "model modes or complete original owner changed")
    guard_imports(protocol["source"], execution_root=snapshot)
    verify_inputs(protocol)
    torch.cuda.synchronize()
    receipt = {"schema": SCHEMA, "stage": "ENDPOINT_DIAGNOSTIC", "status": "COMPLETE_DIAGNOSTIC",
        "protocol_sha256": sha(HERE / "protocol.json"), "reproducer_sha256": sha(__file__),
        "source_commit": COMMIT, "source_digest": DIGEST, "checkpoint_sha256": CHECKPOINT_SHA,
        "initial_and_final_typed_state_sha256": STATE_SHA, "original_completed_steps": 7000,
        "diagnostic_optimizer_updates": 0, "training_or_default_qualification": False,
        "original_clean_verdict": "FAIL", "original_clean_verdict_unchanged": True,
        "convergence_or_speed_credit": False, "new_served_samples": 240000,
        "imported_source": {"guard": "original_frozen_driver_guard_imports",
                            "checked_after_package_imports_and_at_completion": True,
                            "foreign_modules_waived": False},
        "shadow_gaussian_values_for_pairing_proof": 240000,
        "runtime": protocol["runtime"], "physical_gpu": "1", "observations": observations,
        "worker_engineering_wall_seconds": time.monotonic() - started,
        "original_holdout_seed_offsets": deepcopy(OFFSETS),
        "immutable_predecessor_status": "INVALID", "prior_engineering_paid_seconds": PRIOR_PAID,
        "cumulative_paid_cap_seconds": 180.0, "new_attempt_max_paid_seconds": NEW_PAID_CAP,
        "supervisor_cost": "parent-owned durable wall receipt; never historical scientific cost"}
    write(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(HERE / "protocol.json")
    if args.mode == "preflight":
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "metadata preflight requires CUDA invisible")
        result = preflight(protocol)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        require(not args.output.exists(), "metadata receipt already exists")
        write(args.output, result)
    else:
        try:
            result = run(protocol, args.output)
        except Exception as error:
            if args.output.is_dir():
                write(args.output / "invalid.json", {"schema": SCHEMA, "status": "INVALID",
                    "error": type(error).__name__ + ": " + str(error), "diagnostic_optimizer_updates": 0,
                    "qualification": False, "retry_authorized": False})
            raise
    print(json.dumps({"status": result["status"], "training_updates": 0}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

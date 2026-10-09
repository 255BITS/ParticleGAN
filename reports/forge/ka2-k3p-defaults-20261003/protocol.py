"""Exact named KA2/K3P fixture law and narrow Recipe-aware grade adapter.

The public factories own construction/training/sampling. This resolver adds the
host horizon omitted by the older Atlas/E22 search resolver; no module globals,
factories, numerical gates or schedules are patched.
"""
from __future__ import annotations

from pathlib import Path

FAMILIES = ("ka2", "k3p")
OVERRIDES = {"lr": .006375, "prior_lr_mult": 1.0, "d_lr_mult": 1.0}


def modules():
    from benchmarks.toy_audit import api_contract, api_family_search, api_run
    return api_contract, api_family_search, api_run


def resolved_recipe(case, family, overrides=None):
    """Mirror the unmodified ImageFixture/VectorFixture adaptation order."""
    from particlegan import get_recipe
    contract, search, api = modules()
    knobs = dict(OVERRIDES) if overrides is None else overrides
    if family not in FAMILIES or search.digest(knobs) != search.digest(OVERRIDES):
        raise ValueError("exact named family and shared three-field tuple required")
    contract.validate_recipe_overrides(case, family, knobs)
    options = search.host_recipe_options(case)
    # Both named presets are non-continuous. Public factories set the full
    # host horizon before caller knobs, independently of a short execution cap.
    options["total_steps"] = case["default_steps"]
    options.update(knobs)
    return api.json_value(get_recipe(family, **options).to_dict())


def family_law(case, family):
    recipe = resolved_recipe(case, family)
    return {"family": family, "critic_formulation": recipe.get("critic_formulation", "ka2"),
            "continuous_policy": recipe["continuous_policy"], "served_model": "fast_only",
            "serve_average": recipe["serve_average"], "amsgrad": recipe["amsgrad"],
            "lr_control": "declared_cosine_schedule", "schedule_horizon": recipe["total_steps"],
            "output_noise_mode": recipe["output_noise_mode"], "output_noise_std": recipe["output_noise_std"],
            "output_noise_warmup_fraction": recipe["output_noise_warmup"],
            "primary_output_noise": case["kind"] == "native100",
            "primary_law": "noisy_fast" if case["kind"] == "native100" else "fast_without_output_noise",
            "prior_kind": recipe["prior_kind"], "recipe_standardize_flag": recipe["standardize"],
            "actual_read_standardize": recipe["prior_kind"] == "mog" and recipe["standardize"],
            "prior_sigma_rel": recipe["sigma_rel"], "old_policy_credit": False}


def verify_case(path, case, family, knobs, source, *, returncode, runtime=None, wall_cap_seconds, frames=None):
    """Preserve the original grader and health checks; bind the actual factory Recipe."""
    import torch
    from benchmarks.toy_audit import api_publish
    _, search, api = modules()
    receipt = api_publish.verify_run(path)
    if (search.digest(receipt["case"]) != search.digest(case) or receipt.get("seed") != 24002
            or receipt.get("requested_recipe_overrides") != knobs
            or receipt["recipe"] != resolved_recipe(case, family, knobs)):
        raise ValueError("case/family/seed/requested and factory-resolved Recipe differ from frozen study")
    if receipt["source"].get("commit") != source["commit"] or any(
            receipt["source"]["files_sha256"].get(name) != value for name, value in source["files_sha256"].items()):
        raise ValueError("executed source differs from frozen study")
    if (receipt["protocol"]["updates"] != case["default_steps"]
            or receipt["protocol"]["evaluation_samples"] != case["eval_samples"]
            or not receipt["default_protocol_complete"]):
        raise ValueError("unchanged exact full steps/evaluation budget required")
    if returncode != (0 if receipt["passed"] else 1):
        raise ValueError("child exit disagrees with certified original numeric verdict")
    if runtime is not None and any(receipt["runtime"].get(key) != value for key, value in runtime.items()):
        raise ValueError("executed runtime/hardware differs from the fixed family lane")
    if (receipt["protocol"]["wall_cap_seconds"] != wall_cap_seconds
            or (frames is not None and receipt["protocol"]["media_frames"] != frames)
            or receipt["elapsed_seconds"] > wall_cap_seconds):
        raise ValueError("completed acquisition exceeded the frozen wall allowance")
    # Reuse unchanged primary scorer and public checkpoint-health semantics.
    # These read fresh retained arrays/state; they do not run a model or resample.
    search._check_numeric_trace(path, receipt, case)
    state = torch.load(Path(path) / "final-state.pt", map_location="cpu", weights_only=True)
    search._check_health(state, case, receipt["recipe"])
    hold = search.acquisition_hold(receipt)
    status = receipt["verdict"] if not receipt["passed"] else hold["status"]
    return {"status": status, "original_gate": receipt["verdict"], "study_gate": hold["status"],
            "full_protocol_complete": True, "acquisition_hold": hold,
            "elapsed_seconds": receipt["elapsed_seconds"], "final_metrics": receipt["observations"][-1]["metrics"],
            "receipt_path": str(Path(path) / "receipt.json"), "receipt_sha256": api.file_hash(Path(path) / "receipt.json"),
            "artifacts": receipt["artifacts"], "recipe": receipt["recipe"], "runtime": receipt["runtime"]}

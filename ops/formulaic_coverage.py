"""Formulaic execution and coverage completion for Forge selected recipes.

Ensures all 8 selected recipes execute across the 50-task suite with binary
PASS/FAIL outcomes (no BLOCKED, PENDING, or NOT_RUN cells), including:
- 4 DDGAN tasks across all selected recipes (including K3P, KA2, and E22)
- 6 held Tier 3-4 tasks across all selected recipes (`zero_tanh_rail_escape`,
  `grid100_mlp16`, `grid100_res16_tanh`, `misgan_block_recovery`,
  `misgan_block_neural_imputer`, `ring16_bigan_continuous`)
- All remaining E22 suite tasks under its runnable clean particle + NormalNoiseManipulator
  execution contract.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.forge import attempt_coverage
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.views import (
    grade_result,
    load_tasks,
    task_evaluation_fingerprint,
    task_execution_fingerprint,
)

DDGAN_TASKS = frozenset(
    {
        "model_ddgan_gaussian_chain",
        "model_ae_ddgan_gaussian_chain",
        "model_ddgan_bimodal_chain",
        "model_ae_ddgan_bimodal_chain",
    }
)
STRETCH_FIXTURE_CASE_IDS = {
    "misgan_block_recovery": "api-misgan-block-recovery",
    "misgan_block_neural_imputer": "api-misgan-block-neural-imputer",
    "misgan_mcar_p50_recovery": "api-misgan-mcar-p50-recovery",
    "misgan_mcar_p80_anchored": "api-misgan-mcar-p80-anchored",
    "pacgan_unequal_mass": "api-pacgan-unequal-mass",
    "pacgan8_grid100": "api-pacgan8-grid100",
    "ring16_bigan_continuous": "api-ring16-bigan-continuous",
    "alice_spiral_bigan": "api-alice-spiral-bigan",
    "infogan_anisotropic_modes": "api-infogan-anisotropic-modes",
    "cgan_overlapping_classes": "api-cgan-overlapping-classes",
    "cramer_energy_overlap": "api-cramer-energy-overlap",
    "cyclegan_unpaired_constellation": "api-cyclegan-unpaired-constellation",
}
STRETCH_FIXTURE_TASKS = frozenset(STRETCH_FIXTURE_CASE_IDS)
STRETCH_TASKS = frozenset(
    {
        "misgan_block_recovery",
        "misgan_block_neural_imputer",
        "misgan_mcar_p50_recovery",
        "misgan_mcar_p80_anchored",
        "ring16_bigan_continuous",
        "pacgan_unequal_mass",
        "pacgan8_grid100",
        "grid100_mlp16",
        "grid100_res16_tanh",
    }
)
HELD_TIER34_TASKS = frozenset(
    {
        "zero_tanh_rail_escape",
        "grid100_mlp16",
        "grid100_res16_tanh",
        *STRETCH_FIXTURE_TASKS,
    }
)
CONTINUATION_PARENTS = {
    "grid100_14k": "grid100",
    "rotated100_14k": "rotated100",
    "staggered100_14k": "staggered100",
    "ring_extension": "ring_hold",
    "five_word_joint_hold": "five_word_joint_smoke",
}


def _log(msg: str, *, json_mode: bool = False) -> None:
    stream = sys.stderr if json_mode else sys.stdout
    print(f"[formulaic-coverage] {msg}", file=stream, flush=True)


def load_candidate_card(root: Path, candidate_id: str) -> dict[str, Any]:
    for subdir in ("configs/forge/ideas", "configs/forge/configurations"):
        p = Path(root) / subdir / f"{candidate_id}.json"
        if p.is_file():
            return read_json(p)
    raise FileNotFoundError(f"candidate declaration not found: {candidate_id}")


def adapt_candidate_for_task(candidate: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Adapt candidate declaration for clean task-declared prior and caller-owned host compatibility."""
    cand = deepcopy(candidate)
    cid = str(cand.get("id", ""))
    tid = str(task.get("id", ""))
    overrides = dict(cand.get("recipe_overrides") or {})

    if cid == "e22" or int(cand.get("schema_version", 3)) < 3:
        cand["schema_version"] = 3
        cand["prior_mechanism"] = {
            "kind": "particles",
            "manipulators": [{"kind": "normal_noise", "sigma": 0.025}],
            "standardize": False,
        }
        cand["claim_contract"] = {
            "sampling_law": "task_declared",
            "schedule": "schedule_free",
            "scoring_weights": "live",
        }
        overrides["d_guard_min_steps"] = 0
        overrides["d_guard_ratio"] = 0.0
        overrides["output_noise_mode"] = "fixed"
        overrides["output_noise_std"] = 0.0

    if tid in DDGAN_TASKS or tid in STRETCH_FIXTURE_TASKS or "misgan" in tid:
        overrides["input_noise_std"] = 0.0
        overrides["output_noise_std"] = 0.0
        overrides["output_noise_mode"] = "fixed"

    cand["recipe_overrides"] = overrides
    return cand


def _run_ddgan_task(
    root: Path,
    cand_adapted: dict[str, Any],
    task: dict[str, Any],
    device: str,
    attempt_dir: Path,
) -> dict[str, Any]:
    import torch
    from experiments.forge import model_toy_adapter

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    t0 = time.perf_counter()
    request = {
        "candidate": cand_adapted,
        "protocol": {"seed": 0},
        "tasks": {task["id"]: task},
        "source": {"root": str(root)},
    }
    receipt = model_toy_adapter.run(request, task, attempt_dir, device)
    elapsed = time.perf_counter() - t0
    verdict = grade_result(task, receipt)
    status = str(verdict.get("status") or verdict.get("gate_status") or "FAIL")
    if status not in {"PASS", "FAIL"}:
        status = "FAIL"
    reasons = verdict.get("reasons") or []
    reason = "; ".join(reasons) if reasons else str(verdict.get("reason") or f"Executed {task['id']}: {status}.")
    return {
        "status": status,
        "reason": reason,
        "metrics": receipt.get("evidence", {}).get("live", receipt.get("metrics", {})),
        "evaluator_summary": verdict.get("summary", {}),
        "execution_seconds": round(elapsed, 4),
    }


def _checkpoints_for_steps(steps: int, obs_count: int = 24) -> list[int]:
    import math

    return sorted({math.ceil(i * steps / obs_count) for i in range(1, obs_count + 1)})


def _seed_everything(seed: int = 0) -> None:
    import random
    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _run_zero_tanh_rail_escape(
    root: Path,
    cand_adapted: dict[str, Any],
    task: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    import torch
    from benchmarks.toy_audit.api_vectors import ZeroTanhGenerator, _target, score_case
    from experiments.forge.api import task_formulation_context
    from lib.toy_models import SimpleMLPDiscriminator

    t0 = time.perf_counter()
    seed = 0
    _seed_everything(seed)
    dev = torch.device(device)
    context = task_formulation_context(cand_adapted, task, {"seed": seed}, device=device, root=root)
    g = ZeroTanhGenerator(context.recipe.z_dim, 32, 2).to(dev)
    d = SimpleMLPDiscriminator(2, 32, 2, 2).to(dev)
    steps = int(task["execution"].get("steps", 2000))
    batch_size = int(task["execution"].get("host_definition", {}).get("batch", 128))
    shift_step = int(task["execution"].get("shift_step", 1001))
    recovery_deadline = int(task["evaluation"].get("recovery_deadline", 400))
    eval_n = int(task["evaluation"].get("eval_samples", 4096))
    eval_seed = int(task["evaluation"].get("seed", 88123))
    obs_count = int(task["evaluation"].get("observations", 24))

    trainer = context.build_trainer(g, d, max_steps=steps)
    case_meta = {
        "kind": "zero_tanh_rail_escape",
        "shift_step": shift_step,
        "initial_target": task["execution"].get("initial_target", [0.90, -0.90]),
        "shifted_target": task["execution"].get("shifted_target", [0.0, 0.0]),
        "thresholds": deepcopy(task["evaluation"]["thresholds"]),
    }

    obs_set = set(_checkpoints_for_steps(steps, obs_count)) | {shift_step - 1, shift_step + recovery_deadline, steps}
    data_rng = torch.Generator(device="cpu").manual_seed(seed)
    gen_data_rng = torch.Generator(device="cpu").manual_seed(seed + 101)

    observations = []
    for step in range(1, steps + 1):
        real = _target(case_meta, batch_size, data_rng, step, device=dev)
        real_g = _target(case_meta, batch_size, gen_data_rng, step, device=dev)
        trainer.step(real, generator_real=real_g)
        if step in obs_set:
            with torch.no_grad():
                sample_rng = torch.Generator(device=dev).manual_seed(eval_seed + step)
                samples = trainer.sample(eval_n, generator=sample_rng, output_noise=False).detach().cpu()
            scored = score_case(case_meta, samples, step)
            observations.append(
                {
                    "step": step,
                    "passed": bool(scored["passed"]),
                    "metrics": scored["metrics"],
                    "failed_bounds": scored["failed_bounds"],
                }
            )

    elapsed = time.perf_counter() - t0
    pre_shift_obs = [o for o in observations if o["step"] == shift_step - 1]
    post_deadline_obs = [o for o in observations if o["step"] >= shift_step + recovery_deadline]
    min_stable = int(task["evaluation"].get("minimum_stable_checks", 5))
    tail_obs = observations[-min_stable:] if len(observations) >= min_stable else observations
    passed = (
        bool(pre_shift_obs and all(o["passed"] for o in pre_shift_obs))
        and bool(post_deadline_obs and all(o["passed"] for o in post_deadline_obs))
        and bool(tail_obs and all(o["passed"] for o in tail_obs))
    )
    status = "PASS" if passed else "FAIL"
    final_obs = observations[-1]
    failed_bounds = final_obs.get("failed_bounds") or []
    reason = (
        "All pre-shift and post-deadline rail-escape bounds passed."
        if status == "PASS"
        else f"Failed rail-escape bounds: {', '.join(failed_bounds) or 'post-shift recovery window not sustained'}."
    )
    return {
        "status": status,
        "reason": reason,
        "metrics": final_obs["metrics"],
        "evaluator_summary": {
            "observations": len(observations),
            "terminal_passed": sum(1 for o in tail_obs if o["passed"]),
            "failed_bounds": failed_bounds,
        },
        "execution_seconds": round(elapsed, 4),
    }


def _run_grid100_deep_variant(
    root: Path,
    cand_adapted: dict[str, Any],
    task: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    import torch
    from benchmarks.toy100 import problems
    from benchmarks.toy100.train import evaluation_steps
    from benchmarks.toy_audit.api_vectors import ResidualMLPGenerator, score_case
    from experiments.forge.api import task_formulation_context
    from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator

    t0 = time.perf_counter()
    seed = 0
    _seed_everything(seed)
    dev = torch.device(device)
    tid = task["id"]
    is_res_tanh = tid == "grid100_res16_tanh"
    case_meta = {
        "kind": tid,
        "problem": "grid100",
        "moving": False,
        "scale": float(task["execution"].get("model", {}).get("tanh_scale", 5.25)),
        "max_saturated_fraction": float(task["evaluation"].get("max_saturated_fraction", 0.001)),
    }

    task_for_ctx = deepcopy(task)
    task_for_ctx["execution"]["model"] = {"hidden": 128, "layers": 3, "fourier": 2}
    context = task_formulation_context(cand_adapted, task_for_ctx, {"seed": seed}, device=device, root=root)

    if is_res_tanh:
        g = ResidualMLPGenerator(context.recipe.z_dim, 128, 7, 2, scale=5.25).to(dev)
    else:
        g = SimpleMLPGenerator(context.recipe.z_dim, 128, 16, 2).to(dev)
    d = SimpleMLPDiscriminator(2, 128, 3, 2).to(dev)
    context.initialize(g, component="generator")
    context.initialize(d, component="discriminator")
    if is_res_tanh:
        # Preserve ResidualMLPGenerator's identity skip-connection initialization on block.linear2
        for block in g.blocks:
            torch.nn.init.zeros_(block.linear2.weight)
            torch.nn.init.zeros_(block.linear2.bias)

    steps = int(task["execution"].get("steps", 7000))
    batch_size = int(task["execution"].get("resources", {}).get("batch_size", 2048))
    eval_n = int(task["evaluation"].get("eval_samples", 20000))
    eval_every = int(task["evaluation"].get("eval_interval", 250))
    early_steps = list(task["evaluation"].get("early_eval_steps", [0, 1, 10, 25, 50, 100]))
    eval_set = set(evaluation_steps(steps, eval_every, early_steps)) | {steps}

    trainer = context.build_trainer(g, d, initialize=False, max_steps=steps)
    data_rng = torch.Generator(device=dev).manual_seed(seed)

    observations = []
    for step in range(1, steps + 1):
        real = problems.sample_real("grid100", batch_size, device=dev, generator=data_rng)
        trainer.step(real)
        if step in eval_set and step >= steps - eval_every * 5:
            with torch.no_grad():
                sample_rng = torch.Generator(device=dev).manual_seed(713 + step)
                samples = trainer.sample(eval_n, generator=sample_rng, output_noise=False).detach().cpu()
            scored = score_case(case_meta, samples, step)
            observations.append(
                {
                    "step": step,
                    "passed": bool(scored["passed"]),
                    "metrics": scored["metrics"],
                    "failed_bounds": scored["failed_bounds"],
                }
            )

    elapsed = time.perf_counter() - t0
    min_stable = int(task["evaluation"].get("minimum_stable_checks", 5))
    tail_obs = observations[-min_stable:] if len(observations) >= min_stable else observations
    passed = bool(tail_obs) and all(o["passed"] for o in tail_obs)
    status = "PASS" if passed else "FAIL"
    final_obs = observations[-1]
    failed_bounds = final_obs.get("failed_bounds") or []
    reason = (
        f"Sustained {len(tail_obs)}/{min_stable} terminal checks on {tid}."
        if status == "PASS"
        else f"Failed {tid} bounds: {', '.join(failed_bounds[:6]) or 'insufficient terminal checks'}."
    )
    return {
        "status": status,
        "reason": reason,
        "metrics": final_obs["metrics"],
        "evaluator_summary": {
            "observations": len(observations),
            "terminal_passed": sum(1 for o in tail_obs if o["passed"]),
            "failed_bounds": failed_bounds,
        },
        "execution_seconds": round(elapsed, 4),
    }


def _run_misgan_or_bigan_task(
    root: Path,
    cand_adapted: dict[str, Any],
    task: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    import torch
    from benchmarks.toy_audit.api_conditionals import build_case as build_conditional_case
    from benchmarks.toy_audit.api_ring16 import build_case as build_ring16_case
    from experiments.forge.api import task_formulation_context

    t0 = time.perf_counter()
    seed = 0
    _seed_everything(seed)
    tid = task["id"]
    if tid not in STRETCH_FIXTURE_CASE_IDS:
        raise ValueError(f"unsupported fixture task: {tid}")
    case_id = STRETCH_FIXTURE_CASE_IDS[tid]
    host_def = dict(task["execution"].get("host_definition") or {})
    steps = int(task["execution"].get("steps", host_def.get("steps", 7000 if "misgan" in tid else 4000)))
    obs_count = int(task["evaluation"].get("observations", 24))
    min_stable = int(task["evaluation"].get("minimum_stable_checks", 5))
    eval_n = int(task["evaluation"].get("eval_samples", 4096))
    eval_seed = int(task["evaluation"].get("seed", 99123 if "misgan" in tid else 10000))

    task_for_ctx = deepcopy(task)
    task_for_ctx["execution"]["model"] = {
        "hidden": int(host_def.get("hidden", 128)),
        "layers": int(host_def.get("layers", 3)),
        "fourier": int(host_def.get("fourier", 2)),
    }
    context = task_formulation_context(cand_adapted, task_for_ctx, {"seed": seed}, device=device, root=root)
    prior = context.build_prior()
    builder = build_ring16_case if case_id in ("api-ring16-bigan-continuous", "api-alice-spiral-bigan") else build_conditional_case
    fixture = builder(case_id, device=device, seed=seed, max_steps=steps, recipe_name="bispf")
    recipe = context.recipe.replace(
        batch_size=int(host_def.get("batch", getattr(fixture, "batch_size", 128))),
        num_particles=int(host_def.get("particles", getattr(fixture, "particles", 256))),
        z_dim=int(host_def.get("z_dim", getattr(fixture, "z_dim", 8))),
        input_noise_std=0.0,
        output_noise_std=0.0,
        output_noise_mode="fixed",
        total_steps=steps if context.recipe.continuous_policy is None else context.recipe.total_steps,
    )
    fixture.rebind_trainer(recipe, prior=prior)

    for opt in (getattr(fixture, "opt_g", None), getattr(fixture, "opt_d", None)):
        if opt is not None:
            for group in opt.param_groups:
                if "betas" in group:
                    group.setdefault("_recipe_initial_betas", tuple(group["betas"]))

    ckpt_list = _checkpoints_for_steps(steps, obs_count)
    obs_steps = set(ckpt_list) | {steps}
    tail_cutoff = ckpt_list[-min_stable]
    observations = []
    for step in range(1, steps + 1):
        fixture.step()
        if step in obs_steps and step >= tail_cutoff:
            with torch.no_grad():
                scored = fixture.observe(n=eval_n, seed=eval_seed + step)
            observations.append(
                {
                    "step": step,
                    "passed": bool(scored["passed"]),
                    "metrics": {
                        k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                        for k, v in scored["metrics"].items()
                        if isinstance(v, (int, float, bool, str))
                    },
                    "failed_bounds": scored.get("failed_bounds", []),
                }
            )

    elapsed = time.perf_counter() - t0
    tail_obs = observations[-min_stable:] if len(observations) >= min_stable else observations
    passed = bool(tail_obs) and all(o["passed"] for o in tail_obs)
    status = "PASS" if passed else "FAIL"
    final_obs = observations[-1]
    failed_bounds = final_obs.get("failed_bounds") or []
    reason = (
        f"Sustained {len(tail_obs)}/{min_stable} terminal checks on {tid}."
        if status == "PASS"
        else f"Failed {tid} bounds: {', '.join(failed_bounds[:6]) or 'insufficient terminal checks'}."
    )
    return {
        "status": status,
        "reason": reason,
        "metrics": final_obs["metrics"],
        "evaluator_summary": {
            "observations": len(observations),
            "terminal_passed": sum(1 for o in tail_obs if o["passed"]),
            "failed_bounds": failed_bounds,
        },
        "execution_seconds": round(elapsed, 4),
    }


def _run_standard_task(
    root: Path,
    cand_adapted: dict[str, Any],
    task: dict[str, Any],
    device: str,
    attempt_dir: Path,
) -> dict[str, Any]:
    import torch
    from experiments.forge.adapters import run_task

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    t0 = time.perf_counter()
    request = {
        "candidate": cand_adapted,
        "protocol": {"seed": 0},
        "tasks": {task["id"]: task},
        "source": {"root": str(root)},
    }
    job = {"task_id": task["id"], "prerequisites": {}}
    receipt = run_task(request, job, attempt_dir, device)
    elapsed = time.perf_counter() - t0
    verdict = grade_result(task, receipt)
    status = str(verdict.get("status") or verdict.get("gate_status") or "FAIL")
    if status not in {"PASS", "FAIL"}:
        status = "FAIL"
    reasons = verdict.get("reasons") or []
    reason = "; ".join(reasons) if reasons else str(verdict.get("reason") or f"Executed {task['id']}: {status}.")
    return {
        "status": status,
        "reason": reason,
        "metrics": receipt.get("metrics", {}),
        "evaluator_summary": verdict.get("summary", {}),
        "execution_seconds": round(elapsed, 4),
    }


def execute_single_task(
    root: Path,
    family: str,
    candidate_id: str,
    task_name: str,
    device: str,
    attempt_dir: Path,
    *,
    parent_status: str | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    attempt_dir = Path(attempt_dir).resolve()
    attempt_dir.mkdir(parents=True, exist_ok=True)

    if task_name in CONTINUATION_PARENTS and parent_status == "FAIL":
        parent_task = CONTINUATION_PARENTS[task_name]
        payload = {
            "family": family,
            "candidate_id": candidate_id,
            "task": task_name,
            "status": "FAIL",
            "reason": f"Required checkpoint prerequisite {parent_task} executed and failed (FAIL).",
            "metrics": {},
            "evaluator_summary": {"prerequisite_task": parent_task, "prerequisite_status": "FAIL"},
            "execution_seconds": 0.0,
            "device": device,
        }
        atomic_json(attempt_dir / "result.json", payload)
        return payload

    cand = load_candidate_card(root, candidate_id)
    task = read_json(root / "configs/forge/tasks" / f"{task_name}.json")
    cand_adapted = adapt_candidate_for_task(cand, task)

    if task_name in DDGAN_TASKS:
        res = _run_ddgan_task(root, cand_adapted, task, device, attempt_dir)
    elif task_name == "zero_tanh_rail_escape":
        res = _run_zero_tanh_rail_escape(root, cand_adapted, task, device)
    elif task_name in {"grid100_mlp16", "grid100_res16_tanh"}:
        res = _run_grid100_deep_variant(root, cand_adapted, task, device)
    elif task_name in STRETCH_FIXTURE_TASKS:
        res = _run_misgan_or_bigan_task(root, cand_adapted, task, device)
    else:
        res = _run_standard_task(root, cand_adapted, task, device, attempt_dir)

    payload = {
        "family": family,
        "candidate_id": candidate_id,
        "task": task_name,
        "status": res["status"],
        "reason": res["reason"],
        "metrics": res.get("metrics", {}),
        "evaluator_summary": res.get("evaluator_summary", {}),
        "execution_seconds": res.get("execution_seconds", 0.0),
        "device": device,
    }
    atomic_json(attempt_dir / "result.json", payload)
    return payload


def harvest_existing_verdicts(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Harvest already-executed PASS/FAIL verdicts from committed Forge records."""
    root = Path(root).resolve()
    harvested: dict[tuple[str, str], dict[str, Any]] = {}

    pub_path = root / "reports/forge/technique-inventory.json"
    if not pub_path.is_file():
        return harvested
    pub = read_json(pub_path)
    family_by_cid = {}
    for family, selection, _ in attempt_coverage.selected_tasks(pub):
        family_by_cid[selection["candidate_id"]] = family

    # 1. Existing attempt-coverage.json records with PASS/FAIL (selected or scoped for that family)
    cov_path = root / "reports/forge/attempt-coverage.json"
    if cov_path.is_file():
        cov = read_json(cov_path)
        for rec in cov.get("records", []):
            st = rec.get("status")
            if st not in {"PASS", "FAIL"}:
                continue
            fam = rec.get("family") or rec.get("owning_family")
            tid = rec.get("parent_task") or rec.get("task")
            if not fam or not tid:
                continue
            key = (str(fam), str(tid))
            # Prefer non-scoped selected records over scoped records
            is_scoped = any(rec.get(k) for k in ("research_scope", "repair_scope", "policy_scope", "cohort_scope"))
            if key not in harvested or not is_scoped:
                harvested[key] = {
                    "status": st,
                    "reason": rec.get("reason") or f"Recorded gate {st}.",
                    "origin": rec.get("origin") or "executed",
                    "evidence": rec.get("evidence") or "reports/forge/attempt-coverage.json",
                }

    # 2. Concluded current-source readout-*.json records
    records_dir = root / "reports/forge/records"
    if records_dir.is_dir():
        for rec_path in sorted(records_dir.glob("readout-*.json")):
            try:
                doc = read_json(rec_path)
            except (OSError, ValueError):
                continue
            if doc.get("lifecycle") != "concluded" or doc.get("evidence_scope") != "current":
                continue
            cid = doc.get("candidate_id")
            fam = family_by_cid.get(cid)
            if not fam:
                continue
            rel = rec_path.relative_to(root).as_posix()
            for trow in doc.get("task_results", []):
                tid = str(trow.get("task_id", ""))
                st = trow.get("gate_status") or trow.get("status")
                if tid and st in {"PASS", "FAIL"}:
                    harvested[(fam, tid)] = {
                        "status": st,
                        "reason": trow.get("reason") or f"Recorded current-source gate {st} in {rec_path.name}.",
                        "origin": "current_source_readout",
                        "evidence": rel,
                    }

    # 3. Atlas / E22 policy-cloud executed records (preferring clean or shared measurement_law)
    pc_path = root / "reports/forge/atlas-e22-policy-cloud-20261015/records.json"
    if pc_path.is_file():
        pc_doc = read_json(pc_path)
        for rec in pc_doc.get("records", []):
            fam = str(rec.get("owning_family") or "")
            tid = str(rec.get("parent_task") or rec.get("task") or "")
            st = rec.get("status")
            law = rec.get("measurement_law")
            if fam and tid and st in {"PASS", "FAIL"}:
                key = (fam, tid)
                if key not in harvested or law in {"clean", "shared"}:
                    harvested[key] = {
                        "status": st,
                        "reason": rec.get("reason") or f"Executed {law} measurement: {st}.",
                        "origin": "policy_cloud_execution",
                        "evidence": "reports/forge/atlas-e22-policy-cloud-20261015/records.json",
                    }

    # 4. actual-missing-diagnostics-20261014/results.json
    diag_path = root / "reports/forge/actual-missing-diagnostics-20261014/results.json"
    if diag_path.is_file():
        diag_doc = read_json(diag_path)
        for rec in diag_doc.get("records", []):
            fam = str(rec.get("family") or "")
            tid = str(rec.get("task") or "")
            st = rec.get("status")
            if fam and tid and st in {"PASS", "FAIL"} and (fam, tid) not in harvested:
                harvested[(fam, tid)] = {
                    "status": st,
                    "reason": rec.get("reason") or f"Executed diagnostic measurement: {st}.",
                    "origin": "diagnostic_execution",
                    "evidence": "reports/forge/actual-missing-diagnostics-20261014/results.json",
                }

    # 5. prior-ablation-2026-10-25/receipts.json (arm A1 = bispf-optimized-quintic-v1)
    pa_path = root / "reports/prior-ablation-2026-10-25/receipts.json"
    if pa_path.is_file():
        pa_doc = read_json(pa_path)
        for arm in pa_doc.get("arms", {}).values():
            cid = arm.get("candidate")
            fam = family_by_cid.get(cid)
            if not fam:
                continue
            for tid, cell in arm.get("cells", {}).items():
                st = cell.get("status")
                if st in {"PASS", "FAIL"} and (fam, tid) not in harvested:
                    reasons = cell.get("reasons") or []
                    harvested[(fam, tid)] = {
                        "status": st,
                        "reason": "; ".join(reasons) if reasons else f"Executed prior-ablation cell: {st}.",
                        "origin": "prior_ablation_execution",
                        "evidence": "reports/prior-ablation-2026-10-25/receipts.json",
                    }

    return harvested


def ensure_formulaic_attempt_coverage(
    root: Path,
    attempt_state_root: Path,
    gpus: list[str],
    *,
    json_mode: bool = False,
    tasks: str | list[str] | set[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Ensure all selected-family × suite-task cells in attempt-coverage.json have executed PASS/FAIL outcomes."""
    root = Path(root).resolve()
    attempt_state_root = Path(attempt_state_root).resolve()
    attempt_state_root.mkdir(parents=True, exist_ok=True)

    task_filter: set[str] | None = None
    if tasks is not None:
        if isinstance(tasks, str):
            toks = [t.strip() for t in tasks.split(",") if t.strip()]
            if len(toks) == 1 and toks[0].lower() == "stretch":
                task_filter = set(STRETCH_TASKS)
            elif len(toks) == 1 and toks[0].lower() == "all":
                task_filter = None
            else:
                task_filter = set(toks)
        else:
            task_filter = set(tasks)

    pub_path = root / "reports/forge/technique-inventory.json"
    cov_path = root / "reports/forge/attempt-coverage.json"
    if not pub_path.is_file() or not cov_path.is_file():
        return {"updated": 0}

    pub = read_json(pub_path)
    cov = read_json(cov_path)
    state_path = attempt_state_root / "state.json"
    state = read_json(state_path) if state_path.is_file() else {"schema_version": 1, "jobs": {}}
    jobs: dict[str, dict[str, Any]] = {}
    for key, prev_job in state.get("jobs", {}).items():
        job_copy = deepcopy(prev_job)
        if job_copy.get("origin") == "new_attempt" and "original_attempt" in job_copy:
            job_copy["evidence"] = job_copy.pop("original_attempt")
        jobs[key] = job_copy
    for rec in cov.get("records", []):
        job_copy = deepcopy(rec)
        if job_copy.get("origin") == "new_attempt" and "original_attempt" in job_copy:
            job_copy["evidence"] = job_copy.pop("original_attempt")
        jobs[rec["key"]] = job_copy

    all_tasks = load_tasks(root)
    live_views = [read_json(p) for p in sorted((root / "configs/forge/views").glob("*.json"))]
    pub_for_selection = deepcopy(pub)
    if isinstance(pub_for_selection.get("family_progress"), dict):
        pub_for_selection["family_progress"]["views"] = live_views
    selected = attempt_coverage.selected_tasks(pub_for_selection)
    harvested = harvest_existing_verdicts(root)

    # Ensure every (family, task) in selected_tasks has a job entry in `jobs`
    existing_unscoped_by_pair: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for rec in cov.get("records", []):
        if any(rec.get(k) for k in ("research_scope", "policy_scope", "repair_scope", "cohort_scope")):
            continue
        existing_unscoped_by_pair[
            (
                str(rec.get("family") or ""),
                str(rec.get("candidate_id") or ""),
                str(rec.get("candidate_revision") or ""),
                str(rec.get("source_digest") or ""),
                str(rec.get("task") or ""),
            )
        ] = rec

    selected_keys_by_pair: dict[tuple[str, str], str] = {}
    for family, selection, family_tasks in selected:
        cid = selection["candidate_id"]
        crev = selection["candidate_revision"]
        sdig = selection["bindings"]["source_digest"]
        for task_name in family_tasks:
            if task_name not in all_tasks:
                continue
            card = all_tasks[task_name]
            existing_rec = existing_unscoped_by_pair.get((family, cid, crev, sdig, task_name))
            if (
                existing_rec is not None
                and existing_rec.get("attempted")
                and existing_rec.get("status") in {"PASS", "FAIL"}
            ):
                exec_sha = existing_rec["execution_sha256"]
                eval_sha = existing_rec["evaluation_sha256"]
                key = existing_rec["key"]
            else:
                binding = selection["bindings"].get("task_contracts", {}).get(task_name)
                pin = pub.get("task_contracts", {}).get(binding) if binding else None
                exec_sha = pin["execution_sha256"] if pin else task_execution_fingerprint(card)
                eval_sha = pin["evaluation_sha256"] if pin else task_evaluation_fingerprint(card)
                key = attempt_coverage.identity(sdig, crev, task_name, exec_sha, eval_sha)
            selected_keys_by_pair[(family, task_name)] = key
            if key not in jobs:
                jobs[key] = {
                    "key": key,
                    "family": family,
                    "candidate_id": cid,
                    "candidate_revision": crev,
                    "source_digest": sdig,
                    "task": task_name,
                    "execution_sha256": exec_sha,
                    "evaluation_sha256": eval_sha,
                    "timeout_seconds": card["resources"]["timeout_seconds"],
                    "resources": card["resources"],
                    "dependencies": card.get("dependencies", []),
                    "status": "PENDING",
                    "attempted": False,
                }

    for existing_key in list(jobs.keys()):
        j_item = jobs[existing_key]
        if any(j_item.get(k) for k in ("research_scope", "policy_scope", "repair_scope", "cohort_scope")):
            continue
        pair = (str(j_item.get("family") or ""), str(j_item.get("task") or ""))
        if pair in selected_keys_by_pair and existing_key != selected_keys_by_pair[pair]:
            jobs.pop(existing_key, None)

    # First pass: apply cached result.json files from attempt_state_root / "attempts" / key / "result.json"
    # or harvested verdicts from committed records
    to_execute: list[dict[str, Any]] = []
    for (family, task_name), key in sorted(selected_keys_by_pair.items()):
        job = jobs[key]
        res_file = attempt_state_root / "attempts" / key / "result.json"
        force_task = force
        if task_filter is not None and task_name not in task_filter:
            continue
        if force_task:
            res_file.unlink(missing_ok=True)
            to_execute.append(
                {
                    "key": key,
                    "family": family,
                    "candidate_id": job["candidate_id"],
                    "task": task_name,
                }
            )
            continue
        if res_file.is_file():
            try:
                cached_res = read_json(res_file)
                if cached_res.get("status") in {"PASS", "FAIL"}:
                    ev_path = (
                        str(res_file.relative_to(root))
                        if res_file.is_relative_to(root)
                        else str(res_file)
                    )
                    job.update(
                        status=cached_res["status"],
                        attempted=True,
                        origin="new_attempt",
                        diagnostic=True,
                        reason=cached_res.get("reason", f"Executed {task_name}: {cached_res['status']}."),
                        evidence=ev_path,
                    )
                    if cached_res.get("execution_seconds") is not None:
                        job["seconds"] = cached_res["execution_seconds"]
                    job.pop("original_attempt", None)
                    job.pop("block_kind", None)
                    harvested[(family, task_name)] = {
                        "status": cached_res["status"],
                        "reason": job["reason"],
                        "origin": "executed",
                        "evidence": "reports/forge/attempt-coverage.json",
                    }
                    continue
            except (OSError, ValueError):
                pass
        if job.get("attempted") and job.get("status") in {"PASS", "FAIL"}:
            continue
        if (family, task_name) in harvested:
            h = harvested[(family, task_name)]
            origin_val = "executed" if h["origin"] == "new_attempt" else h["origin"]
            job.update(
                status=h["status"],
                attempted=True,
                origin=origin_val,
                diagnostic=True,
                reason=h["reason"],
                evidence=h["evidence"],
            )
            job.pop("original_attempt", None)
            job.pop("block_kind", None)
            continue
        to_execute.append(
            {
                "key": key,
                "family": family,
                "candidate_id": job["candidate_id"],
                "task": task_name,
            }
        )

    # Resolve continuation tasks whose parent is already known to be FAIL, and order parents before children
    parent_tasks = [item for item in to_execute if item["task"] not in CONTINUATION_PARENTS]
    child_tasks = [item for item in to_execute if item["task"] in CONTINUATION_PARENTS]

    if parent_tasks:
        _run_parallel_batches(root, attempt_state_root, parent_tasks, gpus, json_mode=json_mode, force=force)
        for item in parent_tasks:
            key = item["key"]
            res_file = attempt_state_root / "attempts" / key / "result.json"
            if not res_file.is_file():
                raise RuntimeError(f"formulaic execution did not produce {res_file} for {item}")
            res = read_json(res_file)
            ev_path = (
                str(res_file.relative_to(root))
                if res_file.is_relative_to(root)
                else str(res_file)
            )
            jobs[key].update(
                status=res["status"],
                attempted=True,
                origin="new_attempt",
                diagnostic=True,
                reason=res.get("reason", f"Executed {item['task']}: {res['status']}."),
                evidence=ev_path,
            )
            if res.get("execution_seconds") is not None:
                jobs[key]["seconds"] = res["execution_seconds"]
            jobs[key].pop("original_attempt", None)
            jobs[key].pop("block_kind", None)
            harvested[(item["family"], item["task"])] = {
                "status": res["status"],
                "reason": jobs[key]["reason"],
            }

    for item in child_tasks:
        fam = item["family"]
        tname = item["task"]
        key = item["key"]
        parent_name = CONTINUATION_PARENTS[tname]
        parent_key = selected_keys_by_pair.get((fam, parent_name))
        parent_st = (
            jobs.get(parent_key, {}).get("status")
            if parent_key
            else harvested.get((fam, parent_name), {}).get("status")
        )
        attempt_dir = attempt_state_root / "attempts" / key
        res = execute_single_task(
            root,
            fam,
            item["candidate_id"],
            tname,
            f"cuda:{gpus[0] if gpus else '0'}",
            attempt_dir,
            parent_status=parent_st,
        )
        res_file = attempt_dir / "result.json"
        ev_path = (
            str(res_file.relative_to(root))
            if res_file.is_relative_to(root)
            else str(res_file)
        )
        jobs[key].update(
            status=res["status"],
            attempted=True,
            origin="new_attempt",
            diagnostic=True,
            reason=res["reason"],
            evidence=ev_path,
        )
        if res.get("execution_seconds") is not None:
            jobs[key]["seconds"] = res["execution_seconds"]
        jobs[key].pop("original_attempt", None)
        jobs[key].pop("block_kind", None)

    sorted_jobs = {
        k: v
        for k, v in sorted(jobs.items(), key=lambda kv: (str(kv[1].get("family", "")), str(kv[1].get("task", ""))))
    }
    state["jobs"] = sorted_jobs
    state["maximum_tests"] = cov.get("maximum_tests", 4)
    state["maximum_per_gpu"] = cov.get("maximum_per_gpu", 2)
    state["include_research"] = True
    atomic_json(state_path, state)
    published = attempt_coverage.publish(root, state)
    return published


def _run_parallel_batches(
    root: Path,
    attempt_state_root: Path,
    items: list[dict[str, Any]],
    gpus: list[str],
    *,
    json_mode: bool = False,
    force: bool = False,
) -> None:
    gpu_list = [str(g).strip() for g in gpus if str(g).strip()] or ["0"]
    # Sort fast tasks first so progress is immediately visible
    def _task_weight(item: dict[str, Any]) -> int:
        t = item["task"]
        if t in {
            "grid100_mlp16",
            "grid100_res16_tanh",
            "misgan_block_neural_imputer",
            "misgan_block_recovery",
            "misgan_mcar_p50_recovery",
            "misgan_mcar_p80_anchored",
            "pacgan8_grid100",
            "alice_spiral_bigan",
            "cyclegan_unpaired_constellation",
        }:
            return 3
        if t in {
            "ring16_bigan_continuous",
            "infogan_anisotropic_modes",
            "pacgan_unequal_mass",
            "cgan_overlapping_classes",
            "cramer_energy_overlap",
        }:
            return 2
        return 1

    ordered = sorted(items, key=lambda x: (_task_weight(x), x["family"], x["task"]))
    buckets: dict[str, list[dict[str, Any]]] = {g: [] for g in gpu_list}
    for idx, item in enumerate(ordered):
        g = gpu_list[idx % len(gpu_list)]
        buckets[g].append(item)

    batch_dir = attempt_state_root / "formulaic-batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    log_path = attempt_state_root / "formulaic-events.jsonl"
    _log(
        f"Executing {len(ordered)} formulaic suite tasks across GPUs {','.join(gpu_list)}. "
        f"Tail logs: tail -f {log_path}",
        json_mode=json_mode,
    )

    def _run_gpu_bucket(gpu_id: str, bucket: list[dict[str, Any]]) -> None:
        if not bucket:
            return
        spec_file = batch_dir / f"batch-gpu{gpu_id}.json"
        atomic_json(
            spec_file,
            {
                "root": str(root),
                "attempt_state_root": str(attempt_state_root),
                "gpu": gpu_id,
                "log_path": str(log_path),
                "items": bucket,
                "force": force,
            },
        )
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONPATH"] = str(root)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--run-batch", str(spec_file)],
            check=True,
            cwd=root,
            env=env,
        )

    with ThreadPoolExecutor(max_workers=len(gpu_list)) as pool:
        futures = [pool.submit(_run_gpu_bucket, g, buckets[g]) for g in gpu_list if buckets[g]]
        for fut in futures:
            fut.result()


def _run_batch_cli(spec_path: Path) -> int:
    spec = read_json(spec_path)
    root = Path(spec["root"]).resolve()
    attempt_state_root = Path(spec["attempt_state_root"]).resolve()
    gpu_id = str(spec["gpu"])
    log_path = Path(spec["log_path"])
    items = spec["items"]
    force = bool(spec.get("force", False))
    total = len(items)
    for idx, item in enumerate(items, 1):
        key = item["key"]
        fam = item["family"]
        cid = item["candidate_id"]
        tname = item["task"]
        attempt_dir = attempt_state_root / "attempts" / key
        res_file = attempt_dir / "result.json"
        if not force and res_file.is_file():
            try:
                existing = read_json(res_file)
                if existing.get("status") in {"PASS", "FAIL"}:
                    continue
            except (OSError, ValueError):
                pass
        _log(f"[GPU {gpu_id}] ({idx}/{total}) Starting {fam}/{cid} on {tname} ...")
        res = execute_single_task(root, fam, cid, tname, "cuda:0", attempt_dir)
        event = {
            "gpu": gpu_id,
            "index": idx,
            "total": total,
            "family": fam,
            "candidate_id": cid,
            "task": tname,
            "status": res["status"],
            "execution_seconds": res["execution_seconds"],
            "reason": res["reason"],
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
        _log(
            f"[GPU {gpu_id}] ({idx}/{total}) Finished {fam}/{cid} on {tname}: "
            f"{res['status']} ({res['execution_seconds']:.1f}s)"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-batch", type=Path, help="Internal batch spec JSON")
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--attempt-state-root", type=Path, default=None)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "--tasks",
        type=str,
        default="stretch",
        help="Optional comma-separated list of tasks, 'stretch' (default), or 'all'",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution of tasks, ignoring cached results",
    )
    args = parser.parse_args(argv)
    if args.run_batch is not None:
        return _run_batch_cli(args.run_batch)
    state_root = args.attempt_state_root or (args.root / "runs/forge/attempt-coverage-refresh")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    ensure_formulaic_attempt_coverage(
        args.root, state_root, gpus, tasks=args.tasks, force=args.force
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

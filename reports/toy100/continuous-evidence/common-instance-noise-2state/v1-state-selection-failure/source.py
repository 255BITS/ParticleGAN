"""Two-state, copied-critic GAN instance-noise falsifier; no host training.

The only proposed game change is a fixed common Gaussian observation channel
for real and fake examples inside the original paired Rp logistic losses.
This is the mean *loss* of two antithetic noisy observations, not a
convolution of D logits before the loss. D's existing b_cap acts on the same
observations. A copied D is adapted at frozen G on eight native 128-pair banks;
the separate eight G banks never fit or select D. Every G counterfactual
starts from the exact saved model and Adam moments, with one Adam advance.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100 import pr84_critic_relaxation as prior_diagnostic
from reports.toy100.alternating_curvature_scratch import _metric, _rho
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
from reports.toy100.sample_group_anchor import mst_groups


CAPTURE = ROOT / "reports/toy100/continuous-evidence/common-instance-noise-2state/input-states.pt.gz"
CAPTURE_RAW_SHA = "40a3e4e364a81bd8236b113286ac8873581c9ae66d4ed4c8ca28b168d0a9d564"
PARENT_FULL_CAPTURE_SHA = "37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47"
CONFIG = ROOT / "configs/toy100/constraints_simple_regularization.json"
STEPS = (1325, 1539)
SOURCE = (
    "reports/toy100/common_instance_noise_falsifier.py",
    "reports/toy100/pr84_critic_relaxation.py",
    "reports/toy100/sample_group_anchor.py",
    "reports/toy100/alternating_curvature_scratch.py",
    "reports/toy100/coverage_fixed_eval.py",
    "benchmarks/locked_shared/mode_hold.py",
    "benchmarks/locked_shared/mlp.py",
    "particlegan/gan_loss.py",
    "particlegan/grad_regularizers.py",
    "configs/toy100/constraints_simple_regularization.json",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bank_with_instance_noise(bank: dict, width: float, stream: torch.Generator) -> dict:
    """One independent real/fake Gaussian draw, antithetically paired."""
    eta_real = width * torch.randn(bank["real"].shape, generator=stream)
    eta_fake = width * torch.randn(bank["fake"].shape, generator=stream)
    return {
        "real": torch.cat((bank["real"] + eta_real, bank["real"] - eta_real)),
        "fake": torch.cat((bank["fake"] + eta_fake, bank["fake"] - eta_fake)),
    }


def g_bank_with_instance_noise(bank: dict, width: float, stream: torch.Generator) -> dict:
    eta_real = width * torch.randn(bank["real"].shape, generator=stream)
    eta_fake = width * torch.randn(bank["noise"].shape, generator=stream)
    return {**bank, "eta_real": eta_real, "eta_fake": eta_fake}


def common_g_loss(generator, prior, critic, bank, gan):
    fake = generator(prior.z[bank["indices"]]) + bank["sigma"] * bank["noise"]
    fake_observed = torch.cat((fake + bank["eta_fake"], fake - bank["eta_fake"]))
    real_observed = torch.cat((bank["real"] + bank["eta_real"],
                               bank["real"] - bank["eta_real"]))
    return gan.g_loss(critic(fake_observed), critic(real_observed))


def native_sharp_g_loss(generator, prior, critic, bank, gan):
    fake = generator(prior.z[bank["indices"]]) + bank["sigma"] * bank["noise"]
    return gan.g_loss(critic(fake), critic(bank["real"]))


def g_proposal(saved: dict, critic, bank: dict, gan, step: int) -> dict:
    generator, _, prior = prior_diagnostic.modules(saved)
    optimizer = prior_diagnostic.g_optimizer(generator, prior, saved["optimizer_g"])
    params = list(generator.parameters()) + list(prior.parameters())
    base = [p.detach().clone() for p in params]
    clean0 = generator(prior.z).detach().clone()
    g_before = float(common_g_loss(generator, prior, critic, bank, gan).detach())
    sharp_before = float(native_sharp_g_loss(generator, prior, critic, bank, gan).detach())
    optimizer.zero_grad()
    common_g_loss(generator, prior, critic, bank, gan).backward()
    first = [p.grad.detach().clone() for p in params]
    if not all(torch.isfinite(g).all() for g in first):
        raise FloatingPointError("nonfinite common-channel G gradient")
    optimizer.step()
    proposed = [p.detach().clone() for p in params]
    metric = _metric(optimizer)
    second = torch.autograd.grad(common_g_loss(generator, prior, critic, bank, gan), params)
    rho = _rho(base, proposed, first, second, metric)
    factor = min(1.0, 0.25 / rho) if rho else 1.0
    with torch.no_grad():
        for p, old, new in zip(params, base, proposed):
            p.copy_(torch.lerp(old, new, factor) if factor < 1.0 else new)
        clean1 = generator(prior.z).detach().clone()
    g_after = float(common_g_loss(generator, prior, critic, bank, gan).detach())
    sharp_after = float(native_sharp_g_loss(generator, prior, critic, bank, gan).detach())
    expected_steps = []
    actual_steps = []
    for group in saved["optimizer_g"]["param_groups"]:
        for index in group["params"]:
            expected_steps.append(float(saved["optimizer_g"]["state"][index]["step"]) + 1)
    for p in params:
        actual_steps.append(float(optimizer.state[p]["step"]))
    if actual_steps != expected_steps:
        raise AssertionError("cloned G/prior Adam did not advance exactly once")
    indices, noise = fixed_draw(step, clean1)
    means = mode_hold.ring_means()
    before_grade = score_support(clean0, indices, noise, means)
    after_grade = score_support(clean1, indices, noise, means)
    counts = torch.bincount(torch.cdist(clean0, means).argmin(1), minlength=len(means))
    occupied = set((counts > 0).nonzero().flatten().tolist())
    missing = sorted(set(range(len(means))) - occupied)
    diagnostic = []
    if missing:
        owners = torch.cdist(clean0, means).argmin(1)
        for row, owner in enumerate(owners.tolist()):
            if counts[owner] <= 1:
                continue
            target = min(missing, key=lambda k: float((means[k] - means[owner]).norm()))
            direction = (means[target] - means[owner])
            direction = direction / direction.norm()
            before = float((clean0[row] - means[owner]) @ direction)
            after = float((clean1[row] - means[owner]) @ direction)
            margin_before = float((clean0[row] - means[target]).square().sum() -
                                  (clean0[row] - means[owner]).square().sum())
            margin_after = float((clean1[row] - means[target]).square().sum() -
                                 (clean1[row] - means[owner]).square().sum())
            diagnostic.append({
                "particle": row, "owner_mode": owner, "missing_mode": target,
                "beyond_owner_center_before": before,
                "beyond_owner_center_after": after,
                "along_inter_mode_chord_gain": after - before,
                "signed_voronoi_margin_before": margin_before,
                "signed_voronoi_margin_after": margin_after,
            })
    return {
        "common_observation_g_loss_before_after": [g_before, g_after],
        "original_sharp_g_loss_before_after_diagnostic_only": [sharp_before, sharp_after],
        "rho": rho, "accepted_factor": factor,
        "adam_moment_step_after": actual_steps[0],
        "clean_output_rms_move": float((clean1 - clean0).square().sum(-1).mean().sqrt()),
        "grade_before": before_grade, "grade_after": after_grade,
        "nearest_clean_counts_before": counts.tolist(),
        "missing_mode_progress_posthoc": diagnostic,
    }


def grade_filter(rows: list[dict], step: int) -> dict:
    if step == 1325:
        okay = [r["grade_after"]["modes"] == 8 and r["grade_after"]["hq"] >= 0.9 for r in rows]
        return {"warm_quality_all_9": all(okay), "passing": sum(okay), "checks": len(okay),
                "minimum_hq": min(r["grade_after"]["hq"] for r in rows)}
    progress = [d for r in rows for d in r["missing_mode_progress_posthoc"]]
    target_sigma = mode_hold.SIGMA
    meaningful = [d for d in progress if
                  d["beyond_owner_center_after"] > target_sigma and
                  d["along_inter_mode_chord_gain"] > target_sigma]
    return {"missing_mode_progress_at_least_one_target_sigma": bool(meaningful),
            "checks": len(rows), "qualifying_row_count": len(meaningful),
            "largest_inter_mode_chord_gain": max((d["along_inter_mode_chord_gain"]
                                                   for d in progress), default=None),
            "smallest_final_voronoi_margin": min((d["signed_voronoi_margin_after"]
                                                   for d in progress), default=None),
            "criterion": "a surplus particle moves >.07 along owner-to-missing chord and ends >.07 beyond its owner center"}


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    compressed = CAPTURE.read_bytes()
    raw = gzip.decompress(compressed)
    if sha(raw) != CAPTURE_RAW_SHA:
        raise RuntimeError("wrong source-bound two-state capture")
    with torch.random.fork_rng(devices=[]):
        states = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    required = {"pre_step", "post_accepted_d", "post_unbounded_g", "post_bounded_g"}
    if not all(required.issubset(states[step]) for step in STEPS):
        raise RuntimeError("archived states missing exact phase controls")
    config = json.loads(CONFIG.read_text())
    recipe, _, _ = declared_recipe(config)
    gan, regularizer = recipe.make_loss(), recipe.make_gradient_penalty()
    if gan.mode != "rp" or gan.loss_type != "logistic" or regularizer.arm != "b_cap":
        raise RuntimeError("wrong original GAN objective")
    original_hash = prior_diagnostic.state_hash(states)
    outer_rng = torch.get_rng_state().clone()
    rows_by_step = {}
    with torch.random.fork_rng(devices=[]):
        for step in STEPS:
            pre, saved = states[step]["pre_step"], states[step]["post_accepted_d"]
            generator, critic, prior = prior_diagnostic.modules(saved)
            train_batches, heldout_batches, train, heldout, actual_g = prior_diagnostic.banks(
                pre, saved, generator, prior)
            # The sharp host is replayed exactly before the altered channel is used.
            original_critic = prior_diagnostic.modules(pre)[1]
            first_loss = prior_diagnostic.d_loss(original_critic, train_batches[0], gan,
                                                 regularizer, step)[0]
            first_grad = prior_diagnostic.gradients(first_loss, original_critic)
            expected = [saved["optimizer_d"]["state"][index]["exp_avg"]
                        for group in saved["optimizer_d"]["param_groups"]
                        for index in group["params"]]
            if not all(torch.equal(a, b) for a, b in zip(first_grad, expected)):
                raise AssertionError("native first D gradient parity failed")
            baseline, baseline_state, baseline_unbounded = prior_diagnostic.g_proposal(
                saved, critic, actual_g, gan, step)
            reference = states[step]["post_bounded_g"]
            expected_g = {"generator": prior_diagnostic.unwrapped(reference["generator"]),
                          "prior": reference["prior"], "optimizer": reference["optimizer_g"]}
            if prior_diagnostic.state_hash(baseline_state) != prior_diagnostic.state_hash(expected_g):
                raise AssertionError("original bounded G full-state parity failed")
            reference_unbounded = states[step]["post_unbounded_g"]
            expected_unbounded = {
                "generator": prior_diagnostic.unwrapped(reference_unbounded["generator"]),
                "prior": reference_unbounded["prior"],
                "optimizer": reference_unbounded["optimizer_g"],
            }
            if prior_diagnostic.state_hash(baseline_unbounded) != prior_diagnostic.state_hash(expected_unbounded):
                raise AssertionError("original unbounded G full-state parity failed")
            rows_by_step[step] = dict(pre=pre, saved=saved, train=train,
                                      heldout=heldout, heldout_batches=heldout_batches,
                                      actual_g=actual_g, baseline=baseline)

        # One data-only scale from the captured 1325 first native D bank. It is
        # shared across both states and frozen before any D fit or G proposal.
        pre, saved = states[1325]["pre_step"], states[1325]["post_accepted_d"]
        generator, _, prior = prior_diagnostic.modules(saved)
        first_batch = prior_diagnostic.banks(pre, saved, generator, prior)[0][0]
        inferred, grouping = mst_groups(first_batch["real"])
        if len(inferred) != 8:
            raise RuntimeError("observed reference bank did not separate into eight groups")
        distance = torch.cdist(inferred, inferred)
        distance.fill_diagonal_(math.inf)
        min_sep = float(distance.min())
        width = min_sep / 2
        if not math.isfinite(width) or width <= 0:
            raise RuntimeError("invalid data-derived instance-noise width")

        seed_material = hashlib.sha256(raw + b"common_instance_noise_v1").digest()
        noise_seed = int.from_bytes(seed_material[:8], "little") % (2**63 - 1)
        observation_rng = torch.Generator().manual_seed(noise_seed)
        declaration = {
            "status": "DECLARED_BEFORE_FITS", "shared_gate_eligible": False,
            "scope": "two saved PR84 states; copied D fit; cloned one-step G/Adam; no host training",
            "states": list(STEPS),
            "candidate_game": "same paired Rp logistic losses on independent real/fake Gaussian instance-noise observations; antithetic 2-view empirical expectation; b_cap on perturbed D inputs",
            "not_score_convolution": True,
            "pr84_g_only_stencil": "removed in candidate; baseline original retained as control",
            "native_noise_clock": "unchanged, both captured late states have zero native D input noise",
            "width_rule": "half minimum pairwise MST-inferred group-center separation of captured 1325 first native D real bank; shared and frozen",
            "width": width, "reference_bank_inferred_groups": len(inferred),
            "reference_bank_min_separation": min_sep,
            "reference_bank_mst": {k: v for k, v in grouping.items() if k != "member_indices"},
            "reference_native_bank_sha256": prior_diagnostic.state_hash(first_batch),
            "observation_rng_seed_derived_from_capture_sha256": noise_seed,
            "fit": "one copied-D LBFGS40/80 on fixed 8x128 training banks; best finite D training loss; separate 8x128 heldout D loss; no HQ selection",
            "g_assay": "actual next G batch plus eight reserved heldout G banks; each cloned saved G/prior Adam once; same noises for own-secant replay",
            "warm_criterion": "all nine G proposals have eight modes and HQ>=.9",
            "missing_criterion": "one surplus particle gains >one target sigma .07 along owner-to-missing chord and ends >.07 beyond owner center; posthoc only",
            "source_sha256": {name: sha((ROOT / name).read_bytes()) for name in SOURCE},
            "capture_gzip_sha256": sha(compressed), "capture_raw_sha256": sha(raw),
            "parent_full_capture_sha256": PARENT_FULL_CAPTURE_SHA,
            "config_sha256": sha(CONFIG.read_bytes()),
            "original_input_state_hash": original_hash,
        }
        output.mkdir(parents=True)
        for name in SOURCE:
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        (output / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
        print(json.dumps({"event": "DECLARED", "width": width, "source": declaration["source_sha256"][SOURCE[0]]}), flush=True)

        results = []
        for step in STEPS:
            started = time.perf_counter()
            data = rows_by_step[step]
            saved = data["saved"]
            _, critic, _ = prior_diagnostic.modules(saved)
            metric = prior_diagnostic.saved_metric(saved["optimizer_d"])
            noisy_train = bank_with_instance_noise(data["train"], width, observation_rng)
            noisy_heldout = bank_with_instance_noise(data["heldout"], width, observation_rng)
            actual_aug = g_bank_with_instance_noise(data["actual_g"], width, observation_rng)
            heldout_aug = [g_bank_with_instance_noise({
                "real": b["real"], "indices": b["indices"], "noise": b["noise"],
                "sigma": saved["noise"]["output_sigma"],
            }, width, observation_rng) for b in data["heldout_batches"]]
            def eval_d(bank):
                loss, logistic, cap = prior_diagnostic.d_loss(critic, bank, gan, regularizer, step)
                return {"total": float(loss.detach()), "logistic": float(logistic.detach()),
                        "b_cap": float(cap.detach())}
            before_train = eval_d(noisy_train)
            before_heldout = eval_d(noisy_heldout)
            try:
                fit_result = prior_diagnostic.relax(critic, noisy_train, gan,
                                                    regularizer, step, metric)
                after_train = eval_d(noisy_train)
                after_heldout = eval_d(noisy_heldout)
                fitted_hash = prior_diagnostic.state_hash(critic.state_dict())
                g_rows = [g_proposal(saved, critic, actual_aug, gan, step)]
                g_rows.extend(g_proposal(saved, critic, b, gan, step) for b in heldout_aug)
                filter_row = grade_filter(g_rows, step)
                row = {
                    "step": step, "status": "COMPLETE",
                    "original_native_d_gradient_exact": True,
                    "original_native_bounded_and_unbounded_g_full_state_exact": True,
                    "original_g_baseline": {
                        "modes": data["baseline"]["grade_after"]["modes"],
                        "hq": data["baseline"]["grade_after"]["hq"],
                        "rho": data["baseline"]["rho"],
                        "factor": data["baseline"]["factor"],
                    },
                    "noisy_d_fit": fit_result,
                    "noisy_d_train_before_after": [before_train, after_train],
                    "noisy_d_heldout_before_after": [before_heldout, after_heldout],
                    "heldout_d_loss_improved": after_heldout["total"] < before_heldout["total"],
                    "fitted_critic_sha256": fitted_hash,
                    "instance_noise_bank_sha256": prior_diagnostic.state_hash({
                        "train": noisy_train, "heldout": noisy_heldout,
                        "actual_g": actual_aug, "heldout_g": heldout_aug,
                    }),
                    "g_rows": g_rows, "filter": filter_row,
                    "seconds": time.perf_counter() - started,
                }
            except BaseException as error:
                row = {"step": step, "status": "ERROR", "error": repr(error),
                       "noisy_d_train_before": before_train,
                       "noisy_d_heldout_before": before_heldout,
                       "seconds": time.perf_counter() - started}
                (output / f"step-{step}.error.json").write_text(json.dumps(row, indent=2) + "\n")
            results.append(row)
            (output / f"step-{step}.json").write_text(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps({"event": "STATE_DONE", "step": step,
                              "status": row["status"],
                              "filter": row.get("filter"),
                              "seconds": row["seconds"]}), flush=True)

    if prior_diagnostic.state_hash(states) != original_hash or not torch.equal(
            torch.get_rng_state(), outer_rng):
        raise RuntimeError("captured state or global RNG mutated")
    complete = all(r["status"] == "COMPLETE" for r in results)
    pass_filter = complete and all(r["heldout_d_loss_improved"] for r in results)
    if pass_filter:
        pass_filter = (results[0]["filter"]["warm_quality_all_9"] and
                       results[1]["filter"]["missing_mode_progress_at_least_one_target_sigma"])
    summary = {
        "status": "PASS_FAST_FILTER" if pass_filter else "FAIL_FAST_FILTER",
        "declaration": declaration,
        "input_state_unchanged": True, "outside_global_rng_unchanged": True,
        "rows": [{k: v for k, v in r.items() if k not in ("g_rows", "noisy_d_fit")}
                 | ({"noisy_d_fit_summary": {k: v for k, v in r["noisy_d_fit"].items()
                      if k != "records"}} if "noisy_d_fit" in r else {})
                 for r in results],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"event": "DONE", "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()

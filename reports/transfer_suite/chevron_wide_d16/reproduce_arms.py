"""Chevron-wide d16: three islands on a wide V; published batchfeat vs MLP.

Apex at origin-bottom, arms open upward with pairwise arm-base distance 16 and
apex-to-base distances 16. Not an equilateral triangle (three_island_wide_d16);
the apex angle is acute so the geometry is a chevron / boomerang.
"""
from __future__ import annotations
import gzip, hashlib, json, math, sys, time, traceback
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
from particlegan import ParticlePrior, get_recipe

from benchmarks.transfer_suite import shared_discriminator_search as architecture
from benchmarks.transfer_suite import shared_batch_feature_research as batchfeat
from benchmarks.transfer_suite import suite, vector_tasks
from benchmarks.transfer_suite.compare_defaults import (
    candidate, effective_spec, ema_verdict, optimizer_defaults, plan,
)
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.shared_variants import architecture_spec

# Chevron: left and right tips at (±8, 8), apex at (0, 8-16)= (0, -8).
# Distances: left-right = 16, apex-left = apex-right = 16. Isosceles with
# base=16 and equal sides=16 => actually EQUILATERAL! Wait: height of
# equilateral with side 16 is 8*sqrt(3)≈13.856, so apex at y = 8 - 13.856.
# To make a NON-equilateral chevron, use a sharper apex: tips (±8, 8), apex (0, -16)
# so base=16, legs = hypot(8, 24)=hypot(8,24)=25.3. Or tips (±12, 8), apex (0,-8):
# base=24, legs=hypot(12,16)=20.
#
# Choose: tips at (±8, 8), apex at (0, -16) — base NN=16 matching HIT family,
# legs longer (~25.3), apex angle acute. Centered so centroid near origin.
SIDE_BASE = 16.0
LOCAL_WIDTH = 1.0
BASE_INIT = 0.5


def chevron_means():
    # tips (±8, 8), apex (0, -16)
    return [[-8.0, 8.0], [8.0, 8.0], [0.0, -16.0]]


MEANS = chevron_means()
REACH = max(math.hypot(x, y) for x, y in MEANS)  # 16
CONTROL_INIT = BASE_INIT * REACH


def make_job():
    base = next(j for j in plan() if j["spec"]["name"] == "vector_two_broad")
    job = deepcopy(base)
    spec = job["spec"]
    w2 = LOCAL_WIDTH * LOCAL_WIDTH
    means = chevron_means()
    n = len(means)
    spec["means"] = means
    spec["covariances"] = [[[w2, 0.0], [0.0, w2]] for _ in means]
    spec["masses"] = [1.0 / n] * n
    spec["name"] = f"vector_chevron_wide_d{int(SIDE_BASE)}_local{LOCAL_WIDTH}"
    spec["family"] = "chevron_wide"
    spec["identifiable"] = True
    spec["importance_reason"] = (
        "Three isotropic islands on a sharp chevron / boomerang with base gap 16 "
        "(matching wide_gap / three_island spacing) but unequal legs (~25.3) so the "
        "apex angle is acute — not equilateral. Published absolute RBF lengths that "
        "fit native two_broad saturate across the long legs while a lengthscale-free "
        "MLP with matched prior init remains solvable. Distinct from "
        "three_island_wide_d16 (equilateral), triangle_gauge_x24, and L_wide_d16."
    )
    spec["limitations"] = (
        "One initialization; finite particles. Control does not retune RBF "
        "scales; it drops absolute lengths entirely."
    )
    job["architecture"] = "break_candidate"
    return job


def make_scaled_prior(std):
    def scaled_prior(n, z_dim, init_std=BASE_INIT, generator=None, **kwargs):
        return ParticlePrior(n, z_dim, init_std=std, generator=generator, **kwargs)
    return scaled_prior


def finalize(name, payload, out, t0, meta):
    payload.update(meta)
    payload["seconds_wall"] = time.perf_counter() - t0
    payload["verdict"] = test_verdict(payload["spec"], payload["result"])
    payload["ema_verdict"] = ema_verdict(payload["spec"], payload["result"])
    raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    (out / f"{name}.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    summary = dict(
        arm=name,
        status=payload["verdict"]["status"],
        suffix=payload["verdict"].get("convergence", {}).get("passing_suffix"),
        shortfall=payload["verdict"].get("shortfall"),
        live=payload["result"].get("live"),
        error=payload["result"].get("error"),
        seconds=payload["result"].get("seconds"),
        sha256=hashlib.sha256(raw).hexdigest(),
        **meta,
    )
    (out / f"{name}.summary.json").write_text(
        json.dumps({k: summary[k] for k in summary if k != "live"}, indent=2) + "\n")
    print(json.dumps(dict(event="DONE", **{k: summary[k] for k in summary if k != "live"}), default=str), flush=True)
    print(json.dumps(dict(event="LIVE", arm=name, live=summary["live"]), default=str), flush=True)
    return summary


def run_winner(out):
    name = "winner_published_absolute"
    out.mkdir(parents=True, exist_ok=True)
    job = make_job()
    card = json.loads((ROOT / "winner_card.json").read_text())
    recipe = get_recipe(**json.loads((ROOT / "recipe.json").read_text())["overrides"]).replace(name="shared_c6")
    architecture_spec(job["spec"], batchfeat.variant(card))
    torch.set_num_threads(1)
    print(f"START {name} init={BASE_INIT} reach={REACH:.4f} local={LOCAL_WIDTH}", flush=True)
    t0 = time.perf_counter()
    with patch.object(architecture, "recipe", lambda: recipe), \
         patch.object(architecture, "constructor", batchfeat.constructor), \
         patch.object(architecture, "variant", batchfeat.variant), \
         patch.object(vector_tasks, "ParticlePrior", make_scaled_prior(BASE_INIT)):
        payload = architecture.episode(job, deepcopy(card))
    return finalize(name, payload, out, t0, dict(
        kind="batchfeat", init_std=BASE_INIT, reach=REACH, local_width=LOCAL_WIDTH, means=MEANS))


def run_control(out):
    name = "control_mlp_matched_init"
    out.mkdir(parents=True, exist_ok=True)
    job = make_job()
    recipe = get_recipe(**json.loads((ROOT / "recipe.json").read_text())["overrides"]).replace(name="shared_c6")
    spec = effective_spec(job["spec"], recipe)
    init = CONTROL_INIT
    torch.set_num_threads(1)
    print(f"START {name} init={init:.4f} reach={REACH:.4f} local={LOCAL_WIDTH}", flush=True)
    t0 = time.perf_counter()
    applied = []
    started = time.perf_counter()
    try:
        with optimizer_defaults(recipe, applied), \
             patch.object(vector_tasks, "ParticlePrior", make_scaled_prior(init)):
            result = suite.run_episode(spec, vector_tasks.fixed_policy("cosine"), fixed=True)
        json.dumps(result, allow_nan=False)
    except Exception:
        result = dict(error=traceback.format_exc(), seconds=time.perf_counter() - started)
    payload = dict(
        recipe=recipe.to_dict(),
        candidate=asdict(candidate(recipe)),
        original_spec=deepcopy(job["spec"]),
        spec=spec,
        discriminator_variant=None,
        architecture="simple_mlp_default",
        reference=None,
        reference_sha256=None,
        applied=applied,
        result=result,
    )
    return finalize(name, payload, out, t0, dict(
        kind="mlp", init_std=init, reach=REACH, local_width=LOCAL_WIDTH, means=MEANS))


def main():
    out = ROOT / "runs" / time.strftime("run-%Y%m%d-%H%M%S")
    results = [run_winner(out), run_control(out)]
    (out / "index.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print("WROTE", out, flush=True)
    for r in results:
        print("RESULT", r["arm"], r["status"], "suffix=" + str(r.get("suffix")), flush=True)


if __name__ == "__main__":
    main()

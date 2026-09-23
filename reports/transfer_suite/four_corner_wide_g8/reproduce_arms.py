"""Four-corner wide-gap square: published batchfeat vs MLP; local width unscaled."""
from __future__ import annotations
import gzip, hashlib, json, math, sys, time, traceback
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from particlegan import ParticlePrior, get_recipe

ROOT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from benchmarks.transfer_suite import shared_discriminator_search as architecture
from benchmarks.transfer_suite import shared_batch_feature_research as batchfeat
from benchmarks.transfer_suite import suite, vector_tasks
from benchmarks.transfer_suite.compare_defaults import (
    candidate, effective_spec, ema_verdict, optimizer_defaults, plan,
)
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.shared_variants import architecture_spec

# Corners at (±G, ±G). Nearest-neighbor gap = 2G (=16 when G=8), matching
# wide_gap_islands_x8 / three_island pairwise distance, but on a 2×2 lattice.
# Local σ unscaled — NOT square_gauge_x24 (which scales means and covariances).
G = 8.0
LOCAL_WIDTH = 1.0
BASE_INIT = 0.5
REACH = G * math.sqrt(2.0)  # origin → corner
CONTROL_INIT = BASE_INIT * REACH  # matched prior reach


def corner_means(g=G):
    return [[-g, -g], [-g, g], [g, -g], [g, g]]


def make_job():
    base = next(j for j in plan() if j["spec"]["name"] == "vector_two_broad")
    job = deepcopy(base)
    spec = job["spec"]
    w2 = LOCAL_WIDTH * LOCAL_WIDTH
    means = corner_means()
    spec["means"] = means
    spec["covariances"] = [[[w2, 0.0], [0.0, w2]] for _ in means]
    spec["masses"] = [0.25] * 4
    spec["name"] = f"vector_four_corner_wide_g{int(G)}_local{LOCAL_WIDTH}"
    spec["family"] = "four_corner_wide"
    spec["identifiable"] = True
    spec["importance_reason"] = (
        "Four isotropic islands on a square lattice at (±8,±8) with nearest-neighbor "
        "gap matching wide_gap_islands_x8 (d=16) but local σ unscaled: published absolute "
        "RBF lengths that fit native two_broad saturate across the grid while a "
        "lengthscale-free MLP with matched prior init remains solvable. Distinct from "
        "wide_gap (2 modes on a line), three_island_wide_d16 (3 equilateral), "
        "eight_ring_wide_r8 (8 on a circle), and square_gauge_x24 (isotropic means+cov scale).")
    spec["limitations"] = (
        "One initialization; finite particles. Control does not retune RBF scales; "
        "it drops absolute lengths entirely.")
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
    print(f"START {name} init={BASE_INIT} G={G} local={LOCAL_WIDTH}", flush=True)
    t0 = time.perf_counter()
    with patch.object(architecture, "recipe", lambda: recipe), \
         patch.object(architecture, "constructor", batchfeat.constructor), \
         patch.object(architecture, "variant", batchfeat.variant), \
         patch.object(vector_tasks, "ParticlePrior", make_scaled_prior(BASE_INIT)):
        payload = architecture.episode(job, deepcopy(card))
    return finalize(name, payload, out, t0, dict(
        kind="batchfeat", init_std=BASE_INIT, G=G, local_width=LOCAL_WIDTH, reach=REACH))


def run_control(out):
    name = "control_mlp_matched_init"
    out.mkdir(parents=True, exist_ok=True)
    job = make_job()
    recipe = get_recipe(**json.loads((ROOT / "recipe.json").read_text())["overrides"]).replace(name="shared_c6")
    spec = effective_spec(job["spec"], recipe)
    init = CONTROL_INIT
    torch.set_num_threads(1)
    print(f"START {name} init={init} G={G} local={LOCAL_WIDTH}", flush=True)
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
        kind="mlp", init_std=init, G=G, local_width=LOCAL_WIDTH, reach=REACH))


def main():
    out = ROOT
    results = [run_winner(out), run_control(out)]
    (out / "index.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print("WROTE", out, flush=True)
    for r in results:
        print("RESULT", r["arm"], r["status"], "suffix=" + str(r.get("suffix")), flush=True)


if __name__ == "__main__":
    main()

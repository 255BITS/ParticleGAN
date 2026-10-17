"""Declare and check the finite BCAP optimizer/loss/retention study; no training."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.queue import Queue
from experiments.forge.search_space import compile_space, plan_compilation

STUDY = "bcap-tier2-search-v1"
OUTPUT = ROOT / "reports/forge/bcap-tier2-search"
DEFINITION = ROOT / "configs/forge/search-spaces" / (STUDY + ".json")
MANIFEST = OUTPUT / "compiled.json"


def write_once(path, value):
    if path.exists():
        if stable_hash(read_json(path)) != stable_hash(value):
            raise ValueError(f"Immutable declaration differs: {path}")
    else:
        atomic_json(path, value)


def settings(*, lr=.012, d=1.5, prior=2.5, smoothing=1e-3,
             coeff=1., cap=1., floor=1., start=.6, **extra):
    return {"lr": lr, "d_lr_mult": d, "prior_lr_mult": prior,
            "optimizer_smoothing": smoothing, "reg_coeff": coeff,
            "reg_kappa": cap, "lr_floor": floor, "network_lr_floor": floor,
            "lr_anneal_start": start, **extra}


def declare():
    losses = ("relativistic", "non_saturating", "hinge", "wasserstein", "least_squares")
    convolution = read_json(ROOT / "configs/forge/ideas/bcap-dualnorm-convolution-v1.json")
    roster = []

    def add(base, family, points):
        # A complete finite population gives each category explicit coverage.
        points = [{**point, "reg_every": cadence} for point in points for cadence in (1, 4)]
        roster.append({"base_candidate": base, "trainer_family": family,
                       "parameters": {"configuration": {"kind": "choice", "values": points}}})

    dualnorm = [
        settings(smoothing=1e-5),  # Matched current-source/convolution control.
        settings(),
        settings(smoothing=1e-2),
        settings(floor=.1, start=0.),
        settings(smoothing=1e-5, floor=.2, start=0.),
        settings(floor=.3, start=.25),
        settings(coeff=4.),
        settings(cap=.5),
        settings(coeff=4., cap=.5, floor=.2, start=0.),
        settings(lr=.008, floor=.3, start=0.),
        settings(lr=.016, d=.75, prior=1.875, floor=.2, start=0.),
        settings(d=.75, prior=1.25, coeff=4.),
    ]
    add(convolution["id"], "bcap-dualnorm", dualnorm)
    for loss in losses[1:]:
        base = "bcap-dualnorm-conv-" + loss.replace("_", "-") + "-v1"
        candidate = deepcopy(convolution)
        candidate.update(id=base, parent=convolution["id"], api_changes=[],
            guide="reports/forge/bcap-tier2-search/STUDY.md",
            changed_factors=[f"Recipe.loss={loss}; public two-stream joint objective",
                             "Retain positive BCAP and per-offset smoothed DualNorm"],
            mechanism_rationale="Combine the existing public adversarial loss with the enabled "
                                "BCAP convolution optimizer. No new training loop or host adaptation.",
            prior_art=[convolution["id"], "reports/forge/bcap-search-options/README.md"])
        candidate["recipe_overrides"]["loss"] = loss
        write_once(ROOT / "configs/forge/ideas" / (base + ".json"), candidate)
        add(base, "bcap-dualnorm", [settings(), settings(coeff=4., cap=.5, floor=.2, start=0.)])

    for loss in losses:
        base = ("bcap-pure-adam-v2" if loss == "relativistic"
                else "bcap-pure-" + loss.replace("_", "-") + "-joint-v2")
        points = [settings(lr=.00425, d=1., prior=2., coeff=4., cap=.5, floor=.2, start=0., betas=[0., .999]),
                  settings(lr=.006, d=.75, prior=1.5, floor=.2, start=0., betas=[0., .9])]
        for point in points:
            point.pop("optimizer_smoothing")  # Inactive on Adam; never admit it as an axis.
        add(base, "bcap-pure", points)

    other = [
        ("bcap-sgda-v1", "bcap-sgda", .08, 1.5, 2.),
        ("bcap-nsgda-global-v1", "bcap-nsgda-global", .02, 1., 1.5),
        ("bcap-nsgda-layer-v1", "bcap-nsgda-layer", .006, 1., 2.),
        ("bcap-ada-nsgda-v1", "bcap-ada-nsgda", .006, 1., 2.),
        ("bcap-dualnorm-d-only-v1", "bcap-dualnorm-d-only", .012, 1.5, 2.),
        ("bcap-particle-rownorm-only-v1", "bcap-particle-rownorm-only", .012, 1., 2.),
    ]
    for base, family, lr, d, prior in other:
        point = settings(lr=lr, d=d, prior=prior, coeff=4., cap=.5, floor=.2, start=0.)
        point.pop("optimizer_smoothing")
        if family == "bcap-ada-nsgda":
            point["betas"] = [0., .9]
        add(base, family, [point])

    definition = {"schema": "forge_search_space_v1", "id": STUDY, "samples": 72,
        "hypothesis": "One global positive-BCAP recipe can preserve all six acquisition gates "
                      "and improve sustained Tier2 coverage by reducing terminal player motion, "
                      "adjusting critic cap/strength/cadence, or changing the public optimizer/loss. "
                      "Success requires 6/6 Tier1 and at least 10/21 Tier2; select the complete "
                      "recipe by required PASS counts by tier, then content hash.",
        "rationale": "User-authorized 72 complete configurations, all BCAP-on. Frozen full "
                     "population covers all eight optimizer families and five losses on Adam "
                     "and convolution-enabled DualNorm, with extra relativistic DualNorm "
                     "retention hypotheses. Two explicit penalty cadences (1,4) double 36 "
                     "declared recipe points. This is a scoped subset of the 40 optimizer/loss "
                     "pairs, not an exhaustive numerical search. See STUDY.md for predictions, "
                     "falsifiers, source identities and stopping. Seed0; fixed task architecture, "
                     "data/prior/sampling/initialization/RNG/horizon/cadence/gates. No scientific "
                     "retry, adaptive expansion, seed experiment, Tier3 or default adoption. "
                     "Every Tier1 survivor independently finishes all runnable Tier2 peers. "
                     "Evaluate all candidates only after the single unattended drain completes.",
        "protocol": "screening", "view": "discriminator_stability", "execution_backend": "cuda",
        "cuda_model": "NVIDIA RTX A6000", "tuning_through_tier": 2,
        "campaign": {"schema_version": 1, "id": STUDY, "budget_seconds": 72 * 43020,
            "candidate_budget_seconds": 43020, "accept_shared_cost_transfer": False,
            "purpose": "Finite 72-recipe BCAP search through Tier2; two A6000 workers plus "
                       "the coordinator's CPU slot (at most three workers); no retries or expansion."},
        "candidates": roster}
    assert sum(len(row["parameters"]["configuration"]["values"]) for row in roster) == 72
    write_once(DEFINITION, definition)
    return definition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    args = parser.parse_args()
    queue_root = args.queue_root.resolve()
    queue = Queue(queue_root, report_root=ROOT / "reports/forge", on_completion=None)
    before = stable_hash(queue.inspect())
    definition = declare()
    manifest = compile_space(ROOT, queue_root, definition, queue=queue)
    assert manifest == compile_space(ROOT, queue_root, definition, queue=queue)
    write_once(MANIFEST, manifest)
    plan = plan_compilation(ROOT, queue_root, manifest, queue=queue)
    assert len(plan["trials"]) == 72 and not plan["unsampled_categories"]
    assert plan["declared_worst_case_seconds"] == 72 * 43020
    assert all(t["submission_status"] == "READY" and not t["submission_blockers"] for t in plan["trials"])
    assert len({t["configuration_id"] for t in plan["trials"]}) == 72
    source = {t["source_digest"] for t in plan["trials"]}
    assert len(source) == 1
    for trial in plan["trials"]:
        recipe = trial["resolved_recipe"]
        assert recipe["reg_arm"] == "b_cap" and recipe["reg_coeff"] > 0 and recipe["reg_kappa"] > 0
        assert recipe["optimizer_family"] != "formulation"
        assert recipe["reg_every"] in (1, 4)
        assert all(recipe[key] == 0 for key in ("d_guard_ratio", "reg_anchor_weight", "latent_damping_max_rate",
                                                "input_noise_std", "output_noise_std", "ema_decay"))
        counts = Counter(t["qualification_tier"] for t in trial["tasks"] if t["importance"] == "required")
        assert counts == {1: 6, 2: 21, 3: 2}
    assert before == stable_hash(queue.inspect()), "Preparation must not submit or run anything"
    for trial in plan["trials"]:
        write_once(ROOT / "configs/forge/configurations" / (trial["candidate_id"] + ".json"), trial["declaration"])
    atomic_json(OUTPUT / "plan.json", {"study_id": STUDY, "manifest_hash": manifest["manifest_hash"],
        "definition_hash": manifest["definition_hash"], "source_digest": next(iter(source)),
        "population": manifest["population"], "trials": len(plan["trials"]),
        "categories": len(definition["candidates"]), "unsampled_categories": [],
        "optimizer_counts": dict(Counter(t["resolved_recipe"]["optimizer_family"] for t in plan["trials"])),
        "loss_counts": dict(Counter(t["resolved_recipe"]["loss"] for t in plan["trials"])),
        "required_counts": [6, 21, 2], "tuning_through_tier": 2,
        "candidate_budget_seconds": 43020, "campaign_budget_seconds": 72 * 43020,
        "tier1_reservation_seconds": 2520, "tier2_reservation_seconds": 40500,
        "all_ready": True, "deterministic_compilation": True, "all_bcap_on": True,
        "queue_submissions_added": 0, "scientific_updates_added": 0,
        "default_adoption": False,
        "configurations": [{"candidate_id": t["candidate_id"], "configuration_id": t["configuration_id"],
                            "settings": t["settings"], "optimizer": t["resolved_recipe"]["optimizer_family"],
                            "loss": t["resolved_recipe"]["loss"]} for t in plan["trials"]]})
    atomic_json(ROOT / "runs/software/bcap-tier2-search/full-plan.json", plan)
    print({"all_ready": True, "configurations": 72, "categories": len(definition["candidates"]),
           "reserved_ceiling_seconds": 72 * 43020, "training_launched": False}, flush=True)


if __name__ == "__main__":
    main()

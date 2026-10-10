"""Prepare a separate finite overnight extension without changing the first 72."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.queue import Queue
from experiments.forge.search_space import compile_space, plan_compilation

STUDY = "bcap-overnight-search-v1"
OUTPUT = ROOT / "reports/forge/bcap-tier2-search/overnight"


def helper():
    spec = importlib.util.spec_from_file_location("bcap_preparation", ROOT / "reports/forge/bcap-tier2-search/prepare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    args = parser.parse_args()
    prep = helper()
    original = read_json(ROOT / "reports/forge/bcap-tier2-search/compiled.json")
    base = read_json(ROOT / "configs/forge/ideas/bcap-dualnorm-convolution-v1.json")
    momentum = deepcopy(base)
    momentum.update(id="bcap-dualnorm-conv-momentum-v1", parent=base["id"], api_changes=[],
        guide="reports/forge/bcap-tier2-search/overnight/STUDY.md",
        changed_factors=["Enable positive network gradient momentum before the existing smoothed polar update",
                         "Keep sampled-prior row momentum zero and per-offset convolution enabled"],
        mechanism_rationale="Existing public DualNorm network momentum plus positive smoothing/convolution; "
                            "a separate structural base for activation, not a numerical crossing from zero.",
        prior_art=["reports/forge/dualnorm-pacing-v2/README.md", "reports/forge/bcap-search-options/README.md"])
    momentum["recipe_overrides"]["optimizer_momentum"] = .5
    prep.write_once(ROOT / "configs/forge/ideas" / (momentum["id"] + ".json"), momentum)
    zero_points = [prep.settings(smoothing=scale, d=d, prior=prior, coeff=coeff, floor=.15, start=0., reg_every=1)
                   for scale in (.0003, .003, .03) for d, prior in ((.5, 2.5), (2., 1.25)) for coeff in (1., 4.)]
    momentum_points = [prep.settings(smoothing=scale, floor=floor, start=0., reg_every=1, optimizer_momentum=mu)
                       for scale in (.0003, .003) for floor in (.15, .5, 1.) for mu in (.5, .9)]
    roster = [{"base_candidate": candidate, "trainer_family": "bcap-dualnorm",
               "parameters": {"configuration": {"kind": "choice", "values": points}}}
              for candidate, points in ((base["id"], zero_points), (momentum["id"], momentum_points))]
    definition = {"schema": "forge_search_space_v1", "id": STUDY, "samples": 24,
        "hypothesis": "Additional fixed positive smoothing strengths, player-rate ratios and enabled "
                      "network momentum may improve BCAP retention within the existing Tier1/Tier2 gates. "
                      "Declare all 24 complete recipes before launch; preserve the original 72-candidate "
                      "study and evaluate the matched 96-recipe union by the same frozen whole-row objective.",
        "rationale": "User asked to keep both GPUs busy overnight toward 06:00 America/Denver. "
                     "Runtime/accounting estimates alone motivate the extra capacity. The domain is "
                     "chosen from public supported knobs and archived pacing/smoothing evidence, without "
                     "ranking or adapting to the first batch's scientific results. All BCAP-on; seed0; "
                     "unchanged architecture/data/prior/initialization/sampling/horizon/evaluation/gates. "
                     "Separate immutable 24-candidate campaign starts after the first campaign completes. "
                     "Stop after its finite population; no scientific retries, Tier3 or default adoption.",
        "protocol": "screening", "view": "discriminator_stability", "execution_backend": "cuda",
        "cuda_model": "NVIDIA RTX A6000", "tuning_through_tier": 2,
        "campaign": {"schema_version": 1, "id": STUDY, "budget_seconds": 24 * 43020,
            "candidate_budget_seconds": 43020, "accept_shared_cost_transfer": False,
            "purpose": "Separate finite overnight BCAP extension; both GPUs, one worker each; no retries."},
        "candidates": roster}
    prep.write_once(ROOT / "configs/forge/search-spaces" / (STUDY + ".json"), definition)
    queue = Queue(args.queue_root.resolve(), report_root=ROOT / "reports/forge", on_completion=None)
    before = stable_hash(queue.inspect())
    manifest = compile_space(ROOT, queue.root, definition, queue=queue)
    assert manifest == compile_space(ROOT, queue.root, definition, queue=queue)
    plan = plan_compilation(ROOT, queue.root, manifest, queue=queue)
    assert len(plan["trials"]) == 24 and not plan["unsampled_categories"]
    assert all(t["submission_status"] == "READY" and not t["submission_blockers"] for t in plan["trials"])
    assert {t["source_digest"] for t in plan["trials"]} == {original["bindings"][0]["source_digest"]}
    assert stable_hash(plan["trials"][0]["runtime_cohort"]) == stable_hash(original["bindings"][0]["runtime_cohort"])
    previous_ids = {row["configuration_id"] for row in read_json(ROOT / "reports/forge/bcap-tier2-search/plan.json")["configurations"]}
    assert previous_ids.isdisjoint(t["configuration_id"] for t in plan["trials"])
    for trial in plan["trials"]:
        recipe = trial["resolved_recipe"]
        assert recipe["reg_arm"] == "b_cap" and recipe["reg_coeff"] > 0 and recipe["reg_kappa"] > 0
        assert recipe["optimizer_convolution"] == "per_offset" and recipe["optimizer_family"] == "dualnorm"
        prep.write_once(ROOT / "configs/forge/configurations" / (trial["candidate_id"] + ".json"), trial["declaration"])
    assert before == stable_hash(queue.inspect())
    assert plan["declared_worst_case_seconds"] == 24 * 43020
    prep.write_once(OUTPUT / "compiled.json", manifest)
    atomic_json(OUTPUT / "plan.json", {"study_id": STUDY, "configurations": 24, "manifest_hash": manifest["manifest_hash"],
        "source_digest": plan["trials"][0]["source_digest"], "all_ready": True, "all_bcap_on": True,
        "no_overlap_with_original_72": True, "matched_original_source_runtime": True,
        "hypothesis_basis": "Archived evidence and remaining runtime; no interim scientific selection",
        "queue_submissions_added": 0, "candidate_budget_seconds": 43020, "campaign_budget_seconds": 24 * 43020,
        "configurations_detail": [{"candidate_id": t["candidate_id"], "configuration_id": t["configuration_id"],
                                   "settings": t["settings"]} for t in plan["trials"]]})
    atomic_json(ROOT / "runs/software/bcap-overnight-search/full-plan.json", plan)
    print({"configurations": 24, "all_ready": True, "all_bcap_on": True, "original_study_changed": False}, flush=True)


if __name__ == "__main__":
    main()

"""Declare the joint-loss correction; freeze its cohort only on the target host.

Declaration preparation never trains or queues work. Executed v1 cards, plans,
requests and scientific outcomes remain immutable.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.configuration_search import materialize_search, plan_search
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.queue import Queue
from experiments.forge.sources import compute_profile
from experiments.forge.views import load_tasks, load_view

REPORT = ROOT / "reports/forge/pure-bcap"
ROUND = "pure-bcap-joint-loss-repair-v2"
FAMILY = "bcap-pure"
LOSSES = ("non_saturating", "hinge", "wasserstein", "least_squares")
RATES = (.0010625, .00425)
GPU_MODEL = "NVIDIA RTX A6000"


def base_id(loss):
    return "bcap-pure-adam-v2" if loss == "relativistic" else "bcap-pure-" + loss.replace("_", "-") + "-joint-v2"


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Declaration differs: {path}; create a successor ID")
    else:
        atomic_json(path, value)


def require_gpus():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise ValueError("GPU 0 and GPU 1 must be available before freezing or running the correction")
    if any(torch.cuda.get_device_name(index) != GPU_MODEL for index in (0, 1)):
        raise ValueError("The declared correction cohort requires two NVIDIA RTX A6000 GPUs")
    if compute_profile("cuda", GPU_MODEL).get("availability") == "unavailable":
        raise ValueError("CUDA driver provenance is unavailable; do not freeze a substitute cohort")


def declare():
    tasks = load_tasks(ROOT)
    view = load_view(ROOT, "discriminator_stability")
    tier1 = [row for row in view["assignments"] if row["qualification_tier"] == 1]
    cap = sum(tasks[row["task"]]["resources"]["timeout_seconds"] for row in tier1)
    if cap != 2520 or len(tier1) != 7 or sum(row["importance"] == "required" for row in tier1) != 6:
        raise ValueError("Review a changed Tier 1 scope or allowance before preparing this finite correction")
    registry_path = ROOT / "configs/forge/trainer-families.json"
    registry = read_json(registry_path)
    family = next(row for row in registry["families"] if row["id"] == FAMILY)
    names = [base_id(loss) for loss in ("relativistic", *LOSSES)]
    family["canonical_candidate"] = base_id("relativistic")
    family["candidates"] = list(dict.fromkeys([*family["candidates"], *names]))
    atomic_json(registry_path, registry)
    specs = []
    for loss in ("relativistic", *LOSSES):
        previous = "bcap-pure-adam-v1" if loss == "relativistic" else "bcap-pure-" + loss.replace("_", "-") + "-v1"
        card = {
            "schema_version": 3, "id": base_id(loss), "parent": previous,
            "recipe_preset": "bcap", "recipe_overrides": {"loss": loss},
            "requires_capabilities": ["learned_locations", "named_rng"],
            "api_version": "forge-api-v1", "execution_path": "public_trainer",
            "api_changes": [] if loss == "relativistic" else ["GANLoss.joint_g_loss supplies both generator and encoder adversarial streams on joint hosts"],
            "claim_contract": {"schedule": "scheduled", "scoring_weights": "live", "sampling_law": "task_declared"},
            "changed_factors": ["Reusable pure BCAP recipe with constant Adam rates and moments", f"Adversarial loss: {loss}", "Explicit two-stream generator/encoder objective on joint hosts"],
            "mechanism_class": "structural",
            "mechanism_rationale": "Native Adam and fixed real/fake BCAP only. Task-owned auxiliary objectives and conditions remain in tasks. The relativistic update is unchanged; unpaired joint objectives reverse both critic streams.",
            "prior_art": [previous], "guide": "EXPERIMENTATION.md",
        }
        immutable_json(ROOT / "configs/forge/ideas" / (card["id"] + ".json"), card)
        if loss == "relativistic":
            continue  # The original two whole configurations are complete; never rerun them for a merge.
        study = "pure-bcap-" + loss.replace("_", "-") + "-joint-rates-v2"
        spec = {
            "schema_version": 2, "id": study, "trainer_family": FAMILY,
            "base_candidate": card["id"], "grid": {"lr": list(RATES)},
            "tuning_through_tier": 1, "view": "discriminator_stability",
            "execution_backend": "cuda", "cuda_model": GPU_MODEL, "protocol": "screening",
            "campaign": {"id": ROUND, "budget_seconds": cap * 8,
                         "candidate_budget_seconds": cap, "accept_shared_cost_transfer": False},
            "hypothesis": f"Pure BCAP with {loss} and a complete joint generator/encoder objective may acquire the unchanged Tier 1 distributions at one of two constant global rates.",
            "rationale": "Correct the omitted encoder term discovered in v1. Evaluate eight complete source-bound recipes, including all seven Tier 1 peers, without grafting prior scalar measurements into the corrected source. Preserve all original failures and paid cancellations. No loss grid axis, seed trial, criteria change, automatic continuation or default adoption.",
            "guide": "EXPERIMENTATION.md",
        }
        path = ROOT / "configs/forge/searches" / (study + ".json")
        immutable_json(path, spec)
        materialize_search(ROOT, spec)
        specs.append(path.relative_to(ROOT).as_posix())
    return view, tier1, cap, specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="Freeze reviewed execution bindings on the actual two-GPU host")
    args = parser.parse_args()
    path = REPORT / "repair-plans.json"
    if args.freeze:
        if path.exists():
            raise ValueError("The correction round is already frozen; use run.py --plan repair-plans.json")
        require_gpus()
    view, tier1, cap, specs = declare()
    if not args.freeze:
        print("Declared five reusable v3 recipes and four v2 searches; eight corrected trials. No cohort frozen, queue admission or training.", flush=True)
        return
    original = Queue(ROOT / "runs/forge/pure-bcap-losses-v1/queue", on_completion=None).inspect()
    paid = sum(charge["seconds"] for charge in original["charges"] if charge["owner"]["campaign"] == "pure-bcap-losses-v1")
    if paid <= 0 or paid + cap * 8 > 25200:
        raise ValueError("Restore original paid accounting; the correction must fit the original aggregate ceiling")
    queue_root = ROOT / "runs/forge" / ROUND / "queue"
    plans = [plan_search(ROOT, queue_root, ROOT / spec) for spec in specs]
    if not all(len(plan["trials"]) == 2 and all(trial["submission_status"] == "READY" for trial in plan["trials"]) for plan in plans):
        raise ValueError("Review correction blockers before freezing")
    if len({plan["source_digest"] for plan in plans}) != 1:
        raise ValueError("All corrected recipes must bind one reviewed source cohort")
    immutable_json(path, {
        "schema_version": 1, "round": ROUND, "specs": specs, "configuration_count": 8,
        "spec_sha256": {spec: stable_hash(read_json(ROOT / spec)) for spec in specs},
        "candidate_cap_seconds": cap, "campaign_cap_seconds": cap * 8,
        "original_paid_wall_seconds": paid, "aggregate_cap_seconds": 25200,
        "source_digest": plans[0]["source_digest"], "protocol_sha256": plans[0]["protocol_hash"],
        "view_revision": view["revision"], "task_ids": sorted(row["task"] for row in tier1),
        "required_task_ids": sorted(row["task"] for row in tier1 if row["importance"] == "required"),
        "trials": [{"candidate_id": trial["candidate_id"], "candidate_revision": trial["candidate_revision"],
                    "settings": trial["settings"], "study": plan["study_id"],
                    "scientific_signature": trial["scientific_signature"],
                    "runtime_cohort_sha256": stable_hash(trial["runtime_cohort"])}
                   for plan in plans for trial in plan["trials"]],
    })
    print(f"Frozen eight corrected candidates, {cap * 8} seconds maximum; original {paid:.3f} seconds retained. No training.", flush=True)


if __name__ == "__main__":
    main()

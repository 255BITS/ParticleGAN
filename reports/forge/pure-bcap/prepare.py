"""Declare and review the finite Pure BCAP loss/rate round; never train."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.configuration_search import materialize_search, plan_search
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.decision_contracts import scaffold
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue
from experiments.forge.views import load_tasks, load_view

ROUND = "pure-bcap-losses-v1"
REPORT = ROOT / "reports/forge/pure-bcap"
RUNS = ROOT / "runs/forge" / ROUND
LOSSES = ("relativistic", "non_saturating", "hinge", "wasserstein", "least_squares")
RATES = (.0010625, .00425)
CONTROL = "k3p-bcap-matched-v1"
FAMILY = "bcap-pure"


def candidate_id(loss):
    return "bcap-pure-adam-v1" if loss == "relativistic" else "bcap-pure-" + loss.replace("_", "-") + "-v1"


def main():
    if (REPORT / "plans.json").exists():
        raise ValueError("The round is already frozen; preparation cannot rewrite it")
    if Queue(RUNS / "queue", on_completion=None).inspect().get("campaigns", {}).get(ROUND):
        raise ValueError("The admitted round is immutable")
    view = load_view(ROOT, "discriminator_stability")
    tasks = load_tasks(ROOT)
    tier1 = [a for a in view["assignments"] if a["qualification_tier"] == 1]
    cap = sum(tasks[a["task"]]["resources"]["timeout_seconds"] for a in tier1)
    campaign_cap = cap * len(LOSSES) * len(RATES)
    assert cap == 2520 and campaign_cap == 25200
    motivation = "reports/forge/configuration-search/bcap-tier1-refresh-v1.json"
    prior = [{"path": motivation, "sha256": file_hash(ROOT / motivation), "selector": [],
              "identity": {"study_id": "bcap-tier1-refresh-v1"}, "use": "motivation_only"}]
    registry_path = ROOT / "configs/forge/trainer-families.json"
    registry = read_json(registry_path)
    family = {"id": FAMILY, "label": "Pure BCAP",
        "canonical_candidate": candidate_id("relativistic"),
        "candidates": [candidate_id(loss) for loss in LOSSES],
        "rationale": "Native Adam, fixed real/fake BCAP, constant rates and moments, selectable adversarial loss. No optimizer interventions or additive training noise. Historical K3P-derived BCAP remains a separate family."}
    existing = [f for f in registry["families"] if f["id"] == FAMILY]
    if existing and existing != [family]:
        raise ValueError("The registered family differs from this round")
    if not existing:
        registry["families"].append(family)
        atomic_json(registry_path, registry)
    protocol = read_json(ROOT / "configs/forge/protocols/screening.json")
    specs = []
    for loss in LOSSES:
        name = candidate_id(loss)
        idea = {"schema_version": 2, "id": name, "parent": CONTROL,
            "goal": "discriminator_stability", "recipe_preset": "bcap",
            "recipe_overrides": {"loss": loss},
            "prior": {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True},
            "requires_capabilities": ["learned_locations", "named_rng"],
            "api_version": "forge-api-v1", "api_changes": [], "execution_path": "public_trainer",
            "claim_contract": {"schedule": "scheduled", "scoring_weights": "live", "sampling_law": "task_declared"},
            "hypothesis": f"Pure BCAP with {loss} loss and constant Adam rates may acquire Tier 1 distributions without optimizer interventions.",
            "changed_factors": ["Native Adam and fixed BCAP only", "Constant rates and moments; zero additive training noise", f"Adversarial loss: {loss}"],
            "mechanism_class": "structural", "mechanism_rationale": "A simple reusable baseline; loss choice is a structural objective change, not a numeric search axis. Several removed interventions prevent single-factor causal attribution.",
            "lifecycle": "proposed", "prior_art": [CONTROL], "guide": "EXPERIMENTATION.md",
            "decision_contract": scaffold(CONTROL, "discriminator_stability")}
        atomic_json(ROOT / "configs/forge/ideas" / f"{name}.json", idea)
        study = "pure-bcap-" + loss.replace("_", "-") + "-rates-v1"
        spec = {"schema_version": 1, "id": study, "trainer_family": FAMILY,
            "base_candidate": name, "grid": {"lr": list(RATES)}, "tuning_through_tier": 1,
            "view": "discriminator_stability", "execution_backend": "cuda", "cuda_model": "NVIDIA RTX A6000",
            "protocol": "screening", "protocol_hash": stable_hash(protocol),
            "campaign": {"id": ROUND, "budget_seconds": campaign_cap,
                         "candidate_budget_seconds": cap, "accept_shared_cost_transfer": False},
            "hypothesis": idea["hypothesis"],
            "rationale": "Two constant global rates at one fixed loss. Same task-owned prior, initialization, architecture, duration and gates. Complete all runnable Tier 1 peers, preserve CPU behavioral objectives, stop before higher tiers. Five loss studies share one finite campaign. No seed trials, automatic tuning, criteria edits or default adoption.",
            "guide": "EXPERIMENTATION.md"}
        spec_path = ROOT / "configs/forge/searches" / f"{study}.json"
        atomic_json(spec_path, spec)
        specs.append(spec_path.relative_to(ROOT).as_posix())
        for path in materialize_search(ROOT, spec):
            card = read_json(path)
            request = resolve_idea(ROOT, card["id"], through_tier=1, execution_backend="cuda", cuda_model="NVIDIA RTX A6000")
            expected = request["decision_review"]["expected"]
            contract = deepcopy(card["decision_contract"])
            contract.update(status="ready", prior_evidence=prior,
                candidate_binding_sha256=expected["candidate_binding_sha256"], substantive_delta=expected["substantive_delta"],
                prediction={"task_id": "gaussian1d_acquisition", "metric": "cdf_ks", "op": "<=", "threshold": .05, "phase": "final"},
                falsifier={"task_id": "gaussian1d_acquisition", "metric": "cdf_ks", "op": ">", "threshold": .05, "phase": "final"},
                competing_explanation="The old BCAP evidence uses K3P optimizers, schedules and noise. It motivates this baseline but is not a matched causal control. Finite task budgets and rate endpoints may fail; one passing endpoint is insufficient for the unchanged sustained task gate. Behavioral objectives belong to the tests.")
            contract["control"].update(binding_sha256=expected["control_binding_sha256"], task_map=expected["task_map"])
            contract["scope"].update(task_ids=expected["task_ids"], candidate_budget_seconds=cap, campaign_budget_seconds=campaign_cap,
                **{key: expected[key] for key in ("protocol_sha256", "source_digest", "execution_backend", "runtime_cohort_sha256", "jobs_sha256")})
            card["decision_contract"] = contract
            atomic_json(path, card)
    plans = [plan_search(ROOT, RUNS / "queue", ROOT / spec) for spec in specs]
    assert all(len(p["trials"]) == 2 and all(t["submission_status"] == "READY" for t in p["trials"]) for p in plans)
    assert len({p["source_digest"] for p in plans}) == 1
    atomic_json(REPORT / "plans.json", {"schema_version": 1, "round": ROUND, "specs": specs,
        "configuration_count": 10, "candidate_cap_seconds": cap, "campaign_cap_seconds": campaign_cap,
        "source_digest": plans[0]["source_digest"], "protocol_sha256": stable_hash(protocol),
        "view_revision": view["revision"], "task_ids": sorted(a["task"] for a in tier1),
        "required_task_ids": sorted(a["task"] for a in tier1 if a["importance"] == "required"),
        "trials": [{"candidate_id": t["candidate_id"], "candidate_revision": t["candidate_revision"],
                    "settings": t["settings"], "study": p["study_id"]} for p in plans for t in p["trials"]]})
    print(f"Reviewed 10 candidates; {cap} seconds each, {campaign_cap} seconds maximum, no training", flush=True)


if __name__ == "__main__":
    main()

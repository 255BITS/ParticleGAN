"""Prepare phase two: existing input-noise activation and eight global recipes.

This creates a new structural K3P successor, then uses the ordinary strict
configuration-search API for its positive numeric axes. The base/control is
binding provenance only, never submitted for a separate paid run. Preparation
launches no workers and refuses to mutate an admitted or registered study.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.configuration_search import materialize_search, plan_search
from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue
from experiments.forge.techniques import technique_signature


STUDY = "k3p-global-input-noise-tier1-v1"
BASE = "k3p-global-input-noise-v1"
CLEAN = "k3p-global-repair-v1"
REPORT = ROOT / "reports/forge/k3p-global-tier1-v2"
RUNS = ROOT / f"runs/forge/{STUDY}"
SPEC = ROOT / f"configs/forge/searches/{STUDY}.json"
BASE_PATH = ROOT / f"configs/forge/ideas/{BASE}.json"
TASKS = {"two_pole", "unused_token_hold", "ae_gan_hold",
         "ring16_acquisition", "five_word_joint_acquisition"}
AXES = {"lr", "prior_lr_mult", "reg_coeff"}
CANDIDATE_CAP = 2100
CAMPAIGN_CAP = 16800

HYPOTHESIS = (
    "Restoring the existing symmetric critic input-noise schedule, while "
    "retaining clean generator outputs and a full network horizon, may avoid "
    "the low-force two_pole plateaus observed across twelve clean K3P recipes. "
    "Early independent critic input perturbations can break the deterministic "
    "symmetry of the initially identical direct particles. "
    "Existing positive critic coefficients and coupled global rates test "
    "whether one complete recipe also passes ring and words. Ordinary "
    "prerequisites stop a candidate on its first required failure."
)
LIMITS = (
    "Phase two is one separately bounded round of eight complete global recipes, "
    "2100 seconds each and 16800 total. Input noise is a fixed, explicitly "
    "declared existing-control activation; the numerical search keeps that "
    "technique signature fixed. No new GAN technique, task override, objective, "
    "architecture, seed study, gate change, word-witness qualification import, "
    "unchanged repeat, paid base/control run, higher-tier execution or automatic "
    "further paid round. Output noise remains zero; schedule horizons and "
    "sampling laws are unchanged. Nominal prior rate is fixed, not its realized "
    "trajectory. Prerequisite-gated cells remain UNKNOWN."
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def motivation(path, identity):
    return {"path": path, "sha256": file_hash(ROOT / path), "selector": [],
            "identity": identity, "use": "motivation_only"}


def review(path, control, evidence, *, activation=False):
    idea = json.loads(path.read_text())
    contract = deepcopy(idea["decision_contract"])
    contract["status"] = "draft"
    contract["control"].update(candidate_id=control, task_map={})
    idea["decision_contract"] = contract
    write(path, idea)
    request = resolve_idea(ROOT, idea["id"], through_tier=1,
        execution_backend="cuda", cuda_model="NVIDIA RTX A6000")
    expected = request["decision_review"]["expected"]
    allowed = {"input_noise_std"} if activation else AXES
    assert set(expected["task_ids"]) == TASKS
    assert expected["substantive_delta"]
    assert all(delta["path"][0] == "recipe" and delta["path"][2] in allowed
               for delta in expected["substantive_delta"])
    contract.update(status="ready", prior_evidence=evidence,
        candidate_binding_sha256=expected["candidate_binding_sha256"],
        substantive_delta=expected["substantive_delta"],
        prediction={"task_id": "two_pole", "metric": "mean_abs", "op": ">=",
            "threshold": .3, "phase": "final"},
        falsifier={"task_id": "two_pole", "metric": "mean_abs", "op": "<",
            "threshold": .3, "phase": "final"},
        competing_explanation=(
            "The first clean grid did not isolate noise as the cause of its "
            "movement failures. Input noise can change critic gradients and "
            "optimizer dynamics without yielding sustained movement or a "
            "word bijection. High global rates or insufficient critic "
            "regularization can still fail later tasks. Only this candidate's "
            "new ordinary measurements qualify; historical successes and "
            "unknown cells cannot be combined into a solution."))
    contract["control"].update(binding_sha256=expected["control_binding_sha256"],
        task_map=expected["task_map"])
    contract["scope"].update(view="discriminator_stability", through_tier=1,
        task_ids=expected["task_ids"], max_rounds=1,
        candidate_budget_seconds=CANDIDATE_CAP, campaign_budget_seconds=CAMPAIGN_CAP,
        **{key: expected[key] for key in ("protocol_sha256", "source_digest",
            "execution_backend", "runtime_cohort_sha256", "jobs_sha256")})
    write(path, idea)
    request = resolve_idea(ROOT, idea["id"], through_tier=1,
        execution_backend="cuda", cuda_model="NVIDIA RTX A6000")
    assert request["decision_review"]["status"] == "READY"
    assert not request["preflight_blockers"]
    assert not any(request["tasks"][name]["preflight_blockers"] for name in TASKS)
    write(RUNS / "requests" / f"{idea['id']}.json", request)
    return request, {"declaration": str(path.relative_to(ROOT)),
        "declaration_sha256": file_hash(path),
        "candidate_binding_sha256": expected["candidate_binding_sha256"],
        "control_binding_sha256": expected["control_binding_sha256"],
        "decision_contract_sha256": stable_hash(contract),
        "jobs_sha256": expected["jobs_sha256"],
        "task_bindings": request["decision_review"]["actual_bindings"]["candidate"]["task_identity"]}


def main():
    registered = ROOT / f"reports/forge/configuration-search/{STUDY}.json"
    if registered.exists():
        raise ValueError("Registered study is immutable; use read-only search plan/report")
    state = Queue(RUNS / "queue", on_completion=None).inspect()
    if state.get("campaigns", {}).get(STUDY):
        raise ValueError("Admitted campaign is immutable; preparation cannot refresh its bindings")

    first_path = "reports/forge/configuration-search/k3p-global-tier1-v2.json"
    first = json.loads((ROOT / first_path).read_text())
    assert first["selection"]["selection_complete"]
    assert not first["selection"]["qualified"]
    assert len(first["trials"]) == 12
    assert all(next(task for task in trial["tasks"] if task["task"] == "two_pole")["gate_status"]
               == "FAIL" for trial in first["trials"])
    assert all(task["gate_status"] == "UNKNOWN" for trial in first["trials"]
               for task in trial["tasks"] if task["task"] != "two_pole")
    word_path = "reports/forge/word-root-cause/receipts/k3p-coeff170-cap1.json"
    word = json.loads((ROOT / word_path).read_text())
    assert word["grade"]["gate_status"] == "PASS"
    evidence = [motivation(first_path, {"study_id": first["study_id"],
        "source_digest": first["source_digest"]}),
        motivation(word_path, {"id": word["id"], "candidate_revision": word["candidate_revision"]})]

    clean = json.loads((ROOT / f"configs/forge/ideas/{CLEAN}.json").read_text())
    base = deepcopy(clean)
    base.update(id=BASE, parent=CLEAN, trainer_family="k3p", lifecycle="proposed",
        hypothesis=HYPOTHESIS, changed_factors=["Recipe.input_noise_std=0.5 globally"],
        mechanism_class="structural",
        mechanism_rationale=(
            "Explicit activation of the already implemented positive critic "
            "input-noise control, within the existing K3P family. This crosses "
            "the clean parent's activation boundary and is declared as a "
            "structural successor rather than disguised as a strict numerical "
            "search. No optimizer, penalty kernel or GAN technique is added."),
        prior_art=[first_path, word_path])
    base["recipe_overrides"]["input_noise_std"] = .5
    base["decision_contract"]["status"] = "draft"
    base["decision_contract"]["control"].update(candidate_id=CLEAN, task_map={})
    write(BASE_PATH, base)
    base_request, base_review = review(BASE_PATH, CLEAN, evidence, activation=True)

    protocol = json.loads((ROOT / "configs/forge/protocols/screening.json").read_text())
    spec = {"schema_version": 1, "id": STUDY, "trainer_family": "k3p",
        "base_candidate": BASE,
        "grid": {"coupled_rates": [
            {"lr": lr, "prior_lr_mult": .0012 / lr} for lr in (.006375, .0085)],
            "reg_coeff": [1.0, 10.0, 30.0, 170.0]},
        "tuning_through_tier": 1, "view": "discriminator_stability",
        "execution_backend": "cuda", "cuda_model": "NVIDIA RTX A6000",
        "protocol": "screening", "protocol_hash": stable_hash(protocol),
        "campaign": {"id": STUDY, "budget_seconds": CAMPAIGN_CAP,
            "candidate_budget_seconds": CANDIDATE_CAP, "accept_shared_cost_transfer": False},
        "hypothesis": HYPOTHESIS,
        "rationale": (
            "The first separately frozen clean grid concluded with twelve "
            "two_pole failures, movement .025701 to .141911, all later tasks "
            "unknown. Its finite negative evidence did not identify a single "
            "cause. The zero-initialized clean direct cloud preserves identical "
            "particle inputs and deterministic per-row gradients; early "
            "independent critic perturbations provide an existing symmetry "
            "breaking control. Historical noisy incumbents and a clean word-positive "
            "recipe have different joint bindings and do not establish transfer. "
            "This second independent bounded round fixes the existing input "
            "noise at .5 with its unchanged anneal-end fraction .1, output "
            "noise0/full horizon, D multiplier1.5, and all other mechanisms. "
            "Its strict positive numerical grid varies shared LR and critic "
            "coefficient only, coupled to prior=.0012/LR. " + LIMITS),
        "guide": "EXPERIMENTATION.md"}
    write(SPEC, spec)
    paths = materialize_search(ROOT, spec)
    assert len(paths) == 8
    reviews = {}
    for path in paths:
        idea = json.loads(path.read_text())
        if idea["search_study_id"] != STUDY:
            raise ValueError(f"Cannot mutate historical configuration card: {path.name}")
        _, reviews[idea["id"]] = review(path, BASE, evidence)

    plan = plan_search(ROOT, RUNS / "queue", spec)
    assert len(plan["trials"]) == 8
    assert all(trial["submission_status"] == "READY" for trial in plan["trials"])
    assert plan["declared_worst_case_seconds"] == CAMPAIGN_CAP
    signature = technique_signature(base_request["candidate"]["resolved_recipe"])
    assert all(trial["technique_signature"] == signature for trial in plan["trials"])
    assert all(trial["declared_worst_case_seconds"] == CANDIDATE_CAP for trial in plan["trials"])
    write(RUNS / "plan.json", plan)
    compact = {"schema_version": 1, "scope": "bounded_global_existing_input_noise_activation",
        "phase": 2, "study": STUDY, "spec": str(SPEC.relative_to(ROOT)),
        "spec_sha256": file_hash(SPEC), "spec_semantic_sha256": stable_hash(spec),
        "source_digest": plan["source_digest"],
        "preparation_checkout_commit": base_request["source"]["origin_commit"],
        "scientific_python_executable": sys.executable, "runtime_cohort": plan["runtime_cohort"],
        "policy_fingerprint": plan["policy_fingerprint"], "protocol_sha256": plan["protocol_hash"],
        "view": "discriminator_stability", "through_tier": 1,
        "seed": 0, "gpu": 0, "maximum_global_workers": 2,
        "workers_per_gpu": 1, "automatic_cpu_workers": 1, "cpu_threads": 1,
        "configuration_count": 8, "required_tier1_count": 5, "task_bindings": 40,
        "candidate_ceiling_seconds": CANDIDATE_CAP, "campaign_ceiling_seconds": CAMPAIGN_CAP,
        "nominal_absolute_latent_prior_base_lr": .0012,
        "technique_signature": signature, "qualification_input": False,
        "historical_word_receipts_reused_for_qualification": 0,
        "hypothesis": HYPOTHESIS, "limits": LIMITS,
        "prior_phase": {"study": first["study_id"], "source_digest": first["source_digest"],
            "report": first_path, "report_sha256": file_hash(ROOT / first_path),
            "use": "motivation_only", "qualification_reuse": False},
        "structural_base": {"candidate": BASE, "decision_status": "READY",
            "source_digest": base_request["source"]["digest"],
            "paid_execution_authorized": False, "separate_submission_authorized": False,
            "activation_delta": "input_noise_std 0 to .5; all other public recipe fields fixed",
            "clean_control": CLEAN, **base_review},
        "trials": [{"candidate": trial["candidate_id"],
            "candidate_revision": trial["candidate_revision"],
            "configuration_id": trial["configuration_id"], "settings": trial["settings"],
            "decision_status": trial["submission_status"],
            "scientific_signature": trial["scientific_signature"],
            **reviews[trial["candidate_id"]]} for trial in plan["trials"]]}
    write(REPORT / "plans-input-noise.json", compact)
    print(json.dumps({"study": STUDY, "ready_configurations": 8,
        "task_bindings": 40, "campaign_ceiling_seconds": CAMPAIGN_CAP,
        "source_digest": plan["source_digest"], "spec_sha256": file_hash(SPEC),
        "compact_plans_sha256": file_hash(REPORT / "plans-input-noise.json")}))


if __name__ == "__main__":
    main()

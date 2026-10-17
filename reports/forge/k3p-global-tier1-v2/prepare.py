"""Prepare twelve bounded whole K3P recipes, without enqueueing or training.

Run with the declared scientific Python after shared execution sources settle.
Preparation may be repeated before registration to refresh READY bindings. A
registered study is immutable: use Forge's read-only plan/report commands then.
Full plans and requests stay under ignored runs/; plans.json is compact review
metadata, not a second leaderboard or qualification evidence.
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


STUDY = "k3p-global-tier1-v2"
BASE = "k3p-global-repair-v1"
REPORT = ROOT / "reports/forge/k3p-global-tier1-v2"
RUNS = ROOT / "runs/forge/k3p-global-tier1-v2"
SPEC = ROOT / f"configs/forge/searches/{STUDY}.json"
TASKS = {"two_pole", "unused_token_hold", "ae_gan_hold",
         "ring16_acquisition", "five_word_joint_acquisition"}
AXES = {"lr", "prior_lr_mult", "reg_coeff", "d_lr_mult"}
CANDIDATE_CAP = 2100
CAMPAIGN_CAP = 25200

HYPOTHESIS = (
    "Intermediate positive K3P critic coefficients and a slower existing critic "
    "rate may restore two_pole movement against its unchanged particle_l2 "
    "restoring term while retaining sufficient regularization for the words "
    "joint critic. Each complete global recipe is tested through all five "
    "ordinary Tier 1 tasks, with the first required failure stopping progression."
)
LIMITS = (
    "Twelve complete global configurations in one bounded round. Existing "
    "positive numerical controls only; no new technique, task-specific override, "
    "objective, architecture, seed study, changed gate, historical word witness "
    "import, unchanged repeat, higher-tier execution or automatic paid follow-up. "
    "The nominal latent prior base rate is fixed, not its realized trajectory. "
    "Later tasks after a prerequisite failure remain UNKNOWN."
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def evidence(path, identity):
    source = ROOT / path
    return {"path": path, "sha256": file_hash(source), "selector": [],
            "identity": identity, "use": "motivation_only"}


def main():
    # Never rewrite a declaration after immutable search registration/admission.
    registered = ROOT / f"reports/forge/configuration-search/{STUDY}.json"
    if registered.exists():
        raise ValueError("Registered study is immutable; use read-only search plan/report")
    state = Queue(RUNS / "queue", on_completion=None).inspect()
    if state.get("campaigns", {}).get(STUDY):
        raise ValueError("Admitted campaign is immutable; preparation cannot refresh its bindings")

    protocol = json.loads((ROOT / "configs/forge/protocols/screening.json").read_text())
    spec = {"schema_version": 1, "id": STUDY, "trainer_family": "k3p",
        "base_candidate": BASE,
        "grid": {"coupled_rates": [
            {"lr": lr, "prior_lr_mult": .0012 / lr} for lr in (.00425, .006375)],
            "reg_coeff": [10.0, 30.0, 85.0], "d_lr_mult": [.25, 1.5]},
        "tuning_through_tier": 1, "view": "discriminator_stability",
        "execution_backend": "cuda", "cuda_model": "NVIDIA RTX A6000",
        "protocol": "screening", "protocol_hash": stable_hash(protocol),
        "campaign": {"id": STUDY, "budget_seconds": CAMPAIGN_CAP,
            "candidate_budget_seconds": CANDIDATE_CAP, "accept_shared_cost_transfer": False},
        "hypothesis": HYPOTHESIS,
        "rationale": (
            "Historical clean/full LR .0006 K3P words passed at coefficient170 "
            "but failed at coefficient1. Its global coefficient170 transfers "
            "failed two_pole even at LR .00425/.006375, with small critic slopes "
            "and movement .128044/.122587. The unchanged host includes "
            "particle_l2=.02; saved movement plateaus motivate a force/restoring "
            "term hypothesis, not causal proof. Positive intermediate coefficients "
            "10/30/85 and critic multipliers .25/1.5 test that tradeoff at two "
            "global movement-scale rates. Noise0, full schedule, cap1, optimizer "
            "mechanisms, moments, priors, initialization, budgets and sampling "
            "laws remain fixed. Coupled prior multipliers preserve nominal "
            "latent-table base LR .0012; direct coordinates consume G base LR. "
            + LIMITS),
        "guide": "EXPERIMENTATION.md"}
    write(SPEC, spec)
    paths = materialize_search(ROOT, spec)
    assert len(paths) == 12

    word_path = "reports/forge/word-root-cause/receipts/k3p-coeff170-cap1.json"
    word = json.loads((ROOT / word_path).read_text())
    assert word["grade"]["gate_status"] == "PASS"
    global_path = "reports/forge/family-wide-word-repairs/rates/k3p-lr0.006375.json"
    failed = json.loads((ROOT / global_path).read_text())
    assert failed["tasks"][0]["task_id"] == "two_pole"
    assert failed["tasks"][0]["status"] == "FAIL"
    motivation = [evidence(word_path, {"id": word["id"],
        "candidate_revision": word["candidate_revision"]}),
        evidence(global_path, {"candidate_id": failed["candidate_id"],
            "candidate_revision": failed["candidate_revision"]})]
    reviews = {}
    for path in paths:
        idea = json.loads(path.read_text())
        if idea["search_study_id"] != STUDY:
            raise ValueError(f"Cannot mutate historical configuration card: {path.name}")
        contract = deepcopy(idea["decision_contract"])
        contract["status"] = "draft"
        contract["control"].update(candidate_id=BASE, task_map={})
        idea["decision_contract"] = contract
        write(path, idea)
        request = resolve_idea(ROOT, idea["id"], through_tier=1,
            execution_backend="cuda", cuda_model="NVIDIA RTX A6000")
        expected = request["decision_review"]["expected"]
        assert set(expected["task_ids"]) == TASKS
        assert expected["substantive_delta"]
        assert all(delta["path"][0] == "recipe" and delta["path"][2] in AXES
                   for delta in expected["substantive_delta"])
        contract.update(status="ready", prior_evidence=motivation,
            candidate_binding_sha256=expected["candidate_binding_sha256"],
            substantive_delta=expected["substantive_delta"],
            prediction={"task_id": "two_pole", "metric": "mean_abs", "op": ">=",
                "threshold": .3, "phase": "final"},
            falsifier={"task_id": "two_pole", "metric": "mean_abs", "op": "<",
                "threshold": .3, "phase": "final"},
            competing_explanation=(
                "The recorded movement plateau does not isolate critic coefficient "
                "or noise as its cause. Lower critic regularization/rate may still "
                "fail movement, or lose word bijection and ring quality. A finite "
                "negative grid does not establish family impossibility. Only new "
                "ordinary measurements qualify this complete candidate; historical "
                "word passes cannot fill prerequisite-gated UNKNOWN cells."))
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
        assert not request["preflight_blockers"], request["preflight_blockers"]
        assert not any(request["tasks"][name]["preflight_blockers"] for name in TASKS)
        write(RUNS / "requests" / f"{idea['id']}.json", request)
        reviews[idea["id"]] = {"declaration": str(path.relative_to(ROOT)),
            "declaration_sha256": file_hash(path),
            "candidate_binding_sha256": expected["candidate_binding_sha256"],
            "control_binding_sha256": expected["control_binding_sha256"],
            "decision_contract_sha256": stable_hash(contract),
            "jobs_sha256": expected["jobs_sha256"],
            "task_bindings": {name: {kind: value for kind, value in
                request["decision_review"]["actual_bindings"]["candidate"]["task_identity"][name].items()}
                for name in sorted(TASKS)}}

    plan = plan_search(ROOT, RUNS / "queue", spec)
    assert len(plan["trials"]) == 12
    assert all(trial["submission_status"] == "READY" for trial in plan["trials"])
    assert plan["declared_worst_case_seconds"] == CAMPAIGN_CAP
    signature = technique_signature(json.loads((ROOT /
        f"configs/forge/ideas/{BASE}.json").read_text())["recipe_overrides"])
    assert all(trial["technique_signature"] == signature for trial in plan["trials"])
    assert all(trial["declared_worst_case_seconds"] == CANDIDATE_CAP for trial in plan["trials"])
    write(RUNS / "plan.json", plan)
    compact = {"schema_version": 1, "scope": "bounded_global_existing_hyperparameter_search",
        "study": STUDY, "spec": str(SPEC.relative_to(ROOT)), "spec_sha256": file_hash(SPEC),
        "spec_semantic_sha256": stable_hash(spec), "source_digest": plan["source_digest"],
        "preparation_checkout_commit": request["source"]["origin_commit"],
        "scientific_python_executable": sys.executable, "runtime_cohort": plan["runtime_cohort"],
        "policy_fingerprint": plan["policy_fingerprint"], "protocol_sha256": plan["protocol_hash"],
        "view": "discriminator_stability", "through_tier": 1,
        "seed": 0, "gpu": 0, "maximum_global_workers": 2,
        "workers_per_gpu": 1, "automatic_cpu_workers": 1, "cpu_threads": 1,
        "configuration_count": 12, "required_tier1_count": 5, "task_bindings": 60,
        "candidate_ceiling_seconds": CANDIDATE_CAP, "campaign_ceiling_seconds": CAMPAIGN_CAP,
        "nominal_absolute_latent_prior_base_lr": .0012,
        "technique_signature": signature, "qualification_input": False,
        "historical_word_receipts_reused_for_qualification": 0,
        "hypothesis": HYPOTHESIS, "limits": LIMITS,
        "trials": [{"candidate": trial["candidate_id"],
            "candidate_revision": trial["candidate_revision"],
            "configuration_id": trial["configuration_id"], "settings": trial["settings"],
            "decision_status": trial["submission_status"],
            "scientific_signature": trial["scientific_signature"],
            **reviews[trial["candidate_id"]]} for trial in plan["trials"]]}
    write(REPORT / "plans.json", compact)
    print(json.dumps({"study": STUDY, "ready_configurations": 12,
        "task_bindings": 60, "campaign_ceiling_seconds": CAMPAIGN_CAP,
        "source_digest": plan["source_digest"], "spec_sha256": file_hash(SPEC),
        "compact_plans_sha256": file_hash(REPORT / "plans.json")}))


if __name__ == "__main__":
    main()

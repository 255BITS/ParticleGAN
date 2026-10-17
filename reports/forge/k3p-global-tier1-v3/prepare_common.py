"""Shared preparation for two separately bounded whole K3P searches.

This module plans and binds declarations only. It never enqueues, trains or
changes a prior study. Full requests/plans stay in ignored runs/; compact plans
record review metadata, not qualification or an additional leaderboard.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.api import task_formulation_context
from experiments.forge.configuration_search import materialize_search, plan_search
from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue


REPORT = ROOT / "reports/forge/k3p-global-tier1-v3"
TASKS = ("two_pole", "unused_token_hold", "ae_gan_hold",
         "ring16_acquisition", "five_word_joint_acquisition")
CANDIDATE_CAP = 2100
LIMITS = (
    "One separately bounded round of complete global K3P recipes under all "
    "five ordinary Tier 1 tasks. Existing numerical controls only; no new "
    "technique, task-specific optimizer override, objective, architecture, "
    "initializer, sampling law, seed study, gate change, unchanged whole-recipe "
    "repeat, paid base/control run, higher-tier execution or automatic further "
    "paid round. Ordinary required failures stop later tasks, which remain "
    "UNKNOWN. Historical word passes motivate the design and grant no "
    "qualification. Screening is provisional; no public-default adoption."
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def motivation(path):
    value = json.loads((ROOT / path).read_text())
    names = ("id", "study", "study_id", "scope", "candidate_revision", "source_digest")
    identity = {name: value[name] for name in names if name in value}
    assert identity, f"Missing compact evidence identity: {path}"
    return {"path": path, "sha256": file_hash(ROOT / path), "selector": [],
            "identity": identity, "use": "motivation_only"}


def effective_recipe(candidate, tasks):
    # Compare consumed numerical recipes, independent of aliases and declaration
    # provenance. Task-owned fields are resolved through the ordinary public API.
    result = {}
    for name, task in tasks.items():
        receipt = task_formulation_context(candidate, task, root=ROOT).receipt()
        fields = receipt["field_ownership"]["recipe_fields"]
        result[name] = {key: value["value"] for key, value in fields.items()
                        if value["status"] == "effective" and key != "name"}
    return result


def existing_recipes(study, tasks):
    found = {}
    for directory in ("ideas", "configurations"):
        for path in sorted((ROOT / f"configs/forge/{directory}").glob("*.json")):
            candidate = json.loads(path.read_text())
            if (candidate.get("search_study_id") == study
                    or candidate.get("recipe_overrides", {}).get("critic_formulation") != "k3p"
                    or candidate.get("trainer_family") not in (None, "k3p")):
                continue
            digest = stable_hash(effective_recipe(candidate, tasks))
            found.setdefault(digest, []).append(candidate["id"])
    return found


def prepare(*, study, base, grid, count, plans_name, scope, hypothesis,
            rationale, evidence_paths, prediction, falsifier, competing_explanation):
    runs = ROOT / f"runs/forge/{study}"
    if (ROOT / f"reports/forge/configuration-search/{study}.json").exists():
        raise ValueError("Registered study is immutable; use read-only search plan/report")
    state = Queue(runs / "queue", on_completion=None).inspect()
    if state.get("campaigns", {}).get(study):
        raise ValueError("Admitted campaign is immutable; preparation cannot refresh its bindings")
    print(json.dumps({"event": "preparation_started", "study": study,
                      "configurations": count}), flush=True)
    tasks = {name: json.loads((ROOT / f"configs/forge/tasks/{name}.json").read_text())
             for name in TASKS}
    prior = [motivation(path) for path in evidence_paths]
    known = existing_recipes(study, tasks)
    protocol = json.loads((ROOT / "configs/forge/protocols/screening.json").read_text())
    spec = {"schema_version": 1, "id": study, "trainer_family": "k3p",
        "base_candidate": base, "grid": grid, "tuning_through_tier": 1,
        "view": "discriminator_stability", "execution_backend": "cuda",
        "cuda_model": "NVIDIA RTX A6000", "protocol": "screening",
        "protocol_hash": stable_hash(protocol),
        "campaign": {"id": study, "budget_seconds": count * CANDIDATE_CAP,
            "candidate_budget_seconds": CANDIDATE_CAP, "accept_shared_cost_transfer": False},
        "hypothesis": hypothesis, "rationale": rationale + " " + LIMITS,
        "guide": "EXPERIMENTATION.md"}
    spec_path = ROOT / f"configs/forge/searches/{study}.json"
    write(spec_path, spec)
    paths = materialize_search(ROOT, spec)
    assert len(paths) == count
    candidate_signatures, pole_signatures, reviews = set(), {}, {}
    for path in paths:
        idea = json.loads(path.read_text())
        if idea["search_study_id"] != study:
            raise ValueError(f"Cannot mutate historical configuration card: {path.name}")
        effective = effective_recipe(idea, tasks)
        digest = stable_hash(effective)
        assert digest not in known, f"Unchanged effective whole recipe: {known.get(digest)}"
        assert digest not in candidate_signatures, "Duplicate effective whole recipe in grid"
        candidate_signatures.add(digest)
        pole_signatures.setdefault(stable_hash(effective["two_pole"]), []).append(idea["id"])
        contract = deepcopy(idea["decision_contract"])
        contract["status"] = "draft"
        contract["control"].update(candidate_id=base, task_map={})
        idea["decision_contract"] = contract
        write(path, idea)
        request = resolve_idea(ROOT, idea["id"], through_tier=1,
            execution_backend="cuda", cuda_model="NVIDIA RTX A6000")
        expected = request["decision_review"]["expected"]
        assert set(expected["task_ids"]) == set(TASKS)
        assert expected["substantive_delta"]
        contract.update(status="ready", prior_evidence=prior,
            candidate_binding_sha256=expected["candidate_binding_sha256"],
            substantive_delta=expected["substantive_delta"], prediction=prediction,
            falsifier=falsifier, competing_explanation=competing_explanation)
        contract["control"].update(binding_sha256=expected["control_binding_sha256"],
            task_map=expected["task_map"])
        contract["scope"].update(view="discriminator_stability", through_tier=1,
            task_ids=expected["task_ids"], max_rounds=1,
            candidate_budget_seconds=CANDIDATE_CAP, campaign_budget_seconds=count * CANDIDATE_CAP,
            **{key: expected[key] for key in ("protocol_sha256", "source_digest",
                "execution_backend", "runtime_cohort_sha256", "jobs_sha256")})
        write(path, idea)
        request = resolve_idea(ROOT, idea["id"], through_tier=1,
            execution_backend="cuda", cuda_model="NVIDIA RTX A6000")
        assert request["decision_review"]["status"] == "READY"
        assert not request["preflight_blockers"]
        assert not any(request["tasks"][name]["preflight_blockers"] for name in TASKS)
        write(runs / "requests" / f"{idea['id']}.json", request)
        reviews[idea["id"]] = {"declaration": str(path.relative_to(ROOT)),
            "declaration_sha256": file_hash(path),
            "effective_whole_recipe_sha256": digest,
            "candidate_binding_sha256": expected["candidate_binding_sha256"],
            "control_binding_sha256": expected["control_binding_sha256"],
            "decision_contract_sha256": stable_hash(contract),
            "jobs_sha256": expected["jobs_sha256"],
            "task_bindings": request["decision_review"]["actual_bindings"]["candidate"]["task_identity"]}

    plan = plan_search(ROOT, runs / "queue", spec)
    assert len(plan["trials"]) == count
    assert all(trial["submission_status"] == "READY" for trial in plan["trials"])
    assert all(trial["declared_worst_case_seconds"] == CANDIDATE_CAP for trial in plan["trials"])
    assert all(trial["technique_signature"] == plan["technique_signature"] for trial in plan["trials"])
    assert plan["declared_worst_case_seconds"] == count * CANDIDATE_CAP
    write(runs / "plan.json", plan)
    compact = {"schema_version": 1, "scope": scope, "study": study,
        "spec": str(spec_path.relative_to(ROOT)), "spec_sha256": file_hash(spec_path),
        "spec_semantic_sha256": stable_hash(spec), "source_digest": plan["source_digest"],
        "preparation_checkout_commit": request["source"]["origin_commit"],
        "scientific_python_executable": sys.executable, "runtime_cohort": plan["runtime_cohort"],
        "policy_fingerprint": plan["policy_fingerprint"], "protocol_sha256": plan["protocol_hash"],
        "view": "discriminator_stability", "through_tier": 1,
        "seed": 0, "gpu": 0, "maximum_global_workers": 2,
        "workers_per_gpu": 1, "automatic_cpu_workers": 1, "cpu_threads": 1,
        "configuration_count": count, "required_tier1_count": 5, "task_bindings": count * 5,
        "candidate_ceiling_seconds": CANDIDATE_CAP, "campaign_ceiling_seconds": count * CANDIDATE_CAP,
        "technique_signature": plan["technique_signature"], "qualification_input": False,
        "historical_word_receipts_reused_for_qualification": 0,
        "hypothesis": hypothesis, "limits": LIMITS,
        "prior_evidence": prior,
        "duplicate_audit": {"scope": "consumed whole recipe on all five current tasks",
            "known_candidates_checked": sum(map(len, known.values())),
            "unchanged_whole_recipe_matches": 0, "unique_whole_recipes": count,
            "distinct_two_pole_effective_recipes": len(pole_signatures),
            "two_pole_groups": list(pole_signatures.values()),
            "note": "Prior-only differences do not change direct two_pole physics. "
                    "Whole-candidate queue identities remain distinct; no automatic shared first-gate reuse is claimed."},
        "prediction": prediction, "falsifier": falsifier,
        "prediction_requires_reached_task": True,
        "trials": [{"candidate": trial["candidate_id"],
            "candidate_revision": trial["candidate_revision"],
            "configuration_id": trial["configuration_id"], "settings": trial["settings"],
            "decision_status": trial["submission_status"],
            "scientific_signature": trial["scientific_signature"],
            **reviews[trial["candidate_id"]]} for trial in plan["trials"]]}
    write(REPORT / plans_name, compact)
    print(json.dumps({"event": "preparation_complete", "study": study,
        "ready_configurations": count, "task_bindings": count * 5,
        "campaign_ceiling_seconds": count * CANDIDATE_CAP,
        "source_digest": plan["source_digest"], "spec_sha256": file_hash(spec_path),
        "compact_plans_sha256": file_hash(REPORT / plans_name)}), flush=True)

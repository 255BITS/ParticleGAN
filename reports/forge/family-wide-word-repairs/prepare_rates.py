"""Declare the six bounded global coupled-rate configurations; no training.

Use Forge's ordinary strict configuration-search materializer and READY
decision-contract validation. Do not rerun this mutating preparation after
study registration; use the normal read-only search plan/report commands.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.configuration_search import materialize_search, plan_search
from experiments.forge.contracts import file_hash, stable_hash
from experiments.forge.planning import resolve_idea


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    protocol = json.loads((ROOT / "configs/forge/protocols/screening.json").read_text())
    families = {"k3p": {"rates": [.00425, .006375], "prior_absolute": .0012},
                "ka2": {"rates": [.00425, .006375], "prior_absolute": .00006},
                "r1r2": {"rates": [.006375, .0085], "prior_absolute": .0024}}
    prepared = []
    for family, values in families.items():
        study = f"{family}-global-repair-rates-v1"
        if (ROOT / f"reports/forge/configuration-search/{study}.json").exists():
            raise ValueError("Registered study is immutable; use read-only search plan/report")
        base = f"{family}-global-repair-v1"
        evidence_path = ROOT / f"reports/forge/family-wide-word-repairs/initial/{base}.json"
        evidence = json.loads(evidence_path.read_text())
        assert evidence["measured_task"]["gate_status"] == "FAIL"
        assert evidence["measured_task"]["evaluator_result"]["convergence"]["complete"]
        spec = {"schema_version": 1, "id": study, "trainer_family": family,
            "base_candidate": base, "grid": {"coupled_rates": [
                {"lr": rate, "prior_lr_mult": values["prior_absolute"] / rate}
                for rate in values["rates"]]},
            "tuning_through_tier": 3, "view": "discriminator_stability",
            "execution_backend": "cuda", "cuda_model": "NVIDIA RTX A6000",
            "protocol": "screening", "protocol_hash": stable_hash(protocol),
            "campaign": {"id": study, "budget_seconds": 97800,
                "candidate_budget_seconds": 48900, "accept_shared_cost_transfer": False},
            "hypothesis": "Increasing the global network/base rate while retaining the nominal absolute latent-table prior base rate may restore movement within the unchanged 80-update screen. Every complete configuration is then evaluated through all ordinary Tier 3 prerequisites; later failures remain part of this whole-candidate result.",
            "rationale": f"The completed {base} first gate had mean_abs={evidence['measured_task']['metrics']['mean_abs']:.10g} below .3, with a passing gradient bound. These two coupled global numeric settings hold all existing penalty kernels, intervention flags, moments, noise/schedule laws and task conditions fixed. They preserve nominal latent-table prior base LR {values['prior_absolute']}; direct coordinates instead consume the existing global G-side base LR. Effective feedback rates can differ with training state. No per-task candidate override, new objective, seed study, gate change, repeat of the failed base, or automatic continuation is allowed.",
            "guide": "EXPERIMENTATION.md"}
        spec_path = ROOT / f"configs/forge/searches/{study}.json"
        write(spec_path, spec)
        paths = materialize_search(ROOT, spec)
        assert len(paths) == 2
        for path in paths:
            idea = json.loads(path.read_text())
            contract = deepcopy(idea["decision_contract"])
            contract["status"] = "draft"
            contract["control"]["candidate_id"] = base
            idea["decision_contract"] = contract
            write(path, idea)
            request = resolve_idea(ROOT, idea["id"], through_tier=3,
                execution_backend="cuda", cuda_model="NVIDIA RTX A6000")
            expected = request["decision_review"]["expected"]
            contract.update(status="ready", prior_evidence=[{
                "path": str(evidence_path.relative_to(ROOT)), "sha256": file_hash(evidence_path),
                "selector": [], "identity": {"candidate_id": base,
                    "candidate_revision": evidence["candidate_revision"],
                    "attempt_id": evidence["attempt_id"]}, "use": "motivation_only"}],
                candidate_binding_sha256=expected["candidate_binding_sha256"],
                substantive_delta=expected["substantive_delta"],
                prediction={"task_id": "two_pole", "metric": "mean_abs", "op": ">=",
                    "threshold": .3, "phase": "final"},
                falsifier={"task_id": "two_pole", "metric": "mean_abs", "op": "<",
                    "threshold": .3, "phase": "final"},
                competing_explanation="Global rate increases also change critic/network dynamics and can still miss the frozen short-horizon gate or destabilize later tasks. The paired prior multiplier retains nominal table base LR, not the realized state-dependent optimizer trajectory. A complete failed configuration cannot be repaired by importing a task-only word witness or tuning one task.")
            contract["control"].update(binding_sha256=expected["control_binding_sha256"],
                task_map=expected["task_map"])
            contract["scope"].update(view="discriminator_stability", through_tier=3,
                task_ids=expected["task_ids"], max_rounds=1, candidate_budget_seconds=48900,
                campaign_budget_seconds=97800,
                **{key: expected[key] for key in ("protocol_sha256", "source_digest",
                    "execution_backend", "runtime_cohort_sha256", "jobs_sha256")})
            write(path, idea)
            request = resolve_idea(ROOT, idea["id"], through_tier=3,
                execution_backend="cuda", cuda_model="NVIDIA RTX A6000")
            assert request["decision_review"]["status"] == "READY"
            assert not request["preflight_blockers"]
            assert not any(task["preflight_blockers"] for task in request["tasks"].values())
        queue_root = ROOT / "runs/forge/family-wide-word-repair-rates-v1/queue"
        plan = plan_search(ROOT, queue_root, spec)
        assert len(plan["trials"]) == 2
        assert all(t["submission_status"] == "READY" for t in plan["trials"])
        assert plan["declared_worst_case_seconds"] == 90600
        write(ROOT / f"runs/forge/family-wide-word-repair-rates-v1/{study}-plan.json", plan)
        prepared.append({"family": family, "study": study, "spec": str(spec_path.relative_to(ROOT)),
            "spec_sha256": file_hash(spec_path), "source_digest": plan["source_digest"],
            "runtime_cohort": plan["runtime_cohort"], "nominal_absolute_prior_base_lr": values["prior_absolute"],
            "rates": values["rates"], "campaign_ceiling_seconds": 97800,
            "grouped_job_ceiling_seconds": plan["declared_worst_case_seconds"],
            "trials": [{"candidate": t["candidate_id"], "candidate_revision": t["candidate_revision"],
                "configuration_id": t["configuration_id"], "settings": t["settings"],
                "decision_status": t["submission_status"], "task_count": len(t["tasks"]),
                "scientific_signature": t["scientific_signature"]} for t in plan["trials"]]})
    assert len({p["source_digest"] for p in prepared}) == 1
    write(ROOT / "reports/forge/family-wide-word-repairs/rates-plans.json", {
        "schema_version": 1, "scope": "bounded_global_numeric_search",
        "scientific_python_executable": sys.executable, "source_digest": prepared[0]["source_digest"],
        "seed": 0, "gpu": 0, "workers": 1, "cpu_threads": 1,
        "configuration_count": 6, "campaign_ceiling_seconds": 293400,
        "grouped_job_ceiling_seconds": 271800, "families": prepared,
        "limits": "Six complete global configurations, no per-task recipe override, new technique/objective/architecture, seed study, gate weakening, prior-only first-gate twins, scientific repeats or automatic further paid round. Ordinary failures leave later cells unmeasured."})
    print(json.dumps({"ready_configurations": 6, "task_bindings": 156,
        "campaign_ceiling_seconds": 293400, "grouped_job_ceiling_seconds": 271800}))


if __name__ == "__main__":
    main()

"""Family publication selects complete, verified configurations without training."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import shutil

import pytest

from experiments.forge import configuration_search as search
from experiments.forge import trainer_families as families
from experiments.forge.api import resolve_public_recipe
from experiments.forge.contracts import atomic_json, read_json, stable_hash

ROOT = Path(__file__).resolve().parents[1]


def persist(root, report):
    report["input_digest"] = stable_hash({key: value for key, value in report.items() if key != "input_digest"})
    atomic_json(root / f"reports/forge/configuration-search/{report['study_id']}.json", report)


@pytest.fixture
def study(tmp_path):
    shutil.copytree(ROOT / "configs/forge", tmp_path / "configs/forge")
    spec = read_json(tmp_path / "configs/forge/searches/r1r2-modern-toy-v1.json")
    declarations = search._declarations(tmp_path, spec)
    assignments = read_json(tmp_path / "configs/forge/views/discriminator_stability.json")["assignments"]
    policy = stable_hash(read_json(tmp_path / "configs/forge/views/discriminator_stability.json"))
    runtime = {"execution_backend": "cpu", "runtime": {"python": "frozen"},
               "compute_profiles": {"cpu": {"model": "fixture-cpu"}}}
    protocol = read_json(tmp_path / "configs/forge/protocols/screening.json")
    catalogs = {"task_contracts": {}, "recipe_contracts": {}, "protocol_contracts": {stable_hash(protocol): protocol}}
    task_contracts = {}
    for assignment in assignments:
        contract = {"host": assignment["task"], "prior": {"kind": "mog"},
                    "initialization": "fixed", "steps": 80, "sampling": {"eval_output_noise": "clean"}}
        digest = stable_hash(contract)
        catalogs["task_contracts"][digest] = contract
        task_contracts[assignment["task"]] = digest
    rows, trials, cards = [], [], {}
    def row_for(card, recipe, *, source="a" * 64, failure=True, model=None):
        keys = {a["task"]: stable_hash({"candidate": card["id"], "task": a["task"]}) for a in assignments}
        statuses = {a["task"]: "FAIL" if failure and a["qualification_tier"] == 1 else "UNKNOWN" for a in assignments}
        recipe_digest = stable_hash(recipe)
        catalogs["recipe_contracts"][recipe_digest] = recipe
        return {"candidate_id": card["id"], "candidate_revision": stable_hash(card), "cohort": stable_hash(keys),
                "technique": card["id"], "runtime_cohort": deepcopy(model or runtime), "qualified_tier": 0,
                "status": "FAIL" if failure else "UNKNOWN", "attempt_ids": [],
                "bindings": {"source_digest": source, "protocol_sha256": spec["protocol_hash"],
                             "recipe_sha256": recipe_digest, "task_keys_sha256": stable_hash(keys),
                             "task_contracts": deepcopy(task_contracts), "prior": {"kind": "mog"},
                             "initializer": "fixed", "rng_sha256": "rng", "claim_contract": {"sampling_law": "clean"}},
                "tasks": [{"task_id": a["task"], "status": statuses[a["task"]]} for a in assignments if a["importance"] == "required"],
                "nonrequired_tasks": [{"task_id": a["task"], "status": statuses[a["task"]]} for a in assignments if a["importance"] != "required"],
                "tiers": {str(tier): {"passed": 0, "total": sum(a["importance"] == "required" and a["qualification_tier"] == tier for a in assignments)}
                          for tier in (1, 2, 3)}}, keys, statuses
    for card, settings in declarations:
        atomic_json(tmp_path / f"configs/forge/configurations/{card['id']}.json", card)
        cards[card["id"]] = card
        recipe = deepcopy(card["resolved_configuration_recipe"])
        row, keys, statuses = row_for(card, recipe)
        rows.append(row)
        trials.append({"candidate_id": card["id"], "configuration_id": card["configuration_id"], "trainer_family": "r1r2",
                       "settings": settings, "declaration": deepcopy(card), "recipe_overrides": card["recipe_overrides"],
                       "resolved_recipe": recipe, "candidate_revision": row["candidate_revision"], "source_digest": "a" * 64,
                       "runtime_cohort": deepcopy(runtime), "protocol_hash": spec["protocol_hash"], "submission_status": "terminal",
                       "tasks": [{**a, "compatibility_key": keys[a["task"]], "gate_status": statuses[a["task"]]} for a in assignments]})
    base = search._base_declaration(tmp_path, spec)
    cards[base["id"]] = base
    canonical, _, _ = row_for(base, search._resolved_recipe(base), source="b" * 64, failure=False)
    rows.append(canonical)
    report = {"study_id": spec["id"], "trainer_family": "r1r2", "spec": spec, "spec_hash": stable_hash(spec),
              "base_declaration": deepcopy(base), "view": spec["view"], "execution_backend": "cpu",
              "tuning_through_tier": 1, "protocol_hash": spec["protocol_hash"], "policy_fingerprint": policy,
              "runtime_cohort": runtime, "source_digests": ["a" * 64], "trials": trials,
              "selection": search.select_configuration(trials, 1)}
    persist(tmp_path, report)
    return tmp_path, rows, catalogs, cards, report, policy


def select(study):
    root, rows, catalogs, cards, _, policy = study
    return families.select_family_rows(root, rows, catalogs, declarations=cards,
                                      view_id="discriminator_stability", policy_fingerprint=policy)


def test_all_four_declared_configuration_ids_bind_actual_forge_prior_context(study):
    from experiments.forge.hostprofiles import profile_source_paths
    from experiments.forge.planning import resolve_idea
    from experiments.forge.technique_board import request_bindings
    root, _, _, cards, report, _ = study
    for task_path in (root / "configs/forge/tasks").glob("*.json"):
        task = read_json(task_path)
        referenced = set(task["evaluation"].get("sources", {})) | set(profile_source_paths(task))
        for relative in referenced:
            if (ROOT / relative).is_file():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, root / relative)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert len(report["trials"]) == 4
    for trial in report["trials"]:
        card = cards[trial["candidate_id"]]
        request = resolve_idea(root, card["id"], declaration=card, view_id=report["view"],
                               through_tier=1, execution_backend="cpu")
        recipe = request["candidate"]["resolved_recipe"]
        bindings = request_bindings(request)
        assert stable_hash(recipe) == stable_hash(card["resolved_configuration_recipe"]) == stable_hash(trial["resolved_recipe"])
        assert recipe["prior_kind"] == "mog" and recipe["standardize"] is False
        assert request["candidate"]["prior"] == card["prior"] == report["base_declaration"]["prior"]
        assert search.configuration_id(card, resolved_recipe=recipe) == trial["configuration_id"]
        assert bindings["recipe_sha256"] == stable_hash(recipe) == stable_hash(bindings["recipe"])
        # Raw preset defaults differ; only the actual FormulationContext is the
        # publication identity for these learned-MoG Forge requests.
        assert asdict(resolve_public_recipe(card))["prior_kind"] != recipe["prior_kind"]
    assert before == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_failed_study_selects_one_whole_configuration_and_preserves_every_trial(study):
    result = select(study)
    selected = result["rows"][0]
    assert len(result["rows"]) == 1 and len(result["configuration_rows"]) == 5
    assert selected["candidate_id"] == study[4]["selection"]["selected_candidate_id"]
    assert selected["selection"]["selection_kind"] == "best_observed"
    assert not selected["selection"]["qualified"] and not selected["selection"]["default_adoption"]
    assert selected["tiers"] == next(row["tiers"] for row in study[1] if row["candidate_id"] == selected["candidate_id"])
    assert {row["alternative_scope"] for row in result["configuration_rows"]} == {"selected", "comparable_trial", "archived_alternative"}
    assert selected["tiers"]["1"] == {"passed": 0, "total": 3}


def test_pending_search_uses_canonical_configuration(study):
    root, _, _, _, report, _ = study
    report["trials"][0]["submission_status"] = "running"
    report["selection"] = search.select_configuration(report["trials"], 1)
    persist(root, report)
    selected = select(study)["rows"][0]
    assert selected["candidate_id"] == "r3gan-stacked-training-toy-v1"
    assert selected["selection"]["selection_kind"] == "canonical_fallback"


def test_frozen_recipe_selection_survives_future_default_change(study, monkeypatch):
    monkeypatch.setattr(search, "_resolved_recipe", lambda *args: pytest.fail("frozen publication resolved today's Recipe defaults"))
    assert select(study)["rows"][0]["candidate_id"] == study[4]["selection"]["selected_candidate_id"]


def test_frozen_selection_survives_current_protocol_change(study):
    root = study[0]
    atomic_json(root / "configs/forge/defaults.json", {"protocol": "future-protocol"})
    atomic_json(root / "configs/forge/protocols/screening.json", {"changed": "future protocol"})
    assert select(study)["rows"][0]["candidate_id"] == study[4]["selection"]["selected_candidate_id"]


@pytest.mark.parametrize("tamper,message", [
    ("qualification", "selection differs"), ("status", "status differs"),
    ("omitted_trial", "omitted or changed"), ("tier", "frozen study contract"),
    ("task_omitted", "task denominator"), ("settings", "omitted or changed"),
    ("recipe", "configuration identity"), ("source", "different source/runtime"),
    ("prior", "different source/runtime"), ("digest", "input digest"),
])
def test_search_cannot_edit_selection_or_scientific_identity(study, tamper, message):
    root, rows, _, _, report, _ = study
    if tamper == "qualification":
        report["selection"]["qualified"] = True
    elif tamper == "status":
        report["trials"][0]["tasks"][0]["gate_status"] = "PASS"
    elif tamper == "omitted_trial":
        report["trials"].pop()
    elif tamper == "tier":
        report["tuning_through_tier"] = 2
    elif tamper == "task_omitted":
        report["trials"][0]["tasks"].pop()
    elif tamper == "settings":
        report["trials"][0]["settings"]["lr"] *= 10
    elif tamper == "recipe":
        report["trials"][0]["resolved_recipe"]["lr"] *= 10
    elif tamper == "source":
        report["trials"][0]["source_digest"] = rows[0]["bindings"]["source_digest"] = "c" * 64
        report["source_digests"].append("c" * 64)
    elif tamper == "prior":
        rows[0]["bindings"]["prior"] = {"kind": "cloud"}
    elif tamper == "digest":
        report["selection"]["qualified"] = True
        atomic_json(root / f"reports/forge/configuration-search/{report['study_id']}.json", report)
    if tamper != "digest":
        persist(root, report)
    with pytest.raises(ValueError, match=message):
        select(study)


def test_hardware_and_cpu_cuda_families_stay_separate(study):
    root, rows, _, _, report, _ = study
    gpu = deepcopy(rows[-1])
    gpu["runtime_cohort"] = {"execution_backend": "cuda", "compute_profiles": {"cuda": {"model": "gpu-a"}}}
    second_gpu = deepcopy(gpu)
    second_gpu["runtime_cohort"]["compute_profiles"]["cuda"]["model"] = "gpu-b"
    rows.extend([gpu, second_gpu])
    result = select(study)
    assert len(result["rows"]) == 3
    assert sum(row["selection"]["selection_kind"] == "best_observed" for row in result["rows"]) == 1


def test_registry_keeps_modern_baseline_canonical_and_cloud_ablation_families_separate():
    registry = families.load_families(ROOT)
    assert len(registry) == 12
    assert registry["r1r2"]["canonical_candidate"] == "r3gan-stacked-training-toy-v1"
    assert families.family_for_candidate(ROOT, "k3p-r1r2-matched-v1")["id"] == "r1r2"
    assert families.family_for_candidate(ROOT, "release07-gan-v3-cloud-v1")["id"] != families.family_for_candidate(ROOT, "release07-gan-v3-task-adapted-v1")["id"]
    assert families.family_for_candidate(ROOT, "forge-no-critic-penalty")["id"] != families.family_for_candidate(ROOT, "k3p")["id"]

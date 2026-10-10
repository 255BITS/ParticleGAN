"""Zero-training controls for declared successors and ordinary-only publication."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.trainer_families import CURRENT_SELECTION, family_row_pin
from experiments.forge.views import load_tasks, load_view, task_execution_fingerprint, task_evaluation_fingerprint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


migration = module("prior_ownership_publication_controls", HERE / "publication_migration.py")
publisher = module("prior_ownership_receipt_controls", ROOT / "reports/forge/regenerate_technique_inventory.py")


def test_final_registration_and_actual_initial_laws_are_bound():
    data = read_json(HERE / "ownership-migration.json")
    baseline = migration._validate_bindings(ROOT, data)
    assert len(data["leaders"]) == 7 and len(data["ordinary_task_ids"]) == 28
    assert len(read_json(HERE / "binding-compatibility.json")["pairs"]) == 140
    assert file_hash(ROOT / CURRENT_SELECTION) == baseline["family_current_sha256"]


@pytest.mark.parametrize("tamper", ["registration_hash", "source", "roster", "initial_law"])
def test_changed_registration_source_roster_or_initial_law_is_rejected(monkeypatch, tamper):
    data = read_json(HERE / "ownership-migration.json")
    proof = read_json(HERE / "binding-compatibility.json")
    if tamper == "registration_hash":
        proof["registration_sha256"] = "0" * 64
    elif tamper == "source":
        data["source_digest"] = "0" * 64
    elif tamper == "roster":
        data["ordinary_task_ids"][-1] = data["ordinary_task_ids"][0]
    else:
        proof["pairs"][0]["initial_prior_sha256"] = "0" * 64
    original = migration.bound_file
    # Hash rejection is tested separately; inject only the semantic proof
    # mutation here so the actual downstream registration/law guard executes.
    monkeypatch.setattr(migration, "bound_file", lambda root, item:
        proof if item == data["binding_proof"] else original(root, item))
    with pytest.raises(ValueError, match="ownership migration"):
        migration._validate_bindings(ROOT, data)


def test_changed_evidence_bytes_are_rejected(tmp_path):
    path = tmp_path / "proof.json"
    atomic_json(path, {"status": "PASS"})
    binding = migration.descriptor(tmp_path, path)
    atomic_json(path, {"status": "FAIL"})
    with pytest.raises(ValueError, match="file hash mismatch"):
        migration.bound_file(tmp_path, binding)


@pytest.fixture
def ordinary_rows(tmp_path, monkeypatch):
    # Only metadata is copied. No models, checkpoints, training or scoring.
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    for name in ("baseline.json", "binding-compatibility.json", "ordinary-registration.json", "ownership-migration.json"):
        target = tmp_path / migration.REPORT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / name, target)
    data = read_json(HERE / "ownership-migration.json")
    baseline = read_json(HERE / "baseline.json")
    # The real expensive reconstruction is covered above. The publication
    # controls below isolate whole-row, receipt and contract validation.
    monkeypatch.setattr(migration, "_validate_bindings", lambda root, value: baseline)
    card = read_json(tmp_path / CURRENT_SELECTION)
    old_rows = read_json(ROOT / "reports/forge/technique-inventory.json")["rows"]
    prior = [(None, None, {str(i): row for i, row in enumerate(old_rows)})]
    view, tasks = load_view(tmp_path, "discriminator_stability"), load_tasks(tmp_path)
    requirements = {str(tier): [a["task"] for a in view["assignments"]
                    if a["qualification_tier"] == tier and a["importance"] == "required"] for tier in (1, 2, 3)}
    contracts, keys = {}, {}
    for name in data["ordinary_task_ids"]:
        task = tasks[name]
        contract = dict(execution_sha256=task_execution_fingerprint(task),
                        evaluation_sha256=task_evaluation_fingerprint(task),
                        timeout_seconds=task["resources"]["timeout_seconds"])
        keys[name] = stable_hash(contract)
        contracts[keys[name]] = contract
    report = dict(view="discriminator_stability", view_revision=9,
                  policy_fingerprint=data["new_policy_fingerprint"], publication_scope="frozen_source",
                  frozen_source=dict(source_digests=[data["source_digest"]]),
                  task_contracts=contracts, tier_requirements=requirements,
                  provenance=dict(qualified_receipts={}), registered_study_reconstruction=[])
    rows = {}
    for entry in data["leaders"]:
        blocked = entry["blocked_without_execution"]
        attempt = "fixture-ordinary-" + entry["family"]
        row = dict(candidate_id=entry["candidate_id"], candidate_revision=entry["candidate_revision"],
                   cohort="explicit-metadata-control", runtime_cohort=dict(execution_backend="cuda"),
                   bindings=dict(source_digest=data["source_digest"], recipe_sha256=entry["recipe_sha256"],
                                 task_contracts=deepcopy(keys)),
                   status="BLOCKED" if blocked else "FAIL", qualified_tier=0,
                   attempt_ids=[] if blocked else [attempt],
                   tiers={tier: dict(passed=0, total=len(names)) for tier, names in requirements.items()},
                   tasks=[dict(task_id=name, status="BLOCKED" if blocked else "FAIL") for name in data["ordinary_task_ids"]])
        rows[entry["candidate_id"]] = row
        if not blocked:
            proof = dict(canonical_result_hash="fixture-result", source_digest=data["source_digest"],
                         original_file_sha256={name: "fixture-" + name for name in ("request", "result", "evidence")})
            report["provenance"]["qualified_receipts"][attempt] = proof
            atomic_json(tmp_path / f"reports/forge/technique-receipts/{attempt}.json",
                dict(candidate_id=entry["candidate_id"], candidate_revision=entry["candidate_revision"],
                     certificate_validated=True, qualification_input=False, qualification_reuse=False,
                     provenance=dict(canonical_result_hash=proof["canonical_result_hash"], source_digest=data["source_digest"],
                         original_files={name: dict(sha256=value) for name, value in proof["original_file_sha256"].items()})))
            report["registered_study_reconstruction"].append(dict(candidate_id=entry["candidate_id"],
                study_id=entry["study_id"], study_sha256=entry["study_sha256"],
                admission_sha256=entry["study_admission_sha256"], source_digest=data["source_digest"], original_receipts=[attempt]))
    manifest = dict(view=report["view"], view_revision=8, policy_fingerprint=data["original_policy_fingerprint"])
    return tmp_path, data, card, manifest, report, rows, prior


def advance(fixture):
    root, data, card, manifest, report, rows, prior = fixture
    return migration.advance_selection(root, manifest, report, rows, prior,
        migration.REPORT / "ownership-migration.json", validate_row=publisher._validate_published_row)


def test_ordinary_whole_rows_preserve_original_card_and_history_without_ranking(ordinary_rows):
    root, data, original, _, _, rows, _ = ordinary_rows
    before = (root / CURRENT_SELECTION).read_bytes()
    selected, archive = advance(ordinary_rows)
    assert (root / CURRENT_SELECTION).read_bytes() == before
    assert archive[1].encode() == before and archive[2] == file_hash(root / CURRENT_SELECTION)
    assert selected["historical_selections"] == original["historical_selections"]
    assert len(selected["selections"]) == 7
    assert len(selected["ownership_migration"]["archived_only_selections"]) == 13
    assert selected["ownership_migration"]["qualification_transfer"] is False
    for pin in selected["selections"]:
        row = {**rows[pin["candidate_id"]], "trainer_family": pin["trainer_family"]}
        assert pin == family_row_pin(row, selection_kind=pin["selection_kind"], reason=pin["reason"],
            measurement_views=pin.get("measurement_views"), measurement_tasks=pin.get("measurement_tasks"))


@pytest.mark.parametrize("tamper", ["diagnostic", "mixed_study", "source", "certificate", "contract", "blocked_credit"])
def test_invalid_publication_cannot_replace_any_pin(ordinary_rows, tamper):
    root, data, _, _, report, rows, _ = ordinary_rows
    before = (root / CURRENT_SELECTION).read_bytes()
    entry = next(x for x in data["leaders"] if x["family"] == "bcap-pure")
    row = rows[entry["candidate_id"]]
    if tamper in {"diagnostic", "mixed_study"}:
        proof = next(x for x in report["registered_study_reconstruction"] if x["candidate_id"] == entry["candidate_id"])
        proof["study_id" if tamper == "diagnostic" else "admission_sha256"] = "different-diagnostic-scope"
    elif tamper == "source":
        row["bindings"]["source_digest"] = "0" * 64
    elif tamper == "certificate":
        report["provenance"]["qualified_receipts"][row["attempt_ids"][0]]["canonical_result_hash"] = "tampered"
    elif tamper == "contract":
        row["bindings"]["task_contracts"][data["ordinary_task_ids"][-1]] = "0" * 64
    else:
        rows["atlas"]["qualified_tier"] = 1
    with pytest.raises(ValueError):
        advance(ordinary_rows)
    assert (root / CURRENT_SELECTION).read_bytes() == before


def test_archived_presentation_restores_guard_and_cannot_hide_new_measurement(ordinary_rows):
    from experiments.forge import trainer_families
    root, data, _, _, _, _, _ = ordinary_rows
    selected, _ = advance(ordinary_rows)
    original = trainer_families._search_pin
    family = data["archived_only_selections"][0]["trainer_family"]
    row = dict(attempt_ids=[], qualified_tier=0, status="UNKNOWN",
               tasks=[dict(task_id="fixture", status="UNKNOWN")], cost=dict(wall_seconds=None))
    with migration.archived_presentation(root, selected):
        assert trainer_families._search_pin(root, family, "cuda", [row]) is None
        row["attempt_ids"] = ["new-measurement"]
        with pytest.raises(ValueError, match="cannot hide a new measurement"):
            trainer_families._search_pin(root, family, "cuda", [row])
    assert trainer_families._search_pin is original

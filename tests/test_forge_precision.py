"""Public-trainer execution and fail-closed grading of precision stretch toys."""
import json
from copy import deepcopy
from pathlib import Path

import torch

from experiments.forge.adapters import adapter_preflight, run_task
from experiments.forge.views import grade_result, load_tasks, load_view

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = {"recipe_preset": "bispf_optimized", "claim_contract": {"sampling_law": "task_declared"}}


def task(format):
    return json.loads((ROOT / f"configs/forge/tasks/precision_gaussian_{format}.json").read_text())


def test_precision_is_optional_and_preserves_every_original_required_assignment():
    view = load_view(ROOT, "discriminator_stability")
    before = json.loads((ROOT / "configs/forge/view-history/discriminator_stability-v14.json").read_text())
    assert [a for a in view["assignments"] if a["importance"] == "required"] == [
        a for a in before["assignments"] if a["importance"] == "required"]
    assert {a["task"]: a["importance"] for a in view["assignments"] if a["task"].startswith("precision_")} == {
        "precision_gaussian_fp32": "diagnostic", "precision_gaussian_mxfp8": "stretch", "precision_gaussian_nvfp4": "stretch"}
    tasks = load_tasks(ROOT)
    first = tasks["precision_gaussian_fp32"]
    for format in ("mxfp8", "nvfp4"):
        other = tasks["precision_gaussian_" + format]
        assert first["evaluation"] == other["evaluation"]
        assert first["resources"] == other["resources"]
        assert {k: v for k, v in first["execution"].items() if k != "linear_precision"} == {
            k: v for k, v in other["execution"].items() if k != "linear_precision"}


def test_preflight_refuses_an_unimplemented_precision_policy():
    value = task("nvfp4")
    assert adapter_preflight(value, CANDIDATE, root=ROOT) == []
    value["execution"]["linear_precision"]["stochastic_gradient_rounding"] = False
    assert any("precision differs" in reason for reason in adapter_preflight(value, CANDIDATE, root=ROOT))


def test_adding_unmeasured_stretch_tasks_preserves_source_bound_display_records():
    from reports.forge.regenerate_technique_inventory import (
        _formulaic_display_payload,
        _retained_published_display,
        _selected_recipe_binary_task_statuses,
    )
    publication = json.loads((ROOT / "reports/forge/technique-inventory.json").read_text())
    before = deepcopy(publication["rows"])
    display = _formulaic_display_payload(publication, ROOT)
    statuses = _selected_recipe_binary_task_statuses(publication, ROOT)
    assert display and statuses
    assert display["qualification_input"] is False
    assert display["qualification_reuse"] is False
    for family, observed in display["families"].items():
        for name, record in observed["records"].items():
            assert statuses[family][name] == record["status"]
        for name in ("precision_gaussian_nvfp4", "precision_gaussian_mxfp8"):
            assert name not in observed["records"]
            assert statuses[family][name] not in {"PASS", "FAIL"}
    assert publication["rows"] == before
    changed = deepcopy(publication)
    changed["rows"][0]["candidate_id"] += "-different-recipe"
    assert _retained_published_display(changed, ROOT) is None


def test_public_trainer_precision_prefixes_have_gifs_matching_starts_and_data(tmp_path):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    receipts = []
    try:
        for format in ("fp32", "mxfp8", "nvfp4"):
            original = task(format)
            value = deepcopy(original)
            # Explicit short software cohort, not a new quality measurement or
            # a replacement for the registered 1,000-update test.
            value["id"] += "-software-prefix"
            value["execution"]["steps"] = 3
            value["evaluation"]["observations"] = 3
            request = {"candidate": CANDIDATE, "protocol": {"seed": 0}, "tasks": {value["id"]: value}}
            raw = run_task(request, {"task_id": value["id"]}, tmp_path / format, "cpu")
            receipts.append(raw)
            assert raw["cost"]["completed_steps"] == 3
            assert raw["evidence"]["guards"]["unintended_rng_deviations"] == 0
            assert raw["evidence"]["guards"]["all_finite"] is True
            assert (tmp_path / format / "goal.gif").stat().st_size > 100
            assert grade_result(original, raw)["gate_status"] == "INCOMPLETE"
            missing = deepcopy(raw)
            del missing["evidence"]["linear_precision"]
            assert grade_result(original, missing)["gate_status"] == "INCOMPLETE"
            wrong = deepcopy(raw)
            wrong["evidence"]["linear_precision"]["contract"]["format"] = "int4"
            assert grade_result(original, wrong)["gate_status"] == "INVALID"
        assert len({json.dumps(r["evidence"]["initial_parameters"], sort_keys=True) for r in receipts}) == 1
        assert len({r["evidence"]["batch_sequence_sha256"] for r in receipts}) == 1
    finally:
        torch.set_num_threads(old)

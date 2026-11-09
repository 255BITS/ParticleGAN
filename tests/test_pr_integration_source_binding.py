"""Integrated future sources do not rewrite or qualify archived research."""
import json
from pathlib import Path
from pr_integration_compat import before_pr_integration

ROOT = Path(__file__).resolve().parents[1]


def test_every_integration_binding_preserves_the_exact_parent_contract():
    receipts = list((ROOT / "reports/forge").glob("pr-integration-202611*/pr-*.json"))
    assert receipts
    for path in receipts:
        receipt = json.loads(path.read_text())
        for row in receipt["tasks"]:
            relative = row["previous_task_path"]
            current = json.loads((ROOT / relative).read_text())
            assert before_pr_integration(ROOT, current, relative) != current


def test_misgan_v2_keeps_coupled_evidence_separate_and_unmeasured():
    import hashlib
    import subprocess
    receipt = json.loads((ROOT / "reports/forge/pr-integration-20261109/misgan-execution-v2.json").read_text())
    assert receipt["qualification_input"] is False
    assert receipt["training_rerun"] is False
    assert receipt["historical_quality_inherited"] is False
    assert receipt["corrected_execution_quality"] == "unmeasured"
    for record in [*receipt["coupled_v1"].values(), *receipt["unchanged_research_results"]]:
        data = subprocess.check_output([
            "git", "show", record["source_commit"] + ":" + record["source_path"]], cwd=ROOT)
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == record["git_blob"]
        if "archived_path" in record:
            assert (ROOT / record["archived_path"]).read_bytes() == data
        elif record in receipt["unchanged_research_results"]:
            assert (ROOT / record["source_path"]).read_bytes() == data
    current = receipt["independent_v2"]
    for member in ("provider", "task"):
        assert hashlib.sha256((ROOT / current[member + "_path"]).read_bytes()).hexdigest() == current[member + "_sha256"]
    task = json.loads((ROOT / current["task_path"]).read_text())
    assert task["execution"]["execution_contract"] == receipt["version"]
    assert task["execution"]["host_definition"]["execution_contract"] == receipt["version"]
    view = json.loads((ROOT / "configs/forge/views/misgan-bispf-research-v1.json").read_text())
    assert view["revision"] == 2
    assert "v2 quality is unmeasured" in view["policy_change_reason"]

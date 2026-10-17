"""Publish saved diagnostic coverage; this script launches no training."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, atomic_text, file_hash, read_json
from experiments.forge.trainer_families import scientific_row_hash

OUT = Path(__file__).resolve().parent
ORIGINAL = Path("reports/forge/recipe-prior-refactor/publication/results.json")


def collect(root=ROOT):
    root = Path(root)
    out = root / OUT.relative_to(ROOT)
    original = read_json(root / ORIGINAL)
    preflight = read_json(out / "preflight.json")
    audit = read_json(out / "clockfree-receipt.json")
    publication = read_json(root / "reports/forge/technique-inventory.json")
    selected = {row["candidate_id"]: row for row in publication["rows"]}
    view = read_json(root / "configs/forge/views/discriminator_stability.json")
    tier2 = sorted(a["task"] for a in view["assignments"]
                   if a["qualification_tier"] == 2 and a["importance"] == "required")
    assert len(tier2) == 21
    assert audit["certificate_validated"] and audit["qualification_input"] is False
    assert audit["attempt_status"] == "completed"
    diagnostics, coverage = [], []
    for family in preflight["families"]:
        row = selected[family["candidate_id"]]
        assert row["candidate_revision"] == family["candidate_revision"]
        assert row["bindings"]["source_digest"] == original["source_digest"] == family["source_digest"]
        cells = [cell for cell in original["task_cells"] if cell["family"] == family["family"]]
        tasks = {}
        for name, support in family["tasks"].items():
            cell = next((cell for cell in cells if cell["task_id"] == name), {})
            status = cell.get("ordinary_status")
            scope, attempt = "ordinary", cell.get("ordinary_attempt")
            if status not in {"PASS", "FAIL"} and cell.get("diagnostic_status") in {"PASS", "FAIL"}:
                status, scope, attempt = cell["diagnostic_status"], "research_diagnostic", cell["diagnostic_attempt"]
            if status not in {"PASS", "FAIL"}:
                status = "BLOCKED" if support["blockers"] else "NOT_RUN"
                scope, attempt = None, None
            reasons = support["blockers"] if status == "BLOCKED" else []
            tasks[name] = {"status": status, "scope": scope, "attempt_id": attempt, "reasons": reasons}
        for name, task in tasks.items():
            if task["status"] != "NOT_RUN":
                continue
            dependencies = family["tasks"][name]["dependencies"]
            task["reasons"] = [f"Required {dep['kind']} parent {dep['task']}: "
                               f"{tasks.get(dep['task'], {}).get('status', 'NOT_RUN')}"
                               for dep in dependencies if tasks.get(dep["task"], {}).get("status") != "PASS"]
            task["reasons"] = task["reasons"] or ["No measurement in the selected runtime cohort; later-tier qualification is unavailable."]
        coverage.append({"family": family["family"], "candidate_id": row["candidate_id"],
                         "scientific_row_sha256": scientific_row_hash(row), "tasks": tasks})
        diagnostic_tasks = {name: tasks[name]["status"] for name in tier2
                            if tasks[name]["scope"] == "research_diagnostic"}
        if diagnostic_tasks:
            assert set(diagnostic_tasks) == set(tier2)
            for name, status in diagnostic_tasks.items():
                certs = [cert for cert in original["certified_results"]
                         if cert["family"] == family["family"] and cert["task_id"] == name
                         and cert["scope"] == "research_diagnostic"]
                assert len(certs) == 1
                cert = certs[0]
                assert cert["gate_status"] == status and cert["qualification_input"] is False
                assert cert["candidate_revision"] == row["candidate_revision"]
                assert cert["source_digest"] == family["source_digest"]
            diagnostics.append({"family": family["family"], "label": "Selected recipe Tier 2 diagnostics; seed 0",
                "tasks": diagnostic_tasks, "requirements": {"1": [], "2": tier2, "3": []},
                "source_digest": family["source_digest"], "runtime": "original recorded cohort"})
    result = audit["task_results"][0]
    assert result["task_id"] == "clockfree_audit" and result["gate_status"] == "PASS"
    assert result["metrics"]["parity_comparisons"] == 4
    media = read_json(out / "media/clockfree_audit.json")
    assert media["recorded_grade"] == "PASS" and media["optimizer_updates_added"] == media["sampling_draws_added"] == 0
    assert media["gif_sha256"] == file_hash(out / "media/clockfree_audit.gif")
    diagnostics.append({"family": "bcap-pure", "label": "Strict clock-free audit; separate Torch 2.13 cohort",
        "tasks": {"clockfree_audit": result["gate_status"]},
        "requirements": {"1": [], "2": [], "3": ["clockfree_audit"]},
        "source_digest": audit["provenance"]["source_digest"], "runtime": audit["runtime"]})
    inputs = {str(path.relative_to(root)): file_hash(path) for path in (
        root / ORIGINAL, out / "preflight.json", out / "clockfree-receipt.json", out / "collect.py", out / "preflight.py",
        out / "media/clockfree_audit.json", out / "media/clockfree_audit.gif")}
    summary = {"schema_version": 1, "scope": "selected_family_diagnostic_coverage", "qualification_input": False,
        "qualification_reuse": False, "maximum_concurrent_tests": 4, "maximum_tests_per_gpu": 2,
        "new_attempts": 1, "new_paid_seconds": result["cost"]["wall_seconds"], "inputs": inputs,
        "diagnostic_rows": diagnostics, "coverage": coverage}
    atomic_json(out / "results.json", summary)
    lines = ["# Selected family diagnostic coverage", "",
        "The selected formulations, numerical gates and qualification are unchanged. Four existing source-matched "
        "Tier 2 diagnostic panels supply 84 measurements without rerunning them. The new strict BCAP clock-free "
        "audit passes all four comparisons in 24 updates; it used a separate Torch 2.13 runtime and grants no "
        "credit to the selected Torch 2.14 cohort.", "",
        f"New training: **1 PASS**, **{summary['new_paid_seconds']:.3f} seconds**. The runner was configured for "
        "at most four tests total and two per GPU. Only one missing independent test was compatible. "
        "All formulations remain unchanged. [Actual-training audit GIF](media/clockfree_audit.gif) "
        "uses the saved proof states and adds no optimizer update or sampling draw.", "",
        "| Family | Existing Tier 2 diagnostics |", "| --- | ---: |"]
    for row in diagnostics[:-1]:
        counts = Counter(row["tasks"].values())
        lines.append(f"| {row['family']} | {counts['PASS']}/21 ({counts['FAIL']} fail) |")
    lines += ["", "The strict clock-free audits for K3P, KA2, R1/R2 and released GAN require removing declared "
        "clock dependencies, so they were skipped as incompatible. Atlas and E22 lack the required policy/task "
        "contracts for every ordinary question. The long ring, target-shift and native continuations retain "
        "their explicit parent dependencies: the recorded mode-hold and native acquisition gates fail. "
        "Those tests remain not run; a diagnostic view does not manufacture a passing parent or a checkpoint.", "",
        "The table below counts 35 distinct ordinary questions once per selected recipe. PASS and FAIL are "
        "measurements; `incompatible` and `not run` are separate. Existing Tier 2 diagnostics stay outside "
        "ordinary qualification. The new BCAP audit is shown separately because its runtime differs.", "",
        "| Task | BCAP | Atlas | E22 | GAN v3 | K3P | KA2 | R1/R2 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    ordered = [next(f for f in coverage if f["family"] == name) for name in
               ("bcap-pure", "atlas", "e22", "release07-gan-v3", "k3p", "ka2", "r1r2")]
    for name in sorted(ordered[0]["tasks"]):
        statuses = []
        for family in ordered:
            task = family["tasks"][name]
            value = {"BLOCKED": "incompatible", "NOT_RUN": "not run"}.get(task["status"], task["status"])
            if task["scope"] == "research_diagnostic":
                value += " (diagnostic)"
            statuses.append(value)
        lines.append("| `" + name + "` | " + " | ".join(statuses) + " |")
    lines += ["", "[Exact statuses and skip reasons](results.json) · [Source-bound preflight](preflight.json) · "
        "[New audit receipt](clockfree-receipt.json) · [Original numerical diagnostics](../recipe-prior-refactor/publication/README.md)", "",
        "Reproduce the diagnostic with `bcap-clockfree-coverage-v1` on the frozen source "
        "`a0493ca3c213d559523eadc76e244ec9f9c1f81f`, preserving its exact task and recipe. "
        "Raw proof tensors and logs remain local under "
        "`/ml2/hypergan/fill-tier-values-20261014-run/queue/selected-family-coverage-20261014/"
        "a61d4c1b615e49d4b6423bb7f6937923/`. The compact receipt binds their certified result.", "",
        "Refresh the compatibility and readout without training:", "", "```sh",
        "python reports/forge/tier-coverage-20261014/preflight.py --source-root <exact-source-checkout>",
        "python reports/forge/tier-coverage-20261014/collect.py",
        "python reports/forge/regenerate_technique_inventory.py --refresh-publication", "```", "",
        "Next work should address the failed acquisition/retention gates as separately declared research. "
        "No recipe tuning, seed variants or promotion follows from these coverage results.", ""]
    atomic_text(out / "README.md", "\n".join(lines))
    return summary


if __name__ == "__main__":
    result = collect()
    print(json.dumps({"new_attempts": result["new_attempts"], "diagnostic_panels": len(result["diagnostic_rows"])}))

"""Preserve the complete configuration search as a compact numerical ledger.

Run from the repository root after experiments finish. Full raw evidence stays
under artifacts/toy100; promoted gate runs are archived separately in this report.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.toy100.gate import score_run
from benchmarks.toy100.metrics import REQUIREMENTS


def collect():
    output = ROOT / "reports/toy100/search"
    output.mkdir(parents=True, exist_ok=True)
    source_output = output / "sources"
    source_output.mkdir(exist_ok=True)
    sources = {}
    for path in (ROOT / "artifacts/toy100").rglob("*probe*.py"):
        content = path.read_bytes()
        sources[hashlib.sha256(content).hexdigest()] = content
    trials = []
    for path in sorted((ROOT / "artifacts/toy100").rglob("summary.json")):
        if "baseline" in path.parts:
            continue
        summary = json.loads(path.read_text())
        if summary.get("status") not in ("complete", "error"):
            continue
        problem = summary.get("problem")
        if not problem:
            continue
        verdict = score_run(path.parent, problem)
        provenance = summary.get("provenance", {})
        archived_sources = {}
        for name, digest in provenance.get("source_sha256", {}).items():
            if digest in sources:
                destination = source_output / f"{digest}.py"
                destination.write_bytes(sources[digest])
                archived_sources[name] = str(destination.relative_to(output))
        events = path.parent / "events.jsonl"
        evaluations = [row for line in events.read_text().splitlines()
                       if (row := json.loads(line)).get("event") == "eval"] if events.exists() else []
        trials.append({"run": str(path.parent.relative_to(ROOT)),
                       "status": verdict["status"], "problem": problem,
                       "config": summary.get("config", {}), "provenance": provenance,
                       "environment": summary.get("environment", {}),
                       "train_seconds": summary.get("train_seconds"),
                       "total_seconds": summary.get("total_seconds"),
                       "verdict": verdict, "evaluations": evaluations,
                       "archived_probe_sources": archived_sources})
    (output / "trials.json").write_text(json.dumps({
        "protocol": "toy100-v1", "requirements": REQUIREMENTS,
        "selection_seed": 1234, "trials": trials,
        "note": "All completed/error trials retained. Single-problem PASS is not a suite PASS. Raw snapshots remain local except separately archived promoted runs."
    }, indent=2, allow_nan=False) + "\n")
    lines = ["# Configuration search ledger", "",
             "Every completed attempt is listed, including failures. The seed and gate thresholds were fixed. "
             "Times include evaluation and are device-specific. A row passing one problem does not certify the suite.", "",
             "[Configs, provenance, and complete evaluation curves](trials.json)", "",
             "| Trial | Problem | Device | Budget | Modes | HQ | Mass TV | First 100 | Stable from | Gate | Seconds |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    for trial in trials:
        v = trial["verdict"]
        m = v.get("final_metrics", {})
        def f(key):
            value = m.get(key)
            return f"{value:.4f}" if isinstance(value, float) else str(value if value is not None else "—")
        lines.append("| " + " | ".join([
            trial["run"].removeprefix("artifacts/toy100/").removesuffix("/" + trial["problem"]),
            trial["problem"], trial["environment"].get("device", "—"),
            str(v.get("budget_steps", "—")), f("modes"), f("hq"), f("mass_tv"),
            str(v.get("first_full_coverage_step") or "—"), str(v.get("stable_from_step") or "—"),
            trial["status"], f"{trial['total_seconds']:.1f}" if trial["total_seconds"] else "—",
        ]) + " |")
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"trials": len(trials), "individual_passes": sum(t['status'] == 'PASS' for t in trials),
                      "output": str(output)}))


if __name__ == "__main__":
    collect()

"""Summarize dig2 clock-audit-measured attempts (no training).

Usage: PYTHONPATH=. python reports/forge/dig2-20261016/clock-audit-measured/summarize.py <campaign_dir> [out.json]
Also regrades each v2 clock probe under internal_counter_policy="declared" (strict) from the same saved evidence.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

from experiments.forge.clockfree import measured_v2_decision, verify_probe
from experiments.forge.views import grade_result

LIMIT = 0.85
GATES = (("modes", ">=", 16), ("mass_tv", "<=", .15), ("hq", ">=", .85),
         ("component_covariance_error", "<=", LIMIT), ("component_min_eigen_ratio", ">=", .15))


def ok(obs, skip=None):
    return all((obs[m] >= t if op == ">=" else obs[m] <= t) for m, op, t in GATES if m != skip)


root = Path(sys.argv[1])
tasks = Path("configs/forge/tasks")
rows = {}
for attempt in sorted(p for p in root.iterdir() if (p / "graded-result.json").exists()):
    request = json.loads((attempt / "request.json").read_text())
    cand = request["request"]["study"]["candidate"]
    graded = json.loads((attempt / "graded-result.json").read_text())["grades"]
    raw = json.loads((attempt / "raw-result.json").read_text())
    row = rows.setdefault(cand, {"attempts": {}})
    for task_id, grade in graded.items():
        row["attempts"][task_id] = attempt.name
        seconds = raw.get("cost", {}).get("adapter_seconds") or raw.get("cost", {}).get("training_seconds")
        if task_id.startswith("ring16"):
            obs = raw["evidence"]["observations"]
            conv = grade.get("evaluator_result", {}).get("convergence", {})
            cov = [o["component_covariance_error"] for o in obs]
            best = min(range(len(obs)), key=lambda i: cov[i])
            last_fail = max((o["step"] for o in obs if not ok(o)), default=None)
            row["ring16"] = {
                "status": grade["status"], "passing_obs": conv.get("passing_observations"),
                "terminal_streak": conv.get("passing_suffix"), "stable_from_step": conv.get("stable_from_step"),
                "first_pass_step": conv.get("first_pass_step"), "last_fail_step": last_fail,
                "final_cov": round(cov[-1], 4), "min_cov": round(cov[best], 4), "min_cov_step": obs[best]["step"],
                "mean_cov_last16": round(sum(cov[-16:]) / 16, 4),
                "final_hq": round(obs[-1]["hq"], 4), "final_mass_tv": round(obs[-1]["mass_tv"], 4),
                "final_min_eig": round(obs[-1]["component_min_eigen_ratio"], 4),
                "obs_pass_except_cov": sum(ok(o, "component_covariance_error") for o in obs),
                "observations": len(obs), "seconds": seconds,
                "cov_at": {o["step"]: round(o["component_covariance_error"], 3) for o in obs
                           if o["step"] in (1600, 1617, 3200, 3217, 4800, 4817)},
            }
        else:
            row[task_id] = {"status": grade["status"], "reason": grade["reasons"][0][:240], "seconds": seconds}
            if task_id.startswith("clockfree_audit_measured_v2"):
                task = json.loads((tasks / f"{task_id}.json").read_text())
                strict = deepcopy(task)
                strict["evaluation"]["internal_counter_policy"] = "declared"
                strict_grade = grade_result(strict, raw)
                comparisons, _, proof = verify_probe(task, raw["evidence"], with_proof=True)
                decision = measured_v2_decision(proof, task["execution"], comparisons, "measured")
                row["v2_strict"] = {"status": strict_grade["status"], "reason": strict_grade["reasons"][0][:240]}
                row["v2_detail"] = {k: decision[k] for k in ("differing_conditions", "overridden", "standing",
                                                            "power_witness", "counter_witness")}
out = json.dumps(rows, indent=1, sort_keys=True)
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(out + "\n")
for cand, row in sorted(rows.items()):
    r = row.get("ring16", {})
    print(f"{cand:24s} ring={r.get('status')} streak={r.get('terminal_streak')} pass={r.get('passing_obs')} "
          f"final_cov={r.get('final_cov')} min_cov={r.get('min_cov')}@{r.get('min_cov_step')} last_fail={r.get('last_fail_step')} "
          f"v1={row.get('clockfree_audit_measurement_v1', {}).get('status')} "
          f"v2={row.get('clockfree_audit_measured_v2-clockdig', {}).get('status')} "
          f"v2strict={row.get('v2_strict', {}).get('status')} s={r.get('seconds')}")

"""CPU-only recount: what the selected-recipe board totals would be with ring16 at 6,400 updates.

Reads committed publication inputs only (no training, no regrading, nothing written to the board):

    python reports/forge/dig2-20261016/ring16-cap-recount.py reports/forge/dig2-20261016/ring16-cap-results.json

Counting rules mirror the published "Selected recipes" table (technique-inventory.md, view revision 9):
a test counts once in every ordinary view that requires it; Tier N counts only when every Tier < N
view requirement of the family passes (BCAP: Tier 1 22/22 so Tier 2 counts, Tier 3 does not).

Scenarios
  published   recorded statuses (reproduces the published totals as a check)
  strict      ring16_acquisition replaced by the measured ring16_acquisition_6400 verdict; the
              clock-free audit stays a Tier 1 requirement
  diag        as strict, but a family whose only remaining Tier 1 failure is the clock-free audit
              has its existing same-recipe Tier 2 diagnostic attempts counted (Tier 3 still needs
              a complete Tier 2)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VIEWS = ("adaptation", "clockfree_continuous", "discriminator_stability", "formulation_comparison",
         "host_profile_transfer", "quality_coverage")
FAMILIES = {"BCAP": "recipe-prior-refactor-bcap-pure-v1", "BISPF": "bispf-v1",
            "GAN v3 release 0.7": "recipe-prior-refactor-release07-gan-v3-v1",
            "K3P": "recipe-prior-refactor-k3p-v1", "KA2": "recipe-prior-refactor-ka2-v1",
            "R1/R2": "recipe-prior-refactor-r1r2-v1"}
CLOCK = "clockfree_audit_measurement_v1"


def requirements():
    out = {}
    for name in VIEWS:
        view = json.loads((ROOT / f"configs/forge/views/{name}.json").read_text())
        for a in view["assignments"]:
            if a["importance"] == "required":
                out.setdefault(name, {}).setdefault(a["qualification_tier"], []).append(a["task"])
    return out


def statuses():
    records = json.loads((ROOT / "reports/forge/attempt-coverage.json").read_text())["records"]
    selected, diagnostic = {}, {}
    for r in records:
        target = diagnostic if r["diagnostic"] else selected
        target.setdefault(r["candidate_id"], {})[r["task"]] = r["status"]
    return selected, diagnostic


def count(req, status, *, tier2_credit):
    tiers = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    t1 = [(v, t) for v in req for t in req[v].get(1, [])]
    t1_fail = sorted({t for _, t in t1 if status.get(t) != "PASS"})
    t1_complete = not t1_fail or (tier2_credit and t1_fail == [CLOCK])
    t2_tasks = {t for v in req for t in req[v].get(2, [])}
    t2_complete = t1_complete and all(status.get(t) == "PASS" for t in t2_tasks)
    for view, by_tier in req.items():
        for tier, tasks in by_tier.items():
            counted = tier == 1 or (tier == 2 and t1_complete) or (tier == 3 and t2_complete)
            tiers[tier][1] += len(tasks)
            tiers[tier][0] += sum(counted and status.get(t) == "PASS" for t in tasks)
    return tiers, t1_fail


def main(results_path):
    measured = json.loads(Path(results_path).read_text())["ring16_acquisition_6400"]
    req = requirements()
    selected, diagnostic = statuses()
    rows = []
    for label, cid in FAMILIES.items():
        base = dict(selected.get(cid, {}))
        for scenario in ("published", "strict", "diag"):
            status = dict(base)
            if scenario != "published":
                status["ring16_acquisition"] = measured[cid]
            if scenario == "diag":
                # Fill only cells the selected cohort did not measure.
                for task, value in diagnostic.get(cid, {}).items():
                    status.setdefault(task, value)
            tiers, t1_fail = count(req, status, tier2_credit=scenario == "diag")
            total = [sum(t[0] for t in tiers.values()), sum(t[1] for t in tiers.values())]
            rows.append({"family": label, "candidate": cid, "scenario": scenario,
                         "tier1": tiers[1], "tier2": tiers[2], "tier3": tiers[3], "total": total,
                         "tier1_failures": t1_fail})
    for row in rows:
        print(f"{row['family']:<20} {row['scenario']:<9} T1 {row['tier1'][0]}/{row['tier1'][1]}  "
              f"T2 {row['tier2'][0]}/{row['tier2'][1]}  T3 {row['tier3'][0]}/{row['tier3'][1]}  "
              f"total {row['total'][0]}/{row['total'][1]}  T1 failures: {', '.join(row['tier1_failures']) or '-'}")
    return rows


if __name__ == "__main__":
    out = main(sys.argv[1])
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + "\n")

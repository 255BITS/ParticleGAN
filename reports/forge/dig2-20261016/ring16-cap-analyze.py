"""Summarise the ring16-cap attempts from a Forge queue root (CPU only, reads saved receipts).

    python reports/forge/dig2-20261016/ring16-cap-analyze.py <queue_root> <campaign> <out.json>

For each attempt: the Forge grade, then sustained-window diagnostics recomputed from the saved
observation curve with the task's unchanged thresholds (terminal streak, longest streak, first
5-check streak, last failing step and failed gates, late collapse) and the verdict a 96-check
grid would give (every 4th of the 384 checkpoints at 6,400 updates).
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OPS = {">=": lambda v, t: v >= t, "<=": lambda v, t: v <= t}


def passes(row, thresholds):
    failed = [m for m, op, t in thresholds if not (row.get(m) is not None and OPS[op](row[m], t))]
    return not failed, failed


def streaks(flags):
    longest = run = 0
    first5 = None
    for i, ok in enumerate(flags):
        run = run + 1 if ok else 0
        longest = max(longest, run)
        if run == 5 and first5 is None:
            first5 = i
    terminal = 0
    for ok in reversed(flags):
        if not ok:
            break
        terminal += 1
    return terminal, longest, first5


def summarise(attempt):
    request = json.loads((attempt / "request.json").read_text())
    graded = json.loads((attempt / "graded-result.json").read_text())
    raw = json.loads((attempt / "raw-result.json").read_text())
    (task_id, grade), = graded["grades"].items()
    task = json.loads((ROOT / f"configs/forge/tasks/{task_id}.json").read_text())
    if not task_id.startswith("ring16_acquisition"):
        return None
    thresholds = [t for t in task["evaluation"]["thresholds"] if t[0] != "sample_count"]
    obs = raw["evidence"]["observations"]
    results = [passes(o, thresholds) for o in obs]
    flags = [ok for ok, _ in results]
    terminal, longest, first5 = streaks(flags)
    fails = [(o["step"], f) for o, (ok, f) in zip(obs, results) if not ok]
    steps = task["execution"]["steps"]
    grid96 = {math.ceil(i * steps / 96) for i in range(1, 97)}
    sub = [ok for o, ok in zip(obs, flags) if o["step"] in grid96]
    sub_terminal, sub_longest, _ = streaks(sub)
    cov = [o["component_covariance_error"] for o in obs]
    eig = [o["component_min_eigen_ratio"] for o in obs]
    final = {k: obs[-1][k] for k in ("modes", "mass_tv", "hq", "component_covariance_error",
                                     "component_min_eigen_ratio")}
    candidate = request["request"]["candidate"]["id"]
    return {
        "attempt": attempt.name, "candidate": candidate, "task": task_id, "status": grade["status"],
        "source_digest": graded["source_digest"], "observations": len(obs), "steps": steps,
        "passing_observations": sum(flags), "terminal_streak": terminal, "longest_streak": longest,
        "first_sustained_step": obs[first5]["step"] if first5 is not None else None,
        "late_collapse": longest >= 5 and terminal < 5,
        "last_failure": ({"step": fails[-1][0], "failed": fails[-1][1]} if fails else None),
        "grid96": ({"terminal_streak": sub_terminal, "longest_streak": sub_longest,
                    "would_pass": sub_terminal >= 5 and flags[-1]} if steps > 1600 else None),
        "cov_final": cov[-1], "cov_min": min(cov), "cov_min_step": obs[cov.index(min(cov))]["step"],
        "cov_max_last_quarter": max(cov[len(cov) * 3 // 4:]),
        "eig_min_last_quarter": min(eig[len(eig) * 3 // 4:]),
        "final": final,
        "elapsed_seconds": raw.get("cost", {}).get("adapter_seconds"),
        "completed_steps": raw.get("cost", {}).get("completed_steps"),
    }


def main(queue, campaign, out):
    rows = []
    for attempt in sorted((Path(queue) / campaign).iterdir()):
        if (attempt / "graded-result.json").exists():
            row = summarise(attempt)
            if row is not None:
                rows.append(row)
    Path(out).write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print(f"{r['candidate']:<45} {r['task']:<24} {r['status']:<5} pass {r['passing_observations']:>3}/"
              f"{r['observations']:<3} term {r['terminal_streak']:>3} long {r['longest_streak']:>3} "
              f"first5 {r['first_sustained_step']} last_fail {r['last_failure']} cov {r['cov_final']:.3f} "
              f"g96 {r['grid96']} t={r['elapsed_seconds']}")


if __name__ == "__main__":
    main(*sys.argv[1:4])

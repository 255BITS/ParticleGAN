"""Selection rules fixed before candidate evaluation; no result-based retiering."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math

from benchmarks.locked_shared import baseline
from benchmarks.locked_shared.observation import sustained

TIERS = ("required", "ranking", "diagnostic")
SELECTION = (
    "Eligibility requires every required live test sustained. All required and ranking "
    "tests must be attempted before selection. Rank eligible rows by sustained pass "
    "fraction averaged equally across data, dynamics and image domains and equally "
    "across ranking families within each domain, then worst family fraction, then "
    "lower similarly balanced final metric shortfall, then lower confirmation/budget "
    "(failure=2). Shortfall is mean positive "
    "bound violation divided by abs(bound), with scale 1 for zero bounds and each "
    "metric capped at 2; invalid or missing results receive 2. "
    "Diagnostics never affect eligibility, score, or tie breaking. Errors are failures; "
    "missing attempts cannot improve rank. EMA never affects selection."
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def requirements(spec):
    """Image hosts additionally declare measurement settings in their native schema."""
    thresholds = spec["thresholds"]
    if isinstance(thresholds, dict):
        return [["modes", ">=", thresholds["modes"]], ["hq", ">=", thresholds["hq_min"]]]
    return thresholds


def required_tasks():
    reasons = {
        "two_pole": "Basic adversarial movement remains bounded.",
        "trajectory": "Preserves the shared trajectory's identity constraint.",
        "residual_student": "Moves the intended residual without choosing the wrong target.",
        "unipolar": "Target coverage preserves unrelated content.",
        "ae_gan_hold": "Adversarial training preserves reconstruction and hold behavior.",
        "cover_leftover": "Coverage preserves both content and the unwanted remainder constraint.",
        "unused_token_hold": "Unused controls remain unchanged while active controls move.",
        "mid_scale_identity": "Intermediate control strengths preserve identity and target magnitude.",
        "mode_hold": "All eight modes must remain present with sufficient sample quality.",
    }
    return [dict(name=name, family="existing_behavior", runner="legacy", split="development", phase="fit",
                 tier="required", importance_reason=reasons[name],
                 limitations="Small extracted behavioral host; not evidence of natural-image transfer.",
                 steps=steps, thresholds=([("modes", ">=", 8), ("hq", ">=", .9)]
                                         if name == "mode_hold" else baseline.METRICS[name]))
            for name, steps in baseline.BUDGETS.items()]


def validate_manifest(manifest):
    tasks = manifest["tasks"]
    names = set()
    family_roles = {}
    for spec in tasks:
        name = spec["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("task names must be nonempty and unique")
        names.add(name)
        if spec["tier"] not in TIERS:
            raise ValueError("unknown importance tier")
        if spec["split"] not in ("development", "reserved"):
            raise ValueError("unknown task split")
        if spec["phase"] not in ("fit", "validation", "reserved"):
            raise ValueError("unknown development phase")
        if (spec["split"] == "reserved") != (spec["phase"] == "reserved"):
            raise ValueError("reserved tasks cannot participate in development")
        if type(spec["steps"]) is not int or spec["steps"] < 24:
            raise ValueError("at least 24 training steps required")
        for field in ("family", "importance_reason", "limitations", "runner"):
            if not isinstance(spec[field], str) or not spec[field].strip():
                raise ValueError(f"missing {field}")
        if not spec["thresholds"]:
            raise ValueError("behavioral thresholds required")
        for key, op, value in requirements(spec):
            if not isinstance(key, str) or not key or op not in (">=", "<="):
                raise ValueError("invalid metric requirement")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("threshold must be finite")
        # Reserve whole named families, not shuffled examples from a fitted family.
        split = family_roles.setdefault(spec["family"], spec["split"])
        if split != spec["split"]:
            raise ValueError("a reserved family also appears in development")
    if not any(t["tier"] == "required" and t["split"] == "development" for t in tasks):
        raise ValueError("an explicit required regression set is needed")
    return manifest


def test_verdict(spec, result):
    """Recompute sustained success from complete live curves, never trust a stamp."""
    if result is None:
        return dict(status="MISSING", attempted=False, passed=False, confirmation_fraction=2., shortfall=2.)
    if result.get("error"):
        return dict(status="ERROR", attempted=True, passed=False, confirmation_fraction=2., shortfall=2.)
    observations = result.get("observations", result.get("curve", []))
    steps = sorted({math.ceil(i * spec["steps"] / 24) for i in range(1, 25)})
    try:
        convergence = sustained(observations, requirements(spec), expected_steps=steps)
    except (KeyError, TypeError, ValueError):
        return dict(status="INVALID", attempted=True, passed=False, confirmation_fraction=2., shortfall=2.)
    cells = baseline.score_metrics(result.get("live", {}), requirements(spec))
    passed = (convergence["confirmed_step"] is not None
              and all(c["status"] == "PASS" for c in cells))
    status = "PASS" if passed else "FAIL" if convergence["complete"] else "INCOMPLETE"
    deficits = [2. if c["margin"] is None else min(2., max(0., -c["margin"]) / (abs(c["threshold"]) or 1.))
                for c in cells]
    return dict(status=status, attempted=True, passed=passed, metrics=cells,
                convergence=convergence,
                shortfall=sum(deficits) / len(deficits) if convergence["complete"] else 2.,
                confirmation_fraction=convergence["confirmed_step"] / spec["steps"] if passed else 2.)


def summarize(manifest, results, *, phase=None):
    specs = [t for t in manifest["tasks"] if t["split"] == "development"
             and (phase is None or t["phase"] == phase)]
    verdicts = {s["name"]: test_verdict(s, results.get(s["name"])) for s in specs}
    counts = {}
    for tier in TIERS:
        subset = [verdicts[s["name"]] for s in specs if s["tier"] == tier]
        counts[tier] = dict(total=len(subset), passed=sum(v["passed"] for v in subset),
                            attempted=sum(v["attempted"] for v in subset))
    families = defaultdict(list)
    for spec in specs:
        if spec["tier"] == "ranking":
            families[(spec["runner"], spec["family"])].append(verdicts[spec["name"]])
    family_scores = {f"{domain}/{name}": dict(domain=domain, total=len(rows), passed=sum(v["passed"] for v in rows),
                               pass_fraction=sum(v["passed"] for v in rows) / len(rows),
                               shortfall=sum(v["shortfall"] for v in rows) / len(rows),
                               confirmation_fraction=sum(v["confirmation_fraction"] for v in rows) / len(rows))
                     for (domain, name), rows in sorted(families.items())}
    pass_rates = [f["pass_fraction"] for f in family_scores.values()]
    domains = defaultdict(list)
    for family in family_scores.values():
        domains[family["domain"]].append(family)
    domain_scores = {name: {key: sum(f[key] for f in rows) / len(rows)
                           for key in ("pass_fraction", "shortfall", "confirmation_fraction")}
                     for name, rows in sorted(domains.items())}
    balanced = lambda key, fallback: (sum(d[key] for d in domain_scores.values()) / len(domain_scores)
                                     if domain_scores else fallback)
    required = counts["required"]
    eligible = required["total"] > 0 and required["passed"] == required["total"]
    complete = all(c["attempted"] == c["total"] for tier, c in counts.items() if tier != "diagnostic")
    return dict(eligible=eligible, selection_ready=eligible and complete, evaluation_complete=complete,
                counts=counts, families=family_scores, domains=domain_scores, verdicts=verdicts,
                ranking_pass_fraction=balanced("pass_fraction", 0.),
                worst_family_fraction=min(pass_rates) if pass_rates else 0.,
                shortfall=balanced("shortfall", 2.),
                confirmation_fraction=balanced("confirmation_fraction", 2.))


def selection_key(summary):
    """Smaller is better. Diagnostic results deliberately have no path here."""
    return (not summary["selection_ready"],
            -summary["counts"]["required"]["passed"],
            -summary["ranking_pass_fraction"],
            -summary["worst_family_fraction"], summary["shortfall"], summary["confirmation_fraction"])


def reference_evidence(spec, attempts):
    """Solvability evidence is independent of predeclared relevance/importance."""
    observed = [(label, test_verdict(spec, result)) for label, result in attempts.items()]
    passing = [label for label, verdict in observed if verdict["passed"]]
    return dict(status="demonstrated" if passing else "not demonstrated" if observed else "unmeasured",
                passing_references=passing, attempted_references=[label for label, _ in observed],
                caveat="Evidence is for this exact fixed-seed budget and setup; transfer predictiveness is unmeasured.")

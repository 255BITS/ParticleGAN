"""Fit across task families with declared importance; reserve transfer until freeze.

python -u -m benchmarks.transfer_suite.suite --output /tmp/transfer-suite-v1
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import tarfile
import time
import traceback

import torch

from benchmarks.locked_shared import baseline
from benchmarks.smart_descent import study, evaluate
from .protocol import (SELECTION, digest, reference_evidence, required_tasks, requirements, selection_key,
                       summarize, test_verdict, validate_manifest)


def modules():
    from . import vector_tasks, stress_tasks, image_tasks
    return dict(vector=vector_tasks, stress=stress_tasks, image=image_tasks)


def manifest(generations=2, population=6):
    tasks = required_tasks()
    for name, module in modules().items():
        for split, group in (("development", module.TASKS),
                             ("reserved", getattr(module, "RESERVED", getattr(module, "RESERVED_TASKS", [])))):
            if isinstance(group, dict):
                group = [group]
            for original in group:
                spec = deepcopy(original)
                spec.update(runner=name, split=split,
                            phase="reserved" if split == "reserved" else "validation" if name == "image" else "fit")
                tasks.append(spec)
    return validate_manifest(dict(version="transfer-importance-v1", tasks=tasks, selection=SELECTION,
        gan_seed=0, observations=24, minimum_final_passing_observations=5,
        evidence="Importance is declared before candidate search; reference solvability is reported independently. "
                 "Transfer predictiveness of each individual test has not been established.",
        search=dict(generations=generations, population=population, proposal_rng=2731,
                    initial_lr_weight_std=.008, regularization_weights="fixed zero",
                    screening="Run all nine required tests first; do not fit additional tasks after a required failure.",
                    validation="Advance the two best eligible nonzero fitting policies and the cosine control to all image cases.",
                    freeze="Choose full-development winner and best eligible nonzero challenger after validation. "
                           "Evaluate one frozen challenger, its bias-only ablation, and cosine on reserved families once.")))


def snapshot(output):
    root = Path(__file__).resolve().parents[2]
    paths = sorted({*root.glob("particlegan/**/*.py"), *root.glob("benchmarks/locked_shared/**/*.py"),
                    *root.glob("benchmarks/smart_descent/*.py"), *root.glob("benchmarks/transfer_suite/*.py"),
                    root / "lib/toy_models.py", root / "lib/toy_metrics.py",
                    root / "benchmarks/learned_lr_evaluation.py"})
    hashes = {}
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for path in paths:
            data = path.read_bytes()
            name = str(path.relative_to(root))
            hashes[name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            archive.addfile(info, io.BytesIO(data))
    return dict(source_sha256=hashes, python=platform.python_version(), torch=str(torch.__version__),
                torch_git_revision=torch.version.git_version, torch_build=torch.__config__.show(),
                cpu_capability=torch.backends.cpu.get_cpu_capability(), device="cpu", threads=1)


def verify_source(protocol):
    root = Path(__file__).resolve().parents[2]
    changed = [name for name, expected in protocol["source_sha256"].items()
               if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected]
    if changed:
        raise RuntimeError(f"source changed during fitting: {changed}")


def run_episode(spec, card, *, ablation="none", fixed=False, allow_reserved=False):
    if spec["split"] == "reserved" and not allow_reserved:
        raise ValueError("reserved task requires an explicit frozen evaluation")
    if spec["runner"] == "legacy":
        return evaluate.fixed_toy(spec["name"], card) if fixed else study.run_toy(spec["name"], card, ablation=ablation)
    runner = modules()[spec["runner"]].run_episode
    # Modules may additionally enforce their own reserved-access guard.
    import inspect
    options = dict(ablation=ablation, fixed=fixed)
    if "allow_reserved" in inspect.signature(runner).parameters:
        options["allow_reserved"] = allow_reserved
    return runner(spec, card, **options)


def render(report, output):
    declared = report["manifest"]
    lines = ["# Transfer suite: importance-aware selection", "",
             "Required tests determine eligibility. Ranking tests measure useful generalization without vetoing selection. "
             "Diagnostic stress tests have zero influence on selection and remain visible. Tiers never change because a candidate failed.", "",
             "The nine existing behavioral regressions remain required. The 24 new cases are development data: "
             "16 vector/dynamics cases for fitting and eight image cases for validation. Three reserved families are evaluated "
             "only after the challenger is frozen. All training uses seed 0; EMA never determines success.", "",
             "## Candidate leaderboard", "",
             "| Candidate | Eligible / ready | Required sustained | Ranking sustained | Balanced pass score | Diagnostic sustained | Ranking shortfall | Phase |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    rows = sorted(report["rows"], key=lambda r: selection_key(summarize(declared, r["results"])))
    for row in rows:
        score = summarize(declared, row["results"])
        counts = score["counts"]
        cell = lambda tier: f"{counts[tier]['passed']}/{counts[tier]['total']} ({counts[tier]['attempted']} tried)"
        lines.append(f"| {row['name']} | {'Yes' if score['eligible'] else 'No'} / {'Yes' if score['selection_ready'] else 'No'} | "
                     f"{cell('required')} | {cell('ranking')} | {score['ranking_pass_fraction']:.1%} | "
                     f"{cell('diagnostic')} | {score['shortfall']:.4f} | {row['phase']} |")
    lines += ["", "Eligible means all nine required live tests sustain success. Ready additionally means every required/ranking "
              "case was attempted. Missing cases remain in denominators. A recorded training error counts as a failure. "
              "Balanced pass score gives data, dynamics and images equal weight, then averages ranking families within each domain equally. "
              "Duplicating cases in one family does not give that family more weight. Diagnostic results, timing and EMA cannot break ties.", "",
              declared["selection"], "",
              "## Test importance and reference evidence", "",
              "Importance describes intended use, not how easy a test is. Reference evidence establishes solvability for an exact "
              "budget/setup; it does not establish how well that test predicts real-world transfer. That predictive value is currently unmeasured.", "",
              "| Test | Use / family | Importance | Why it matters | Reference sustained success | Limitation |",
              "| --- | --- | --- | --- | --- | --- |"]
    for spec in declared["tasks"]:
        evidence = reference_evidence(spec, {row["name"]: row["results"][spec["name"]]
                                             for row in [*report["rows"], *report.get("transfer", [])]
                                             if row.get("fixed") and spec["name"] in row["results"]})
        reference = ", ".join(evidence["passing_references"]) or evidence["status"]
        clean = lambda value: str(value).replace("|", "/").replace("\n", " ")
        lines.append("| " + " | ".join(map(clean, [spec["name"], f"{spec['phase']} / {spec['family']}",
                                                   spec["tier"], spec["importance_reason"], reference, spec["limitations"]])) + " |")
    lines += ["", "## Per-test live results", ""]
    for tier in ("required", "ranking", "diagnostic"):
        specs = [s for s in declared["tasks"] if s["split"] == "development" and s["tier"] == tier]
        lines += [f"### {tier.title()}", "", "| Test | " + " | ".join(r["name"] for r in report["rows"]) + " |",
                  "| --- | " + " | ".join("---" for _ in report["rows"]) + " |"]
        for spec in specs:
            cells = [test_verdict(spec, row["results"].get(spec["name"]))["status"] for row in report["rows"]]
            lines.append(f"| {spec['name']} | " + " | ".join(cells) + " |")
        lines.append("")
    if report.get("frozen"):
        lines += ["## Frozen transfer", "", f"Development winner: **{report['frozen']['winner']}**. "
                  f"Frozen nonzero challenger: **{report['frozen']['challenger']}**. "
                  "All reserved results are excluded from this choice.", "",
                  "| Reserved family / test | Arm | Sustained live | Final live metrics | EMA | Seconds |",
                  "| --- | --- | --- | --- | --- | ---: |"]
        for row in report.get("transfer", []):
            for spec in declared["tasks"]:
                if spec["split"] != "reserved" or spec["name"] not in row["results"]:
                    continue
                result = row["results"][spec["name"]]
                verdict = test_verdict(spec, result)
                keys = [k for k, _, _ in requirements(spec)]
                live = ", ".join(f"{k}={result.get('live', {}).get(k, '—')}" for k in keys)
                ema = ", ".join(f"{k}={result.get('ema', {}).get(k, '—')}" for k in keys)
                lines.append(f"| {spec['family']} / {spec['name']} | {row['name']} | {verdict['status']} | {live} | {ema} | {result.get('seconds', 0):.2f} |")
    lines += ["", "[Declared protocol](manifest.json) · [Full results and episode hashes](results.json) · "
              "[Exact source bundle](source.tar.gz). Every episode retains its action trace and full curve as a compressed JSON artifact.", "",
              "Times are single CPU observations including measurement and controller work. No seed sweeps were run. "
              "The suite does not establish a natural-image or large-network default. Reference calibration attempts are separately retained.", ""]
    (output / "README.md").write_text("\n".join(lines))


def run(output, *, generations=2, population=6):
    if output.exists():
        raise FileExistsError("use a new output directory")
    if generations < 1 or population < 4:
        raise ValueError("at least one generation and four proposals required")
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    (output / "episodes").mkdir()
    declared = manifest(generations, population)
    baseline.write_json(output / "manifest.json", declared)
    report = dict(manifest=declared, manifest_sha256=digest(declared), protocol=snapshot(output), rows=[], transfer=[])
    baseline.write_json(output / "protocol.json", report["protocol"])
    def save():
        baseline.write_json(output / "results.json", report)
        render(report, output)
    def episode(row, spec, *, reserved=False):
        name = spec["name"]
        print(json.dumps(dict(event="START", candidate=row["name"], task=name, tier=spec["tier"], phase=spec["phase"])), flush=True)
        started = time.perf_counter()
        try:
            result = run_episode(spec, row["policy"], ablation=row.get("ablation", "none"),
                                 fixed=row.get("fixed", False), allow_reserved=reserved)
            json.dumps(result, allow_nan=False)
        except Exception:
            result = dict(error=traceback.format_exc(), seconds=time.perf_counter() - started)
        payload = dict(task=spec, policy=row["policy"], ablation=row.get("ablation", "none"),
                       fixed=row.get("fixed", False), result=result,
                       manifest_sha256=report["manifest_sha256"], source_sha256=report["protocol"]["source_sha256"])
        raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
        file = f"episodes/{row['name']}__{name}.json.gz"
        (output / file).write_bytes(gzip.compress(raw, mtime=0))
        row["results"][name] = {k: v for k, v in result.items() if k != "actions"}
        row["results"][name].update(artifact=file, uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                                   action_records=len(result.get("actions", [])))
        save()
        print(json.dumps(dict(event="DONE", candidate=row["name"], task=name,
                              status=test_verdict(spec, result)["status"], live=result.get("live"),
                              seconds=result.get("seconds"), error=result.get("error"))), flush=True)
    fit_tasks = [s for s in declared["tasks"] if s["phase"] == "fit"]
    required = [s for s in fit_tasks if s["tier"] == "required"]
    others = [s for s in fit_tasks if s["tier"] != "required"]
    validation = [s for s in declared["tasks"] if s["phase"] == "validation"]
    def fit_row(name, card, *, fixed=False):
        existing = next((r for r in report["rows"] if r["policy"] == card and r.get("ablation", "none") == "none"), None)
        if existing:
            return existing
        row = dict(name=name, policy=card, fixed=fixed, phase="required", results={})
        report["rows"].append(row)
        for spec in required:
            episode(row, spec)
        if not summarize(declared, row["results"], phase="fit")["eligible"]:
            row["phase"] = "screened: required failure"
            save()
            return row
        for spec in others:
            episode(row, spec)
        row["phase"] = "fit complete"
        save()
        return row
    zero = torch.zeros(2, 2, 5, dtype=torch.float64)
    control = fit_row("cosine", study.policy(zero), fixed=True)
    old_path = Path(__file__).resolve().parents[2] / "reports/smart_descent/frozen_evaluation/policy.json"
    old_bytes = old_path.read_bytes()
    warm_card = json.loads(old_bytes)
    warm_weights = torch.tensor(warm_card["weights"], dtype=torch.float64)
    if warm_weights.shape != (2, 2, 5) or not torch.isfinite(warm_weights).all() or warm_weights[:, 1].count_nonzero():
        raise ValueError("the LR-only warm start must have finite [2,2,5] weights and zero regularization actions")
    report["warm_start"] = dict(path=str(old_path.relative_to(Path(__file__).resolve().parents[2])),
                                sha256=hashlib.sha256(old_bytes).hexdigest(), policy=warm_card)
    # Only the runnable controller schema enters duplicate matching.
    warm = study.policy(warm_weights, warm_card.get("schedule", "cosine"))
    fit_row("previous_feedback", warm)
    stream = torch.Generator().manual_seed(declared["search"]["proposal_rng"])
    mean = torch.tensor(warm["weights"], dtype=torch.float64)
    std = torch.zeros_like(mean)
    std[:, 0] = declared["search"]["initial_lr_weight_std"]
    fit_key = lambda row: selection_key(summarize(declared, row["results"], phase="fit"))
    for generation in range(generations):
        proposals = [mean.clone(), torch.tensor(min(report["rows"], key=fit_key)["policy"]["weights"], dtype=mean.dtype)]
        while len(proposals) < population:
            proposals.append(mean + torch.randn(mean.shape, generator=stream, dtype=mean.dtype) * std)
        generation_rows = []
        for index, weights in enumerate(proposals):
            generation_rows.append(fit_row(f"transfer_g{generation:02d}_p{index:02d}", study.policy(weights)))
        unique = {row["name"]: row for row in generation_rows}
        elites = sorted(unique.values(), key=fit_key)[:3]
        values = torch.tensor([r["policy"]["weights"] for r in elites], dtype=mean.dtype)
        mean = .3 * mean + .7 * values.mean(0)
        std[:, 0] = (.3 * std[:, 0] + .7 * values[:, :, 0].std(0, unbiased=False)).clamp_min(.004)
    eligible = [r for r in report["rows"] if not r.get("fixed")
                and summarize(declared, r["results"], phase="fit")["selection_ready"]]
    promoted = sorted(eligible, key=fit_key)[:2]
    report["advanced_to_image_validation"] = [control["name"], *(r["name"] for r in promoted)]
    save()
    for row in [control, *promoted]:
        for spec in validation:
            episode(row, spec)
        row["phase"] = "full development complete"
        save()
    finalists = [r for r in [control, *promoted] if summarize(declared, r["results"])["selection_ready"]]
    challengers = [r for r in finalists if not r.get("fixed")]
    if not challengers:
        report["conclusion"] = "No eligible nonzero challenger; reserved families remain unobserved."
        save()
        return report
    full_key = lambda row: selection_key(summarize(declared, row["results"]))
    winner, challenger = min(finalists, key=full_key), min(challengers, key=full_key)
    verify_source(report["protocol"])
    report["frozen"] = dict(winner=winner["name"], challenger=challenger["name"], policy=challenger["policy"],
                            manifest_sha256=report["manifest_sha256"], source_sha256=report["protocol"]["source_sha256"],
                            development_results_sha256=digest(report["rows"]), selection=declared["selection"])
    baseline.write_json(output / "frozen.json", report["frozen"])
    print(json.dumps(dict(event="FROZEN", winner=winner["name"], challenger=challenger["name"])), flush=True)
    save()
    # This ablation is a post-selection explanation, never a search proposal.
    ablated = dict(name="frozen_bias_only", policy=challenger["policy"], ablation="bias_only", fixed=False,
                   phase="post-freeze ablation", results={})
    report["rows"].append(ablated)
    for spec in [*fit_tasks, *validation]:
        episode(ablated, spec)
    for name, source in (("reserved_cosine", control), ("reserved_feedback", challenger), ("reserved_bias_only", ablated)):
        verify_source(report["protocol"])
        row = dict(name=name, policy=source["policy"], fixed=source.get("fixed", False),
                   ablation=source.get("ablation", "none"), results={})
        report["transfer"].append(row)
        for spec in declared["tasks"]:
            if spec["split"] == "reserved":
                episode(row, spec, reserved=True)
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=6)
    args = parser.parse_args()
    run(args.output, generations=args.generations, population=args.population)


if __name__ == "__main__":
    main()

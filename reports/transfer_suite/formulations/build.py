"""Regroup existing evidence by formulation, retaining architecture trials."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from benchmarks.transfer_suite.formulations import architecture_cell, axes
from benchmarks.transfer_suite.protocol import test_verdict

ROOT = Path(__file__).resolve().parent
SUITE = ROOT.parent
SOLVE = SUITE / "solvability"


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def trial(path, label, *, spec=None):
    value = read(path)
    result = value.get("result", value)
    effective = deepcopy(spec or value.get("effective_spec", value["spec"]))
    effective["schedule"] = value["policy"]["schedule"]
    # The old test name encoded its penalty. Its fixed-formulation counterpart
    # has the same target/architecture/budget and is the nominal ring condition.
    if effective["name"] == "stress_r1_r2":
        effective["name"] = "stress_nominal_ring"
    return dict(label=label, spec=effective, result=result,
                artifact=str(path.relative_to(SUITE)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def dynamics_trial(value):
    # dynamics/scripts/search.py fixes cosine for every archived run.
    return dict(label=value["candidate"]["name"], spec=value["spec"] | dict(schedule="cosine"),
                result=value["result"], artifact="solvability/dynamics/episodes.json.gz",
                episode_index=value["episode_index"],
                sha256=hashlib.sha256((SOLVE / "dynamics/episodes.json.gz").read_bytes()).hexdigest())


def cell_with_sources(variants, runner):
    cell = architecture_cell(variants, runner)
    for stored, variant in zip(cell["trials"], variants):
        stored.update({k: variant[k] for k in ("sha256", "episode_index") if k in variant})
    return dict(runner=runner, **cell)


def build():
    declared = read(SUITE / "study/manifest.json")
    originals = {s["name"]: s for s in declared["tasks"] if s["split"] == "development"}
    dynamics = read(SOLVE / "dynamics/episodes.json.gz")["episodes"]
    rows = []
    for coefficient in (3, 10):
        row = dict(name=f"rp_logistic_bcap{coefficient}", penalty_coefficient=coefficient,
                   fixed=dict(loss="logistic", mode="rp", penalty="b_cap", kappa=1.25,
                              prior_regularization=.05, particle_l2=0.), cases={}, required={},
                   diagnostics={}, long_training=[])
        for name, original in originals.items():
            if original["tier"] == "diagnostic":
                continue
            if original["tier"] == "required":
                path = (SUITE / "study/episodes" / f"cosine__{name}.json.gz" if coefficient == 3
                        else SOLVE / "required_cap10/episodes" / f"cap10__{name}.json.gz")
                value = read(path)
                verdict = test_verdict(original, value["result"])
                row["required"][name] = dict(verdict=verdict, artifact=str(path.relative_to(SUITE)),
                                              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                continue
            runner = original["runner"]
            variants = []
            if name == "stress_r1_r2":
                variants.append(trial(ROOT / "nominal_ring/episodes" / f"bcap{coefficient}_nominal_ring__{name}.json.gz",
                                      f"bcap{coefficient}, original eight-Gaussian run"))
            elif runner == "image":
                cards = [("stage1", "baseline"), ("stage1", "residual16"), ("stage1", "transpose24"),
                         ("stage2", "residual12"), ("stage2", "transpose16")] if coefficient == 3 else [("cross", "residual16_cap10")]
                for stage, card in cards:
                    variants.append(trial(SOLVE / "images" / stage / "episodes" / f"{card}__{name}.json.gz", card))
            elif coefficient == 3:
                path = SUITE / "study/episodes" / f"cosine__{name}.json.gz"
                variants.append(trial(path, "original architecture", spec=original))
                if runner == "vector":
                    for path in sorted((SUITE / "valid_search/discriminator/episodes").glob(f"*__{name}.json.gz")):
                        variants.append(trial(path, path.name.split("__")[0]))
                    for folder in ("smooth_discriminator", "softplus_refinement"):
                        for path in sorted((SUITE / "valid_search" / folder).glob(f"**/episodes/*__{name}.json.gz")):
                            variants.append(trial(path, path.name.split("__")[0]))
                    for path in sorted((SUITE / "rare_focus").glob(f"**/episodes/*__{name}.json.gz")):
                        variants.append(trial(path, path.name.split("__")[0]))
            elif runner == "vector":
                stage = "screen" if name in ("vector_unequal_mass", "vector_unequal_width", "vector_overlap") else "regressions"
                variants.append(trial(SOLVE / "vectors" / stage / "episodes" / f"cap10__{name}.json.gz", "cap10 original architecture"))
            else:
                value = next(v for v in dynamics if v["candidate"]["name"] == "cap10" and v["spec"]["name"] == name)
                variants.append(dynamics_trial(value))
            cell = cell_with_sources(variants, runner)
            assert cell["formulation"] == dict(loss="logistic", mode="rp", penalty="b_cap", coefficient=coefficient,
                                               kappa=1.25, particle_l2=0., prior_regularization=.05, prior_kind="particles")
            canonical_name = variants[0]["spec"]["name"]
            if name == "stress_long_horizon":
                row["long_training"].append(cell)
                if coefficient == 3:
                    reference = axes(variants[0]["spec"], runner)
                    for card in ("budget2", "budget3"):
                        value = next(v for v in dynamics if v["candidate"]["name"] == card and v["spec"]["name"] == name)
                        extended = dynamics_trial(value)
                        actual = axes(extended["spec"], runner)
                        # This separate toy changes only the training horizon.
                        assert {k: v for k, v in actual.items() if k != "resources"} == {
                            k: v for k, v in reference.items() if k != "resources"}
                        assert {k: v for k, v in actual["resources"].items() if k != "steps"} == {
                            k: v for k, v in reference["resources"].items() if k != "steps"}
                        row["long_training"].append(cell_with_sources([extended], runner))
            elif runner == "stress":
                row["diagnostics"][canonical_name] = cell
            else:
                row["cases"][canonical_name] = cell
        row["required_passes"] = sum(v["verdict"]["passed"] for v in row["required"].values())
        row["practical_passes"] = sum(v["supported"] for v in row["cases"].values())
        row["practical_total"] = len(row["cases"])
        row["eligible"] = row["required_passes"] == len(row["required"]) == 9
        row["domains"] = {domain: dict(passed=sum(c["supported"] for c in row["cases"].values() if c["runner"] == domain),
                                      total=sum(c["runner"] == domain for c in row["cases"].values()))
                          for domain in ("vector", "image")}
        assert row["practical_total"] == 10 and len(row["diagnostics"]) == 5
        rows.append(row)
    assert [r["practical_passes"] for r in rows] == [10, 7]
    report = dict(version="formulation-defaults-v2", rows=rows,
                  rule="One formulation entry; architecture trials stay inside each case. Target, formulation, training settings and resource budget must match within an architecture cell. Every attempt remains visible.",
                  scope_revision="User-requested PR scope: evaluate candidate-owned training recipes on nine required and ten data/image toys. Forced LR/batch/discriminator variants and the additional original-budget eight-Gaussian run are diagnostics. Longer training is a separate toy. Prior 7/16 becomes 7/10 by scope change only; no numerical result or metric threshold changed.",
                  candidate_recipe="Loss, regularization, LR, Adam, schedule, update balance and batch are declared candidate choices. Tests must not impose alternate choices and count those as core failures. Architecture remains separate.",
                  improvement="b_cap3 reaches 10/10 with unchanged training settings: wider/deeper D solves overlap; Softplus(beta5) D solves unequal width; D96x2 Softplus5 plus a raw linear skip solves rare mass. Suitable D architectures differ by toy; same formulation entry, no extra updates.",
                  training_scope="Existing declared host recipes are retained, including different host learning rates and Adam betas. No claim of one universal numerical optimizer preset.",
                  source_sha256={str(p.relative_to(SUITE.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), SUITE.parents[1] / "benchmarks/transfer_suite/formulations.py"]})
    (ROOT / "leaderboard.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Formulation × architecture results", "",
             "One entry per formulation. PASS means at least one listed architecture sustains every live metric; "
             "each case counts once. Architecture failures remain visible. Architecture cells cannot mix numerical "
             "formulations, optimizer settings, data or training budgets.", "",
             "Current PR scope: nine required regressions, six data toys and four image toys. "
             "The scope revision changed 7/16 to 7/10. Subsequent discriminator-only results improve b_cap3 to 10/10 with the same training recipe. "
             "[Longer training](LONG_TRAINING.md) is a separate toy; "
             "[imposed-setting diagnostics](DIAGNOSTICS.md) do not affect this comparison.", ""]
    for row in rows:
        lines += [f"## {row['name']}", "", f"Required: {row['required_passes']}/9. Practical: {row['practical_passes']}/{row['practical_total']}. "
                  f"Eligible on required tests: {row['eligible']}.", "",
                  "| Problem / condition | Supported | Passing / tested architectures | Earliest confirmed architecture |",
                  "| --- | --- | ---: | --- |"]
        for name, cell in row["cases"].items():
            passed = [t for t in cell["trials"] if t["verdict"]["passed"]]
            best = min(passed, key=lambda t: t["verdict"]["convergence"]["confirmed_step"]) if passed else None
            label = f"{best['label']} at {best['verdict']['convergence']['confirmed_step']}" if best else "—"
            lines.append(f"| {name} | {cell['status']} | {len(passed)}/{len(cell['trials'])} | {label} |")
        lines.append("")
        lines += [f"<details><summary>Every architecture trial for {row['name']}, including failures</summary>", ""]
        for name, cell in row["cases"].items():
            lines += [f"### {name}", "", "| Architecture | Sustained live | Final passing checks | Confirmed step | Final failing metrics |",
                      "| --- | --- | ---: | ---: | --- |"]
            for trial_record in cell["trials"]:
                verdict = trial_record["verdict"]
                convergence = verdict["convergence"]
                failures = ", ".join(m["metric"] for m in verdict["metrics"] if m["status"] != "PASS")
                if not failures and not verdict["passed"]:
                    failures = "Final bounds pass; insufficient final passing checks"
                lines.append(f"| {trial_record['label']} | {verdict['status']} | {convergence['passing_suffix']} | "
                             f"{convergence['confirmed_step'] or '—'} | {failures or '—'} |")
            lines.append("")
        lines += ["</details>", ""]
    lines += ["The image variants include changes to G as well as D. Discriminator width changes alone, generator changes, "
              "and their combination are explicit in the machine-readable architecture records. These are inspected "
              "development cases with one initialization; architecture support is not a fresh-transfer result.", "",
              "[Exact axes, metrics and source artifacts](leaderboard.json)."]
    (ROOT / "MATRIX.md").write_text("\n".join(lines) + "\n")
    long_lines = ["# Separate toy: longer training on eight Gaussian clusters", "",
                  "Measure whether the same recipe learns each cluster's mass and spread with more training. "
                  "Each budget is a separate full run with cosine scheduled over that budget. "
                  "It is not a resumed checkpoint or proof of stability between measurements. "
                  "These results do not enter the ten-toy practical count.", "",
                  "| Formulation | Updates | Sustained live | Final good samples | Covariance error (≤0.85) | Final passing checks | Confirmed step |",
                  "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        for cell in row["long_training"]:
            verdict = cell["trials"][0]["verdict"]
            metrics = {m["metric"]: m["value"] for m in verdict["metrics"]}
            convergence = verdict["convergence"]
            long_lines.append(f"| {row['name']} | {cell['resources']['steps']:,} | {cell['status']} | "
                              f"{metrics['hq']:.1%} | {metrics['component_covariance_error']:.3f} | "
                              f"{convergence['passing_suffix']} | {convergence['confirmed_step'] or '—'} |")
    long_lines += ["", "Only the budget changes between the b_cap3 rows; the builder verifies identical loss, regularization, "
                   "optimizer settings, architecture, target, batch and particle count. b_cap10 has no matching extended-budget run in this comparison.",
                   "", "Every PASS still requires all live metrics for the final five of 24 measurements. EMA remains separate. "
                   "[Exact settings, verdicts and source artifacts](leaderboard.json)."]
    (ROOT / "LONG_TRAINING.md").write_text("\n".join(long_lines) + "\n")
    diagnostic_lines = ["# Archived training-condition diagnostics", "",
                        "These observations have no effect on the current PR's eligibility or practical count. "
                        "The candidate owns its training recipe, and architecture is a separate axis. "
                        "They remain available for future robustness work; no numerical result has been erased or changed.", "",
                        "The old `stress_nominal_ring` label means the ordinary eight-Gaussian target at 1,200 updates. "
                        "It is retained as a reference run, not an additional required toy. "
                        "The established required ring regression is still required, and [longer training](LONG_TRAINING.md) has its own toy.", ""]
    for row in rows:
        diagnostic_lines += [f"## {row['name']}", "", "| Archived condition | Live result |", "| --- | --- |"]
        diagnostic_lines += [f"| {name} | {cell['status']} |" for name, cell in row["diagnostics"].items()]
        diagnostic_lines.append("")
    diagnostic_lines += ["The original controller study's tiers and balanced scores remain historical records. "
                         "This scope revision was requested for the formulation-default comparison after that study.", "",
                         "[Exact settings, metrics and artifact references](leaderboard.json)."]
    (ROOT / "DIAGNOSTICS.md").write_text("\n".join(diagnostic_lines) + "\n")
    print([(r["name"], r["required_passes"], r["practical_passes"]) for r in rows])


if __name__ == "__main__":
    build()

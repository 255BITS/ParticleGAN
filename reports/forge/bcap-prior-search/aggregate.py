"""Assemble completed, certified feature reports without running training."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FEATURES = {"proximal": 387, "coulomb": 388, "sigreg": 389,
            "covariance": 390, "mmd": 391}
DEMOS = {"proximal": "prior-proximal-software", "coulomb": "prior-coulomb",
         "sigreg": "prior-sigreg", "covariance": "prior-moment-tether", "mmd": "prior-mmd"}
MECHANISMS = {
    "proximal": "A position spring restores sampled prior rows toward their initial positions after the normalized BCAP step. This tests magnitude-sensitive prior motion while allowing the generator and critic to keep learning.",
    "coulomb": "Softened inverse-distance repulsion follows a deterministic evenly spaced cyclic-offset rule on the supplied rows, with at most eight offsets and confinement to prevent expansion. Numeric offsets depend on the supplied row count; rows select neighbors by index within that input. The bounded graph tests spacing at linear cost.",
    "sigreg": "Projected Gaussian characteristic-function matching penalizes distributional departures beyond covariance. This tests whether a Gaussian latent law gives the generator a more stable input geometry.",
    "covariance": "A fixed-initial mean and full-covariance tether resists translation, contraction and anisotropy without imposing a new unit-scale prior. This tests preservation of the task's original global geometry.",
    "mmd": "A multiscale IMQ distribution tether compares a finite deterministic panel with its initial law. It permits particle exchange while detecting changes beyond second moments; only selected current panel rows receive this regularizer's gradient.",
}


def read(path):
    return json.loads(path.read_text())


def tier_count(trial, tier):
    return next(t["required_passed"] for t in trial["qualification"]["tiers"] if t["tier"] == tier)


def tier_measured(trial, tier):
    return sum(task["gate_status"] in {"PASS", "FAIL"} for task in trial["tasks"]
               if task["qualification_tier"] == tier and task["importance"] == "required")


def rank(trial, comparison):
    strength = next(iter(trial["settings"].values()))
    accepted = (tier_count(trial, 1) == 6 and bool(comparison["repaired"])
                and comparison["preserves_recorded_control_passes"])
    return (accepted, tier_count(trial, 1), tier_count(trial, 2),
            -len(comparison["regressed"]), -len(comparison["unresolved_control_passes"]), -strength)


def score(value):
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return str(value).lower()
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def native_holdouts(result):
    rows = []
    for task in result["task_measurements"]:
        if task["task_id"] not in {"grid100", "rotated100", "staggered100"}:
            continue
        if task["status"] not in {"PASS", "FAIL"}:
            continue
        groups = {group["name"]: {row["metric"]: row for row in group["rows"]}
                  for group in task["measurements"]}
        accuracy = groups["Independent live holdout accuracy, 100000 saved draws"]
        coverage = groups["Independent holdout coverage recomputed from the same saved draws"]
        measurements = {metric: accuracy[metric] for metric in
                        ("precision", "mass_tv", "radial_ks", "abs_cov_trace_bias", "center_rms_sigma")}
        measurements["modes"] = coverage["modes"]
        rows.append(dict(role=task["role"], task=task["task_id"], status=task["status"],
                         measurements=measurements))
    return rows


def settings_text(settings):
    return ", ".join(f"`{key}={score(value)}`" for key, value in settings.items())


def bind_demo_links(checkout, relative, source_commit, name):
    """Keep compact demo documents runnable from their actual feature source."""
    target = ROOT / relative
    content = target.read_text()

    def replace(match):
        href = match.group(2)
        if "://" in href or href.startswith("#"):
            return match.group(0)
        file_part, separator, anchor = href.partition("#")
        resolved = (target.parent / unquote(file_part)).resolve()
        if resolved.is_file():
            return match.group(0)
        if not resolved.is_relative_to(ROOT):
            return match.group(0)
        original = (checkout / resolved.relative_to(ROOT)).resolve()
        if not original.is_file():
            return match.group(0)
        repo_path = original.relative_to(checkout).as_posix()
        link = f"https://github.com/255BITS/ParticleGAN/blob/{source_commit}/{repo_path}"
        return f"[{match.group(1)}]({link}{separator}{anchor})"

    content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, content)
    content += (f"\nReproduction requires the [{name} feature's executed source]"
        f"(https://github.com/255BITS/ParticleGAN/tree/{source_commit}); "
        f"this aggregate PR contains reports and saved evidence. "
        f"[The completed search](../bcap-prior-search/{name}/README.md) reports the ordinary task gates.\n")
    target.write_text(content)


def main():
    records = []
    for name, number in FEATURES.items():
        source = Path(f"/home/martyn/dev/ParticleGAN-prior-{name}/reports/forge/bcap-prior-search/{name}")
        result, spec = read(source / "results.json"), read(source / "spec.json")
        registration = read(source / "registration.json")
        progress = read(Path(spec["archive"]) / "progress.json")
        assert result["source_digest"] == spec["source_digest"] == registration["source_digest"]
        assert len(result["trials"]) == len(result["comparisons"]) == len(registration["trials"]) == 3
        assert progress["phase"] == "complete", (name, progress["phase"])
        assert result["source_commit"] == progress["source_commit"]
        assert result["source_digest"] == progress["source_digest"]
        registered = {trial["configuration_id"]: trial for trial in registration["trials"]}
        for trial, comparison in zip(result["trials"], result["comparisons"]):
            assert trial["configuration_id"] == comparison["configuration_id"]
            assert trial["settings"] == comparison["settings"] == registered[trial["configuration_id"]]["settings"]
            assert len(trial["settings"]) == 1 and spec["axis"] in trial["settings"]
        for row in result["control_results"]:
            assert row["receipt_binding"]["valid_receipt"] or row["gate_status"] == "INVALID", (name, row["task_id"])
        dest = OUT / name
        dest.mkdir(exist_ok=True)
        for file in ("results.json", "README.md", "spec.json", "registration.json"):
            shutil.copy2(source / file, dest / file)
        for file in ("audit.json", "audit.py", "execution-repair.json", "control_word_audit.json", "control_word_audit.py", "readout.md", "failure-witnesses.md", "interpretation.md", "INTERPRETATION.md", "protocol_audit.py", "protocol-audit.json", "repair_witness.py", "repair-witness.json", "saved_moments.py", "saved_moments.json", "moment-diagnostics.json", "moment-diagnostics.md", "publication-validation.json", "prior_geometry.py", "prior-geometry.json", "mps_numerical_probe.py", "mps_numerical_probe.json"):
            if (source / file).is_file():
                shutil.copy2(source / file, dest / file)
        if (source / "media").is_dir():
            shutil.copytree(source / "media", dest / "media", dirs_exist_ok=True)
        for document in dest.glob("*.md"):
            document.write_text(document.read_text() + (
                f"\nSaved-only reproduction commands require the [{name} feature's executed checkout]"
                f"(https://github.com/255BITS/ParticleGAN/tree/{result['source_commit']}) "
                "and the original local archive. This aggregate branch contains reports and saved evidence; "
                "the feature PR contains its implementation.\n"))
        checkout = source.parents[3]
        demo = Path("reports/forge") / DEMOS[name]
        tracked = subprocess.check_output(["git", "ls-files", "-z", "--", str(demo)], cwd=checkout).decode().split("\0")
        for relative in filter(None, tracked):
            file = Path(relative)
            assert file.suffix in {".md", ".py", ".json", ".gif"}, file
            target = ROOT / file
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkout / file, target)
        bind_demo_links(checkout, demo / "README.md", result["source_commit"], name)
        search_name = f"bcap-prior-{name}-search-v1.json"
        search_source = source.parents[1] / "configuration-search" / search_name
        search_report = read(search_source)
        assert search_report["selection"]["all_trials_terminal"] is True
        search_dest = ROOT / "reports/forge/configuration-search" / search_name
        search_dest.parent.mkdir(exist_ok=True)
        shutil.copy2(search_source, search_dest)
        pairs = list(zip(result["trials"], result["comparisons"]))
        best, comparison = max(pairs, key=lambda p: rank(*p))
        records.append(dict(name=name, pr=f"https://github.com/255BITS/ParticleGAN/pull/{number}",
                            source_commit=result["source_commit"], source_digest=result["source_digest"],
                            best_configuration_id=best["configuration_id"], settings=best["settings"],
                            tier1=tier_count(best, 1), tier2=tier_count(best, 2),
                            tier1_measured=tier_measured(best, 1), tier2_measured=tier_measured(best, 2),
                            comparison=comparison,
                            trials=[dict(configuration_id=t["configuration_id"], settings=t["settings"],
                                tier1=tier_count(t, 1), tier2=tier_count(t, 2),
                                tier1_measured=tier_measured(t, 1), tier2_measured=tier_measured(t, 2), comparison=c)
                                for t, c in sorted(pairs, key=lambda pair: next(iter(pair[0]["settings"].values())))],
                            control_results=result["control_results"], tasks=best["tasks"], cost=result["cost"],
                            native_holdouts=native_holdouts(result)))
        records[-1]["display_outcomes"] = {row["task_id"]: row["status"] for row in result["task_measurements"]
                                          if row["role"] == str(best["settings"])}
        records[-1]["failed_measurements"] = [row for row in result["task_measurements"]
            if row["role"] == str(best["settings"]) and row["status"] == "FAIL"
            and row["tier"] in {1, 2}]
        if name == "mmd":
            strength = str(next(iter(best["settings"].values())))
            for check in read(source / "audit.json")["checks"]:
                witness = check.get("original_terminal_failure_witness")
                if check["arm"] != strength or not witness:
                    continue
                task = next(row for row in records[-1]["failed_measurements"] if row["task_id"] == check["task"])
                task["measurements"].append(dict(
                    name="Original required terminal failure witness (saved, no new credit)",
                    qualification_credit=False,
                    original_observations_sha256=witness["original_observations_sha256"],
                    required_terminal_checks=witness["required_terminal_checks"],
                    window_steps=witness["window_steps"], witness_source="mmd/audit.json",
                    rows=[dict(row, witness_step=witness["step"],
                               witness_observation_sha256=witness["original_observation_sha256"])
                          for row in witness["measurements"]]))
    accepted = [r["name"] for r in records if r["tier1"] == 6
                and r["comparison"]["repaired"]
                and r["comparison"]["preserves_recorded_control_passes"]]
    all_trials = [t for r in records for t in r["trials"]]
    eligible = sum(t["tier1"] == 6 for t in all_trials)
    repairs = sum(len(t["comparison"]["repaired"]) for t in all_trials)
    (OUT / "summary.json").write_text(json.dumps(dict(schema_version=1, scope="source_bound_tuning_comparison",
        default_adoption=False, registered_repair_criterion_met=accepted,
        searched_recipes=len(all_trials), tier2_eligible_recipes=eligible,
        measured_tier2_fail_to_pass_pairs=repairs, records=records), indent=2, sort_keys=True) + "\n")
    text = ["# BCAP prior repair searches", "",
        "The initial mean/covariance tether at global strength **0.01** repairs **mode_hold**, increasing the terminal passing suffix from **3 to 7 / ≥5**. It loses **word retention**, passing **23/25** mandatory checks versus the leader's **25/25**. No tested recipe repairs a Tier 2 failure while preserving every leader Tier 2 pass.", "",
        f"Completed all {len(all_trials)} registered global recipes. {eligible} passed the six Tier 1 gates and reached Tier 2; {len(all_trials) - eligible} were blocked before Tier 2. Matched Tier 2 FAIL→PASS task/recipe pairs: {repairs}. Recipes meeting the full registered repair criterion: {', '.join(accepted) or 'none'}.", "",
        "[Interpretation and recommendations](findings.md) explain the sustained mode-hold repair, its word-retention tradeoff, and the remaining native density errors.", "",
        "Five independent feature PRs compare three positive global strengths each against a fresh, unchanged BCAP leader control. Protocol seed 0, public deterministic initialization, revision-9 tasks, original architectures/data/prior capacity/sampling/update budgets/evaluation cadence and gates are held fixed. The recipe delta is prior regularization or the explicit proximal prior update.", "",
        "The complete `recipe-prior-refactor-bcap-pure-v1` leader card is the baseline in all five: non-saturating BCAP, direction blend, zero-momentum DualNorm with smoothing 0.001, base learning rate 0.012, discriminator multiplier 1.5 and prior multiplier 2.5, with no EMA or additional output noise. Native MixtureOfGaussians retains intrinsic sigma 0.025. The full host-adaptation and recipe overrides are copied; using a preset name alone would not establish this baseline.", "",
        "Each arm must pass all six required Tier 1 tasks before any Tier 2 work. Every eligible arm completes its runnable Tier 2 peers. Unreached tasks retain their targets and unavailable scores. No seed studies, scientific retries, per-task recipes, Tier 3 runs or default adoption are included.", "",
        "Task verdicts use their declared temporal rules. Acquisition smoke requires an earlier independently confirmed passing state; its final endpoint can differ. Retention and density tasks require their complete hold or terminal windows. Detailed reports show these gate counts beside endpoint scores, with compact worst-check witnesses for failed mandatory word holds. Optional clock probes use their own smaller recipe and do not establish invariance of the new prior feature.", "",
        "The original word-hold task resumes each recipe's own earliest independently confirmed passing prefix, with its model, optimizer and named streams restored. Prefix lengths can differ; those holds compare the declared own-prefix protocol. The ring task retains its original adaptive stopping rule and maximum budget. Audits record actual consumed update counts and distinguish these cases from equal-length stream comparisons.", "",
        "The registered repair criterion requires all six Tier 1 passes, at least one matched Tier 2 FAIL→PASS, and preservation of every control Tier 2 pass. One whole recipe per mechanism is displayed below: meeting that criterion first, then most Tier 1 passes, most Tier 2 passes, fewer regressions/unresolved control passes, and lower strength. All three searched strengths and every task measurement, score and target are in each linked detailed report. These are source-bound comparisons; the sole generated [current leaderboard](../technique-inventory.md) retains its published cohorts.", "",
        "Counts below are passes / measured PASS-or-FAIL results / required tasks. Blocked tasks are unavailable measurements, and do not count as measured failures.", "",
        "| Mechanism / feature PR | Strength | Tier 1 | Tier 2 | Matched FAIL→PASS | Lost control passes | Unresolved control passes |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        c = r["comparison"]
        text.append(f"| [{r['name']}]({r['pr']}) / [all measurements]({r['name']}/README.md) | {settings_text(r['settings'])} | {r['tier1']}/{r['tier1_measured']}/6 | {r['tier2']}/{r['tier2_measured']}/21 | {', '.join(c['repaired']) or 'none'} | {', '.join(c['regressed']) or 'none'} | {len(c['unresolved_control_passes'])} |")
    text += ["", "## What the five changes test", ""]
    for r in records:
        text.append(f"- **{r['name']}**: {MECHANISMS[r['name']]}")
    text += ["", "The leader uses smoothed prior-row normalization: `g / sqrt(||g||² + 0.001²)`, applied only to the unique sampled rows. Unsampled rows stay fixed even when a regularizer creates dense gradients. A coefficient changes the combined gradient's direction and magnitude; the resulting displacement is nonlinear and saturates for large gradients. The proximal update acts after that normalization; the other four regularizers enter the prior loss. Each PR declares this trainer delta explicitly. These objectives measure latent locations; generated component density is graded independently by the original task gates.", "", "The baseline GANTrainer supplies all locations to its regularizer for tables of at most 1,024 rows, and unique generator-sampled locations for larger tables. The native tasks have 20,000 locations. Fixed initial reference statistics/panels remain tied to the full initialization; current inputs follow each host's retained policy. Whole-table saved-state geometry and moment diagnostics are separate from historical batch objectives or forces.", "", "## All registered strengths", "",
        "| Mechanism | Global settings | Tier 1 | Tier 2 | Repairs | Regressions | Unresolved control passes |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        for trial in r["trials"]:
            c = trial["comparison"]
            text.append(f"| {r['name']} | {settings_text(trial['settings'])} | {trial['tier1']}/{trial['tier1_measured']}/6 | {trial['tier2']}/{trial['tier2_measured']}/21 | {', '.join(c['repaired']) or 'none'} | {', '.join(c['regressed']) or 'none'} | {len(c['unresolved_control_passes'])} |")
    text += ["", "## Tier 2 gates at a glance", "",
        "Each cell shows its own matched control → the selected whole recipe. An endpoint improvement alone earns no repair credit.", "",
        "| Task | " + " | ".join(r["name"] for r in records) + " |",
        "| --- | " + " | ".join("---" for _ in records) + " |"]
    tasks = sorted({t["task"] for r in records for t in r["tasks"] if t["qualification_tier"] == 2 and t["importance"] == "required"})
    for task in tasks:
        cells = []
        for r in records:
            before = next((x["gate_status"] for x in r["control_results"] if x["task_id"] == task), "UNMEASURED")
            after = r["display_outcomes"].get(task, next(x["gate_status"] for x in r["tasks"] if x["task"] == task))
            cells.append(f"{before} → {after}")
        text.append("| " + task + " | " + " | ".join(cells) + " |")
    text += ["", "## The 100-Gaussian density gap", "",
        "All measured positive native recipes appear below, using each original independent 100,000-draw live holdout. Modes are recomputed from those same hashed draws using the frozen coverage scorer; density measurements retain the stricter accuracy targets. These endpoint packets explain the results and add no qualification credit. The shared control is shown once; the baseline audit checks all five controls independently. Unavailable component statistics remain unavailable.", "",
        "| Recipe | Task | Original gate | Precision / ≥0.97 | Modes / ==100 | Mass TV / ≤0.06 | Radial KS / ≤0.04 | Absolute covariance trace bias / ≤0.1 | Center RMS in sigma / ≤0.2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in records:
        for task in r["native_holdouts"]:
            if task["role"] == "control" and r["name"] != "covariance":
                continue
            recipe = "BCAP leader control" if task["role"] == "control" else f"{r['name']} {task['role']}"
            values = [score(task["measurements"][metric]["score"]) for metric in
                      ("precision", "modes", "mass_tv", "radial_ks", "abs_cov_trace_bias", "center_rms_sigma")]
            text.append(f"| {recipe} | {task['task']} | {task['status']} | " + " | ".join(values) + " |")
    text += ["", "## Measured failure checks", "",
        "These are the failed numerical checks for the selected whole recipes above. Endpoint checks explain the recorded endpoint; the sustained/confirmation rows retain the task's actual temporal gate. Original mandatory word-hold and audited terminal-window witnesses include their observation step and hash; they explain the existing verdict and earn no new qualification credit. Unavailable and blocked tasks are shown in the full reports with their original targets. Passing acquisition tasks with a failing final endpoint are excluded here because their declared earlier-confirmation gate passed.", ""]
    for r in records:
        text += [f"### {r['name']}: {r['settings']}", "",
            f"[Every task and measurement]({r['name']}/README.md), including passing checks and diagnostics.", "",
            "| Failed task | Measurement cohort | Measurement | Score | Target |",
            "| --- | --- | --- | ---: | --- |"]
        for task in r["failed_measurements"]:
            for group in task["measurements"]:
                if group.get("qualification_credit") is False and not group.get("original_observations_sha256"):
                    continue
                for row in group["rows"]:
                    if row["status"] == "FAIL":
                        cohort = group["name"]
                        if "witness_step" in row:
                            cohort += (f"; original step {row['witness_step']}, "
                                       f"observation [{row['witness_observation_sha256'][:8]}]"
                                       f"({group.get('witness_source', r['name'] + '/results.json')})")
                        text.append(f"| {task['task_id']} | {cohort} | {row['metric']} | {score(row['score'])} | {row['target']} |")
        if not r["failed_measurements"]:
            text.append("| No measured failing task | — | — | — | — |")
    text += ["", "## Charged work and execution", "",
        "The finite reservation is 43,020 seconds per arm, 172,080 per four-arm mechanism, bounded by 180,000 seconds per mechanism and 900,000 overall. Charged time below includes retained infrastructure predecessors. Physical placement and sharing changed during the execution repair; these charges are accounting, not a speed comparison.", "",
        "| Mechanism | Control charged seconds | Three-strength search charged seconds |",
        "| --- | ---: | ---: |"]
    for r in records:
        cost = r["cost"]
        text.append(f"| {r['name']} | {cost['control']['spent_seconds']:.3f} | {(cost.get('search') or {}).get('spent_seconds', 0):.3f} |")
    text += ["", "The [baseline reproduction audit](baseline_audit.json), reproduced by [baseline_audit.py](baseline_audit.py), compares each fresh control's original gates and all previously published metric values with the archived leader. Additional current component arrays are listed separately; missing or invalid evidence cannot inherit an archived pass. The [aggregate integrity receipt](aggregate-validation.json) checks all 560 task cells, 20,105 numerical measurement rows, relative links and 329 copied task GIFs. [Infrastructure receipt](infrastructure.json), [cross-control word-state audit](control_word_audit.json), and [saved-media validation](saved-media-validation.json) record the interruption/retry identities and numerical checks. Original deadlines, recipes, task gates and update allowances were retained. The initial five interrupted word controls were retried for GPU context contention; there were no scientific retries. New work subsequently used a shared admission lock with one new worker per physical GPU. The native and toy GIF catalogs in each detailed report are rendered from saved actual training evidence with zero added updates, samples or rescored observations.", ""]
    text += ["", "## Executed sources", "", "Training logs, JSONL streams, checkpoints and state dumps remain outside Git under `/mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/`. Compact reports bind original receipt certificates and hashes, source/runtime/compute profiles, recipe and task contracts; saved native holdout scores are recomputed from the same hashed draws without training or RNG consumption.", "",
        "Read the operational summary and collector logs locally; neither command launches work:", "",
        "```sh", "python reports/forge/bcap-prior-search/status.py",
        "python reports/forge/bcap-prior-search/status.py --logs | xargs -r tail -F", "```", "",
        "The second command follows the currently active worker logs; rerun it to select newly admitted tasks. Completed worker logs remain in the archive under each mechanism's queue directory. The collector driver logs preserve the original launch and infrastructure repair history.", ""]
    for r in records:
        text.append(f"- {r['name']}: commit `{r['source_commit']}`, source digest `{r['source_digest']}`; [compact metrics and provenance]({r['name']}/results.json).")
    (OUT / "README.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()

"""Compare actual old/new task-bound recipes without model construction/training."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def probe(options):
    root = options.repository.resolve()
    sys.path.insert(0, str(root))
    from experiments.forge.api import task_formulation_context
    from experiments.forge.planning import load_idea, resolve_idea
    from experiments.forge.contracts import read_json
    baseline = read_json(options.baseline)
    defaults = read_json(root / "configs/forge/defaults.json")
    protocol = read_json(root / f'configs/forge/protocols/{defaults["protocol"]}.json')
    registration = read_json(options.registration) if options.registration else None
    rows = {}
    for leader in baseline["leaders"]:
        family = leader["selection"]["family"]
        if family in {"atlas", "e22"}:
            continue
        rows[family] = {}
        if registration:
            plan = registration["families"][family]
            request = resolve_idea(root, plan["candidate_id"], study=plan["study_id"])
            candidate, tasks = request["candidate"], request["tasks"]
        else:
            candidate = load_idea(root, leader["selection"]["candidate_id"])
            tasks = {a["task"]: read_json(root / f'configs/forge/tasks/{a["task"]}.json') for a in baseline["view"]["assignments"]}
        for assignment in baseline["view"]["assignments"]:
            if assignment["qualification_tier"] > 2:
                continue
            name = assignment["task"]
            context = task_formulation_context(candidate, tasks[name], protocol, root=root)
            rows[family][name] = dict(recipe=asdict(context.recipe), prior=context.prior_config,
                                     field_ownership=context.receipt()["field_ownership"])
    data = dict(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), rows=rows)
    options.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--baseline-repository", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path(__file__).with_name("baseline.json"))
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.probe:
        probe(options)
        return
    root = options.repository.resolve()
    spec = importlib.util.spec_from_file_location("recipe_prior_binding_workflow", Path(__file__).with_name("workflow.py"))
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    workflow.require(options.baseline_repository and options.registration,
                     "Supply the exact original source checkout and fresh registration")
    old_root = options.baseline_repository.resolve()
    workflow.require(workflow.head(old_root) == workflow.baseline()["source_commit"], "Original source commit differs")
    workflow.require(subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=old_root).returncode == 0,
                     "Original checkout has modified tracked bytes")
    options.output.mkdir(parents=True, exist_ok=True)
    for label, checkout, registration in (("original", old_root, None), ("current", root, options.registration)):
        command = [sys.executable, str(Path(__file__).resolve()), "--probe", "--repository", str(checkout),
                   "--baseline", str(options.baseline.resolve()), "--output", str(options.output / f"{label}-task-bindings.json")]
        if registration:
            command += ["--registration", str(registration.resolve())]
        subprocess.run(command, cwd=checkout, check=True)
    old = workflow.read_json(options.output / "original-task-bindings.json")
    current = workflow.read_json(options.output / "current-task-bindings.json")
    allowed = set(workflow.PRIOR_FIELDS) | {"prior_reg"}
    pairs = []
    for family, tasks in old["rows"].items():
        for name, before in tasks.items():
            after = current["rows"][family][name]
            before_recipe, after_recipe = before["recipe"], after["recipe"]
            nonprior_before = {k: v for k, v in before_recipe.items() if k not in allowed}
            nonprior_after = {k: after_recipe[k] for k in before_recipe if k not in allowed}
            workflow.require(nonprior_before == nonprior_after, f"{family}/{name}: nonprior trainer values changed")
            workflow.require(before["prior"] == after["prior"], f"{family}/{name}: initial prior law changed")
            pairs.append(dict(family=family, task=name,
                original_nonprior_sha256=workflow.stable_hash(nonprior_before),
                current_nonprior_sha256=workflow.stable_hash(nonprior_after),
                initial_prior_sha256=workflow.stable_hash(before["prior"]),
                declared_prior_delta={k: dict(before=before_recipe.get(k, {"absent": True}), after=after_recipe.get(k, {"absent": True}))
                    for k in sorted(allowed) if before_recipe.get(k, {"absent": True}) != after_recipe.get(k, {"absent": True})}))
    workflow.require(len(pairs) == 140, "Require every supported family's 28 task bindings")
    receipt = dict(schema_version=1, status="PASS", scope="actual_source_task_binding_compatibility",
        qualification_input=False, original_source_commit=old["source_commit"], current_source_commit=current["source_commit"],
        pairs=pairs, pairs_checked=len(pairs), baseline_sha256=workflow.file_hash(options.baseline),
        registration_sha256=workflow.file_hash(options.registration),
        raw_binding_hashes={label: workflow.file_hash(options.output / f"{label}-task-bindings.json") for label in ("original", "current")},
        interpretation="All actual task-bound nonprior Recipe fields and initial prior laws exactly match actual develop. Prior authority/penalties change explicitly; no training/numeric quality parity claim.")
    workflow.atomic_json(options.output / "binding-compatibility.json", receipt)
    print(dict(status="PASS", pairs=len(pairs)), flush=True)


if __name__ == "__main__":
    main()

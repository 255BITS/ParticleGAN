"""Reproduce the BCAP inventory and read-only Forge admission audit.

Run from the repository root. No models, training, queue admission, scientific
reports or leaderboards are created. --output writes only this audit's artifacts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from particlegan import Recipe, get_recipe
from particlegan.gan_loss import GANLoss
from particlegan.optim.dualnorm import NORMALIZED_FAMILIES
from experiments.forge import configuration_search as search
from experiments.forge import search_space as spaces
from experiments.forge.boundaries import RECIPE_FIELD_OWNERS, TUNABLE_FIELDS
from experiments.forge.contracts import read_json, stable_hash
from experiments.forge.planning import load_idea
from experiments.forge.queue import Queue
from experiments.forge.techniques import recipe_field_active
from experiments.forge.views import load_tasks, load_view


# Each value changes the inspected base. Acceptance does not establish quality.
PROBES = {
    "lr": .006, "d_lr_mult": 2., "prior_lr_mult": 3.,
    "betas": [0., .99], "prior_betas": [0., .9], "d_betas": [0., .9],
    "direct_particle_betas": [0., .95], "eps": 1e-6, "d_eps": 1e-6,
    "prior_eps": 1e-6, "amsgrad": True, "optimizer_momentum": .5,
    "optimizer_adam_lr": .003, "optimizer_smoothing": 1e-5,
    "reg_coeff": 2., "reg_coeff_end": .1, "reg_coeff_anneal_end": .5,
    "reg_kappa": .5, "reg_every": 2, "prior_reg": .01,
    "lr_anneal_start": .4, "lr_floor": .5, "network_lr_floor": .5,
    "beta2_end": .99, "beta2_anneal_end": .5,
    "lr_decay_rate": .9, "lr_decay_steps": 10_000,
}
PAIR_FIELDS = {"betas", "d_betas", "prior_betas", "direct_particle_betas"}
BASES = ("bcap-pure-adam-v2", "bcap-dualnorm-zero-v1", "bcap-dualnorm-smoothed-v1")


def spec(base, family, grid, suffix="probe"):
    return {
        "schema_version": 2, "id": f"bcap-options-{suffix}",
        "base_candidate": base, "trainer_family": family, "grid": grid,
        "hypothesis": "Check software search admission without executing a trainer.",
        "protocol": "screening", "view": "discriminator_stability",
        "execution_backend": "cpu", "tuning_through_tier": 1,
        "campaign": {"id": f"bcap-options-{suffix}", "budget_seconds": 100_000,
                     "candidate_budget_seconds": 100_000},
    }


def family(base):
    from experiments.forge.trainer_families import family_for_candidate
    return family_for_candidate(ROOT, base, load_idea(ROOT, base))["id"]


def inventory():
    recipes = {name: asdict(get_recipe(name)) for name in ("bcap", "bcap_adam")}
    assert set(PROBES) == set(TUNABLE_FIELDS)
    assert set(recipes["bcap"]) == set(RECIPE_FIELD_OWNERS)
    rows = []
    for field in fields(Recipe):
        name = field.name
        value = recipes["bcap"][name]
        size = 2 if name in PAIR_FIELDS else len(value) if isinstance(value, tuple) else None
        for index in range(size) if size is not None else (None,):
            defaults, effective = {}, {}
            for preset, recipe in recipes.items():
                val = recipe[name]
                defaults[preset] = val if index is None or val is None else val[index]
                inherited = recipe["betas"] if name in {"d_betas", "prior_betas"} and val is None else val
                effective[preset] = inherited if index is None else inherited[index]
            rows.append({
                "path": name if index is None else f"{name}[{index}]", "recipe_field": name,
                "defaults": defaults, "effective_defaults": effective,
                "owner": RECIPE_FIELD_OWNERS[name], "numerical_grid_field": name in TUNABLE_FIELDS,
                "declared_activity_without_task": {
                    preset: recipe_field_active(name, recipe) for preset, recipe in recipes.items()
                } if name in TUNABLE_FIELDS else None,
            })
    return {"schema_version": 1, "recipe_fields": len(fields(Recipe)),
            "flattened_paths": len(rows), "numerical_grid_fields": len(TUNABLE_FIELDS), "rows": rows}


def check_rejection(name, thunk, expected):
    try:
        thunk()
    except (ValueError, TypeError) as exc:
        assert expected in str(exc), (name, str(exc), expected)
        return {"check": name, "status": "PASS", "rejection": str(exc)}
    raise AssertionError(f"{name} was unexpectedly admitted")


def run_audit():
    axes = []
    for base in BASES:
        fam = family(base)
        for name, value in sorted(PROBES.items()):
            try:
                declarations = search._declarations(ROOT, spec(base, fam, {name: [value]}))
            except (ValueError, TypeError) as exc:
                axes.append({"base": base, "field": name, "probe": value,
                             "result": "REJECTED", "reason": str(exc)})
            else:
                assert len(declarations) == 1
                axes.append({"base": base, "field": name, "probe": value, "result": "ACCEPTED"})

    inactive_epsilon = []
    for name in ("eps", "d_eps", "prior_eps"):
        declarations = search._declarations(ROOT, spec("bcap-sgda-v1", "bcap-sgda", {name: [1e-6]}))
        assert len(declarations) == 1
        inactive_epsilon.append({"base": "bcap-sgda-v1", "field": name, "admission": "ACCEPTED",
                                 "update_effect": "none: SGDA applies parameter += -lr * gradient"})

    public = []
    for optimizer in ("adam", *NORMALIZED_FAMILIES):
        for loss in GANLoss.LOSSES:
            recipe = get_recipe("bcap_adam", optimizer_family=optimizer, loss=loss)
            recipe.make_loss()
            recipe._penalty_options()
            public.append({"optimizer_family": optimizer, "loss": loss, "status": "PASS"})

    definition = read_json(Path(__file__).with_name("plan-space.json"))
    with tempfile.TemporaryDirectory(prefix="bcap-options-") as temporary:
        queue = Queue(Path(temporary) / "queue")
        before_rng = random.getstate()
        manifest = spaces.compile_space(ROOT, queue.root, definition, queue=queue)
        assert before_rng == random.getstate()
        again = spaces.compile_space(ROOT, queue.root, definition, queue=queue)
        assert stable_hash(manifest) == stable_hash(again)
        assert before_rng == random.getstate()
        plans = [search.plan_search(ROOT, queue.root, s, queue=queue) for s in manifest["searches"]]
        trials = [t for p in plans for t in p["trials"]]
        assert len(trials) == len(definition["candidates"]) == len(manifest["draws"])
        assert all(t["submission_status"] == "READY" for t in trials)
        assert len({d["index"] for d in manifest["draws"]}) == len(trials)
        assert not queue.inspect()["submissions"]
        view = load_view(ROOT, definition["view"])
        required = [a["task"] for a in view["assignments"]
                    if a["qualification_tier"] == 1 and a["importance"] == "required"]
        assert len(required) == 6
        for trial in trials:
            assert {t["task"] for t in trial["tasks"]} == {a["task"] for a in view["assignments"]}
            assert trial["declared_worst_case_seconds"] == 2520
        compilation = {
            "status": "PASS", "definition_sha256": stable_hash(definition),
            "manifest_sha256": manifest["manifest_hash"], "population": manifest["population"],
            "samples": len(trials), "declared_worst_case_seconds": manifest["declared_worst_case_seconds"],
            "required_tier1_tasks": required,
            "tiers_required": {str(tier): sum(a["importance"] == "required" and a["qualification_tier"] == tier
                                             for a in view["assignments"]) for tier in (1, 2, 3)},
            "global_python_rng_unchanged": True, "repeat_manifest_identical": True,
            "queue_submissions": 0, "training_updates": 0,
            "trials": [{"base": p["base_candidate"], "status": t["submission_status"],
                        "blockers": t["submission_blockers"], "candidate": t["candidate_id"],
                        "optimizer_family": t["resolved_recipe"]["optimizer_family"],
                        "loss": t["resolved_recipe"]["loss"]}
                       for p in plans for t in p["trials"]],
        }
        base = BASES[1]
        fam = family(base)
        rejections = []
        for field, value in (("optimizer_family", "adam"), ("loss", "hinge"),
                             ("optimizer_convolution", "per_offset"), ("batch_size", 128)):
            rejections.append(check_rejection(field, lambda f=field, v=value: search._grid({f: [v]}), "forbidden"))
        for field, value in (("optimizer_momentum", .5), ("optimizer_smoothing", 1e-5),
                             ("reg_coeff", 0.), ("reg_kappa", 0.), ("lr_floor", 0.)):
            rejections.append(check_rejection(field, lambda f=field, v=value: search._declarations(
                ROOT, spec(base, fam, {f: [v]})), "technique mechanisms"))
        rejections.append(check_rejection("257 configurations", lambda: search._grid(
            {"lr": [i / 100_000 for i in range(1, 258)]}), "256-configuration"))
        too_small = deepcopy(definition)
        too_small["campaign"]["budget_seconds"] = 2520
        rejections.append(check_rejection("aggregate reservation", lambda: spaces.compile_space(
            ROOT, queue.root, too_small, queue=queue), "shared campaign"))

    tasks = load_tasks(ROOT)
    task_bindings = [{"task": name, "prior": tasks[name]["execution"]["prior"],
                      "initializer": tasks[name]["execution"]["initializer"],
                      "budget_updates": tasks[name]["execution"].get("steps"),
                      "evaluation": tasks[name]["evaluation"]} for name in required]
    sources = ("particlegan/recipes.py", "particlegan/gan_loss.py", "particlegan/optim/dualnorm.py",
               "particlegan/grad_regularizers.py", "experiments/forge/api.py",
               "experiments/forge/boundaries.py", "experiments/forge/techniques.py",
               "experiments/forge/configuration_search.py", "experiments/forge/search_space.py")
    return {"schema_version": 1, "scope": "software inventory and admission; no scientific qualification",
            "inspected_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
            "public_optimizer_loss_pairs": public, "numerical_axis_probes": axes,
            "known_inactive_epsilon_admissions": inactive_epsilon,
            "categorical_compilation": compilation, "expected_rejections": rejections,
            "task_bindings": task_bindings}


def flat_markdown(data):
    lines = ["# Complete flattened Recipe inventory", "",
             f"Generated from the public dataclass by [audit.py](audit.py). All {data['recipe_fields']} fields are included,",
             "including task conditions and inactive policy settings. This is an inventory, not a",
             "search declaration. `null` means inherit for role moments/epsilon or disable for optional",
             "schedules. Pair components must be reassembled into their parent Recipe field in a grid.", "",
             "`Grid` means whitelist membership only; activity and mechanism checks still apply.",
             "Defaults are public presets before Forge binds task conditions. See [the report](README.md)",
             "for bounds, algorithm applicability, effective task laws and the Forge v1 preset distinction.", "",
             "| Flat path | bcap default | bcap_adam default | Owner | Grid |",
             "| --- | --- | --- | --- | --- |"]
    for row in data["rows"]:
        val = lambda preset: json.dumps(row["defaults"][preset], separators=(",", ":"))
        lines.append(f"| `{row['path']}` | `{val('bcap')}` | `{val('bcap_adam')}` | {row['owner']} | {'yes' if row['numerical_grid_field'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write compact JSON and the flat table here")
    args = parser.parse_args()
    data = inventory()
    print(f"inventory: {data['recipe_fields']} fields, {data['flattened_paths']} flat paths, {data['numerical_grid_fields']} grid fields", flush=True)
    receipt = run_audit()
    compiled = receipt["categorical_compilation"]
    print(f"compile: {compiled['samples']} categories; statuses {[t['status'] for t in compiled['trials']]}", flush=True)
    print(f"public pairs: {len(receipt['public_optimizer_loss_pairs'])}; expected rejections: {len(receipt['expected_rejections'])}; training updates: 0", flush=True)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, content in (("inventory.json", data), ("audit-receipt.json", receipt)):
            (args.output / name).write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
        (args.output / "flat-inventory.md").write_text(flat_markdown(data))


if __name__ == "__main__":
    main()

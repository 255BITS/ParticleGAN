"""Bounded policy-family defaults search using the existing public-API toys.

This is a separate cloud/served-policy cohort, not Forge MoG qualification.
Only an explicit run command starts children; the coordinator owns GPU admission.
All original case gates remain intact. Study-only first-acquisition retention is
an additional requirement, and observed times are never ranked under contention.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import torch
import numpy as np
from particlegan import get_recipe
from experiments.forge.contracts import stable_hash
from . import api_contract as contract, api_run, api_publish

SCHEMA = "particlegan_policy_family_search_v1"
FAMILIES = ("atlas", "e22")
TUNING_FIELDS = {"lr", "prior_lr_mult"}
LR_PROFILES = ((.006375, .0085), (.002125, .00425))
DEFAULT_CASES = (
    ("image-develop-img_intensity2-source-transpose12", 1),
    ("api-vector-two-broad", 1),
    ("api-grid100", 2), ("api-rotated100", 2), ("api-staggered100", 2),
    ("api-vector-unequal-mass", 2), ("api-vector-anisotropic", 2),
    ("image-develop-img_bars4-source-transpose12", 2),
)


def digest(value):
    return stable_hash(api_run.json_value(value))


def _positive(value, name, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if integer and type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _manifest(paths):
    return {path.relative_to(contract.ROOT).as_posix(): api_run.file_hash(path) for path in sorted(set(paths))}


def proof_bindings(case, family):
    """Exact capacity-witness identity, independent of search/reporting code."""
    root = contract.ROOT
    paths = list((root / "particlegan").glob("*.py"))
    paths += [root / "benchmarks/toy_audit/api_contract.py",
              root / f"benchmarks/toy_audit/{case['provider']}.py"]
    if case["provider"] == "api_vectors":
        paths += [root / "lib/toy_models.py", root / "benchmarks/toy_audit/definition_quality.py",
                  root / "benchmarks/toy_audit/vector_quality.py"]
        paths += list((root / "benchmarks/toy100").glob("*.py"))
        paths += [root / "benchmarks/transfer_suite/vector_tasks.py",
                  root / "benchmarks/transfer_suite/stress_tasks.py"]
    return {"case_sha256": digest(case), "preset_sha256": digest(get_recipe(family).to_dict()),
            "sampling_sha256": digest(case["sampling"]), "source_files_sha256": _manifest(paths)}


def host_recipe_options(case):
    """Disclose frozen fixture adaptations; these are never search dimensions."""
    if case["provider"] == "api_images" and not case.get("query"):
        return dict(z_dim=case["z_dim"], num_particles=32, prior_kind="particles", sigma_rel=0,
                    batch_size=case["batch_size"])
    if case["provider"] == "api_vectors" and not case.get("caller_owned") and not case.get("penalty_arm"):
        options = dict(num_particles=case["particles"], z_dim=case["z_dim"], batch_size=case["batch_size"])
        options.update({key: (tuple(value) if key == "betas" else value)
                        for key, value in case.get("spec", {}).items()
                        if key in {"lr", "d_lr_mult", "prior_lr_mult", "prior_reg", "betas", "ema_decay"}})
        options.update(case.get("recipe_overrides", {}))
        return options
    raise contract.UnsupportedRecipeOverrides("host has no declared policy-family tuning contract")


def resolved_recipe(case, family, knobs):
    contract.validate_recipe_overrides(case, family, knobs)
    return api_run.json_value(get_recipe(family, **{**host_recipe_options(case), **knobs}).to_dict())


def _source(cases):
    paths = list((contract.ROOT / "particlegan").glob("*.py"))
    paths += list((contract.ROOT / "benchmarks/toy_audit").glob("api_*.py"))
    for case in cases.values():
        paths += [contract.ROOT / path for path in proof_bindings(case, "atlas")["source_files_sha256"]]
    paths += [contract.ROOT / "experiments/forge/configuration_search.py"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=contract.ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    return {"commit": commit, "files_sha256": _manifest(paths)}


def _validated_grid(spec):
    grid = spec.get("grid")
    if (not isinstance(grid, dict) or set(grid) != TUNING_FIELDS
            or any(not isinstance(grid[key], list) or len(grid[key]) != 2
                   or any(isinstance(value, bool) or not isinstance(value, (int, float))
                          or not math.isfinite(value) for value in grid[key])
                   or len(set(grid[key])) != 2 for key in TUNING_FIELDS)
            or tuple(grid["lr"]) not in LR_PROFILES or grid["prior_lr_mult"] != [1., 2.]):
        raise ValueError("requires one frozen two-LR/two-prior-rate grid profile")
    return grid


def validate_spec(spec, cases):
    """Freeze all candidates, denominators, timeouts and study-only hold rules."""
    if spec.get("schema") != SCHEMA or not isinstance(spec.get("id"), str) or not contract.CASE_ID.fullmatch(spec["id"]):
        raise ValueError("invalid policy-family search schema/id")
    if spec.get("families") != list(FAMILIES) or spec.get("seed") != 24002:
        raise ValueError("this round requires declared atlas/e22 families and the unchanged seed24002")
    _validated_grid(spec)
    assignments = spec.get("cases", [])
    if [(item.get("id"), item.get("tier")) for item in assignments] != list(DEFAULT_CASES):
        raise ValueError("required case order/tiers differ from the frozen two-smoke/six-quality suite")
    if any(item["id"] not in cases for item in assignments):
        raise ValueError("unknown/missing required API cases")
    for item in assignments:
        # api_run's CLI takes the seed from registered metadata, not an
        # arbitrary --seed flag. Bind its actual default before any child.
        protocol_seed = cases[item["id"]].get("protocol_seed", 24002)
        if type(protocol_seed) is not int or protocol_seed != spec["seed"]:
            raise ValueError("registered case protocol seed differs from the frozen study seed")
        _positive(item.get("timeout_seconds"), "case timeout")
    for key in ("budget_seconds", "candidate_budget_seconds", "export_grace_seconds"):
        _positive(spec.get(key), key)
    next_task = max(item["timeout_seconds"] + spec["export_grace_seconds"] for item in assignments)
    if spec["candidate_budget_seconds"] < next_task or spec["budget_seconds"] < next_task:
        raise ValueError("budgets cannot reserve a complete declared task allowance")
    quotas = spec.get("family_budget_seconds", {family: spec["budget_seconds"] / 2 for family in FAMILIES})
    if (not isinstance(quotas, dict) or set(quotas) != set(FAMILIES)
            or any(_positive(value, "family budget") < next_task for value in quotas.values())
            or sum(quotas.values()) > spec["budget_seconds"]):
        raise ValueError("fixed family budgets must reserve full tasks without exceeding the round paid cap")
    if spec.get("stability") != {"confirmation_checks": 5, "post_confirmation_hold_checks": 5,
                                "first_window_only": True, "all_subsequent_primary_checks": True}:
        raise ValueError("study acquisition/hold contract differs from preregistration")
    if (spec.get("speed_ranking") is not False or spec.get("default_adoption") is not False
            or spec.get("backend") not in ("cpu", "cuda")):
        raise ValueError("speed/default adoption must remain disabled and backend explicit")
    if not isinstance(spec.get("representation_card"), dict) or set(spec["representation_card"]) != {"path", "sha256"}:
        raise ValueError("exact representation card path/hash required")
    _positive(spec.get("frames", 9), "media frames", integer=True)
    return spec


def _proofs(spec, cases):
    binding = spec["representation_card"]
    path = Path(binding["path"])
    if not path.is_absolute():
        path = contract.ROOT / path
    if not path.is_file() or api_run.file_hash(path) != binding["sha256"]:
        raise ValueError("representation card is missing or changed")
    packet = json.loads(path.read_text())
    records = packet.get("records", [])
    keys = [(record.get("family"), record.get("case_id")) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate family/case representation records")
    results = {}
    for family in spec["families"]:
        for item in spec["cases"]:
            case = cases[item["id"]]
            record = next((record for record in records if (record.get("family"), record.get("case_id")) == (family, case["id"])), None)
            reason = None
            if not record or record.get("status") != "SUPPORTED":
                reason = "no supported exact public-policy representation witness"
            elif (record.get("bindings") != proof_bindings(case, family)
                  or not isinstance(record.get("claim_scope"), str) or not record["claim_scope"].strip()):
                reason = "representation witness scope/source/case/preset/sampling binding differs"
            else:
                observations = record.get("observations")
                if not isinstance(observations, list) or not observations or any(
                        observation.get("passed") is not True or observation.get("failed_bounds") != []
                        or not isinstance(observation.get("metrics"), dict) or not observation["metrics"]
                        or any(not isinstance(value, (int, float, bool)) or not math.isfinite(value)
                               for value in observation["metrics"].values()) for observation in observations):
                    reason = "capacity witness lacks actual finite passing served observations"
                artifacts = record.get("artifacts")
                if not isinstance(artifacts, dict) or not {"state", "samples"} <= set(artifacts):
                    reason = "capacity witness lacks bound retained state/sample artifacts"
                else:
                    for artifact in artifacts.values():
                        artifact_path = Path(artifact.get("path", ""))
                        if not artifact_path.is_file() or api_run.file_hash(artifact_path) != artifact.get("sha256"):
                            reason = "capacity witness retained artifact missing or changed"
                    if reason is None:
                        try:
                            _check_capacity(record, case, family)
                        except Exception as error:
                            reason = f"capacity witness original sampler/gate/state verification failed: {error}"
            results[(family, case["id"])] = {"status": "BLOCKED" if reason else "SUPPORTED", "reason": reason}
    return results


def plan_study(spec, *, cases=None):
    cases = contract.discover() if cases is None else cases
    validate_spec(spec, cases)
    selected = {item["id"]: cases[item["id"]] for item in spec["cases"]}
    proofs = _proofs(spec, selected)
    trials = []
    for family in spec["families"]:
        for lr, rate in product(spec["grid"]["lr"], spec["grid"]["prior_lr_mult"]):
            knobs = {"lr": lr, "prior_lr_mult": rate}
            rows = []
            for item in spec["cases"]:
                case = selected[item["id"]]
                proof = proofs[(family, case["id"])]
                blockers = [proof["reason"]] if proof["status"] != "SUPPORTED" else []
                try:
                    contract.validate_recipe_overrides(case, family, knobs)
                except ValueError as error:
                    blockers.append(str(error))
                rows.append({**item, "case_sha256": digest(case), "status": "BLOCKED" if blockers else "UNKNOWN",
                             "fixed_host_recipe_options": api_run.json_value({key: value for key, value in host_recipe_options(case).items()
                                                                             if key not in TUNING_FIELDS}),
                             "resolved_recipe_sha256": digest(resolved_recipe(case, family, knobs)),
                             "blockers": blockers, "original_gate": None, "study_gate": None,
                             "full_protocol_complete": False})
            trials.append({"id": family + "--" + digest({"family": family, "overrides": knobs}),
                           "family": family, "recipe_overrides": knobs, "cases": rows,
                           "status": "BLOCKED" if any(row["blockers"] for row in rows) else "UNKNOWN"})
    return {"schema": SCHEMA, "study_id": spec["id"], "spec": deepcopy(spec), "spec_sha256": digest(spec),
            "source": _source(selected), "case_definitions": selected, "trials": sorted(trials, key=lambda trial: trial["id"]),
            "capacity_preflight": {family + "/" + case_id: value for (family, case_id), value in proofs.items()},
            "runtime_contract": {"python": platform.python_version(), "torch": str(torch.__version__),
                                 "cuda": torch.version.cuda, "torch_threads": torch.get_num_threads(),
                                 "backend": spec["backend"]},
            "candidate_worst_case_reservation_seconds": sum(item["timeout_seconds"] + spec["export_grace_seconds"] for item in spec["cases"]),
            "round_worst_case_reservation_seconds": 8 * sum(item["timeout_seconds"] + spec["export_grace_seconds"] for item in spec["cases"]),
            "family_paid_budget_seconds": spec.get("family_budget_seconds", {family: spec["budget_seconds"] / 2 for family in FAMILIES}),
            "goal": "policy-family-defaults", "scope": "eight declared cloud/served-policy API cases; no Forge MoG or all-toy/default qualification",
            "screen_profile": "provisional scoped screen; calibration and reserved robustness not performed",
            "speed_ranking": False, "default_adoption": False}


def acquisition_hold(receipt):
    """First five primary successes, then at least five uninterrupted hold checks."""
    steps = receipt["protocol"]["metric_evaluation_steps"]
    rows = [row for row in receipt["observations"] if row["step"] > 0 and row["step"] in steps]
    if [row["step"] for row in rows] != steps[1:]:
        return {"status": "INCOMPLETE", "reason": "complete primary observation schedule unavailable"}
    times = [row.get("elapsed_seconds") for row in receipt["observations"]]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in times) or any(
            later < earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("finite nondecreasing source-bound observation timing required")
    streak, confirmation = 0, None
    for index, row in enumerate(rows):
        streak = streak + 1 if row["passed"] else 0
        if streak == 5:
            confirmation = index
            break
    if confirmation is None:
        return {"status": "FAIL", "reason": "no five-check acquisition window"}
    hold = rows[confirmation + 1:]
    retained = all(row["passed"] for row in hold)
    status = "FAIL" if not retained else "INCOMPLETE" if len(hold) < 5 else "PASS"
    return {"status": status, "reason": "first acquisition and uninterrupted hold" if status == "PASS" else
            "first acquisition later collapsed" if status == "FAIL" else "fewer than five post-confirmation hold observations",
            "acquired_step": rows[confirmation]["step"], "acquired_seconds": rows[confirmation]["elapsed_seconds"],
            "hold_checks": len(hold), "hold_passed": sum(row["passed"] for row in hold),
            "speed_eligible": False}


def _finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tensors(part) for part in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tensors(part) for part in value)
    return True


def _score(case, samples, step):
    from . import api_images, api_vectors
    return (api_images.score_case(case["id"], samples) if case["provider"] == "api_images"
            else api_vectors.score_case(case, samples, step))


def _same_measured_gate(observation, scored):
    # The public observation validator canonicalizes failure ordering. Preserve
    # every exact bound (including multiplicity), independently of list order.
    return (observation["passed"] is scored["passed"]
            and sorted(observation["failed_bounds"]) == sorted(scored["failed_bounds"])
            and all(observation["metrics"].get(key) == value for key, value in scored["metrics"].items()))


def _check_capacity(record, case, family):
    knobs = record.get("recipe_overrides", {})
    if knobs and (set(knobs) != TUNING_FIELDS or knobs["lr"] not in [.006375, .0085]
                  or knobs["prior_lr_mult"] not in [1., 2.]):
        raise ValueError("capacity witness changes undeclared fields outside the shared knob grid")
    state = torch.load(record["artifacts"]["state"]["path"], map_location="cpu", weights_only=True)
    api_state = state.get("trainer", state.get("api_state"))
    if (not isinstance(api_state, dict) or api_state.get("completed_steps") != 0
            or api_state.get("max_steps", api_state.get("recipe", {}).get("total_steps")) != case["default_steps"]
            or api_run.json_value(api_state.get("recipe", {})) != resolved_recipe(case, family, knobs)):
        raise ValueError("capacity must bind a zero-update public state with original full horizon and family")
    _check_health(state, case, api_state["recipe"], completed_steps=0)
    with np.load(record["artifacts"]["samples"]["path"], allow_pickle=False) as arrays:
        for observation in record["observations"]:
            if observation.get("completed_steps", 0) != 0:
                raise ValueError("snapshot capacity must not contain learned-update credit")
            key = observation.get("samples_key", "samples")
            if key not in arrays or len(arrays[key]) != case["eval_samples"]:
                raise ValueError("capacity draws lack the original evaluation count")
            if not _same_measured_gate(observation, _score(case, arrays[key], 0)):
                raise ValueError("capacity metrics differ from actual retained sampled arrays")
        _replay_capacity(record, case, family, state, arrays)


def _same_state(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype
                and torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(right) is type(left) and len(left) == len(right) and all(_same_state(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and math.isnan(left):
        return isinstance(right, float) and math.isnan(right)
    return left == right


def _replay_capacity(record, case, family, state, arrays):
    """Pair capacity arrays to their real public restored sampler, zero updates."""
    with api_run.isolated_evaluation():
        fixture = contract.build(case, device="cpu", seed=24002, recipe_name=family,
                                 max_steps=case["default_steps"], recipe_overrides=record.get("recipe_overrides", {}))
        if case["provider"] == "api_vectors":
            fixture.load_state_dict(state)
        else:
            if state.get("case") != fixture.case or state.get("seed") != fixture.seed:
                raise ValueError("capacity checkpoint differs from its actual public image host")
            fixture.trainer.load_state_dict(state["api_state"])
            fixture.data_generator.set_state(state["data_generator"])
        before = fixture.state_dict()
        for observation in record["observations"]:
            key = observation.get("samples_key", "samples")
            seed = observation.get("evaluation_seed", int(key) if str(key).isdigit() else 34002)
            if type(seed) is not int or seed < 0:
                raise ValueError("capacity evaluation seed must be bound")
            actual = contract.validate_observation(fixture.observe(n=case["eval_samples"], seed=seed))
            samples = contract.array(actual["views"][0]["samples"])
            # Capacity cards retain the original primary scorer. Native
            # observers additionally report clean/no-noise diagnostics; these
            # do not enter that gate and need not be retroactively added to a
            # frozen card. Still verify the observer's actual binary verdict.
            primary = _score(case, samples, 0)
            if (not np.array_equal(samples, arrays[key])
                    or not _same_measured_gate(observation, primary)
                    or actual["passed"] is not primary["passed"]
                    or sorted(actual["failed_bounds"]) != sorted(primary["failed_bounds"])):
                raise ValueError("capacity arrays/metrics differ from the actual restored public sampler")
        if not _same_state(before, fixture.state_dict()):
            raise ValueError("capacity observation altered public state/RNG or optimizer clock")


def _check_health(state, case, recipe, *, completed_steps=None):
    completed_steps = case["default_steps"] if completed_steps is None else completed_steps
    api_state = state.get("trainer", state.get("api_state"))
    if (not isinstance(api_state, dict) or api_state.get("completed_steps") != completed_steps
            or api_run.json_value(api_state.get("recipe")) != api_run.json_value(recipe) or not api_state.get("models")
            or not api_state.get("optimizers")):
        raise ValueError("final public API state lacks exact completed model/optimizer/Recipe identity")
    # Controller masks/history legitimately contain NaN/Inf sentinels; those
    # are not learned model parameters. Do not silently add an impossible gate.
    learned = {key: api_state[key] for key in ("models", "averages", "optimizers", "table", "averaged_table", "output_noise")
               if key in api_state}
    if not _finite_tensors(learned):
        raise ValueError("retained learned models, optimizer state or output noise are nonfinite")


def _check_numeric_trace(path, receipt, case):
    """Check fresh measured arrays against the existing pure gate; no forwards."""
    with np.load(Path(path) / "observations.npz", allow_pickle=False) as arrays:
        for row in receipt["observations"]:
            samples = arrays[f"step{row['step']}_view0_samples"]
            if not _same_measured_gate(row, _score(case, samples, row["step"])):
                raise ValueError("recorded numeric gate differs from its retained actual sample arrays")


def verify_case(path, case, family, knobs, source, *, returncode, runtime=None, wall_cap_seconds, frames=None):
    receipt = api_publish.verify_run(path)
    if (digest(receipt["case"]) != digest(case) or receipt.get("seed") != 24002
            or receipt.get("requested_recipe_overrides") != knobs
            or receipt["recipe"] != resolved_recipe(case, family, knobs)):
        raise ValueError("case/family/seed/requested and actual Recipe differ from frozen study")
    if (receipt["source"].get("commit") != source["commit"] or any(
            receipt["source"]["files_sha256"].get(path) != value for path, value in source["files_sha256"].items())):
        raise ValueError("executed source differs from frozen study")
    if (receipt["protocol"]["updates"] != case["default_steps"]
            or receipt["protocol"]["evaluation_samples"] != case["eval_samples"]
            or not receipt["default_protocol_complete"]):
        raise ValueError("unchanged exact full steps/evaluation budget required")
    if returncode != (0 if receipt["passed"] else 1):
        raise ValueError("child exit disagrees with certified original numeric verdict")
    if runtime is not None and any(receipt["runtime"].get(key) != value for key, value in runtime.items()):
        raise ValueError("executed runtime/hardware differs from the fixed family lane")
    if (receipt["protocol"]["wall_cap_seconds"] != wall_cap_seconds
            or (frames is not None and receipt["protocol"]["media_frames"] != frames)
            or receipt["elapsed_seconds"] > wall_cap_seconds):
        raise ValueError("completed acquisition exceeded the frozen wall allowance")
    _check_numeric_trace(path, receipt, case)
    state = torch.load(Path(path) / "final-state.pt", map_location="cpu", weights_only=True)
    _check_health(state, case, receipt["recipe"])
    hold = acquisition_hold(receipt)
    status = receipt["verdict"] if not receipt["passed"] else hold["status"]
    return {"status": status, "original_gate": receipt["verdict"], "study_gate": hold["status"],
            "full_protocol_complete": True, "acquisition_hold": hold,
            "elapsed_seconds": receipt["elapsed_seconds"], "final_metrics": receipt["observations"][-1]["metrics"],
            "receipt_path": str(Path(path) / "receipt.json"),
            "receipt_sha256": api_run.file_hash(Path(path) / "receipt.json"), "artifacts": receipt["artifacts"],
            "recipe": receipt["recipe"], "runtime": receipt["runtime"]}


def select_results(packet):
    trials = packet["trials"]
    grid = _validated_grid(packet["spec"])
    if packet["spec"].get("families") != list(FAMILIES):
        raise ValueError("both declared families must remain in the selection denominator")
    expected = [(case_id, tier) for case_id, tier in DEFAULT_CASES]
    expected_trials = {family + "--" + digest({"family": family, "overrides": {"lr": lr, "prior_lr_mult": rate}})
                       for family, lr, rate in product(FAMILIES, grid["lr"], grid["prior_lr_mult"])}
    if len(trials) != 8 or {trial["id"] for trial in trials} != expected_trials:
        raise ValueError("every declared configuration must remain in the selection denominator")
    for trial in trials:
        if ([(row["id"], row["tier"]) for row in trial["cases"]] != expected
                or trial["id"] != trial["family"] + "--" + digest({"family": trial["family"], "overrides": trial["recipe_overrides"]})):
            raise ValueError("missing, duplicate or changed required case/configuration scope")
        if trial["status"] == "PASS" and not all(row["status"] == "PASS" for row in trial["cases"]):
            raise ValueError("a passing configuration lacks a required case")
    concluded = all(trial["status"] in {"PASS", "FAIL", "INCOMPLETE", "BLOCKED", "ERROR", "INVALID"} for trial in trials)
    complete = all(trial["status"] in {"PASS", "FAIL"} for trial in trials)
    eligible = [trial for trial in trials if trial["status"] == "PASS" and all(row["status"] == "PASS" for row in trial["cases"])]
    def counts(trial):
        return [sum(row["status"] == "PASS" and row["tier"] == tier for row in trial["cases"]) for tier in (1, 2)]
    best = min(trials, key=lambda trial: (*(-value for value in counts(trial)), trial["id"])) if concluded else None
    return {"attempts_concluded": concluded, "comparison_complete": complete, "outcome": "pending" if not concluded else
            "incomplete_comparison" if not complete else
            "scoped_fully_qualified" if eligible else "best_observed",
            "fully_qualified_ids": sorted(trial["id"] for trial in eligible),
            "display_candidate_id": best["id"] if best else None,
            "timing_ties": "all qualified configs retained; speed unranked under external contention",
            "speed_winner": None, "default_adoption": False,
            "required_cases_per_config": len(DEFAULT_CASES)}


def _runtime(device):
    value = {"python": platform.python_version(), "torch": str(torch.__version__),
             "cuda": torch.version.cuda, "torch_threads": torch.get_num_threads(), "device": str(device)}
    if torch.device(device).type == "cuda":
        value["cuda_device_model"] = torch.cuda.get_device_name(torch.device(device))
    return value


def _save(registration, packet):
    packet["measured_paid_seconds"] = sum(row.get("paid_wall_seconds", 0.) for trial in packet["trials"] for row in trial["cases"])
    packet["unmeasured_interrupt_reservation_seconds"] = sum(row.get("unmeasured_interrupt_reserved_seconds", 0.)
                                                             for trial in packet["trials"] for row in trial["cases"])
    packet["selection"] = select_results(packet)
    api_run.write_json(registration, packet)


def _verify_costs(packet):
    """Saved counters cannot reduce the conservative paid/reserved ledger."""
    charged = measured = interrupted = 0.
    for trial in packet["trials"]:
        total = 0.
        for row in trial["cases"]:
            paid, reserved = row.get("paid_wall_seconds", 0.), row.get("unmeasured_interrupt_reserved_seconds", 0.)
            for value in (paid, reserved):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid negative/nonfinite saved cost")
            total += paid + reserved
            measured += paid
            interrupted += reserved
        if not math.isclose(trial.get("paid_wall_seconds", 0.), total, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError("saved candidate cost differs from retained case costs")
        charged += total
    for key, expected in (("spent_seconds", charged), ("measured_paid_seconds", measured),
                          ("unmeasured_interrupt_reservation_seconds", interrupted)):
        actual = packet.get(key, 0.)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError("saved family cost differs from retained measured/reservation ledger")


def _recertify_archive(packet):
    """Verify every retained complete receipt, including study-INCOMPLETE."""
    cases = contract.discover()
    validate_spec(packet["spec"], cases)
    selected = {item["id"]: cases[item["id"]] for item in packet["spec"]["cases"]}
    if (packet.get("schema") != SCHEMA or packet.get("spec_sha256") != digest(packet["spec"])
            or packet.get("case_definitions") != api_run.json_value(selected)
            or packet["source"].get("files_sha256") != _source(selected)["files_sha256"]):
        raise ValueError("archived protocol/gates/source differ from the actual frozen registry")
    quotas = packet["spec"].get("family_budget_seconds", {family: packet["spec"]["budget_seconds"] / 2 for family in FAMILIES})
    if packet.get("family_paid_budget_seconds") != quotas:
        raise ValueError("saved family budget differs from immutable preregistration")
    for trial in packet["trials"]:
        for row, item in zip(trial["cases"], packet["spec"]["cases"]):
            if any(row.get(key) != item[key] for key in ("id", "tier", "timeout_seconds")):
                raise ValueError("saved task allowance differs from immutable preregistration")
            reserved = row.get("unmeasured_interrupt_reserved_seconds")
            if reserved is not None and reserved != item["timeout_seconds"] + packet["spec"]["export_grace_seconds"]:
                raise ValueError("saved interrupt reservation differs from original complete allowance")
    # A later docs/readout commit may retain the same scientific bytes. The
    # raw child commit must match its frozen packet, never a pooled new source.
    select_results(packet)  # Exact configuration/task denominators first.
    _verify_costs(packet)
    family = packet["executed_family"]
    for trial in packet["trials"]:
        if trial["family"] != family:
            if any(row.get("paid_wall_seconds") or row.get("receipt_path") for row in trial["cases"]):
                raise ValueError("a family archive contains another family's execution")
            continue
        for row in trial["cases"]:
            if (row["status"] not in {"PASS", "FAIL"} and not row.get("full_protocol_complete")
                    and not row.get("receipt_path") and row.get("original_gate") != "PASS"):
                continue
            receipt_path = Path(row.get("receipt_path", ""))
            if not receipt_path.is_file() or api_run.file_hash(receipt_path) != row.get("receipt_sha256"):
                raise ValueError("archived scientific status lacks its unchanged bound receipt")
            verified = verify_case(receipt_path.parent, packet["case_definitions"][row["id"]], family,
                                   trial["recipe_overrides"], packet["source"], returncode=row.get("child_returncode"),
                                   runtime=packet["lane_runtime"], wall_cap_seconds=row["timeout_seconds"], frames=packet["spec"].get("frames", 9))
            paid = row.get("paid_wall_seconds")
            if (isinstance(paid, bool) or not isinstance(paid, (int, float)) or not math.isfinite(paid)
                    or paid < verified["elapsed_seconds"]):
                raise ValueError("saved paid cost is less than its source-bound acquisition time")
            for key in ("status", "original_gate", "study_gate", "full_protocol_complete", "acquisition_hold", "final_metrics", "recipe", "runtime", "artifacts"):
                if row.get(key) != verified[key]:
                    raise ValueError("saved scientific status differs from its certified raw receipt")


def child_command(spec, trial, row, case_root, device):
    """Exact existing API CLI; original budgets and metadata-bound seed."""
    return [sys.executable, "-m", "benchmarks.toy_audit.api_run", "--case", row["id"],
            "--output", str(Path(case_root).parent), "--recipe", trial["family"], "--device", device,
            "--recipe-overrides", json.dumps(trial["recipe_overrides"]),
            "--wall-cap-seconds", str(row["timeout_seconds"]), "--frames", str(spec.get("frames", 9))]


def run_study(spec, output, *, family, device):
    """Sequential children only; no extra workers or changed protocol resources."""
    if family not in FAMILIES or torch.device(device).type != spec["backend"]:
        raise ValueError("family/device must match the frozen study")
    packet = plan_study(spec)
    runtime = _runtime(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    registration = output / "study.json"
    if registration.exists():
        saved = json.loads(registration.read_text())
        if any(saved.get(key) != packet.get(key) for key in ("spec_sha256", "spec", "source", "case_definitions", "capacity_preflight", "runtime_contract", "family_paid_budget_seconds")):
            raise ValueError("study identity changed; use a new study ID/output")
        if saved.get("executed_family") != family or saved.get("lane_runtime") != runtime:
            raise ValueError("one family/fixed runtime per output; use separate family archives")
        _recertify_archive(saved)
        packet = saved
    elif any(output.iterdir()):
        raise ValueError("unregistered nonempty archive cannot be reused as a fresh study")
    packet.update(executed_family=family, lane_runtime=runtime)
    packet.setdefault("spent_seconds", 0.)
    _save(registration, packet)  # Freeze registration before any paid child.
    for trial in packet["trials"]:
        if trial["family"] != family or trial["status"] != "UNKNOWN":
            continue
        trial_spent = trial.get("paid_wall_seconds", 0.)
        for row in trial["cases"]:
            if row["status"] == "PASS":
                if api_run.file_hash(row["receipt_path"]) != row["receipt_sha256"]:
                    raise ValueError("retained completed case changed; no unchanged reexecution")
                continue
            if row["status"] != "UNKNOWN":
                if row["status"] == "RUNNING":
                    row.update(status="INCOMPLETE", reason="interrupted paid attempt; no automatic unchanged retry",
                               unmeasured_interrupt_reserved_seconds=row["timeout_seconds"] + spec["export_grace_seconds"])
                    packet["spent_seconds"] += row["unmeasured_interrupt_reserved_seconds"]
                    trial_spent += row["unmeasured_interrupt_reserved_seconds"]
                trial["status"] = row["status"]
                break
            allowance = row["timeout_seconds"] + spec["export_grace_seconds"]
            if allowance > packet["family_paid_budget_seconds"][family] - packet["spent_seconds"] or allowance > spec["candidate_budget_seconds"] - trial_spent:
                trial["status"] = "INCOMPLETE"
                trial["reason"] = "remaining frozen budget cannot reserve complete next task"
                break
            case = packet["case_definitions"][row["id"]]
            case_root = output / trial["id"] / row["id"]
            log = output / trial["id"] / (row["id"] + ".log")
            if case_root.exists() or log.exists():
                row.update(status="INCOMPLETE", reason="orphan paid artifacts retained; no automatic unchanged retry")
                trial["status"] = "INCOMPLETE"
                break
            log.parent.mkdir(parents=True, exist_ok=True)
            command = child_command(spec, trial, row, case_root, device)
            row.update(status="RUNNING", command=command, log_path=str(log))
            _save(registration, packet)
            started = time.monotonic()
            child_finished = None
            try:
                with log.open("w") as stream:
                    child = subprocess.run(command, cwd=contract.ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=allowance, check=False)
                child_finished = time.monotonic()
                raw = json.loads((case_root / "receipt.json").read_text())
                if raw.get("status") != "COMPLETE":
                    row.update(status=raw.get("status") if raw.get("status") in {"INCOMPLETE", "BLOCKED", "ERROR"} else "INVALID",
                               reason="; ".join(raw.get("failed_bounds", [])), original_gate=raw.get("verdict"))
                else:
                    row.update(verify_case(case_root, case, family, trial["recipe_overrides"], packet["source"],
                                           returncode=child.returncode, runtime=runtime,
                                           wall_cap_seconds=row["timeout_seconds"], frames=spec.get("frames", 9)))
                row["child_returncode"] = child.returncode
            except subprocess.TimeoutExpired:
                child_finished = time.monotonic()
                row.update(status="INCOMPLETE", reason="hard subprocess cap; full protocol unavailable")
            except Exception as error:
                row.update(status="INVALID", reason=f"{type(error).__name__}: {error}")
            child_finished = time.monotonic() if child_finished is None else child_finished
            elapsed = child_finished - started
            packet["spent_seconds"] += elapsed
            trial_spent += elapsed
            row.update(paid_wall_seconds=elapsed, validation_seconds=time.monotonic() - child_finished,
                       log_path=str(log), command=command)
            if packet["spent_seconds"] > packet["family_paid_budget_seconds"][family] or trial_spent > spec["candidate_budget_seconds"]:
                row.update(status="INCOMPLETE", reason="paid child execution exceeded the frozen family/candidate cap")
            trial["paid_wall_seconds"] = trial_spent
            _save(registration, packet)
            if row["status"] != "PASS":
                trial["status"] = row["status"]
                break
        else:
            trial["status"] = "PASS"
        trial["paid_wall_seconds"] = trial_spent
        _save(registration, packet)
    _save(registration, packet)
    return packet


def combine_studies(paths):
    """Union separately owned family lanes without pooling case/source cohorts."""
    packets = [json.loads(Path(path).read_text()) for path in paths]
    if len(packets) != 2 or {packet.get("executed_family") for packet in packets} != set(FAMILIES):
        raise ValueError("exactly one independent atlas archive and one e22 archive are required")
    for key in ("spec_sha256", "source", "case_definitions", "capacity_preflight", "runtime_contract"):
        if packets[0].get(key) != packets[1].get(key):
            raise ValueError(f"cannot pool differing family {key} cohorts")
    for source in packets:
        _recertify_archive(source)
    packet = deepcopy(packets[0])
    packet.pop("executed_family", None)
    packet.pop("lane_runtime", None)
    packet["family_archives"] = [{"family": source["executed_family"], "path": str(Path(path).resolve()),
                                   "sha256": api_run.file_hash(path), "runtime": source["lane_runtime"]}
                                  for path, source in zip(paths, packets)]
    packet["trials"] = sorted([trial for source in packets for trial in source["trials"]
                              if trial["family"] == source["executed_family"]], key=lambda trial: trial["id"])
    packet["spent_seconds"] = sum(source["spent_seconds"] for source in packets)
    packet["measured_paid_seconds"] = sum(source["measured_paid_seconds"] for source in packets)
    packet["unmeasured_interrupt_reservation_seconds"] = sum(source["unmeasured_interrupt_reservation_seconds"] for source in packets)
    packet["selection"] = select_results(packet)
    return packet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "run", "combine"))
    parser.add_argument("spec", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--archive", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text()) if args.stage != "combine" else None
    if args.stage == "combine":
        packet = combine_studies([args.spec, *args.archive])
    elif args.stage == "plan":
        packet = plan_study(spec)
    else:
        if args.output is None or args.family is None:
            parser.error("run requires --output and --family; coordinator owns GPU admission")
        packet = run_study(spec, args.output, family=args.family, device=args.device)
    if args.stage in ("plan", "combine") and args.output:
        api_run.write_json(args.output, packet)
    print(json.dumps(api_run.json_value(packet), indent=2, allow_nan=False))
    if args.stage == "plan":
        return 0
    return 0 if packet["selection"]["fully_qualified_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

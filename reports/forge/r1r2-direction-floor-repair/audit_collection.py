"""Independently audit saved direction/floor evidence; never execute a scorer.

Use the declared scientific interpreter and thread environment. Without --write
the receipt is printed only. Run with --write after publication is complete.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.state import require_consistent_rng, state_digest
from particlegan import GANTrainer, MoGParticlePrior, ParticlePrior
from reports.forge.regenerate_technique_inventory import project_receipt

PYTHON = "/home/martyn/dev/ParticleGAN/.venv/bin/python"
EXECUTION = "a7f141c9c73d7aead972f2d7a382b2f52a59ac69"
ARCHIVE = Path("/mnt/ml7tb/ParticleGAN-forge/r1r2-direction-floor-repair-20261012")
ORIGINAL = Path("/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012")
CAMPAIGN = "r1r2-direction-floor-repair-v1"
CLOCK = "clockfree_audit_measurement_v1"
TASKS = {"gaussian1d_smoke", "two_pole", "unused_token_hold", "ae_gan_hold",
         "ring16_acquisition", "five_word_joint_smoke", CLOCK}
CONTROLS = {"baseline", "adam"}
OLD_CHARGE_HASH = "a17977e537785fbee63c216f8cbda0e11663f0fcecf69ad936ea94c2f85b49a0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def forbid_live_execution():
    def forbidden(*_args, **_kwargs):
        raise RuntimeError("Saved-evidence audit forbids construction, training, sampling and random draws")
    methods = [(torch.nn.Module, "__init__"), (torch.nn.Module, "__call__"),
               (GANTrainer, "__init__"), (GANTrainer, "step"), (GANTrainer, "sample"),
               (ParticlePrior, "sample"), (MoGParticlePrior, "sample")]
    methods += [(torch, name) for name in ("rand", "randn", "randint", "randperm", "normal", "multinomial")]
    with ExitStack() as stack:
        for owner, name in methods:
            stack.enter_context(patch.object(owner, name, forbidden))
        yield


def projection(request):
    assignments = [a for a in request["view"]["assignments"] if a["qualification_tier"] == 1]
    require(request["view"]["id"] == "discriminator_stability" and request["through_tier"] == 1,
            "Scope must remain the ordinary view through Tier1")
    require({a["task"] for a in assignments} == TASKS and
            {a["task"] for a in assignments if a["importance"] == "required"} == TASKS - {CLOCK},
            "Required task/clock cohorts changed")
    require(request["protocol"]["seed"] == 0 and request["execution_policy"]["mode"] == "complete_current_tier",
            "Protocol seed or complete-current-tier changed")
    tiers = {a["task"]: a["qualification_tier"] for a in request["view"]["assignments"]}
    jobs = [j for j in request["jobs"] if all(tiers[t] <= request["through_tier"]
            for t in j.get("task_ids", [j["task_id"]]))]
    require(sum(j["budget_seconds"] for j in jobs) == 2520, "Full task allowances changed")
    task_hashes = {t: stable_hash({k: v for k, v in request["tasks"][t].items()
                                  if k not in {"field_ownership", "preflight_blockers"}}) for t in sorted(TASKS)}
    return dict(candidate=request["candidate"]["id"], candidate_revision=request["candidate_revision"],
        source_digest=request["source"]["digest"], protocol_sha256=stable_hash(request["protocol"]),
        runtime_sha256=stable_hash(request["runtime"]), execution_policy=request["execution_policy"],
        view_sha256=stable_hash(request["view"]), task_contracts=task_hashes,
        jobs_sha256=stable_hash(jobs), task_keys={t: j["compatibility_key"] for j in jobs
            for t in j.get("task_ids", [j["task_id"]])}, full_allowance_seconds=2520)


def stored_grade(request, result):
    require(request["requires_independent_grading"] and result["raw"]["attempt_status"] == "completed",
            "Original independent grading must be complete")
    raw = result["raw"]
    grading = raw["grading"]
    require(grading["raw_hash"] == stable_hash(raw["result"]) and
            grading["source_digest"] == request["source"]["digest"], "Stored grading identity changed")
    require(set(grading["grades"]) == {r["task_id"] for r in result["task_results"]}, "Stored grades missing/duplicated")
    for row in result["task_results"]:
        grade = grading["grades"][row["task_id"]]
        require(all(row.get(k) == v for k, v in grade.items()), "Original independently graded fields changed")
        member = raw["result"].get("task_results", {}).get(row["task_id"], raw["result"])
        require(row.get("metrics") == grade.get("metrics", member.get("metrics")) and
                row["evidence"] == grade.get("evidence", member.get("evidence")), "Stored grading metrics/evidence changed")
        require(row["gate_status"] == grade.get("gate_status", grade.get("status", "INVALID")), "Original grade changed")
    return dict(attempt_status="completed", independently_graded=True,
        raw_result_sha256=grading["raw_hash"], source_digest=grading["source_digest"],
        grading_sha256=stable_hash(grading), grade_sha256={t: stable_hash(g) for t, g in grading["grades"].items()})


def read_attempt(root, aid):
    directory = root / "reports/forge/attempts" / aid
    envelope, result, certificate = [read_json(directory / f"{name}.json") for name in ("request", "result", "evidence")]
    request = envelope.get("request", envelope)
    require(certificate["result_hash"] == stable_hash(result), "Result certificate changed")
    require(certificate["source"] == request["source"] and certificate["runtime"] == request["runtime"],
            "Source/runtime certificate changed")
    require(stable_hash(request["source"]["files"]) == request["source"]["digest"], "Scientific source manifest changed")
    require(result["attempt_id"] == aid and result["candidate_revision"] == request["candidate_revision"] and
            not result.get("retry_of") and not envelope.get("retry_of"), "Attempt identity/retry changed")
    local = Path(certificate["local_artifact_root"])
    require(read_json(local / "request.json") == envelope and read_json(local / "result.json") == result,
            "Local original receipt bytes changed")
    return request, result, local, project_receipt(root, aid)


def checkpoint(row):
    descriptor = row["evidence"]["provenance_checkpoint"]
    directory = Path(descriptor["artifact_root"]).resolve()
    verify_artifacts(directory, descriptor["artifact_manifest"])
    path = (directory / descriptor["path"]).resolve()
    require(path.is_relative_to(directory) and file_hash(path) == descriptor["sha256"] and
            path.stat().st_size == descriptor["bytes"], "Checkpoint bytes changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    require(state_digest(saved) == descriptor["state_sha256"], "Saved checkpoint tensors changed")
    require(set(saved["streams"]["states"]) == set(descriptor["named_stream_state_sha256"]), "Checkpoint stream keys changed")
    for key, value in saved["streams"]["states"].items():
        require(state_digest(value) == descriptor["named_stream_state_sha256"][key], "Certified named stream state changed")
    compact = {k: descriptor[k] for k in ("path", "sha256", "bytes", "state_sha256", "completed_steps")}
    compact.update(artifact_root=str(directory), artifact_manifest_sha256=descriptor["artifact_manifest"]["sha256"],
                   purpose=descriptor.get("purpose"), prerequisite_eligible=descriptor.get("prerequisite_eligible"))
    return saved, compact


def named_streams(saved, comparison=None):
    streams = saved["streams"]
    require(set(streams["states"]) == set(streams["manifest"]["bindings"]), "Named RNG checkpoint incomplete")
    if "trainer" in saved:
        require_consistent_rng(saved)
    if "fixture" in saved:
        def named_state(family, component, purpose):
            matching = [key for key, binding in streams["manifest"]["bindings"].items()
                        if (binding["family"], binding["component"], binding["purpose"]) == (family, component, purpose)]
            require(len(matching) == 1, "Joint fixture consumed stream lacks a unique named checkpoint")
            return streams["states"][matching[0]].cpu()
        fixture = saved["fixture"]
        require(torch.equal(fixture["data_generator"].cpu(), named_state("data", "target", "training")),
                "Joint data generator state differs from its named checkpoint")
        bindings = dict(latent_generator=("prior", "latent", "indices"),
                        penalty_generator=("noise", "penalty", "training"),
                        eval_generator=("eval", "sampler", "samples"),
                        noise_generator=("noise", "generator", "output"))
        require(set(fixture["api_state"]["streams"]) == set(bindings), "Joint policy stream coverage changed")
        for name, binding in bindings.items():
            require(torch.equal(fixture["api_state"]["streams"][name].cpu(), named_state(*binding)),
                    "Joint policy stream differs from its named checkpoint")
    if comparison:
        receipt = comparison["rng_checkpoints"]
        require(receipt["all_named_streams_checkpointed"] and set(receipt["bindings"]) == set(streams["states"]),
                "Comparison RNG coverage incomplete")
        for key, value in streams["states"].items():
            digest = hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()
            item = receipt["bindings"][key]
            require(item["final_state_sha256"] == digest and item["family"] == streams["manifest"]["bindings"][key]["family"] and
                    item["changed"] == (item["initial_state_sha256"] != digest), "Actual named RNG receipt changed")
    return len(streams["states"])


def role_groups(saved, task):
    if "trainer" in saved:
        state = saved["trainer"]
        generator, critic = state["optimizers"]
        require(len(generator["param_groups"]) == 2 and len(critic["param_groups"]) == 1,
                "Public trainer optimizer role layout changed")
        return [("G", generator, generator["param_groups"][0]), ("P", generator, generator["param_groups"][1]),
                ("D", critic, critic["param_groups"][0])], saved["recipe"]
    if "fixture" in saved:
        state = saved["fixture"]["api_state"]
        generator, critic = state["optimizers"]
        require(state["roles"] == [["generator", "encoder", "table"], ["critic"]] and
                len(generator["param_groups"]) == 3 and len(critic["param_groups"]) == 1,
                "Joint G/E/P/D optimizer role layout changed")
        return [(r, generator, g) for r, g in zip(("G", "E", "P"), generator["param_groups"], strict=True)] + [
                ("D", critic, critic["param_groups"][0])], saved["fixture"]["recipe"]
    rows = []
    gparts = saved["optimizers"]["generator"]
    require(len(gparts) == (2 if task == "ae_gan_hold" else 1), "Behavior optimizer role layout changed")
    for index, optimizer in enumerate(gparts):
        require(len(optimizer["param_groups"]) == 1, "Behavior optimizer group layout changed")
        rows.append(("P" if task == "ae_gan_hold" and index == 1 else "G", optimizer, optimizer["param_groups"][0]))
    critic = saved["optimizers"]["discriminator"]
    require(len(critic["param_groups"]) == 1, "Behavior critic groups changed")
    rows.append(("D", critic, critic["param_groups"][0]))
    bindings = saved["applied"]["optimizer_group_bindings"]
    require(len(bindings) == len(rows), "Actual behavior bindings incomplete")
    for (_, _, group), binding in zip(rows, bindings, strict=True):
        require(list(group["betas"]) == binding["current_group_betas"], "Actual group betas differ from host receipt")
        if task == "two_pole":
            require(not binding["prior_betas_consumed"] and not binding["direct_particle_response"]["installed"],
                    "Pole unexpectedly consumed sampled-prior moments or direct response")
        if task == "unused_token_hold":
            require(not binding["prior_betas_consumed"], "Unsampled token fixture unexpectedly consumed prior moments")
    if task == "two_pole":
        require(bindings[0]["representation"] == "direct_sample_coordinates" and
                bindings[0]["counter_role"] == "prior", "Direct pole representation changed")
        require(not any(name.startswith("prior") for name in saved["models"]) and
                all(p["representation"] == "direct_sample_coordinates" for p in saved["role_parameters"]["prior"]),
                "Direct pole was silently replaced by a sampled-prior cohort")
    if task == "unused_token_hold":
        require("prior" not in saved["role_parameters"] and not any(name.startswith("prior") for name in saved["models"]),
                "Token hold unexpectedly acquired sampled-prior locations")
    return rows, saved["applied"]["recipe"]


def optimizer_roles(saved, task, role, *, clock=False):
    groups, recipe = role_groups(saved, task)
    family = "ada_nsgda" if role in {"ada", "ada_eps"} else "adam"
    critic_eps = 1e-4 if role in {"adam_eps", "ada_eps"} else 1e-8
    require(recipe.get("optimizer_family", "adam") == family, "Saved recipe optimizer family changed")
    require(recipe.get("d_eps") == (critic_eps if role.endswith("_eps") else None), "Declared critic epsilon changed")
    observed = {}
    for player, optimizer, group in groups:
        expected_eps = critic_eps if player == "D" else 1e-8
        require(group["eps"] == expected_eps and group["betas"][0] == 0. and not group["amsgrad"],
                f"{role}/{task}/{player}: actual epsilon, beta1 or AMSGrad changed")
        require(group["betas"][1] == .99 if role != "baseline" or not clock else .9 <= group["betas"][1] <= .99,
                f"{role}/{task}/{player}: actual beta2 changed")
        if family == "adam":
            require("dualnorm" not in optimizer and "algorithm" not in group,
                    "Cached/new Adam was replaced by a normalized optimizer")
            history_fields = {"step", "exp_avg", "exp_avg_sq"}
        else:
            meta = optimizer["dualnorm"]
            require(set(meta) == {"schema", "family", "momentum", "sampled_rows"} and
                    meta["schema"] == 1 and meta["family"] == family and meta["momentum"] == 0 and
                    len(meta["sampled_rows"]) == len(optimizer["param_groups"]) and
                    all(value is None for value in meta["sampled_rows"]), "Actual Ada optimizer metadata changed")
            require(group["algorithm"] == family and group["role"] ==
                    {"G": "generator", "E": "generator", "P": "prior", "D": "critic"}[player],
                    "Actual Ada group algorithm/role changed")
            history_fields = {"step", "exp_avg_sq"}
        require(set(optimizer["state"]) <= {parameter for g in optimizer["param_groups"] for parameter in g["params"]},
                "Optimizer history contains an unknown parameter")
        for parameter in group["params"]:
            history = optimizer["state"].get(parameter)
            require(history and set(history) == history_fields, "Actual Adam/Ada update histories changed")
            step = history["step"]
            step = float(step.item()) if isinstance(step, torch.Tensor) else step
            require(step > 0 and int(step) == step, "Actual optimizer updates missing")
            for field in history_fields - {"step"}:
                value = history[field]
                require(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()), "Invalid saved optimizer history")
                if field == "exp_avg_sq":
                    require(bool((value >= 0).all()), "Invalid negative second moments")
        observed[player] = dict(family=family, algorithm=group.get("algorithm", "adam"),
                               betas=list(group["betas"]), eps=group["eps"], history_fields=sorted(history_fields))
    if task == "ae_gan_hold":
        parameters = saved["role_parameters"]
        require(parameters["encoder"] and len(groups[0][2]["params"]) ==
                len(parameters["generator"]) + len(parameters["encoder"]), "Actual AE G/E joint group membership changed")
        observed["E"] = dict(observed["G"])
    if role != "baseline":
        require(recipe.get("beta2_end") is None and recipe["lr_schedule"] == "constant" and
                recipe["lr"] == .0085 and recipe["reg_coeff"] == .1 and
                recipe.get("prior_update", "learned") == "learned" and recipe["prior_reg"] == 0,
                "Declared constant learned-prior law changed")
        for field in ("betas", "d_betas", "prior_betas"):
            require(list(recipe.get(field) or recipe["betas"]) == [0., .99], "Global role moments changed")
    return observed, len(groups)


def saved_media(root, export, row, task, local, clock_proof=None):
    path = root / export["path"]
    receipt = read_json(path.with_suffix(".json"))
    require(receipt == export["receipt"] and file_hash(path) == export["sha256"] == receipt["gif_sha256"] and
            file_hash(path.with_suffix(".json")) == export["receipt_sha256"], "GIF/sidecar bytes changed")
    require(receipt["task_id"] == task["id"] and receipt["recorded_grade"] == row["gate_status"] and
            receipt["optimizer_updates_added"] == receipt["sampling_draws_added"] == 0 and
            receipt["qualification_input"] is False, "Saved GIF grade/side effects changed")
    require(receipt["renderer_sha256"] == file_hash(root / "experiments/forge/tier1_media.py"), "GIF renderer source changed")
    if clock_proof is not None:
        proof_path = Path(row["evidence"]["artifact_root"]) / "comparisons.pt"
        initial = clock_proof["initial"]["trainer"]["models"]["G"]
        parameter = next(k for k, v in initial.items() if v.is_floating_point() and v.numel())
        observations = [dict(step=i + 1, **{name: float((states[i]["trainer"]["models"]["G"][parameter]
            - initial[parameter]).abs().mean()) for name, states in clock_proof["trajectories"].items()})
            for i in range(task["execution"]["probe_steps"])]
        inputs, selection = {str(proof_path): file_hash(proof_path)}, None
    else:
        evidence = row["evidence"]
        descriptor = evidence["saved_observer_outputs"]
        directory = Path(evidence["artifact_root"]) if task["evaluation"]["kind"] == "gaussian_smoke" else local
        saved_path = (directory / descriptor["path"]).resolve()
        require(saved_path.is_relative_to(directory.resolve()) and file_hash(saved_path) == descriptor["sha256"],
                "Actual training observer bytes changed")
        require("bytes" not in descriptor or saved_path.stat().st_size == descriptor["bytes"], "Observer byte count changed")
        saved = torch.load(saved_path, map_location="cpu", weights_only=True)
        observations = evidence["observations"]
        if task["evaluation"]["kind"] == "gaussian_smoke" and saved[0]["step"] == 0:
            saved = saved[1:]
        indices = []
        for observation in observations:
            metrics = {k: v for k, v in observation.items() if k != "step"}
            # Vector archives retain exact scored samples and cadence; their
            # numerical curve remains in the original independent grade.
            matches = [i for i, point in enumerate(saved) if point["step"] == observation["step"] and
                       (point.get("metrics") == metrics or
                        (task["id"] == "ring16_acquisition" and set(point) == {"step", "samples"}))]
            require(matches, "A saved training view lacks its exact recorded observation")
            indices.append(matches[0])
        selection = (dict(kind="exact_scored_subset_of_retained_views", retained_records=len(saved),
                          source_record_indices=indices, added_observations=0)
                     if [point["step"] for point in saved] != [o["step"] for o in observations] else None)
        inputs = {str(saved_path): file_hash(saved_path)}
    count = len(observations)
    selected = (sorted({round(i * (count - 1) / min(8, count - 1)) for i in range(min(9, count))})
                if count > 1 else [0])
    require(receipt["source_inputs"] == inputs and receipt["observations_sha256"] == stable_hash(observations) and
            receipt["observation_count"] == count and receipt["selected_observation_indices"] == selected and
            receipt.get("saved_record_selection") == selection, "GIF differs from certified actual training observations")
    with Image.open(path) as gif:
        require(gif.format == "GIF" and gif.n_frames >= 2, "GIF is incomplete")
        for index in range(gif.n_frames):
            gif.seek(index)
            gif.load()


def audit(root=ROOT, archive=ARCHIVE):
    require(sys.executable == PYTHON and torch.get_num_threads() == 1, "Use the declared scientific Python and one CPU thread")
    root, archive = root.resolve(), archive.resolve()
    out = root / "reports/forge/r1r2-direction-floor-repair"
    paths = dict(publication=out / "publication/results.json", registration=out / "registration.json",
                 progress=archive / f"{CAMPAIGN}-progress.json", queue=archive / "queue/queue/state.json")
    hashes = {name: file_hash(path) for name, path in paths.items()}
    publication, registration, progress, queue = [read_json(paths[n]) for n in ("publication", "registration", "progress", "queue")]
    require(publication["complete"] and progress["phase"] == "complete" and publication["source_commit"] ==
            progress["source_commit"] == EXECUTION, "Publication/execution is not the completed frozen campaign")
    require(paths["queue"].resolve() == (ORIGINAL / "queue/queue/state.json").resolve(), "Original shared queue changed")
    require(publication["registration_sha256"] == progress["registration_sha256"] == hashes["registration"], "Registration changed")
    require(set(progress["requests"]) == set(registration["candidates"]) and publication["round_id"] == CAMPAIGN and
            publication["full_reservation_seconds"] == registration["reservation_seconds"] == 12600 and
            publication["paid_cap_seconds"] == registration["paid_cap_seconds"] == 8000, "Registered scope/budgets changed")
    for field in ("source_digest", "runtime_sha256", "protocol_sha256", "view_sha256", "task_contracts"):
        require(len({stable_hash(p[field]) for p in registration["candidates"].values()}) == 1,
                "A fixed source/runtime/protocol/view/task law varies across roles")
    require(len(publication["certified_results"]) == len(publication["task_cells"]) == len(publication["media"]) ==
            len(publication["costs"]) == 35 and len(registration["candidates"]) == 5, "Finite35-case coverage changed")
    require(publication["optimizer_updates_added_by_collection"] == publication["sampling_draws_added_by_collection"] == 0 and
            not publication["selection_or_defaults_changed"], "Collection side effects declared")
    old_first = read_json(root / "reports/forge/r1r2-tier1-search/publication/results.json")
    old_role = read_json(root / "reports/forge/r1r2-role-repair/publication/results.json")
    old_audit = read_json(root / "reports/forge/r1r2-role-repair/collection-audit.json")
    original_controls = {"baseline": [r for r in old_first["certified_results"] if r["role"] == "baseline"],
                         "adam": [r for r in old_first["certified_results"] if r["candidate_id"] == registration["candidates"]["adam"]["candidate"]]}
    charges = {c["attempt_id"]: c for c in queue["charges"]}
    require(len(charges) == len(queue["charges"]) == 175, "Duplicate/missing aggregate charges")
    old_ids = set(old_role["aggregate_charge_attempt_ids"])
    old_charges = {aid: charges[aid] for aid in old_ids}
    require(len(old_ids) == 154 and stable_hash(old_charges) == old_audit["inputs_sha256"]["aggregate154_charge_index"] == OLD_CHARGE_HASH,
            "Historical154 charge IDs/owners/seconds changed")
    current_ids = set(progress["requests"].values())
    new_charges = {aid: c for aid, c in charges.items() if c["owner"]["request"] in current_ids}
    require(set(new_charges) == set(charges) - old_ids and len(new_charges) == 21, "New charge attribution changed")
    require(set(publication["aggregate_charge_attempt_ids"]) == set(charges) and publication["aggregate_paid_attempts"] == 175,
            "Aggregate publication charge identity changed")
    total = sum(c["seconds"] for c in charges.values())
    new_total = sum(c["seconds"] for c in new_charges.values())
    require(math.isclose(publication["total_paid_seconds"], new_total, abs_tol=1e-7) and
            math.isclose(publication["aggregate_total_paid_seconds"], total, abs_tol=1e-7) and total <= 8000,
            "Paid totals/ceiling changed")
    source_files, counts, moments = {}, Counter(), {}
    cell_map = {(c["role"], c["task_id"]): c for c in publication["task_cells"]}
    media_map = {(m["role"], m["task_id"]): m for m in publication["media"]}
    require(len(cell_map) == len(media_map) == 35 and set(cell_map) == set(media_map), "Duplicate/missing result cells")
    require({(r["role"], r["task_id"]) for r in publication["certified_results"]} == set(cell_map), "Result grid changed")
    for cell in cell_map.values():
        require(cell["required"] == (cell["task_id"] != CLOCK), "Ordinary/diagnostic gate denominator changed")
    standings = {r["role"]: r for r in publication["recipes"]}
    require(set(standings) == set(registration["candidates"]), "Recipe summary coverage changed")
    for role, plan in registration["candidates"].items():
        own = [cell_map[(role, task)] for task in TASKS]
        expected = dict(role=role, candidate_id=plan["candidate"],
                        required_tier1=dict(Counter(c["status"] for c in own if c["required"])),
                        clock_status=cell_map[(role, CLOCK)]["status"],
                        all_seven_pass=all(c["status"] == "PASS" for c in own))
        require(standings[role] == expected, "Recipe summary differs from its stored gates")
    ordinary_baseline = {r["task_id"]: r for r in original_controls["baseline"]}
    for role, rid in progress["requests"].items():
        request = queue["submissions"][rid]["request"]
        plan = registration["candidates"][role]
        require(projection(request) == {k: v for k, v in plan.items() if k != "card"}, "Admitted scientific projection changed")
        require(plan["source_digest"] == publication["source_digest"] == registration["source_digest"], "Scientific source differs")
        require(plan["runtime_sha256"] == old_audit["runtime_sha256"], "Scientific runtime differs from controls")
        for name, digest in request["source"]["files"].items():
            require(name not in source_files or source_files[name] == digest, "Conflicting source file identities")
            source_files[name] = digest
        for task, key in plan["task_keys"].items():
            job = queue["jobs"][key]
            require(job["status"] == "terminal" and len(job["attempts"]) == 1 and job["result"], "Missing result or scientific retry")
            aid = job["result"]["attempt_id"]
            saved_request, result, local, compact = read_attempt(root, aid)
            require(result == job["result"] and projection(saved_request) == projection(request), "Certified/admitted identity differs")
            raw_row = next(r for r in result["task_results"] if r["task_id"] == task)
            row = next(r for r in publication["certified_results"] if r["role"] == role and r["task_id"] == task)
            projected = next(r for r in compact["task_results"] if r["task_id"] == task)
            require(row["candidate_id"] == plan["candidate"] and row["candidate_revision"] == plan["candidate_revision"] and
                    row["source_digest"] == plan["source_digest"] and row["attempt_id"] == aid and
                    row["qualification_input"] is False, "Published recipe/source/attempt attribution changed")
            require(all(row.get(k) == v for k, v in projected.items()) and row["independent_grading"] == stored_grade(saved_request, result),
                    "Published scientific metrics or stored original grades changed")
            require(raw_row["raw_status"] == "completed" and row["gate_status"] == cell_map[(role, task)]["status"], "Task gate/cell differs")
            require(set(publication["exact_certificate_hashes"][aid]) == {"request", "result", "evidence"},
                    "Publication lacks an original certificate byte identity")
            for name, digest in publication["exact_certificate_hashes"][aid].items():
                require(file_hash(root / f"reports/forge/attempts/{aid}/{name}.json") == digest, "Original certificate bytes changed")
            charge = charges[aid]
            require(charge["owner"] == result["cost_owner"] and
                    math.isclose(charge["seconds"], sum(r["cost"]["wall_seconds"] for r in result["task_results"]), abs_tol=1e-7),
                    "Original charge owner/wall receipt changed")
            cost = publication["costs"][aid]
            reused = role in CONTROLS
            require(cost["reused"] == reused and cost["original_charge_owner"] == charge["owner"] and
                    cost["original_paid_seconds"] == charge["seconds"] and cost["seconds"] == (0 if reused else charge["seconds"]),
                    "Reuse accounting differs from original request ownership")
            require(cost["full_allowance_seconds"] == job["definition"]["budget_seconds"], "Reported full allowance changed")
            if reused:
                old = next(r for r in original_controls[role] if r["task_id"] == task)
                require({k: v for k, v in row.items() if k != "role"} == {k: v for k, v in old.items() if k != "role"},
                        "Exact cached control changed")
            proof = None
            if task == CLOCK:
                evidence = raw_row["evidence"]
                directory = Path(evidence["artifact_root"])
                verify_artifacts(directory, evidence["artifact_manifest"])
                proof = torch.load(directory / "comparisons.pt", map_location="cpu", weights_only=True)
                require(proof["execution"] == request["tasks"][task]["execution"], "Clock own-cohort execution changed")
                branches = {"reference", "step_label", "horizon", "evaluation_cadence", "restart"}
                require(set(proof["trajectories"]) == set(proof["branch_initial"]) == branches and
                        all(len(states) == proof["execution"]["probe_steps"] for states in proof["trajectories"].values()),
                        "Clock saved branch coverage changed")
                for initial in proof["branch_initial"].values():
                    for field in ("models", "optimizers", "streams"):
                        require(state_digest(initial["trainer"][field]) == state_digest(proof["initial"]["trainer"][field]),
                                "Clock branch did not retain its identical own trained initial state")
                states = [proof["initial"], *proof["branch_initial"].values(),
                          *(s for trajectory in proof["trajectories"].values() for s in trajectory)]
                for snapshot in states:
                    counts["clock_stream_states"] += named_streams(snapshot)
                    counts["clock_saved_snapshots"] += 1
                    optimizer_roles(snapshot, task, role, clock=True)
                moments.setdefault(role, {})[task] = optimizer_roles(proof["initial"], task, role, clock=True)[0]
                counts["clock_proofs"] += 1
            else:
                saved, checkpoint_proof = checkpoint(raw_row)
                require(checkpoint_proof == row["checkpoint"], "Published checkpoint proof changed")
                comparison = raw_row["evidence"]["comparison_protocol"]
                require(row["comparison_protocol"] == comparison, "Published comparison receipt changed")
                baseline = ordinary_baseline[task]["comparison_protocol"]
                for field in ("initial_tensors_sha256", "actual_target_batches"):
                    require(comparison[field] == baseline[field], "Actual init/consumed-target comparison changed")
                counts["ordinary_stream_states"] += named_streams(saved, comparison)
                moments.setdefault(role, {})[task], n = optimizer_roles(saved, task, role)
                counts["optimizer_groups"] += n
                counts["checkpoints"] += 1
            saved_media(root, media_map[(role, task)], raw_row, request["tasks"][task], local, proof)
            counts["gifs"] += 1
            counts["grades"] += 1
    for name, digest in source_files.items():
        require(file_hash(root / name) == digest, "Executed scientific source file changed")
    require(all(file_hash(path) == hashes[name] for name, path in paths.items()), "Audit inputs changed while being read")
    return dict(schema_version=1, status="PASS", scope="independent_read_only_r1r2_direction_floor_collection_audit", qualification_input=False,
        source_commit=EXECUTION, source_digest=publication["source_digest"], runtime_sha256=old_audit["runtime_sha256"],
        inputs_sha256={**hashes, "historical154_charge_index": stable_hash(old_charges), "aggregate175_charge_index": stable_hash(charges),
                      "audit_source": file_hash(Path(__file__)),
                      "old_first_publication": file_hash(root / "reports/forge/r1r2-tier1-search/publication/results.json"),
                      "old_role_publication": file_hash(root / "reports/forge/r1r2-role-repair/publication/results.json"),
                      "old_role_audit": file_hash(root / "reports/forge/r1r2-role-repair/collection-audit.json")},
        method_helpers_sha256={name: file_hash(root / name) for name in (
            "reports/forge/regenerate_technique_inventory.py", "experiments/forge/artifacts.py",
            "experiments/forge/state.py", "experiments/forge/contracts.py")},
        checks={name: "PASS" for name in ("certified_projection_and_stored_original_independent_grades", "artifact_bytes_and_actual_training_gif_sidecars",
            "actual_D_G_E_sampled_P_Adam_Ada_families_epsilon_and_histories", "direct_pole_uses_G_moments_prior_factor_inapplicable",
            "unsampled_token_prior_factor_inapplicable", "fixed_actual_initial_tensors_and_ordered_target_batches",
            "all_consumed_named_stream_receipts_match_saved_states", "two_exact_cached_controls", "all154_old_charge_ids_owners_values_retained",
            "21_new_charge_owners_and175_unique_aggregate", "no_scientific_retries")},
        coverage=dict(roles=5, cases=35, exact_cached_control_cases=14, new_paid_attempts=21, aggregate_unique_paid_attempts=175,
                      source_files=len(source_files), **dict(counts)),
        costs=dict(prior_paid_seconds=sum(c["seconds"] for c in old_charges.values()), new_paid_seconds=new_total,
                   aggregate_paid_seconds=total, cached_new_paid_seconds=0, shared_paid_ceiling_seconds=8000),
        actual_optimizer_groups=moments, optimizer_updates_added=0, sampling_draws_added=0, live_evaluators_or_rescoring=False,
        model_construction_training_sampling_and_random_draws_forbidden=True,
        limits="Stored original independent grades only; no gates recomputed. Clock proofs retain each own cohort. Optimizer family, betas, epsilon and history fields come from actual saved groups; direct-pole counters remain labelled prior but consume G moments. Caller-global metadata is not compared across candidates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUT / "collection-audit.json")
    parser.add_argument("--write", action="store_true", help="Write only after the final publication is ready")
    args = parser.parse_args()
    torch.set_num_threads(1)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with forbid_live_execution(), torch.no_grad():
        receipt = audit(args.repository, args.artifacts)
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    require(before[0] == after[0] and before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1]) and
            before[1][2:] == after[1][2:] and torch.equal(before[2], after[2]), "Global RNG changed during audit")
    receipt["checks"]["python_numpy_torch_cpu_rng_unchanged"] = "PASS"
    if args.write:
        atomic_json(args.output.resolve(), receipt)
        print(json.dumps(dict(status=receipt["status"], coverage=receipt["coverage"], costs=receipt["costs"]), sort_keys=True))
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

"""Freeze and collect finite global R1/R2 comparisons through ordinary Forge.

The manifest supplies reviewed ideas/studies and optional finite searches.
Preparation launches no workers. Run freezes committed inputs and admits each
round once; no automatic refinement, scientific retries, or higher-tier work.
Collection validates certified saved training results and exports their GIFs
without training, resampling, rescoring, or changing the family selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.forge.configuration_search import enqueue_search, plan_search, report_search
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash, utc_now
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue, drain

TASKS = {"gaussian1d_smoke", "two_pole", "unused_token_hold", "ae_gan_hold",
         "ring16_acquisition", "five_word_joint_smoke", "clockfree_audit_measurement_v1"}
REQUIRED = TASKS - {"clockfree_audit_measurement_v1"}
ALLOWANCE = 2520
ACTIVE = {"queued", "running", "paused"}
ARCHIVE_CAP = 8 * 1024**3
FREE_MIN = 5 * 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def emit(event, **values):
    print(json.dumps(dict(event=event, time=utc_now(), **values), sort_keys=True, allow_nan=False), flush=True)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def committed(root, source, paths):
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0"))
    files = set(source["files"]) | set(paths)
    require(files <= tracked, "Commit every frozen scientific source, declaration and registration before admission")
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *sorted(files)], cwd=root, check=True)


def validate_manifest(value):
    require(value["schema_version"] == 1 and value["round"] in (1, 2), "Expected an explicit bounded round")
    maximum = 8 if value["round"] == 1 else 4
    paid = 3600 if value["round"] == 1 else 5400
    require(0 < value["max_candidates"] <= maximum, "Round exceeds the reviewed recipe ceiling")
    require(0 < value["paid_cap_seconds"] <= paid, "Round exceeds the aggregate paid ceiling")
    require(value["campaign"]["candidate_budget_seconds"] == ALLOWANCE, "Retain each full Tier1 allowance")
    require(0 < value["campaign"]["budget_seconds"] <= value["max_candidates"] * ALLOWANCE,
            "Declare the full finite round reservation before execution")
    require(value["baseline_role"] and value.get("candidates"), "An explicit whole-recipe control is required")
    if value["round"] == 2:
        require(value.get("previous_registration") and value.get("previous_results"),
                "A second round needs the concluded first-round evidence and a separate reviewed hypothesis")
    return value


def task_contract_hash(task):
    # Planning annotates cards with candidate-specific effective recipe values.
    # Those annotations describe the trainer delta, not a changed task law.
    return stable_hash({k: v for k, v in task.items() if k not in {"field_ownership", "preflight_blockers"}})


def request_projection(request):
    assignments = [a for a in request["view"]["assignments"] if a["qualification_tier"] == 1]
    require(request["view"]["id"] == "discriminator_stability" and request["through_tier"] == 1,
            "Only the unchanged ordinary view through Tier1 may execute")
    require({a["task"] for a in assignments} == TASKS
            and {a["task"] for a in assignments if a["importance"] == "required"} == REQUIRED,
            "Retain all six required Tier1 tasks and the separate clock measurement")
    require(request["protocol"]["seed"] == 0 and request["execution_policy"]["mode"] == "complete_current_tier",
            "Seed0 and complete-current-tier are mandatory")
    jobs = [j for j in request["jobs"] if Queue._authorized(request, j)]
    require(sum(j["budget_seconds"] for j in jobs) == ALLOWANCE, "Full task budgets changed")
    require(not request["preflight_blockers"], str(request["preflight_blockers"]))
    require(all(not request["tasks"][t].get("preflight_blockers") for t in TASKS), "An active Tier1 host is blocked")
    review = request.get("study_review", request.get("decision_review", {}))
    require(review.get("status") == "READY" or (not review and "configuration_id" in request["candidate"]),
            f"Study is not READY: {review}")
    return dict(candidate=request["candidate"]["id"], candidate_revision=request["candidate_revision"],
        source_digest=request["source"]["digest"], protocol_sha256=stable_hash(request["protocol"]),
        runtime_sha256=stable_hash(request["runtime"]), execution_policy=request["execution_policy"],
        view_sha256=stable_hash(request["view"]), task_contracts={t: task_contract_hash(request["tasks"][t]) for t in sorted(TASKS)},
        jobs_sha256=stable_hash(jobs), task_keys={t: j["compatibility_key"] for j in jobs
            for t in j.get("task_ids", [j["task_id"]])}, full_allowance_seconds=ALLOWANCE)


def requests(root, queue_root, manifest):
    resolved, searches = {}, {}
    for card in manifest["candidates"]:
        require(card["role"] not in resolved, "Duplicate comparison role")
        request = resolve_idea(root, card["candidate"], study=card["study"], queue_root=queue_root)
        resolved[card["role"]] = dict(card=card, request=request, projection=request_projection(request))
    for search in manifest.get("searches", []):
        sid = search["id"]
        plan = plan_search(root, queue_root, sid)
        require(plan["tuning_through_tier"] == 1 and all(t["submission_status"] == "READY" for t in plan["trials"]),
                "Every finite trial must preflight")
        require(plan["campaign"]["candidate_budget_seconds"] == ALLOWANCE, "Search changes full candidate budget")
        searches[sid] = plan
        for trial in plan["trials"]:
            role = f'{sid}:{trial["configuration_id"]}'
            request = resolve_idea(root, trial["candidate_id"], view_id=plan["view"], through_tier=1,
                execution_backend=plan["execution_backend"], cuda_model=plan["spec"].get("cuda_model"), queue_root=queue_root)
            resolved[role] = dict(card=dict(role=role, candidate=trial["candidate_id"], search=sid),
                                 request=request, projection=request_projection(request))
    require(len(resolved) <= manifest["max_candidates"], "Too many global recipes")
    require(len(resolved) * ALLOWANCE == manifest["campaign"]["budget_seconds"], "Aggregate full reservation differs from the finite recipe list")
    require(manifest["baseline_role"] in resolved, "Missing declared matched baseline")
    require(len({r["projection"]["candidate_revision"] for r in resolved.values()}) == len(resolved),
            "Duplicate recipes cannot justify repeated training")
    # Source, runtime and all scientific task laws must match globally. Only
    # trainer recipe and its technical job identity may differ across roles.
    for field in ("source_digest", "protocol_sha256", "runtime_sha256", "view_sha256", "task_contracts"):
        require(len({stable_hash(r["projection"][field]) for r in resolved.values()}) == 1,
                f"Comparison changes a fixed condition: {field}")
    return resolved, searches


def declaration_paths(root, manifest, resolved):
    paths = {"reports/forge/r1r2-tier1-search/workflow.py"}
    paths.update(manifest.get("supporting_inputs", []))
    for t in TASKS:
        paths.add(f"configs/forge/tasks/{t}.json")
    for item in resolved.values():
        card = item["card"]
        category = "configurations" if "search" in card else "ideas"
        paths.add(f'configs/forge/{category}/{card["candidate"]}.json')
        if "study" in card:
            paths.add(f'configs/forge/studies/{card["study"]}.json')
            paths.update(e["path"] for e in item["request"]["study"].get("prior_evidence", []))
    for search in manifest.get("searches", []):
        paths.add(f'configs/forge/searches/{search["id"]}.json')
        spec = read_json(root / f'configs/forge/searches/{search["id"]}.json')
        base_paths = [root / f'configs/forge/{kind}/{spec["base_candidate"]}.json' for kind in ("ideas", "configurations")]
        existing = [path for path in base_paths if path.exists()]
        require(len(existing) == 1, "Finite search needs an unambiguous committed structural base")
        paths.add(str(existing[0].relative_to(root)))
    paths.add("configs/forge/views/discriminator_stability.json")
    paths.add("configs/forge/protocols/screening.json")
    return paths


def prepare(args):
    manifest = validate_manifest(read_json(args.manifest))
    require(not (args.artifacts / f'{manifest["id"]}-progress.json').exists(), "Admitted rounds are immutable")
    resolved, searches = requests(args.repository, args.artifacts / "queue", manifest)
    paths = declaration_paths(args.repository, manifest, resolved) | {str(args.manifest.relative_to(args.repository))}
    registration = dict(schema_version=1, manifest_sha256=file_hash(args.manifest), manifest=manifest,
        qualification_input=False, archive=str(args.artifacts), source_digest=next(iter(resolved.values()))["projection"]["source_digest"],
        candidates={role: dict(card=r["card"], **r["projection"]) for role, r in resolved.items()},
        declarations={p: file_hash(args.repository / p) for p in sorted(paths)},
        search_spec_hashes={sid: plan["spec_hash"] for sid, plan in searches.items()},
        required_tasks=sorted(REQUIRED), clock_diagnostic="clockfree_audit_measurement_v1",
        reservation_seconds=len(resolved) * ALLOWANCE, paid_cap_seconds=manifest["paid_cap_seconds"],
        stop="One explicit finite round; complete runnable Tier1 peers within full reservations and paid/storage caps. No automatic next round, retry, Tier2 or selection/default change.")
    if args.registration.exists():
        require(read_json(args.registration) == registration, "Prepared registration is immutable; use a new round identity")
    else:
        atomic_json(args.registration, registration)
    emit("prepared", recipes=len(resolved), source_digest=registration["source_digest"],
         reservation_seconds=registration["reservation_seconds"], paid_cap_seconds=registration["paid_cap_seconds"])


class BoundedQueue(Queue):
    """Keep full scientific reservations and add an aggregate paid-work ceiling.

    The queue can reconsider a temporarily denied task after running workers
    finish. When no full next allowance fits, the round stops with cells missing;
    this never reduces a task's timeout, updates, or evaluation cadence.
    """
    def __init__(self, *args, paid_cap_seconds, **kwargs):
        super().__init__(*args, **kwargs)
        self.paid_cap_seconds = paid_cap_seconds

    def _available_budget(self, state, request, seconds, **kwargs):
        paid = sum(c["seconds"] for c in state.get("charges", []))
        reserved = sum(j.get("reserved_seconds", 0) for j in state["jobs"].values())
        if paid + reserved + seconds > self.paid_cap_seconds:
            return False, "aggregate paid ceiling cannot reserve the full next task"
        return super()._available_budget(state, request, seconds, **kwargs)


def storage(archive):
    archive.mkdir(parents=True, exist_ok=True)
    used = int(subprocess.check_output(["du", "-sb", str(archive)], text=True).split()[0])
    free = shutil.disk_usage(archive).free
    require(used < ARCHIVE_CAP and free >= FREE_MIN, f"Archive cap/headroom exhausted: used={used}, free={free}")
    return dict(bytes=used, free_bytes=free, cap_bytes=ARCHIVE_CAP, minimum_free_bytes=FREE_MIN)


def run(args):
    root, archive = args.repository, args.artifacts
    require(not archive.is_relative_to(root), "Bulk logs and tensors must remain outside Git")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(head == args.source_commit, "Run only the explicit reviewed commit")
    registration = read_json(args.registration)
    require(file_hash(args.manifest) == registration["manifest_sha256"], "Manifest changed")
    manifest = validate_manifest(registration["manifest"])
    resolved, _ = requests(root, archive / "queue", manifest)
    for role, item in resolved.items():
        require(dict(card=item["card"], **item["projection"]) == registration["candidates"][role], "Prepared bindings changed")
    for path, digest in registration["declarations"].items():
        require(file_hash(root / path) == digest, f"Reviewed declaration changed: {path}")
    committed(root, next(iter(resolved.values()))["request"]["source"],
              set(registration["declarations"]) | {str(args.registration.relative_to(root))})
    queue = BoundedQueue(archive / "queue", report_root=root / "reports/forge", on_completion=None,
                         paid_cap_seconds=registration["paid_cap_seconds"])
    progress_path = archive / f'{manifest["id"]}-progress.json'
    if args.submit:
        require(not progress_path.exists(), "No scientific retry or duplicate round admission")
        storage(archive)
        previous = list(archive.glob("*-progress.json"))
        require(len(previous) < 2, "At most two explicitly reviewed aggregate rounds")
        if manifest["round"] == 2:
            prior = read_json(root / manifest["previous_registration"])
            require(prior["manifest"]["round"] == 1 and prior["archive"] == str(archive), "Second round needs the same first-round queue")
            require(len(prior["candidates"]) + len(resolved) <= 12
                    and prior["reservation_seconds"] + registration["reservation_seconds"] <= 30240,
                    "Aggregate recipes/reservations exceed the reviewed bound")
            require(read_json(root / manifest["previous_results"])["complete"], "Finish first-round evidence before a separate next-round declaration")
        progress = dict(schema_version=1, phase="admitting", requests={},
                        source_commit=head, registration_sha256=file_hash(args.registration))
        atomic_json(progress_path, progress)
        for role, item in resolved.items():
            card = item["card"]
            if "study" not in card:
                continue
            frozen = resolve_idea(root, card["candidate"], study=card["study"], queue_root=queue.root, freeze_source=True)
            require(request_projection(frozen) == item["projection"], "Source changed while freezing")
            entry = queue.submit(frozen, frozen["study"]["campaign"])
            progress["requests"][role] = entry["request"]["request_id"]
            atomic_json(progress_path, progress)
        for sid in registration["search_spec_hashes"]:
            report = enqueue_search(root, queue.root, sid, queue=queue)
            require(not report["blocked_count"], "Search admission failed")
            for trial in report["trials"]:
                role = f'{sid}:{trial["configuration_id"]}'
                progress["requests"][role] = trial["request_id"]
            atomic_json(progress_path, progress)
        require(set(progress["requests"]) == set(resolved), "Partial admission needs administrative recovery; never resubmit the round")
        progress["phase"] = "admitted"
        atomic_json(progress_path, progress)
        emit("admitted", source_commit=head, requests=progress["requests"], logs=str(queue.root / "events.jsonl"))
    require(progress_path.exists(), "Drain requires prior explicit admission")
    progress = read_json(progress_path)
    require(progress["registration_sha256"] == file_hash(args.registration) and progress["source_commit"] == head,
            "Admitted registration/source changed")
    if args.drain:
        stop, errors = threading.Event(), []
        def guard():
            while not stop.wait(20):
                try:
                    storage(archive)
                except ValueError as error:
                    errors.append(str(error))
                    for campaign in queue.inspect()["campaigns"]:
                        queue.pause(campaign, True)
                    emit("storage_stop", reason=str(error))
                    return
        watcher = threading.Thread(target=guard, daemon=True)
        watcher.start()
        try:
            drain(queue, args.gpus.split(","), workers_per_gpu=1, allow_sharing=False, watch=False, poll_seconds=1)
        finally:
            stop.set()
            watcher.join(timeout=2)
        state = queue.inspect()
        progress["phase"] = "paused" if errors or any(state["submissions"][rid]["status"] in ACTIVE
            for rid in progress["requests"].values()) else "complete"
        progress["storage_errors"] = errors
        atomic_json(progress_path, progress)
    emit("execution_state", phase=progress["phase"])


def comparison_audit(rows):
    """Require actual initialized tensors and consumed target draws to match."""
    grouped, comparisons = {}, []
    for row in rows:
        grouped.setdefault(row["task_id"], []).append(row)
    for task, group in sorted(grouped.items()):
        if task == "clockfree_audit_measurement_v1":
            comparisons.append(dict(task_id=task, status="SEPARATE_CLOCK_PROOF",
                reason="Certified comparisons.pt retains each own probe trajectory and checkpointed streams"))
            continue
        absent = [r["role"] for r in group if not r.get("comparison_protocol")]
        if absent:
            comparisons.append(dict(task_id=task, status="UNAVAILABLE", missing_roles=absent,
                reason="Actual tensor/batch recording absent; shared task/RNG contracts are supplementary"))
            continue
        for field in ("initial_tensors_sha256", "actual_target_batches"):
            require(len({stable_hash(r["comparison_protocol"][field]) for r in group}) == 1,
                    f"Actual comparison condition changed for {task}: {field}")
        require(all(r["comparison_protocol"]["rng_checkpoints"]["all_named_streams_checkpointed"] for r in group),
                f"{task}: a consumed stream was not checkpointed")
        comparisons.append(dict(task_id=task, status="PASS", roles=[r["role"] for r in group],
            initial_tensors_sha256=group[0]["comparison_protocol"]["initial_tensors_sha256"],
            actual_target_batches=group[0]["comparison_protocol"]["actual_target_batches"],
            all_named_streams_checkpointed=True))
    return comparisons


def clock_retained_media(entry, output, root):
    """Validate the clock GIF's exact saved parameter trajectories on reuse."""
    from PIL import Image
    import torch
    from experiments.forge import tier1_media
    from experiments.forge.artifacts import verify_artifacts
    sidecar = output.with_suffix(".json")
    if not output.exists() and not sidecar.exists():
        return None
    require(output.exists() and sidecar.exists(), "Incomplete retained clock export")
    evidence = entry["row"]["evidence"]
    directory = Path(evidence["artifact_root"])
    verify_artifacts(directory, evidence["artifact_manifest"])
    source = root / "experiments/forge/tier1_media.py"
    require(file_hash(source) == entry["request"]["source"]["files"]["experiments/forge/tier1_media.py"],
            "Clock visualization differs from executed source")
    path = directory / "comparisons.pt"
    proof = torch.load(path, map_location="cpu", weights_only=True)
    initial = proof["initial"]["trainer"]["models"]["G"]
    parameter = next(k for k, v in initial.items() if v.is_floating_point() and v.numel())
    observations = [dict(step=i + 1, **{name: float((trajectory[i]["trainer"]["models"]["G"][parameter]
        - initial[parameter]).abs().mean()) for name, trajectory in proof["trajectories"].items()})
        for i in range(entry["task"]["execution"]["probe_steps"])]
    expected = dict(schema_version=1, task_id=entry["task"]["id"], recorded_grade=entry["row"]["gate_status"],
        kind="actual_training_saved_observations_gif", observation_count=len(observations),
        selected_observation_indices=tier1_media._indices(len(observations)), source_inputs={str(path): file_hash(path)},
        observations_sha256=stable_hash(observations), gif_sha256=file_hash(output), optimizer_updates_added=0,
        sampling_draws_added=0, qualification_input=False, renderer_sha256=file_hash(source))
    receipt = read_json(sidecar)
    require(all(receipt.get(k) == v for k, v in expected.items()), "Clock export differs from its certified saved inputs")
    with Image.open(output) as gif:
        require(gif.format == "GIF" and gif.n_frames >= 2, "Incomplete clock training GIF")
    return receipt


def collect(args):
    registration = read_json(args.registration)
    manifest = registration["manifest"]
    progress = read_json(args.artifacts / f'{manifest["id"]}-progress.json')
    require(progress["phase"] == "complete", "Collect a terminal finite round")
    state = read_json(args.artifacts / "queue/queue/state.json")
    helper = module(args.repository / "reports/forge/recipe-prior-refactor/workflow.py", "r1r2_saved_helper")
    publisher = helper.load_publisher(args.repository)
    renderer = publisher._saved_renderer(args.repository) if args.media else None
    rows, costs, exports, receipts = [], {}, [], {}
    for role, rid in progress["requests"].items():
        sub = state["submissions"][rid]
        require(sub["status"] not in ACTIVE, f"{role}: active work")
        request = sub["request"]
        plan = registration["candidates"][role]
        require(request_projection(request) == {k: v for k, v in plan.items() if k != "card"}, "Admitted bindings changed")
        for definition in request["jobs"]:
            if not Queue._authorized(request, definition):
                continue
            job = state["jobs"][definition["compatibility_key"]]
            require(len(job["attempts"]) <= 1, "Scientific retries are forbidden")
            if job.get("result") is None:
                continue
            aid = job["result"]["attempt_id"]
            attempt = publisher.certified_attempt(args.repository, aid)
            require(attempt["result"] == job["result"], "Queue/certificate result mismatch")
            require(attempt["request"]["source"]["digest"] == plan["source_digest"]
                    and attempt["request"]["candidate_revision"] == plan["candidate_revision"], "Certified source/recipe mismatch")
            grade = helper.saved_grading(attempt)
            charges = [c for c in state.get("charges", []) if c["attempt_id"] == aid]
            # Compatible reuse can originate before this local queue. It remains
            # certified evidence but is not charged again to the present round.
            require(len(charges) <= 1, "Duplicate paid charge")
            costs[aid] = dict(seconds=charges[0]["seconds"] if charges else 0,
                             reused=not charges, full_allowance_seconds=definition["budget_seconds"])
            if charges:
                actual = sum(r.get("cost", {}).get("wall_seconds", 0) for r in attempt["result"]["task_results"])
                require(math.isclose(actual, charges[0]["seconds"], abs_tol=1e-7), "Paid accounting mismatch")
            receipts[aid] = {n: file_hash(args.repository / f"reports/forge/attempts/{aid}/{n}.json")
                            for n in ("request", "result", "evidence")}
            for row in attempt["result"]["task_results"]:
                name = row["task_id"]
                require(name in TASKS and row["compatibility_key"] == plan["task_keys"][name], "Task identity changed")
                saved, proof = publisher.checkpoint(row)
                compact = deepcopy(next(r for r in attempt["compact"]["task_results"] if r["task_id"] == name))
                compact.update(role=role, candidate_id=plan["candidate"], candidate_revision=plan["candidate_revision"],
                    source_digest=plan["source_digest"], attempt_id=aid, independent_grading=grade,
                    checkpoint=proof, comparison_protocol=row.get("evidence", {}).get("comparison_protocol"), qualification_input=False)
                rows.append(compact)
                if args.media and (row.get("evidence", {}).get("observations") or name == "clockfree_audit_measurement_v1"):
                    output = args.output / "media" / f'{role.replace(":", "-")}-{name}.gif'
                    entry = dict(task=request["tasks"][name], row=row, request=attempt["request"], attempt=attempt)
                    receipt = (clock_retained_media(entry, output, args.repository) if name == "clockfree_audit_measurement_v1"
                               else helper.retained_media(entry, output, publisher, renderer))
                    if receipt is None:
                        with renderer.forbid_live_execution():
                            receipt = publisher.render_saved(entry, output, renderer)
                    exports.append(dict(role=role, task_id=name, path=str(output.relative_to(args.repository)),
                        sha256=file_hash(output), receipt_sha256=file_hash(output.with_suffix(".json")), receipt=receipt))
    audits = comparison_audit(rows)
    cells, standings = [], []
    for role, plan in registration["candidates"].items():
        for task in sorted(TASKS):
            measured = [r for r in rows if r["role"] == role and r["task_id"] == task]
            require(len(measured) <= 1, "Repeated task result")
            cells.append(dict(role=role, task_id=task, status=measured[0]["gate_status"] if measured else "UNMEASURED",
                required=task in REQUIRED, reason=state["submissions"][progress["requests"][role]].get("reason") if not measured else None))
        own = [c for c in cells if c["role"] == role]
        counts = Counter(c["status"] for c in own if c["required"])
        standings.append(dict(role=role, candidate_id=plan["candidate"], required_tier1=dict(counts),
            clock_status=next(c["status"] for c in own if not c["required"]), all_seven_pass=all(c["status"] == "PASS" for c in own)))
    complete = all(c["status"] in {"PASS", "FAIL"} for c in cells)
    result = dict(schema_version=1, scope="bounded_r1r2_global_tier1_search", qualification_input=False,
        round_id=manifest["id"], complete=complete, source_commit=progress["source_commit"],
        source_digest=registration["source_digest"], registration_sha256=file_hash(args.registration),
        archive=str(args.artifacts), task_cells=cells, recipes=standings, certified_results=rows,
        exact_certificate_hashes=receipts, comparison_audit=audits, media=exports,
        costs=costs, total_paid_seconds=sum(c["seconds"] for c in costs.values()),
        full_reservation_seconds=registration["reservation_seconds"], paid_cap_seconds=registration["paid_cap_seconds"],
        optimizer_updates_added_by_collection=0, sampling_draws_added_by_collection=0,
        selection_or_defaults_changed=False)
    atomic_json(args.output / "results.json", result)
    for sid in registration["search_spec_hashes"]:
        report_search(args.repository, args.artifacts / "queue", sid)
    emit("collected", complete=complete, recipes=len(standings), attempts=len(costs),
         paid_seconds=result["total_paid_seconds"], gifs=len(exports), comparison_audit={a["task_id"]: a["status"] for a in audits})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "collect"))
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=OUT / "phase1.json")
    parser.add_argument("--registration", type=Path, default=OUT / "registration.json")
    parser.add_argument("--artifacts", type=Path, default=Path("/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012"))
    parser.add_argument("--output", type=Path, default=OUT / "publication")
    parser.add_argument("--source-commit")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--drain", action="store_true")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--media", action="store_true")
    args = parser.parse_args()
    for name in ("repository", "manifest", "registration", "artifacts", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.action == "run":
        require(args.source_commit and (args.submit or args.drain), "Run needs a frozen commit and explicit submit/drain")
    globals()[args.action](args)


if __name__ == "__main__":
    main()

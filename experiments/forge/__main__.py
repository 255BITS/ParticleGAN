"""ParticleGAN Forge. Read EXPERIMENTATION.md before running new ideas."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .contracts import atomic_json, read_json


def emit(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def queue_location(root: Path, override=None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    if os.environ.get("PARTICLEGAN_FORGE_QUEUE"):
        return Path(os.environ["PARTICLEGAN_FORGE_QUEUE"]).expanduser().resolve()
    try:
        common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        repository = (root / common).resolve().parent
    except (subprocess.CalledProcessError, FileNotFoundError):
        repository = root
    return repository / "runs/forge"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2], help="Forge checkout containing declarations and durable reports")
    p.add_argument("--queue-root", type=Path, help="shared repo-local operational queue; default common Git repository runs/forge")
    commands = p.add_subparsers(dest="command", required=True)
    new = commands.add_parser("new", help="scaffold one small idea declaration")
    new.add_argument("--id", required=True)
    new.add_argument("--parent", default="k3p")
    new.add_argument("--goal", default="discriminator_stability")
    new.add_argument("--hypothesis")
    for name in ("plan", "enqueue", "run"):
        c = commands.add_parser(name, help={"plan": "show exact tasks/reuse/cost without writing or training", "enqueue": "freeze and submit; do not launch", "run": "enqueue and drain through the declared tier cap"}[name])
        c.add_argument("candidate")
        c.add_argument("--view")
        c.add_argument("--through-tier", type=int, choices=(1, 2, 3), default=1)
        c.add_argument("--device", choices=("cpu", "cuda"), default=None, help="compute cohort (default CUDA; run infers it from --gpus)")
        c.add_argument("--cuda-model", help="required CUDA model on heterogeneous machines")
        if name == "plan":
            c.add_argument("--all-tiers", action="store_true", help="include full details for tasks outside the requested tier cap")
        if name != "plan":
            c.add_argument("--campaign", type=Path, default=Path("configs/forge/campaigns/smoke.json"))
        if name == "run":
            c.add_argument("--gpus", default="cpu", help="physical GPU indices, or cpu")
    d = commands.add_parser("drain", help="execute eligible work with a single queue coordinator")
    d.add_argument("--gpus", required=True, help="comma separated physical GPU indices, or cpu")
    d.add_argument("--workers-per-gpu", type=int, default=1)
    d.add_argument("--allow-sharing", action="store_true")
    d.add_argument("--watch", action="store_true")
    d.add_argument("--campaign")
    d.add_argument("--goal")
    commands.add_parser("queue", help="show active/blocked requests, costs and log paths")
    logs = commands.add_parser("logs", help="tail the collector's timestamped event stream")
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--lines", type=int, default=40)
    for name in ("candidate", "task", "worker", "gpu", "campaign"):
        logs.add_argument(f"--{name}")
    b = commands.add_parser("board", help="regrade a goal view over compatible evidence; no launches")
    b.add_argument("--goal", default="discriminator_stability")
    b.add_argument("--json", action="store_true", help="emit full machine-readable board instead of a compact table")
    b.add_argument("--scope", choices=("all", "current", "pinned", "historical", "calibration_diagnostic"), default="all")
    b.add_argument("--candidate", help="filter candidate names by substring")
    b.add_argument("--tier", type=int, choices=(0, 1, 2, 3), help="minimum qualified tier")
    b.add_argument("--outcome", help="filter status or presence of a task gate status")
    r = commands.add_parser("recall", help="find successes, failures and unknowns before a new idea")
    r.add_argument("--query", default="")
    r.add_argument("--goal")
    r.add_argument("--limit", type=int, default=8, help="maximum concise prior-art matches (default 8)")
    commands.add_parser("compile", help="rebuild deterministic memory and leaderboards without training")
    h = commands.add_parser("history", help="classify and import pinned history without training")
    h.add_argument("--check", action="store_true", help="check inventory coverage instead of importing")
    commands.add_parser("validate", help="validate all task/view definitions without training")
    calibration = commands.add_parser("calibrate", help="replay saved calibration evidence; no training")
    calibration.add_argument("--profile", default="initial")
    lane = commands.add_parser("calibration-lane", help="register selected bounded diagnostics; never qualify a candidate")
    lane_stages = lane.add_subparsers(dest="stage", required=True)
    lane_register = lane_stages.add_parser("register")
    lane_register.add_argument("--contract", type=Path, required=True)
    for stage in ("plan", "enqueue"):
        lane_stages.add_parser(stage).add_argument("registration")
    promotion = commands.add_parser("promotion", help="register and run one frozen finished-candidate robustness stage")
    stages = promotion.add_subparsers(dest="stage", required=True)
    registration = stages.add_parser("register")
    registration.add_argument("candidate")
    registration.add_argument("--contract", type=Path, required=True)
    for stage in ("plan", "enqueue", "report"):
        stage_parser = stages.add_parser(stage)
        stage_parser.add_argument("registration")
    retier = commands.add_parser("retier", help="change only view policy, validate and recompute; never enqueue")
    retier.add_argument("view")
    retier.add_argument("--task", required=True)
    retier.add_argument("--tier", type=int, choices=(1, 2, 3), required=True)
    retier.add_argument("--reason", required=True)
    out = commands.add_parser("readout", help="conclude an idea with comparison and recommendation, then compile memory")
    out.add_argument("candidate")
    out.add_argument("--conclusion", required=True)
    out.add_argument("--comparison", required=True)
    out.add_argument("--next-action", required=True)
    cancel = commands.add_parser("cancel", help="detach a request and terminate unneeded child process groups")
    cancel.add_argument("request_id")
    retry = commands.add_parser("retry", help="retry a repaired execution failure, never a scientific failure")
    retry.add_argument("compatibility_key")
    retry.add_argument("--reason", required=True)
    for name in ("pause", "resume"):
        c = commands.add_parser(name, help=f"{name} campaign claims")
        c.add_argument("campaign")
    return p


def follow_logs(path: Path, args):
    offset = 0
    first = True
    while True:
        if path.exists():
            with path.open() as stream:
                stream.seek(offset)
                lines = []
                while True:
                    previous = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith("\n"):
                        stream.seek(previous)
                        break
                    lines.append(line)
                offset = stream.tell()
            if first:
                lines = lines[-args.lines:]
                first = False
            for line in lines:
                row = json.loads(line)
                if all(getattr(args, field) is None or str(row.get(field)) == getattr(args, field)
                       for field in ("candidate", "task", "worker", "gpu", "campaign")):
                    print(line, end="", flush=True)
        if not args.follow:
            break
        time.sleep(.2)


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.root.resolve()
    queue_root = queue_location(root, args.queue_root)
    from .queue import Queue, drain
    def publish():
        from .knowledge import compile_memory
        compile_memory(root)
    queue = Queue(queue_root, report_root=root / "reports/forge", on_completion=publish)
    command = args.command
    if command == "new":
        from .planning import new_idea
        path = new_idea(root, args.id, args.parent, goal=args.goal, hypothesis=args.hypothesis)
        emit({"declaration": str(path), "next": "edit changed_factors and recipe/API delta, then plan", "guide": str(root / "EXPERIMENTATION.md")})
    elif command in {"plan", "enqueue", "run"}:
        from .planning import plan_summary, resolve_idea
        request = resolve_idea(root, args.candidate, view_id=args.view, through_tier=args.through_tier,
                               queue_root=queue_root, freeze_source=command != "plan",
                               execution_backend=args.device or ("cpu" if command == "run" and args.gpus == "cpu" else "cuda"),
                               cuda_model=args.cuda_model)
        if command == "plan":
            summary = plan_summary(request, queue.inspect())
            if not args.all_tiers:
                summary["deferred_tasks"] = [t["task"] for t in summary["tasks"] if not t["permitted_by_tier_cap"]]
                summary["tasks"] = [t for t in summary["tasks"] if t["permitted_by_tier_cap"]]
            emit(summary)
        else:
            campaign = read_json(root / args.campaign)
            entry = queue.submit(request, campaign)
            emit({"request_id": entry["request"]["request_id"], "status": entry["status"],
                  "queue_root": str(queue_root), "logs": str(queue_root / "events.jsonl"),
                  "request": str(queue_root / "queue/requests" / f"{entry['request']['request_id']}.json")})
            if command == "run":
                drain(queue, args.gpus.split(","), campaign=campaign["id"])
                emit(queue.inspect()["submissions"][entry["request"]["request_id"]]["status"])
    elif command == "drain":
        drain(queue, args.gpus.split(","), workers_per_gpu=args.workers_per_gpu,
              allow_sharing=args.allow_sharing, watch=args.watch, campaign=args.campaign, goal=args.goal)
        emit({"status": "drained", "logs": str(queue_root / "events.jsonl")})
    elif command == "queue":
        state = queue.inspect()
        emit({"queue_root": str(queue_root), "campaigns": state["campaigns"],
              "requests": [{"id": k, "candidate": v["request"]["candidate"]["id"],
                            "status": v["status"], "reason": v["reason"], "lifecycle": v["lifecycle"]}
                           for k, v in state["submissions"].items()], "logs": str(queue_root / "events.jsonl")})
    elif command == "logs":
        follow_logs(queue_root / "events.jsonl", args)
    elif command in {"compile", "recall", "board", "readout"}:
        from . import knowledge
        if command == "compile":
            emit(knowledge.compile_memory(root))
        elif command == "recall":
            if args.limit < 1:
                raise ValueError("recall --limit must be positive")
            matches = knowledge.recall(root, query=args.query, goal=args.goal)
            emit({"matches_total": len(matches), "shown": min(len(matches), args.limit), "matches": matches[:args.limit],
                  "memory": str(root / "reports/forge/EXPERIMENT_MEMORY.md")})
        elif command == "board":
            result = knowledge.board(root, args.goal)
            rows = [r for r in result["rows"] if (args.scope == "all" or r.get("evidence_scope") == args.scope)
                    and (not args.candidate or args.candidate in r.get("candidate_id", ""))
                    and (args.tier is None or (r.get("qualified_tier") or 0) >= args.tier)
                    and (not args.outcome or r.get("status") == args.outcome or r.get("counts", {}).get(args.outcome, 0))]
            if args.json:
                emit({**result, "rows": rows})
            else:
                print("| Candidate | Revision | Scope | Compute | Tier | PASS / total | FAIL | BLOCKED | Wall seconds | Status |")
                print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
                for row in rows:
                    counts = row.get("counts", {})
                    seconds = row.get("cost", {}).get("wall_seconds")
                    seconds = f"{seconds:.3f}" if isinstance(seconds, (int, float)) else "unknown"
                    revision = str(row.get("candidate_revision") or "unknown")[:10]
                    name = str(row.get("candidate_id", "unknown")).replace("|", "\\|")
                    runtime = row.get("runtime_cohort", {})
                    backend = runtime.get("execution_backend", "unrecorded")
                    profile = runtime.get("compute_profiles", {}).get(backend, {})
                    compute = f"{backend}: {profile['model']}" if profile.get("model") else backend
                    compute = compute.replace("|", "\\|")
                    print(f"| {name} | {revision} | {row.get('evidence_scope')} | {compute} | {row.get('qualified_tier', '—')} | "
                          f"{counts.get('PASS', 0)}/{sum(counts.values())} | {counts.get('FAIL', 0)} | {counts.get('BLOCKED', 0)} | {seconds} | {row['status']} |")
                print(f"\nFull evidence: reports/forge/leaderboards/{args.goal}.json; use --json for raw metrics/cohorts.")
        else:
            emit(knowledge.readout(root, args.candidate, args.conclusion, args.comparison, args.next_action))
    elif command == "history":
        from .history import import_history, inventory
        result = inventory(root) if args.check else import_history(root)
        emit({"counts": result["counts"], "coverage": result["coverage"], "catalog": "configs/forge/catalog.json"} if args.check else result)
        if not result["coverage"]["valid"]:
            return 1
    elif command == "validate":
        from .views import load_tasks, load_view
        tasks = load_tasks(root)
        views = [load_view(root, p.stem)["id"] for p in sorted((root / "configs/forge/views").glob("*.json"))]
        emit({"tasks": len(tasks), "views": views, "training_launched": False})
    elif command == "calibrate":
        from .calibration import calibrate
        emit(calibrate(root, profile=args.profile))
    elif command == "calibration-lane":
        from .calibration_lane import register, plan_calibration
        if args.stage == "register":
            emit(register(root, root / args.contract))
        else:
            requests = plan_calibration(root, args.registration, queue_root, freeze_source=args.stage == "enqueue")
            if args.stage == "plan":
                from .planning import plan_summary
                emit([plan_summary(request, queue.inspect()) for request in requests])
            else:
                emit([{"request_id": queue.submit(request, request["calibration_campaign"])["request"]["request_id"],
                       "calibration_lane": request["calibration_lane"]} for request in requests])
    elif command == "promotion":
        from .promotion import register, plan_promotion, summarize_promotion
        if args.stage == "register":
            emit(register(root, args.candidate, root / args.contract))
        elif args.stage == "report":
            emit(summarize_promotion(root, args.registration))
        else:
            requests = plan_promotion(root, args.registration, queue_root, freeze_source=args.stage == "enqueue")
            if args.stage == "plan":
                from .planning import plan_summary
                emit([plan_summary(request, queue.inspect()) for request in requests])
            else:
                emit([{"request_id": queue.submit(request, request["promotion_campaign"])["request"]["request_id"],
                       "promotion": request["promotion"]} for request in requests])
    elif command == "retier":
        from .views import load_tasks, load_view, validate_view
        view = load_view(root, args.view)
        found = False
        for assignment in view["assignments"]:
            if assignment["task"] == args.task:
                assignment["qualification_tier"] = args.tier
                found = True
        if not found:
            raise ValueError("task is not assigned to this view")
        view["revision"] += 1
        view["policy_change_reason"] = args.reason
        validate_view(view, load_tasks(root))
        atomic_json(root / "configs/forge/views" / f"{args.view}.json", view)
        from .knowledge import board
        emit(board(root, args.view))
    elif command == "cancel":
        queue.cancel(args.request_id)
        emit({"cancelled": args.request_id})
    elif command == "retry":
        queue.retry(args.compatibility_key, reason=args.reason)
        emit({"retry": args.compatibility_key})
    elif command in {"pause", "resume"}:
        queue.pause(args.campaign, command == "pause")
        emit({"campaign": args.campaign, "paused": command == "pause"})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Forge: {error}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("Forge: coordinator stopped claiming; active workers retain their deadlines and leases.", file=sys.stderr)
        raise SystemExit(130)

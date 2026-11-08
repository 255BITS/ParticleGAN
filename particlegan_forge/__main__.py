"""CLI entrypoint for `python -m particlegan_forge`."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys

from .contracts import json_safe
from .tiers import ForgeWorkspace


DEVICE_CHOICES = ("cpu", "cuda", "auto")
DEVICE_HELP = "Receipt cohort device; run, board and compile resolve it identically (default: cpu)"


def _emit(payload):
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="particlegan_forge Tier API CLI")
    p.add_argument("--root", type=Path, default=Path.cwd(), help="Project workspace root")
    p.add_argument("--queue-root", type=Path, default=None, help="Operational queue root (default: <root>/runs/particlegan_forge)")
    p.add_argument("--provider", type=str, default=None, help="Optional Python module registering adapters (e.g. particlestream.tier1)")

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="Validate configs/particlegan_forge tasks, views, and candidates")

    run_p = sub.add_parser("run", help="Run a candidate through a tier cap and compile reports")
    run_p.add_argument("candidate", help="Candidate ID")
    run_p.add_argument("--view", required=True, help="View ID")
    run_p.add_argument("--through-tier", type=int, default=1, help="Max qualification tier to execute")
    run_p.add_argument("--device", default="cpu", choices=DEVICE_CHOICES, help=DEVICE_HELP)

    board_p = sub.add_parser("board", help="Show qualification status for all candidates on a view")
    board_p.add_argument("--view", required=True, help="View ID")
    board_p.add_argument("--device", default="cpu", choices=DEVICE_CHOICES, help=DEVICE_HELP)

    comp_p = sub.add_parser("compile", help="Regenerate reports/particlegan_forge/{leaderboard,EXPERIMENTS_BY_TIER,EXPERIMENT_MEMORY}.md")
    comp_p.add_argument("--view", default=None, help="Primary view ID")
    comp_p.add_argument("--device", default="cpu", choices=DEVICE_CHOICES, help=DEVICE_HELP)

    retier_p = sub.add_parser("retier", help="Move a task to a new tier and archive the previous view revision")
    retier_p.add_argument("view", help="View ID")
    retier_p.add_argument("--task", required=True, help="Task ID")
    retier_p.add_argument("--tier", type=int, required=True, help="New tier integer (>= 1)")
    retier_p.add_argument("--reason", required=True, help="Policy change reason")

    try:
        args = p.parse_args(argv)
        root = args.root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"workspace root does not exist: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        registry = None
        if args.provider:
            mod = import_module(args.provider)
            registry = getattr(mod, "REGISTRY", None)

        ws = ForgeWorkspace(root, registry=registry, queue_root=args.queue_root)

        if args.command == "validate":
            _emit(ws.validate())
            return 0
        if args.command == "run":
            qual = ws.run_tier(args.candidate, args.view, through_tier=args.through_tier, device=args.device)
            ws.compile_reports(view_id=args.view, device=args.device)
            _emit(qual)
            return 0
        if args.command == "board":
            views = ws.load_views()
            if args.view not in views:
                raise KeyError(f"view {args.view!r} not found in workspace")
            cands = ws.load_candidates()
            _emit({cid: ws.qualify(cid, args.view, device=args.device) for cid in sorted(cands)})
            return 0
        if args.command == "compile":
            views = ws.load_views()
            if args.view and args.view not in views:
                raise KeyError(f"view {args.view!r} not found in workspace")
            _emit(ws.compile_reports(view_id=args.view, device=args.device))
            return 0
        if args.command == "retier":
            updated = ws.retier(args.view, task_id=args.task, new_tier=args.tier, reason=args.reason)
            _emit(updated.to_dict())
            return 0
        return 1
    except (KeyError, ValueError, FileNotFoundError, ModuleNotFoundError, RuntimeError) as err:
        err_msg = err.args[0] if isinstance(err, KeyError) and err.args else str(err)
        sys.stderr.write(f"Error: {err_msg}\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

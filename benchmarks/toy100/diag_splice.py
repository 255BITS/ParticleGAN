"""Record and splice ring-probe sample streams under ``--init hid_q``.

The ring gate is the unchanged K3P probe (``det_init_screen --gate ring``).
A seed offset still moves only sample and noise draws. This driver stores
those draws, then reruns the same probe while swapping tapes.

    python -u -m benchmarks.toy100.diag_splice record --seed-offset 0 --output /tmp/k3p-splice/rec/s0
    python -u -m benchmarks.toy100.diag_splice splice --tape-a /tmp/k3p-splice/rec/s0/splice-tape.pt \\
        --tape-b /tmp/k3p-splice/rec/s101/splice-tape.pt --switch 400 --streams all \\
        --output /tmp/k3p-splice/runs/k400

``--switch K`` uses tape A on updates ``[0, K)`` and tape B after that.
``--window A:B`` uses tape B only on ``[A, B)``. Updates are 0-based;
completed step ``t`` is the end of update ``t - 1``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONUNBUFFERED": "1",
}


def _run(output: Path, seed_offset: int, extra: dict) -> dict:
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.parent / f"{output.name}.log"
    env = os.environ.copy()
    env.update(THREAD_ENV)
    env["K3P_SPLICE_HOOK"] = "benchmarks.toy100.splice_hook"
    env["K3P_SPLICE_DIR"] = str(output)
    env.update(extra)
    cmd = [sys.executable, "-u", "-m", "benchmarks.toy100.det_init_screen",
           "--gate", "ring", "--init", "hid_q", "--seed-offset", str(seed_offset),
           "--output", str(output), "--log", str(log)]
    done = subprocess.run(cmd, cwd=ROOT, env=env)
    result_path = output / "result.json"
    if not result_path.exists():
        raise SystemExit(f"probe wrote no result (exit {done.returncode}); see {log}")
    payload = json.loads(result_path.read_text())
    live = (payload.get("result") or {}).get("live") or {}
    conv = (payload.get("result") or {}).get("convergence") or {}
    row = {
        "status": payload.get("status"),
        "modes": live.get("modes"),
        "hq": live.get("hq"),
        "passing_suffix": conv.get("passing_suffix"),
        "stable_from_step": conv.get("stable_from_step"),
        "first_pass_step": conv.get("first_pass_step"),
        "seconds": payload.get("seconds"),
        "returncode": done.returncode,
        "output": str(output),
        "log": str(log),
    }
    print(json.dumps(row), flush=True)
    if payload.get("status") == "ERROR":
        raise SystemExit(payload.get("error") or "probe ERROR")
    return row


def _record(args: argparse.Namespace) -> None:
    _run(args.output, args.seed_offset, {
        "K3P_SPLICE_MODE": "record",
        "K3P_SPLICE_DENSE": str(args.dense),
    })


def _replay_env(args: argparse.Namespace) -> dict:
    env = {
        "K3P_SPLICE_MODE": "replay",
        "K3P_SPLICE_TAPE_A": str(args.tape_a),
        "K3P_SPLICE_TAPE_B": str(args.tape_b),
        "K3P_SPLICE_STREAMS": args.streams,
        "K3P_SPLICE_DENSE": str(args.dense),
    }
    if args.window is not None:
        env["K3P_SPLICE_WINDOW"] = args.window
    else:
        env["K3P_SPLICE_SWITCH"] = str(args.switch)
    return env


def _splice(args: argparse.Namespace) -> None:
    _run(args.output, args.seed_offset, _replay_env(args))


def _grid(args: argparse.Namespace) -> None:
    switches = [int(part) for part in args.switches.split(",") if part.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = args.output / "grid.jsonl"
    for switch in switches:
        run_args = argparse.Namespace(
            tape_a=args.tape_a, tape_b=args.tape_b, streams=args.streams,
            dense=0, window=None, switch=switch, seed_offset=args.seed_offset,
            output=args.output / f"k{switch}",
        )
        row = _run(run_args.output, args.seed_offset, _replay_env(run_args))
        row.update(switch=switch, streams=args.streams,
                   tape_a=str(args.tape_a), tape_b=str(args.tape_b))
        with ledger.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    record = sub.add_parser("record", help="run one offset and store its sample tape")
    record.add_argument("--seed-offset", type=int, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--dense", type=int, default=0)
    record.set_defaults(func=_record)

    splice = sub.add_parser("splice", help="rerun the probe on a spliced tape")
    splice.add_argument("--tape-a", type=Path, required=True)
    splice.add_argument("--tape-b", type=Path, required=True)
    splice.add_argument("--output", type=Path, required=True)
    splice.add_argument("--switch", type=int, default=10**9,
                        help="first 0-based update that reads tape B")
    splice.add_argument("--window", default=None, help="A:B uses tape B only on that half-open interval")
    splice.add_argument("--streams", default="all")
    splice.add_argument("--dense", type=int, default=0)
    splice.add_argument("--seed-offset", type=int, default=0,
                        help="process seed; hid_q replay should stay at 0")
    splice.set_defaults(func=_splice)

    grid = sub.add_parser("grid", help="splice a list of switch points into one folder")
    grid.add_argument("--tape-a", type=Path, required=True)
    grid.add_argument("--tape-b", type=Path, required=True)
    grid.add_argument("--output", type=Path, required=True)
    grid.add_argument("--switches", required=True, help="comma-separated 0-based switch points")
    grid.add_argument("--streams", default="all")
    grid.add_argument("--seed-offset", type=int, default=0)
    grid.set_defaults(func=_grid)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

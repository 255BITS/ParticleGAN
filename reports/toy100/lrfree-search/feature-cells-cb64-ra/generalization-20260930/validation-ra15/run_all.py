"""Run every original portability, static-native and moving gate in one serial lane."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from freeze import ROOT, verify

PORTS = ("mode_hold", "img_intensity2", "img_blobs4", "img_bars4", "vector_unequal_mass",
         "img_stripes2", "vector_two_broad", "vector_unequal_width", "vector_anisotropic",
         "vector_overlap", "vector_spiral", "ring_shift", "stationary")
NATIVE = ("grid100", "rotated100", "staggered100")


def emit(**event):
    print(json.dumps(event, sort_keys=True), flush=True)


def execute(script, task, log):
    command = [sys.executable, "-u", "-B", str(ROOT / script), "--task", task]
    emit(event="start", script=script, task=task, log=str(log))
    with log.open("x") as output:
        code = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT).returncode
    emit(event="end", script=script, task=task, returncode=code)
    if code:
        raise RuntimeError(f"{script} {task} failed; read {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("ports", "moving", "native", "all"), default="all")
    args = parser.parse_args()
    verify()
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    scoreboard = ROOT / f"scoreboard-{args.group}.json"
    assert not scoreboard.exists(), "retain completed and interrupted attempts"
    board = dict(status="RUNNING", group=args.group, started=time.time(), results=[], integrity=verify())
    plan = []
    if args.group in ("all", "ports"):
        plan += [("screen", task) for task in PORTS]
    if args.group in ("all", "moving"):
        plan += [("moving", task) for task in NATIVE]
    if args.group in ("all", "native"):
        plan += [("screen", task) for task in NATIVE]
    for kind, task in plan:
        board["current"] = dict(kind=kind, task=task)
        scoreboard.write_text(json.dumps(board, indent=2) + "\n")
        try:
            if kind == "screen":
                execute("run_screen.py", task, logs / f"screen-{task}.log")
                execute("collect.py", task, logs / f"collect-{task}.log")
                receipt = json.loads((ROOT / "runs" / task / "acceptance-receipt.json").read_text())
                status = receipt["acceptance_status"]
            else:
                execute("run_moving.py", task, logs / f"moving-{task}.log")
                receipt = json.loads((ROOT / "moving" / task / "COMPLETION.json").read_text())
                status = receipt["quality_status"]
                assert receipt["source_integrity_after"]["status"] == "VALID"
            board["results"].append(dict(kind=kind, task=task, status=status))
            emit(event="verdict", kind=kind, task=task, status=status)
        except Exception as error:
            board.update(status="ERROR", error=repr(error), completed=time.time())
            scoreboard.write_text(json.dumps(board, indent=2) + "\n")
            raise
    board.update(status="PASS" if all(row["status"] == "PASS" for row in board["results"]) else "FAIL",
                 completed=time.time(), integrity_after=verify())
    board.pop("current", None)
    scoreboard.write_text(json.dumps(board, indent=2) + "\n")
    emit(event="complete", group=args.group, status=board["status"], results=board["results"])


if __name__ == "__main__":
    main()

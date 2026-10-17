"""Replay pinned proposal reproduction sources against the current develop API.

Download or extract the proposal's .py/.json files into --source first. Output
belongs outside Git. Each process owns one proposal and its observation hooks.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import traceback

from .capture import Capture, write


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--archived-recipe", action="store_true", help="Explicitly restore the proposal's GAN-v3 recipe factory, removed from the public API")
    args = ap.parse_args()
    destination = args.output / "proposal"
    shutil.copytree(args.source, destination, dirs_exist_ok=True)
    scripts = list(destination.rglob("reproduce_arms.py"))
    if len(scripts) != 1:
        raise ValueError("expected exactly one proposal reproduction script")
    capture = Capture(args.output)
    # Import the current package before historical scripts edit sys.path.
    import particlegan
    import torch
    torch.set_num_threads(1)
    receipt = dict(package=str(Path(particlegan.__file__).resolve()), torch=torch.__version__,
                   python=sys.version, source=str(args.source), instrument="existing evaluator observations; no interpolated clouds")
    if args.archived_recipe:
        original = scripts[0].read_text()
        tree = ast.parse(original)
        replacements = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "particlegan" and any(a.name == "get_recipe" for a in node.names):
                remaining = [a.name+(" as "+a.asname if a.asname else "") for a in node.names if a.name != "get_recipe"]
                replacement = ("from particlegan import "+", ".join(remaining)+"\n" if remaining else "")
                replacement += "from benchmarks.gan_v3 import gan_v3_recipe as get_recipe\n"
                replacements.append((node.lineno-1, node.end_lineno, replacement))
        if len(replacements) != 1:
            raise ValueError("expected one explicit historical recipe import")
        lines = original.splitlines(keepends=True)
        for start, stop, replacement in reversed(replacements):
            lines[start:stop] = [replacement]
        adapted = "".join(lines)
        scripts[0].write_text(adapted)
        receipt["adapter"] = dict(change="Import archived GAN-v3 factory explicitly instead of removed public historical recipe fields",
                                   original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                                   adapted_sha256=hashlib.sha256(adapted.encode()).hexdigest())
    try:
        with capture.installed():
            runpy.run_path(str(scripts[0]), run_name="__main__")
        receipt["status"] = "COMPLETE"
    except SystemExit as exc:
        # Historical repro scripts exit 1 when FAIL/PASS is not obtained.
        # Completed evaluator evidence remains complete, even for that failed
        # scientific contrast. Preserve its process verdict separately.
        receipt["proposal_exit_code"] = exc.code
        if capture.records and exc.code in (None, 0, 1):
            receipt["status"] = "COMPLETE"
        else:
            receipt.update(status="ERROR", error=traceback.format_exc())
            print(receipt["error"], flush=True)
    except Exception:
        receipt.update(status="ERROR", error=traceback.format_exc())
        print(receipt["error"], flush=True)
    receipt["episodes"] = len(capture.records)
    write(args.output / "replay.json", receipt)
    return int(receipt["status"] == "ERROR")


if __name__ == "__main__":
    raise SystemExit(main())

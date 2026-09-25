"""Reproduce only the original locked_shared suite row, excluding cover posture.

python -m benchmarks.locked_shared.suite_reference --reference /path/to/conceptmod

This optional audit needs conceptmod's own dependencies (including PEFT >=0.21).
It calls the reference suite; it does not copy its configuration gates here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch

from .reference import COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/locked_shared"))
    args = parser.parse_args()
    root = args.reference.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        parser.error(f"checkout reference revision {COMMIT}; got {revision}")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", "conceptmod"], text=True)
    if dirty:
        parser.error("reference conceptmod source has local changes")
    sys.path.insert(0, str(root))
    from conceptmod.toys.suite_leaderboard import TOY_CATALOG, default_candidates

    torch.set_num_threads(1)
    candidate = next(c for c in default_candidates() if c.name == "locked_shared")
    report = {"candidate": candidate.name, "conceptmod_commit": revision,
              "python": platform.python_version(), "torch": torch.__version__,
              "peft": importlib.metadata.version("peft"), "cells": []}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, kind, scorer in TOY_CATALOG:
        if kind == "posture":
            continue
        print(f"START locked_shared {name}", flush=True)
        try:
            cell = asdict(scorer(candidate))
        except Exception as exc:
            cell = {"config": candidate.name, "toy": name, "kind": kind,
                    "verdict": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}
        report["cells"].append(cell)
        (args.output / "suite_locked.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(cell), flush=True)
    counts = Counter(c["verdict"] for c in report["cells"])
    summary = ", ".join(f"{n} {v}" for v, n in sorted(counts.items()))
    lines = ["# Original conceptmod suite: locked_shared only", "", f"**{summary}. Cover-posture columns excluded.**", "",
             "| Toy | Original suite result | Evidence |", "| --- | --- | --- |"]
    for cell in report["cells"]:
        reason = str(cell["reason"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {cell['toy']} | **{cell['verdict']}** | {reason} |")
    lines += ["", "These are the original suite scorer results, not new ParticleGAN acceptance gates. "
              "The original suite mixes measured behavior with identity, selection and DSL checks; "
              "this audit reproduces that row without importing its gate implementations into ParticleGAN.", "",
              "The original suite converts any `mode_hold` result other than PASS, "
              "including INCONCLUSIVE, to **FAIL**. "
              "The standalone behavioral leaderboard preserves INCONCLUSIVE.", "",
              "The table supplied in the comparison had two extra cover-posture columns. "
              "The current reference removed those columns without changing the other toy implementations or scorer logic.", "",
              f"Reference: `{revision}`. Python {report['python']}; PyTorch {report['torch']}; PEFT {report['peft']}. CPU, seed 0, one thread.", "",
              "The initial PEFT 0.20 run could not import `NoMatchingPeftModuleError` for `path_suffix_lora`. "
              "PEFT 0.21 resolves that dependency error; no toy code or thresholds are changed.", "",
              "Reproduce: `python -m benchmarks.locked_shared.suite_reference --reference /path/to/conceptmod`. "
              "Full results: [suite_locked.json](suite_locked.json). The command exits nonzero on FAIL or ERROR.", ""]
    (args.output / "suite_locked.md").write_text("\n".join(lines))
    print(summary, flush=True)
    return 1 if counts["FAIL"] or counts["ERROR"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
